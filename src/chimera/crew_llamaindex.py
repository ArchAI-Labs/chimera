########### SUPPRESS WARNINGS - MUST BE FIRST #####################
import os
import sys
import warnings

# Set environment variables BEFORE any imports
os.environ['PYDANTIC_SKIP_VALIDATING_CORE_SCHEMAS'] = '1'
os.environ['PYTHONWARNINGS'] = 'ignore::DeprecationWarning'

# Aggressive warning suppression
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Suppress Pydantic-specific warnings
try:
    from pydantic import warnings as pydantic_warnings
    pydantic_warnings.PydanticDeprecatedSince20 = type('PydanticDeprecatedSince20', (UserWarning,), {})
    pydantic_warnings.PydanticDeprecatedSince211 = type('PydanticDeprecatedSince211', (UserWarning,), {})
except ImportError:
    pass

try:
    from pydantic.warnings import PydanticDeprecatedSince20, PydanticDeprecatedSince211
    warnings.filterwarnings("ignore", category=PydanticDeprecatedSince20)
    warnings.filterwarnings("ignore", category=PydanticDeprecatedSince211)
except ImportError:
    pass

# Additional filters
warnings.filterwarnings("ignore", message=".*Pydantic.*")
warnings.filterwarnings("ignore", message=".*__fields__.*")
warnings.filterwarnings("ignore", message=".*__fields_set__.*")
warnings.filterwarnings("ignore", message=".*model_computed_fields.*")
warnings.filterwarnings("ignore", message=".*model_fields.*")
warnings.filterwarnings(
    "ignore",
    message=r".*`duckduckgo_search`\) has been renamed to `ddgs`.*",
    category=RuntimeWarning,
)

# Monkey-patch warnings to be even more aggressive
_original_warn = warnings.warn
def _silent_warn(message, category=UserWarning, stacklevel=1):
    msg_str = str(message).lower()
    if any(x in msg_str for x in ['pydantic', '__fields__', 'deprecated', 'model_fields']):
        return
    _original_warn(message, category, stacklevel)

warnings.warn = _silent_warn

####################


from typing import Dict, Any, Optional
import yaml
import requests
from dotenv import load_dotenv

# --- LlamaIndex imports (with fallbacks for version differences) ---
try:
    # Newer LlamaIndex (0.10+)
    from llama_index.core.agent.workflow import ReActAgent
except ImportError:
    from llama_index.core.agent import ReActAgent

from llama_index.core.tools import FunctionTool, QueryEngineTool
from llama_index.core.workflow import (
    Workflow,
    StartEvent,
    StopEvent,
    step,
    Event,
    Context,
)
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.llms import ChatMessage

# -----------------------------------------------------------
# Local imports + fallbacks
# -----------------------------------------------------------
try:
    from utils.utils import print_output, check_memory_dir, LLM_Config
    from utils.storage_config import (
        get_long_term_memory,
        get_short_term_memory,
        get_entity_memory,
    )
    from tools.duckduckgo_tool import MyCustomDuckDuckGoTool
    from tools.dalle_tool import download_image_tool, DallETool
    from tools.qdrant_tool import search_knowledge, upsert_knowledge
    from utils.storage_qdrant import QdrantStorage
except ImportError:
    # Fallbacks when utils module doesn't exist
    def print_output(*args, **kwargs):
        """Fallback print output"""
        print(*args, **kwargs)

    def check_memory_dir():
        """Ensure memory directory exists"""
        os.makedirs("memory", exist_ok=True)

    def LLM_Config(provider, model, base_url=None, temperature=0.7,
               max_tokens=2000, timeout=60, callbacks=None):
        if provider == "openai":
            from llama_index.llms.openai import OpenAI
            return OpenAI(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=os.getenv("OPENAI_API_KEY"),
                timeout=timeout,
            )
        elif provider == "anthropic":
            from llama_index.llms.anthropic import Anthropic
            return Anthropic(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=os.getenv("ANTHROPIC_API_KEY"),
                timeout=timeout,
            )
        elif provider == "ollama":
            from llama_index.llms.ollama import Ollama
            safe_model = model or "llama3.1:8b"
            # Use a reasonable context size (4096 or 8192)
            num_ctx = int(os.getenv("NUM_CTX", "4096"))
            return Ollama(
                model=safe_model,
                base_url=base_url or "http://localhost:11434",
                temperature=temperature,
                request_timeout=float(timeout),
                context_window=num_ctx,
                additional_kwargs={
                    "num_ctx": num_ctx,
                    "num_gpu": 33
                }
            )
        else:
            from llama_index.llms.ollama import Ollama
            return Ollama(
                model="llama3.1:8b",
                context_window=4096,
                additional_kwargs={"num_ctx": 4096}
            )

    from tools.duckduckgo_tool import MyCustomDuckDuckGoTool
    from tools.dalle_tool import download_image_tool, DallETool
    from tools.qdrant_tool import search_knowledge, upsert_knowledge

    # Mock storage config functions
    def get_long_term_memory():
        return None

    def get_short_term_memory():
        return None

    def get_entity_memory():
        return None

    class QdrantStorage:
        pass

# -----------------------------------------------------------
# Environment loading
# -----------------------------------------------------------
load_dotenv()


# -----------------------------------------------------------
# Custom Events for Workflow
# -----------------------------------------------------------
class EditorialPlanEvent(Event):
    """Event containing editorial plan results"""
    result: str


class ContentGenerationEvent(Event):
    """Event containing generated content"""
    result: str
    content_type: str
    editorial_plan: str  # Pass editorial_plan forward


class PostWritingEvent(Event):
    """Event containing written post"""
    result: str
    editorial_plan: str
    generated_content: str


class VisualsEvent(Event):
    """Event containing visual assets info"""
    result: str
    editorial_plan: str
    generated_content: str
    linkedin_posts: str


class PlanningCompleteEvent(Event):
    """Event indicating planning is complete"""
    result: str


class LinkedInCrew:
    """
    Fully automated LinkedIn content creation pipeline using
    LlamaIndex agents and workflows.
    """

    def __init__(self, inputs: Optional[Dict[str, Any]] = None):
        self.inputs = inputs if inputs is not None else {}
        print(f"USER INPUTS RECEIVED: {self.inputs}")

        # Ensure memory directories exist
        check_memory_dir()

        # Load configurations
        self.agents_config = self._load_yaml_config("config/agents.yaml")
        self.tasks_config = self._load_yaml_config("config/tasks.yaml")

        # Normalize env (manager model falls back to main MODEL if not set)
        provider = os.getenv("PROVIDER") or "openai"
        model = os.getenv("MODEL")
        manager_model = os.getenv("MANAGER_MODEL") or model
        base_url = os.getenv("BASE_URL")
        temperature = float(os.getenv("TEMPERATURE", "0.7"))
        max_tokens = int(os.getenv("MAX_TOKENS", "2000"))
        timeout = float(os.getenv("TIMEOUT", "60"))

        # Initialize LLM configurations (will sanitize Ollama names if needed)
        self.llm = self._create_llm(
            provider=provider,
            model=model,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

        self.manager_llm = self._create_llm(
            provider=provider,
            model=manager_model,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

        # Initialize tools & agents
        self._initialize_tools()
        self._test_tools()  # Test tools before using them
        self._initialize_agents()

        # Create workflow
        workflow_inputs = {
            **self.inputs,
            "agents_config": self.agents_config,
            "tasks_config": self.tasks_config,
        }
        self.workflow = LinkedInWorkflow(
            agents=self.agents,
            llm=self.llm,  # Pass direct LLM access
            manager_llm=self.manager_llm,
            inputs=workflow_inputs,
            tools_working=self.tools_working,  # Pass tool status
            timeout=None,
            verbose=True,
        )

    def _load_yaml_config(self, config_path: str) -> Dict[str, Any]:
        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Warning: Config file {config_path} not found. Using defaults.")
            return {}

    def _create_llm(
        self,
        provider: str,
        model: Optional[str],
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        timeout: float = 60.0,
    ):
        return LLM_Config(
            provider=provider,
            model=model,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            callbacks=[print_output],
        )

    def _test_tools(self):
        """Test if tools are working properly"""
        self.tools_working = {
            "web_search": False,
            "file_writer": True,  # File writer should always work
            "knowledge_base": False,
        }
        
        print("\n🔧 Testing tools...")
        
        # Test web search
        try:
            custom_search = MyCustomDuckDuckGoTool()
            result = custom_search.run("test query")
            if result and len(str(result)) > 10:
                self.tools_working["web_search"] = True
                print("✅ Web search tool: WORKING")
            else:
                print("⚠️ Web search tool: NOT WORKING (empty results)")
        except Exception as e:
            print(f"⚠️ Web search tool: NOT WORKING ({str(e)[:50]})")
        
        # Test knowledge base
        try:
            result = search_knowledge("test")
            self.tools_working["knowledge_base"] = True
            print("✅ Knowledge base: WORKING")
        except Exception as e:
            print(f"⚠️ Knowledge base: NOT WORKING ({str(e)[:50]})")
        
        print("")

    def _initialize_tools(self):
        """Initialize all tools for agents"""
        # Web search tool
        if os.environ.get("SERPER_API_KEY"):
            try:
                from crewai_tools import SerperDevTool
                web_search_fn = SerperDevTool()
                self.web_search_tool = FunctionTool.from_defaults(
                    fn=lambda query: web_search_fn.run(query),
                    name="web_search",
                    description="Search the web for information on a given topic",
                )
            except ImportError:
                # Fallback to DuckDuckGo if SerperDevTool not available
                custom_search = MyCustomDuckDuckGoTool()
                self.web_search_tool = FunctionTool.from_defaults(
                    fn=lambda query: custom_search.run(query),
                    name="web_search",
                    description="Search the web for information on a given topic",
                )
        else:
            custom_search = MyCustomDuckDuckGoTool()
            self.web_search_tool = FunctionTool.from_defaults(
                fn=lambda query: custom_search.run(query),
                name="web_search",
                description="Search the web for information on a given topic",
            )

        # File writer tool
        def write_file(filename: str, content: str) -> str:
            try:
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(content)
                print(f"📝 Successfully wrote {len(content)} characters to {filename}")
                return f"Successfully wrote to {filename}"
            except Exception as e:
                print(f"❌ Error writing file: {str(e)}")
                return f"Error writing file: {str(e)}"

        self.file_writer_tool = FunctionTool.from_defaults(
            fn=write_file, name="file_writer", description="Write content to a file"
        )

        # DALL-E image generation tool (graceful no-op if no OPENAI_API_KEY)
        dalle = DallETool(model="dall-e-3", size="1024x1024", quality="standard", n=1)
        self.dalle_tool = FunctionTool.from_defaults(
            fn=lambda prompt: dalle.run(prompt),
            name="generate_image",
            description="Generate an image using DALL-E based on a text prompt",
        )

        # Image download tool
        self.download_image_tool = FunctionTool.from_defaults(
            fn=download_image_tool,
            name="download_image",
            description="Download an image from a URL",
        )

        # Web scraper tool
        try:
            from crewai_tools import ScrapeWebsiteTool
            scraper = ScrapeWebsiteTool()
            self.scraper_tool = FunctionTool.from_defaults(
                fn=lambda url: scraper.run(url),
                name="scrape_website",
                description="Scrape content from a website URL",
            )
        except ImportError:
            from bs4 import BeautifulSoup

            def simple_scraper(url: str) -> str:
                try:
                    response = requests.get(url, timeout=10)
                    soup = BeautifulSoup(response.content, "html.parser")
                    text = soup.get_text(separator="\n", strip=True)
                    return text[:5000]
                except Exception as e:
                    return f"Error scraping {url}: {str(e)}"

            self.scraper_tool = FunctionTool.from_defaults(
                fn=simple_scraper,
                name="scrape_website",
                description="Scrape content from a website URL",
            )

        # Knowledge base tools (RAG)
        self.search_knowledge_tool = FunctionTool.from_defaults(
            fn=search_knowledge,
            name="search_knowledge",
            description="Search the knowledge base for relevant information",
        )
        self.upsert_knowledge_tool = FunctionTool.from_defaults(
            fn=upsert_knowledge,
            name="upsert_knowledge",
            description="Add or update information in the knowledge base",
        )

    def _initialize_agents(self):
        """Initialize all agents with their respective tools and configurations"""
        self.agents = {}

        # Only initialize agents that NEED tools
        # Generalist Expert Agent (needs web search)
        self.agents["generalist_expert"] = ReActAgent(
            tools=[self.web_search_tool],
            llm=self.llm,
            memory=ChatMemoryBuffer.from_defaults(token_limit=4000),
            max_iterations=15,
            verbose=True,
        )

        # Product Expert Agent (uses RAG + Scraper)
        print("Activated agent: Product Expert (uses RAG + Scraper)")
        self.agents["product_expert"] = ReActAgent(
            tools=[
                self.search_knowledge_tool,
                self.upsert_knowledge_tool,
                self.scraper_tool,
            ],
            llm=self.llm,
            memory=ChatMemoryBuffer.from_defaults(token_limit=4000),
            max_iterations=20,
            verbose=True,
        )

        # Designer Agent (needs DALL-E)
        self.agents["designer"] = ReActAgent(
            tools=[self.dalle_tool, self.download_image_tool],
            llm=self.llm,
            memory=ChatMemoryBuffer.from_defaults(token_limit=4000),
            max_iterations=15,
            verbose=True,
        )

        # Planner Agent (needs file writer)
        self.agents["planner"] = ReActAgent(
            tools=[self.file_writer_tool],
            llm=self.llm,
            memory=ChatMemoryBuffer.from_defaults(token_limit=4000),
            max_iterations=10,
            verbose=True,
        )

    async def kickoff(self) -> Dict[str, Any]:
        print("Starting LinkedIn Content Creation Workflow...")
        result = await self.workflow.run()
        return result

    def run(self) -> Dict[str, Any]:
        import asyncio
        return asyncio.run(self.kickoff())


class LinkedInWorkflow(Workflow):
    """Workflow that orchestrates the LinkedIn content creation pipeline"""

    def __init__(
        self, 
        agents: Dict[str, ReActAgent], 
        llm,
        manager_llm,
        inputs: Dict[str, Any],
        tools_working: Dict[str, bool],
        **kwargs
    ):
        super().__init__(**kwargs)
        self.agents = agents
        self.llm = llm  # Direct LLM access for simple tasks
        self.manager_llm = manager_llm
        self.inputs = inputs
        self.tools_working = tools_working
        self.expert_type = inputs.get("expert_type", "generalista")
        
        # Use workflow instance to store data (version-agnostic)
        self.workflow_data = {}

    # ---------- Direct LLM call (no agent loop) ----------
    async def _call_llm(self, llm, prompt: str) -> str:
        """Call LLM directly without ReAct agent loop - for simple tasks"""
        print(f"\n🤖 Calling LLM directly...")
        print(f"📝 Prompt preview: {prompt[:150]}...")
        
        messages = [ChatMessage(role="user", content=prompt)]
        response = await llm.achat(messages)
        result = str(response.message.content)
        
        print(f"✅ LLM Response length: {len(result)} characters")
        print(f"📄 Preview: {result[:200]}...")
        return result

    # ---------- ReAct agent call (with tool use) ----------
    async def _ask_agent(self, agent: ReActAgent, prompt: str, max_iterations: int = 15):
        """
        Call a ReActAgent for tasks that need tools.
        """
        print(f"\n🔧 Calling ReActAgent (max_iterations={max_iterations})...")
        print(f"📝 Prompt preview: {prompt[:150]}...")
        
        try:
            # Try the new API with max_iterations parameter
            if hasattr(agent, "run"):
                handler = agent.run(prompt, max_iterations=max_iterations)
                result = await handler
                print(f"✅ Agent completed")
                print(f"📄 Result preview: {str(result)[:200]}...")
                return result
            if hasattr(agent, "aquery"):
                result = await agent.aquery(prompt)
                print(f"✅ Agent completed")
                return result
            if hasattr(agent, "achat"):
                result = await agent.achat(prompt)
                print(f"✅ Agent completed")
                return result
            raise AttributeError("Agent has no usable run/aquery/achat methods")
        except Exception as e:
            print(f"❌ Agent error: {str(e)[:100]}")
            raise

    @step
    async def editorial_plan(self, ctx: Context, ev: StartEvent) -> EditorialPlanEvent:
        print("\n" + "="*60)
        print("=== STEP 1: Editorial Planning ===")
        print("="*60)

        topic = self.inputs.get("topic", "AI and Technology")
        num_posts = self.inputs.get("num_posts", 5)
        frequency = self.inputs.get("frequency", "weekly")

        prompt = f"""You are an expert content strategist. Create an editorial plan for {num_posts} LinkedIn posts about {topic}.
The posts should be published {frequency}.

Include:
1. Post topics and angles
2. Key messages for each post
3. Target audience considerations
4. Content mix (educational, promotional, thought leadership)

Provide a detailed, structured editorial plan."""

        # Use direct LLM call - no tools needed for planning
        result = await self._call_llm(self.manager_llm, prompt)

        # Store in workflow instance
        self.workflow_data["editorial_plan"] = result
        print(f"\n✅ Editorial plan created: {len(result)} characters")
        return EditorialPlanEvent(result=result)

    @step
    async def content_generation(
        self, ctx: Context, ev: EditorialPlanEvent
    ) -> ContentGenerationEvent:
        print("\n" + "="*60)
        print("=== STEP 2: Content Generation ===")
        print("="*60)

        editorial_plan = ev.result
        topic = self.inputs.get("topic", "AI and Technology")

        # Check if we should try using tools or just use LLM
        if self.expert_type == "generalista" and self.tools_working.get("web_search"):
            print("⚠️ Web search not working - using direct LLM instead...")
            
        if self.expert_type == "prodotto" and self.tools_working.get("knowledge_base"):
            print("⚠️ Knowledge base not working - using direct LLM instead...")
        
        # ALWAYS use direct LLM for now since tools are having issues
        print("Using direct LLM for content generation...")
        prompt = f"""You are an expert content creator. Based on this editorial plan, generate detailed, engaging content about {topic}.

Editorial Plan:
{editorial_plan}

Generate comprehensive content covering:
- Industry trends and insights
- Technical explanations
- Best practices
- Real-world applications
- Examples and case studies

Format the output as structured, detailed content ready to be transformed into LinkedIn posts.
Make it substantive and valuable - at least 500 words of quality content."""

        result = await self._call_llm(self.llm, prompt)
        content_type = "direct_llm"

        # Store in workflow instance
        self.workflow_data["generated_content"] = result
        print(f"\n✅ Content generated: {len(result)} characters")
        
        return ContentGenerationEvent(
            result=result, 
            content_type=content_type,
            editorial_plan=editorial_plan
        )

    @step
    async def write_linkedin_post(
        self, ctx: Context, ev: ContentGenerationEvent
    ) -> PostWritingEvent:
        print("\n" + "="*60)
        print("=== STEP 3: Writing LinkedIn Post ===")
        print("="*60)

        content = ev.result
        editorial_plan = ev.editorial_plan

        prompt = f"""You are an expert LinkedIn copywriter. Transform this content into {self.inputs.get('num_posts', 5)} engaging, ready-to-publish LinkedIn posts.

Editorial Plan:
{editorial_plan[:500]}...

Generated Content:
{content}

Create {self.inputs.get('num_posts', 5)} separate LinkedIn posts, each one should:
- Start with a compelling hook in the first line
- Use short paragraphs (2-3 sentences max)
- Include 3-5 relevant hashtags at the end
- Have a clear call-to-action
- Be between 150-300 words
- Use emojis strategically (1-2 per post)
- Tell a story or provide unique value

Format as:

=== POST 1 ===
[First post content here]
[hashtags]

=== POST 2 ===
[Second post content here]
[hashtags]

And so on..."""

        # Use direct LLM - no tools needed for copywriting
        result = await self._call_llm(self.llm, prompt)

        # Store in workflow instance
        self.workflow_data["linkedin_posts"] = result
        print(f"\n✅ LinkedIn posts written: {len(result)} characters")
        
        return PostWritingEvent(
            result=result,
            editorial_plan=editorial_plan,
            generated_content=content
        )

    @step
    async def create_visuals(self, ctx: Context, ev: PostWritingEvent) -> VisualsEvent:
        print("\n" + "="*60)
        print("=== STEP 4: Creating Visual Assets ===")
        print("="*60)

        posts = ev.result

        # For now, just generate concepts without DALL-E since it may not be available
        prompt = f"""You are a visual designer. Based on these LinkedIn posts, create detailed visual design concepts.

LinkedIn Posts:
{posts[:1000]}

For each post, provide:
1. Visual concept description (what the image should show)
2. Color scheme (primary and secondary colors)
3. Design style (minimalist, bold, professional, etc.)
4. Key visual elements to include
5. Text overlay suggestions (if any)

Format as:

=== VISUAL 1 ===
Concept: [description]
Colors: [color scheme]
Style: [style]
Elements: [key elements]

=== VISUAL 2 ===
[and so on...]"""

        # Use direct LLM instead of agent since DALL-E might not be available
        result = await self._call_llm(self.llm, prompt)

        # Store in workflow instance
        self.workflow_data["visuals"] = result
        print(f"\n✅ Visual concepts created: {len(result)} characters")
        
        return VisualsEvent(
            result=result,
            editorial_plan=ev.editorial_plan,
            generated_content=ev.generated_content,
            linkedin_posts=posts
        )

    @step
    async def plan_posts(self, ctx: Context, ev: VisualsEvent) -> StopEvent:
        print("\n" + "="*60)
        print("=== STEP 5: Planning and Saving ===")
        print("="*60)

        # Retrieve from event
        editorial_plan = ev.editorial_plan
        content = ev.generated_content
        posts = ev.linkedin_posts
        visuals = ev.result

        # Create the final content plan
        final_plan = f"""# LinkedIn Content Calendar

## Editorial Plan
{editorial_plan}

---

## Generated Content
{content}

---

## LinkedIn Posts (Ready to Publish)
{posts}

---

## Visual Asset Concepts
{visuals}

---

## Publishing Schedule
- Frequency: {self.inputs.get('frequency', 'weekly')}
- Number of posts: {self.inputs.get('num_posts', 5)}
- Topic: {self.inputs.get('topic', 'AI and Technology')}

## Engagement Strategy
1. Post at optimal times (9 AM, 12 PM, or 5 PM on weekdays)
2. Respond to comments within 1 hour
3. Engage with relevant posts in your network
4. Track metrics: likes, comments, shares, profile views

## Success Metrics
- Target engagement rate: 3-5%
- Target reach: 1000+ impressions per post
- Target profile views: 50+ per week
"""

        # Write directly to file
        try:
            with open("linkedin_content_plan.md", "w", encoding="utf-8") as f:
                f.write(final_plan)
            print(f"📝 Saved content plan to linkedin_content_plan.md ({len(final_plan)} characters)")
            result = f"Successfully saved content plan with {len(final_plan)} characters"
        except Exception as e:
            print(f"❌ Error saving file: {str(e)}")
            result = f"Error saving file: {str(e)}"

        print("\n" + "="*60)
        print("✅ Content planning complete!")
        print("="*60)
        
        return StopEvent(
            result={
                "status": "success",
                "editorial_plan": editorial_plan,
                "generated_content": content,
                "linkedin_posts": posts,
                "visuals": visuals,
                "final_plan": result,
            }
        )