########### SUPPRESS WARNINGS - MUST BE REMOVED #####################
import os
import sys
import warnings

os.environ['PYDANTIC_SKIP_VALIDATING_CORE_SCHEMAS'] = '1'
os.environ['PYTHONWARNINGS'] = 'ignore::DeprecationWarning,ignore::ResourceWarning'

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ResourceWarning)

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

warnings.filterwarnings("ignore", message=".*Pydantic.*")
warnings.filterwarnings("ignore", message=".*__fields__.*")
warnings.filterwarnings("ignore", message=".*__fields_set__.*")
warnings.filterwarnings("ignore", message=".*model_computed_fields.*")
warnings.filterwarnings("ignore", message=".*model_fields.*")
warnings.filterwarnings("ignore", message=".*unclosed.*")
warnings.filterwarnings("ignore", message=".*socket.*")
warnings.filterwarnings("ignore", message=".*transport.*")
warnings.filterwarnings("ignore", message=r".*`duckduckgo_search`\) has been renamed to `ddgs`.*", category=RuntimeWarning)

_original_warn = warnings.warn
def _silent_warn(message, category=UserWarning, stacklevel=1):
    msg_str = str(message).lower()
    if any(x in msg_str for x in ['pydantic', '__fields__', 'deprecated', 'model_fields', 'unclosed', 'socket', 'transport']):
        return
    _original_warn(message, category, stacklevel)

warnings.warn = _silent_warn

###############################################

from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import yaml
import requests
import asyncio
from dotenv import load_dotenv

try:
    from llama_index.core.agent.workflow import ReActAgent
except ImportError:
    from llama_index.core.agent import ReActAgent

from llama_index.core.tools import FunctionTool
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
from llama_index.core.agent.workflow import AgentStream

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
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False
    
    def print_output(*args, **kwargs):
        print(*args, **kwargs)

    def check_memory_dir():
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
            safe_model = model or "qwen3:8b"
            num_ctx = int(os.getenv("NUM_CTX", "4096"))
            return Ollama(
                model=safe_model,
                base_url=base_url or "http://localhost:11434",
                thinking=False,
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
                model="qwen3:8b",
                context_window=4096,
                additional_kwargs={"num_ctx": 4096}
            )

    try:
        from tools.duckduckgo_tool import MyCustomDuckDuckGoTool
    except ImportError:
        class MyCustomDuckDuckGoTool:
            def run(self, query: str) -> str:
                return f"Search results for: {query} (tool not available)"

    try:
        from tools.dalle_tool import download_image_tool, DallETool
    except ImportError:
        def download_image_tool(url: str) -> str:
            return f"Would download image from: {url}"
        
        class DallETool:
            def __init__(self, **kwargs):
                pass
            def run(self, prompt: str) -> str:
                return f"Would generate image for: {prompt}"

    try:
        from tools.qdrant_tool import search_knowledge, upsert_knowledge
    except ImportError:
        def search_knowledge(query: str) -> str:
            return f"Search knowledge base for: {query} (not available)"
        
        def upsert_knowledge(data: str) -> str:
            return f"Would store in knowledge base: {data[:50]}..."

    def get_long_term_memory():
        return None

    def get_short_term_memory():
        return None

    def get_entity_memory():
        return None

    class QdrantStorage:
        pass

load_dotenv()


def ensure_utf8_path(p: Path) -> Path:
    """Ensure parent directories exist for a given path."""
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def save_text_utf8(path: Path, text: str) -> Path:
    """
    Write text as UTF-8 with LF newlines. Handles emoji and special characters.
    Never fails on encoding issues - replaces problematic characters if needed.
    Works reliably on Windows by forcing UTF-8 encoding.
    """
    path = ensure_utf8_path(path)
    
    path_str = str(path)
    
    try:
        with open(path_str, 'wb') as f:
            utf8_bytes = text.encode('utf-8', errors='replace')
            f.write(utf8_bytes)
    except Exception as e:
        print(f"⚠️  Warning saving to {path_str}: {e}")
        try:
            sanitized = text.encode("utf-8", errors="ignore").decode("utf-8")
            with open(path_str, 'wb') as f:
                f.write(sanitized.encode('utf-8'))
        except Exception as e2:
            print(f"❌ Error saving file {path_str}: {e2}")
            try:
                ascii_safe = text.encode("ascii", errors="ignore").decode("ascii")
                with open(path_str, 'w', encoding='utf-8') as f:
                    f.write(ascii_safe)
                print(f"⚠️  Saved ASCII-only version to {path_str}")
            except Exception as e3:
                print(f"❌ Complete failure saving {path_str}: {e3}")
                raise
    return path


def resolve_path(base_dir: Path, maybe_rel: str) -> Path:
    """Join relative paths under base_dir; leave absolute paths as-is."""
    p = Path(maybe_rel)
    return p if p.is_absolute() else (base_dir / p)



class EditorialPlanEvent(Event):
    result: str


class ContentGenerationEvent(Event):
    result: str
    content_type: str
    editorial_plan: str


class PostWritingEvent(Event):
    result: str
    editorial_plan: str
    generated_content: str


class VisualsEvent(Event):
    result: str
    editorial_plan: str
    generated_content: str
    linkedin_posts: str


class PlanningCompleteEvent(Event):
    result: str


class LinkedInCrew:
    """Main class for LinkedIn content creation workflow."""
    
    def __init__(self, inputs: Optional[Dict[str, Any]] = None):
        self.inputs = inputs if inputs is not None else {}
        print(f"USER INPUTS RECEIVED: {self.inputs}")

        requested_out = self.inputs.get("output_dir")
        if requested_out:
            self.output_dir = Path(requested_out)
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = Path("output") / f"run_{ts}"
        
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)
            print(f"📁 Created output dir: {self.output_dir}")
        else:
            print(f"📁 Using existing output dir: {self.output_dir}")

        check_memory_dir()

        self.agents_config = self._load_yaml_config("config/agents.yaml")
        self.tasks_config = self._load_yaml_config("config/tasks.yaml")

        provider = os.getenv("PROVIDER", "openai")
        model = os.getenv("MODEL")
        manager_model = os.getenv("MANAGER_MODEL") or model
        base_url = os.getenv("BASE_URL")
        temperature = float(os.getenv("TEMPERATURE", "0.7"))
        max_tokens = int(os.getenv("MAX_TOKENS", "2000"))
        timeout = float(os.getenv("TIMEOUT", "60"))

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

        self._initialize_tools()
        self._test_tools()
        self._initialize_agents()

        workflow_inputs = {
            **self.inputs,
            "agents_config": self.agents_config,
            "tasks_config": self.tasks_config,
            "output_dir": str(self.output_dir),
        }
        self.workflow = LinkedInWorkflow(
            agents=self.agents,
            llm=self.llm,
            manager_llm=self.manager_llm,
            inputs=workflow_inputs,
            tools_working=self.tools_working,
            timeout=None,
            verbose=True,
        )

    def _load_yaml_config(self, config_path: str) -> Dict[str, Any]:
        """Load YAML configuration file with error handling."""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"⚠️  Config file {config_path} not found. Using defaults.")
            return {}
        except Exception as e:
            print(f"⚠️  Error loading {config_path}: {e}. Using defaults.")
            return {}

    def _create_llm(self, provider: str, model: Optional[str], base_url: Optional[str] = None,
                    temperature: float = 0.7, max_tokens: int = 2000, timeout: float = 60.0):
        """Create LLM instance with proper configuration."""
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
        """Test which tools are working properly."""
        self.tools_working = {
            "web_search": False,
            "file_writer": True,
            "knowledge_base": False,
        }
        
        print("\n🔧 Testing tools...")
        
        try:
            custom_search = MyCustomDuckDuckGoTool()
            result = custom_search.run("test query")
            if result and len(str(result)) > 10:
                self.tools_working["web_search"] = True
                print("✅ Web search tool: WORKING")
            else:
                print("⚠️  Web search tool: NOT WORKING (empty results)")
        except Exception as e:
            print(f"⚠️  Web search tool: NOT WORKING ({str(e)[:50]})")
        
        try:
            result = search_knowledge("test")
            self.tools_working["knowledge_base"] = True
            print("✅ Knowledge base: WORKING")
        except Exception as e:
            print(f"⚠️  Knowledge base: NOT WORKING ({str(e)[:50]})")
        
        print("")

    def _initialize_tools(self):
        """Initialize all tools used by agents."""
        
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

        def write_file(filename: str, content: str) -> str:
            """Write content to file with UTF-8 encoding."""
            try:
                dest = resolve_path(self.output_dir, filename)
                save_text_utf8(dest, content)
                print(f"📝 Successfully wrote {len(content)} chars to {dest}")
                return f"Successfully wrote to {dest}"
            except Exception as e:
                print(f"❌ Error writing file: {e}")
                return f"Error writing file: {e}"

        self.file_writer_tool = FunctionTool.from_defaults(
            fn=write_file,
            name="file_writer",
            description="Write content to a file with UTF-8 encoding"
        )

        dalle = DallETool(model="dall-e-3", size="1024x1024", quality="standard", n=1)
        self.dalle_tool = FunctionTool.from_defaults(
            fn=lambda prompt: dalle.run(prompt),
            name="generate_image",
            description="Generate an image using DALL-E based on a text prompt",
        )

        self.download_image_tool = FunctionTool.from_defaults(
            fn=download_image_tool,
            name="download_image",
            description="Download an image from a URL",
        )

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
                """Simple web scraper fallback."""
                try:
                    response = requests.get(url, timeout=10)
                    response.raise_for_status()
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
        """Initialize all ReAct agents."""
        self.agents = {}

        self.agents["generalist_expert"] = ReActAgent(
            tools=[self.web_search_tool],
            llm=self.llm,
            memory=ChatMemoryBuffer.from_defaults(token_limit=4000),
            max_iterations=15,
            verbose=True,
        )

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

        self.agents["designer"] = ReActAgent(
            tools=[self.dalle_tool, self.download_image_tool],
            llm=self.llm,
            memory=ChatMemoryBuffer.from_defaults(token_limit=4000),
            max_iterations=15,
            verbose=True,
        )

        self.agents["planner"] = ReActAgent(
            tools=[self.file_writer_tool],
            llm=self.llm,
            memory=ChatMemoryBuffer.from_defaults(token_limit=4000),
            max_iterations=10,
            verbose=True,
        )

    def cleanup(self):
        """Cleanup resources and close all connections."""
        try:
            if hasattr(self.llm, 'close'):
                self.llm.close()
            if hasattr(self.manager_llm, 'close'):
                self.manager_llm.close()
            
            for agent_name, agent in self.agents.items():
                try:
                    if hasattr(agent, 'close'):
                        agent.close()
                    if hasattr(agent, 'llm') and hasattr(agent.llm, 'close'):
                        agent.llm.close()
                except Exception:
                    pass

            import gc
            gc.collect()
            
        except Exception as e:
            print(f"⚠️  Warning during cleanup: {e}")

    async def kickoff(self) -> Dict[str, Any]:
        """Run the workflow asynchronously."""
        print("\n" + "="*60)
        print("Starting LinkedIn Content Creation Workflow...")
        print("="*60)
        try:
            result = await self.workflow.run()
            return result
        finally:
            self.cleanup()
            await asyncio.sleep(0.1)

    def run(self) -> Dict[str, Any]:
        """Run the workflow synchronously."""
        try:
            return asyncio.run(self.kickoff())
        finally:
            self.cleanup()


class LinkedInWorkflow(Workflow):
    """Workflow for creating LinkedIn content."""
    
    def __init__(self, agents: Dict[str, ReActAgent], llm, manager_llm,
                 inputs: Dict[str, Any], tools_working: Dict[str, bool], **kwargs):
        super().__init__(**kwargs)
        self.agents = agents
        self.llm = llm
        self.manager_llm = manager_llm
        self.inputs = inputs
        self.tools_working = tools_working
        self.expert_type = inputs.get("expert_type", "generalista")
        self.workflow_data = {}
        self.output_dir = Path(self.inputs.get("output_dir", "output"))

    async def _call_llm(self, llm, prompt: str) -> str:
        """Call LLM directly without agent."""
        print(f"\n🤖 Calling LLM directly...")
        print(f"📝 Prompt preview: {prompt[:150]}...")
        
        messages = [ChatMessage(role="user", content=prompt)]
        response = await llm.achat(messages)
        result = str(response.message.content)
        
        print(f"✅ LLM Response length: {len(result)} characters")
        print(f"📄 Preview: {result[:200]}...")
        return result

    async def _ask_agent_with_fallback(self, agent: ReActAgent, prompt: str, 
                                       max_iterations: int = 15, fallback_llm=None) -> str:
        """Call agent with fallback to direct LLM on max iterations error."""
        print(f"\n🔧 Calling ReActAgent (max_iterations={max_iterations})...")
        print(f"📝 Prompt preview: {prompt[:150]}...")
        
        try:
            if hasattr(agent, "run"):
                handler = agent.run(prompt, max_iterations=max_iterations)
                async for ev in handler.stream_events():
                    if isinstance(ev, AgentStream):
                        print(f"{ev.delta}", end="", flush=True)
                result = await handler
                print(f"✅ Agent completed")
                print(f"📄 Result preview: {str(result)[:200]}...")
                return str(result)
            elif hasattr(agent, "aquery"):
                result = await agent.aquery(prompt)
                print(f"✅ Agent completed")
                return str(result)
            elif hasattr(agent, "achat"):
                result = await agent.achat(prompt)
                print(f"✅ Agent completed")
                return str(result)
            else:
                raise AttributeError("Agent has no usable run/aquery/achat methods")
        except Exception as e:
            error_msg = str(e)
            if "Max iterations" in error_msg or "WorkflowRuntimeError" in error_msg:
                print(f"⚠️  Agent hit max iterations - falling back to direct LLM...")
                if fallback_llm:
                    return await self._call_llm(fallback_llm, prompt)
                else:
                    print(f"❌ No fallback LLM provided")
                    raise
            else:
                print(f"❌ Agent error: {error_msg[:100]}")
                raise

    @step
    async def editorial_plan(self, ctx: Context, ev: StartEvent) -> EditorialPlanEvent:
        """Step 1: Create editorial plan."""
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

        result = await self._call_llm(self.manager_llm, prompt)

        self.workflow_data["editorial_plan"] = result
        print(f"\n✅ Editorial plan created: {len(result)} characters")
        
        save_text_utf8(self.output_dir / "editorial_plan.md", result)
        
        return EditorialPlanEvent(result=result)

    @step
    async def content_generation(self, ctx: Context, ev: EditorialPlanEvent) -> ContentGenerationEvent:
        """Step 2: Generate content based on editorial plan."""
        print("\n" + "="*60)
        print("=== STEP 2: Content Generation ===")
        print("="*60)

        editorial_plan = ev.result
        topic = self.inputs.get("topic", "AI and Technology")

        if self.expert_type == "generalista":
            if self.tools_working.get("web_search"):
                print("✅ Using Generalist Expert with web search...")
                prompt = f"""Based on this editorial plan, generate detailed technical content about {topic}.

Editorial Plan:
{editorial_plan}

Your task:
1. Use the web_search tool to find current information about {topic}
2. Generate comprehensive technical content covering:
   - Industry trends and insights
   - Technical explanations
   - Best practices
   - Real-world applications

Search for recent information and incorporate it into your response.
Format the output as structured, detailed content ready to be transformed into LinkedIn posts."""
                
                agent = self.agents["generalist_expert"]
                result = await self._ask_agent_with_fallback(
                    agent, prompt, max_iterations=15, fallback_llm=self.llm
                )
                content_type = "technical_with_search"
            else:
                print("⚠️  Web search not working - using direct LLM instead...")
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
                content_type = "technical_direct"

        elif self.expert_type == "prodotto":
            if self.tools_working.get("knowledge_base"):
                print("✅ Using Product Expert with knowledge base...")
                prompt = f"""Based on this editorial plan, generate product-focused content about {topic}.

Editorial Plan:
{editorial_plan}

Your task:
1. Use the search_knowledge tool to find relevant product information
2. Generate product-focused content covering:
   - Product features and benefits
   - Use cases and success stories
   - Competitive advantages
   - Customer pain points and solutions

Format the output as structured content ready for LinkedIn posts."""
                
                agent = self.agents["product_expert"]
                result = await self._ask_agent_with_fallback(
                    agent, prompt, max_iterations=20, fallback_llm=self.llm
                )
                content_type = "product_with_kb"
            else:
                print("⚠️  Knowledge base not working - using direct LLM instead...")
                prompt = f"""You are an expert content creator. Based on this editorial plan, generate detailed product-focused content about {topic}.

Editorial Plan:
{editorial_plan}

Generate comprehensive content covering:
- Product features and benefits
- Use cases and success stories
- Competitive advantages
- Customer pain points and solutions
- Examples and case studies

Format the output as structured, detailed content ready to be transformed into LinkedIn posts.
Make it substantive and valuable - at least 500 words of quality content."""

                result = await self._call_llm(self.llm, prompt)
                content_type = "product_direct"
        else:
            print("Using direct LLM (no expert type specified)...")
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
            content_type = "general"

        self.workflow_data["generated_content"] = result
        print(f"\n✅ Content generated ({content_type}): {len(result)} characters")
        
        save_text_utf8(self.output_dir / "generated_content.md", result)
        
        return ContentGenerationEvent(
            result=result, 
            content_type=content_type,
            editorial_plan=editorial_plan
        )

    @step
    async def write_linkedin_post(self, ctx: Context, ev: ContentGenerationEvent) -> PostWritingEvent:
        """Step 3: Transform content into LinkedIn posts."""
        print("\n" + "="*60)
        print("=== STEP 3: Writing LinkedIn Post ===")
        print("="*60)

        content = ev.result
        editorial_plan = ev.editorial_plan
        num_posts = self.inputs.get('num_posts', 5)

        prompt = f"""You are an expert LinkedIn copywriter. Transform this content into {num_posts} engaging, ready-to-publish LinkedIn posts.

Editorial Plan:
{editorial_plan[:500]}...

Generated Content:
{content}

Create {num_posts} separate LinkedIn posts, each one should:
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

        result = await self._call_llm(self.llm, prompt)

        self.workflow_data["linkedin_posts"] = result
        print(f"\n✅ LinkedIn posts written: {len(result)} characters")
        
        posts_dir = self.output_dir / "posts"
        posts_dir.mkdir(exist_ok=True)
        save_text_utf8(posts_dir / "linkedin_posts.md", result)
        
        return PostWritingEvent(
            result=result,
            editorial_plan=editorial_plan,
            generated_content=content
        )

    @step
    async def create_visuals(self, ctx: Context, ev: PostWritingEvent) -> VisualsEvent:
        """Step 4: Create visual concepts for posts."""
        print("\n" + "="*60)
        print("=== STEP 4: Creating Visual Assets ===")
        print("="*60)

        posts = ev.result

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

        result = await self._call_llm(self.llm, prompt)

        self.workflow_data["visuals"] = result
        print(f"\n✅ Visual concepts created: {len(result)} characters")
        
        images_dir = self.output_dir / "images"
        images_dir.mkdir(exist_ok=True)
        save_text_utf8(images_dir / "visuals_info.md", result)
        
        return VisualsEvent(
            result=result,
            editorial_plan=ev.editorial_plan,
            generated_content=ev.generated_content,
            linkedin_posts=posts
        )

    @step
    async def plan_posts(self, ctx: Context, ev: VisualsEvent) -> StopEvent:
        """Step 5: Create final content plan and save all results."""
        print("\n" + "="*60)
        print("=== STEP 5: Planning and Saving ===")
        print("="*60)

        editorial_plan = ev.editorial_plan
        content = ev.generated_content
        posts = ev.linkedin_posts
        visuals = ev.result

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

        try:
            save_text_utf8(self.output_dir / "final_content_plan.md", final_plan)
            save_text_utf8(self.output_dir / "linkedin_content_plan.md", final_plan)
            print(f"✅ Saved content plan ({len(final_plan)} characters)")
            result_msg = f"Successfully saved content plan with {len(final_plan)} characters"
        except Exception as e:
            print(f"❌ Error saving file: {str(e)}")
            result_msg = f"Error saving file: {str(e)}"

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
                "final_plan": result_msg,
                "output_directory": str(self.output_dir),
            }
        )