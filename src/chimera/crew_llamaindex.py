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
warnings.filterwarnings("ignore", message=r".*`ddgs`\) has been renamed to `ddgs`.*", category=RuntimeWarning)

_original_warn = warnings.warn
def _silent_warn(message, category=UserWarning, stacklevel=1):
    msg_str = str(message).lower()
    if any(x in msg_str for x in ['pydantic', '__fields__', 'deprecated', 'model_fields', 'unclosed', 'socket', 'transport']):
        return
    _original_warn(message, category, stacklevel)

warnings.warn = _silent_warn

###############################################

from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime
import yaml
from bs4 import BeautifulSoup
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
        try:
            from ddgs import DDGS
        except ImportError:
            try:
                from duckduckgo_search import DDGS
            except ImportError:
                DDGS = None

        class MyCustomDuckDuckGoTool:
            def run(self, query: str) -> str:
                if DDGS is None:
                    return f"Error: DuckDuckGo search library not available for query: {query}"
                try:
                    with DDGS() as ddg:
                        results = ddg.text(query, max_results=5)
                    lines = []
                    for i, item in enumerate(results or [], 1):
                        title = item.get("title") or item.get("source") or "No title"
                        href = item.get("href") or item.get("link") or ""
                        snippet = item.get("body") or item.get("snippet") or ""
                        lines.append(f"{i}. {title}\n{href}\n{snippet}\n")
                    return "\n".join(lines) if lines else "No results found."
                except Exception as e:
                    return f"Error performing DuckDuckGo search: {str(e)}"

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


# Simplified event structure
class EditorialPlanEvent(Event):
    result: str


class LinkedInPostsEvent(Event):
    """Posts are created with research included - no intermediate step"""
    posts: str
    research_notes: str


class VisualsEvent(Event):
    """Visual prompts ready for image generation"""
    prompts: List[str]
    concepts: str


class AgentConfig:
    """Helper class to manage agent configurations from YAML."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        self.role = config_dict.get("role", "")
        self.goal = config_dict.get("goal", "")
        self.backstory = config_dict.get("backstory", "")
        self.tools = config_dict.get("tools", [])
        self.max_iterations = config_dict.get("max_iterations", 15)
    
    def build_system_prompt(self) -> str:
        """Build a comprehensive system prompt from config."""
        parts = []
        
        if self.role:
            parts.append(f"# Your Role\n{self.role}")
        
        if self.goal:
            parts.append(f"# Your Goal\n{self.goal}")
        
        if self.backstory:
            parts.append(f"# Your Background\n{self.backstory}")
        
        if parts:
            return "\n\n".join(parts) + "\n\n---\n\n"
        return ""
    
    def __repr__(self):
        return f"AgentConfig(role='{self.role[:30]}...', goal='{self.goal[:30]}...')"


class TaskConfig:
    """Helper class to manage task configurations from YAML."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        self.description = config_dict.get("description", "")
        self.expected_output = config_dict.get("expected_output", "")
        self.agent = config_dict.get("agent", "")
    
    def build_task_prompt(self, context: str = "") -> str:
        """Build a task prompt from config."""
        parts = []
        
        if self.description:
            parts.append(f"# Task\n{self.description}")
        
        if context:
            parts.append(f"# Context\n{context}")
        
        if self.expected_output:
            parts.append(f"# Expected Output\n{self.expected_output}")
        
        return "\n\n".join(parts) if parts else ""
    
    def __repr__(self):
        return f"TaskConfig(agent='{self.agent}', description='{self.description[:30]}...')"


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

        self.agents_config = self._load_configs("config/agents.yaml")
        self.tasks_config = self._load_configs("config/tasks.yaml")
        
        print(f"\n📋 Loaded {len(self.agents_config)} agent configs: {list(self.agents_config.keys())}")
        print(f"📋 Loaded {len(self.tasks_config)} task configs: {list(self.tasks_config.keys())}")

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

        self._ingest_product_sites()

        workflow_inputs = {
            **self.inputs,
            "output_dir": str(self.output_dir),
            "agents_config": self.agents_config,
            "tasks_config": self.tasks_config,
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

    def _load_configs(self, config_path: str) -> Dict[str, Any]:
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                raw_config = yaml.safe_load(f) or {}
            
            if "agents.yaml" in config_path:
                return {
                    name: AgentConfig(cfg) 
                    for name, cfg in raw_config.items()
                }
            elif "tasks.yaml" in config_path:
                return {
                    name: TaskConfig(cfg) 
                    for name, cfg in raw_config.items()
                }
            else:
                return raw_config
                
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

    def _ingest_product_sites(self):
        """Scrape PRODUCT_SITES env and upsert content into Qdrant."""
        sites_str = os.getenv("PRODUCT_SITES", "").strip()
        if not sites_str:
            print("ℹ️ No PRODUCT_SITES provided; skipping ingestion.")
            return

        urls = [u.strip() for u in sites_str.split(",") if u.strip()]
        if not urls:
            print("ℹ️ PRODUCT_SITES is empty after parsing; skipping ingestion.")
            return

        collection = os.getenv("QDRANT_COLLECTION", "linkedin_knowledge")
        print(f"🔎 Ingesting {len(urls)} product site(s) into Qdrant collection '{collection}'")

        try:
            chunk_size = int(os.getenv("CHUNK_SIZE", "512"))
        except Exception:
            chunk_size = 512
        try:
            chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "50"))
        except Exception:
            chunk_overlap = 50
        chunk_size = max(1, chunk_size)
        chunk_overlap = max(0, min(chunk_overlap, chunk_size - 1))

        for url in urls:
            try:
                resp = requests.get(url, timeout=20)
                resp.raise_for_status()
                content_type = resp.headers.get("Content-Type", "").lower()

                text = ""
                if "text/markdown" in content_type or url.endswith(".md") or "raw.githubusercontent.com" in url:
                    text = resp.text
                elif "text/plain" in content_type:
                    text = resp.text
                else:
                    try:
                        soup = BeautifulSoup(resp.content, "html.parser")
                        text = soup.get_text(separator="\n", strip=True)
                    except Exception as parse_err:
                        text = resp.text
                        print(f"⚠️ HTML parse error for {url}: {parse_err}. Using raw text.")

                if not text or len(text.strip()) < 50:
                    print(f"⚠️ Skipping {url}: content too short or empty")
                    continue

                max_chars = 50_000
                clipped = text[:max_chars]

                metadata = {"url": url, "source": url, "type": "product_site"}
                result = upsert_knowledge(
                    clipped,
                    metadata=metadata,
                    collection_name=collection,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )

                print(f"✓ Ingested from {url}: {result}")
            except Exception as e:
                print(f"❌ Error ingesting {url}: {e}")

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
            if result and isinstance(result, str) and len(result) > 10 and "Error" not in result:
                self.tools_working["web_search"] = True
                print("✅ Web search tool: WORKING")
            else:
                print("⚠️  Web search tool: NOT WORKING (empty or error)")
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
        try:
            custom_search = MyCustomDuckDuckGoTool()
            self.web_search_tool = FunctionTool.from_defaults(
                fn=lambda query: custom_search.run(query),
                name="web_search",
                description="Search the web for information on a given topic",
            )
        except Exception as e:
            print(f"Error in initializing tool: {e}")

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
            def simple_scraper(url: str) -> str:
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
        except Exception as e:
            print(f"Failed to run scraper: {e}")

        collection = os.getenv("QDRANT_COLLECTION", "linkedin_knowledge")
        self.search_knowledge_tool = FunctionTool.from_defaults(
            fn=lambda query: search_knowledge(query, collection_name=collection),
            name="search_knowledge",
            description="Search the knowledge base for relevant information",
        )
        self.upsert_knowledge_tool = FunctionTool.from_defaults(
            fn=lambda text, metadata=None: upsert_knowledge(text, metadata=metadata, collection_name=collection),
            name="upsert_knowledge",
            description="Add or update information in the knowledge base",
        )

    def _initialize_agents(self):
        """Initialize all ReAct agents with configurations from YAML."""
        self.agents = {}

        agent_config = self.agents_config.get("generalist_expert")
        if agent_config:
            print(f"✅ Initializing Generalist Expert with config: {agent_config}")
            max_iter = agent_config.max_iterations
        else:
            print("⚠️  No config for generalist_expert, using defaults")
            max_iter = 15

        self.agents["generalist_expert"] = ReActAgent(
            tools=[self.web_search_tool],
            llm=self.llm,
            memory=get_short_term_memory(),
            max_iterations=max_iter,
            verbose=True,
        )

        agent_config = self.agents_config.get("product_expert")
        if agent_config:
            print(f"✅ Initializing Product Expert with config: {agent_config}")
            max_iter = agent_config.max_iterations
        else:
            print("⚠️  No config for product_expert, using defaults")
            max_iter = 20

        self.agents["product_expert"] = ReActAgent(
            tools=[
                self.search_knowledge_tool,
                self.upsert_knowledge_tool,
                self.scraper_tool,
            ],
            llm=self.llm,
            memory=get_long_term_memory(),
            max_iterations=max_iter,
            verbose=True,
        )

        agent_config = self.agents_config.get("designer")
        if agent_config:
            print(f"✅ Initializing Designer with config: {agent_config}")
            max_iter = agent_config.max_iterations
        else:
            print("⚠️  No config for designer, using defaults")
            max_iter = 15

        self.agents["designer"] = ReActAgent(
            tools=[self.dalle_tool, self.download_image_tool],
            llm=self.llm,
            memory=get_short_term_memory(),
            max_iterations=max_iter,
            verbose=True,
        )

        agent_config = self.agents_config.get("planner")
        if agent_config:
            print(f"✅ Initializing Planner with config: {agent_config}")
            max_iter = agent_config.max_iterations
        else:
            print("⚠️  No config for planner, using defaults")
            max_iter = 10

        self.agents["planner"] = ReActAgent(
            tools=[self.file_writer_tool],
            llm=self.llm,
            memory=get_entity_memory(),
            max_iterations=max_iter,
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
    """Streamlined workflow with reduced redundancy."""
    
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
        
        self.agents_config: Dict[str, AgentConfig] = inputs.get("agents_config", {})
        self.tasks_config: Dict[str, TaskConfig] = inputs.get("tasks_config", {})

    def _get_agent_prompt(self, agent_name: str, task_context: str) -> str:
        """Build a complete prompt combining agent config and task context."""
        agent_config = self.agents_config.get(agent_name)
        
        if agent_config:
            system_prompt = agent_config.build_system_prompt()
            return f"{system_prompt}{task_context}"
        else:
            return task_context

    def _get_task_prompt(self, task_name: str, context: str = "") -> str:
        """Build a task prompt from task configuration."""
        task_config = self.tasks_config.get(task_name)
        
        if task_config:
            return task_config.build_task_prompt(context)
        else:
            return context

    async def _call_llm(self, llm, prompt: str) -> str:
        """Call LLM directly without agent."""
        print(f"\n🤖 Calling LLM directly...")
        print(f"📝 Prompt preview: {prompt[:150]}...")
        
        messages = [ChatMessage(role="user", content=prompt)]
        response = await llm.achat(messages)
        result = str(response.message.content)
        
        print(f"✅ LLM Response length: {len(result)} characters")
        return result

    async def _ask_agent(self, agent_name: str, prompt: str, 
                        max_iterations: Optional[int] = None) -> str:
        """Call an agent with its configuration-based prompt."""
        agent = self.agents.get(agent_name)
        if not agent:
            raise ValueError(f"Agent '{agent_name}' not found")
        
        full_prompt = self._get_agent_prompt(agent_name, prompt)
        
        if max_iterations is None:
            agent_config = self.agents_config.get(agent_name)
            max_iterations = agent_config.max_iterations if agent_config else 15
        
        print(f"\n🔧 Calling {agent_name} (max_iterations={max_iterations})...")
        
        try:
            if hasattr(agent, "run"):
                handler = agent.run(full_prompt, max_iterations=max_iterations)
                async for ev in handler.stream_events():
                    if isinstance(ev, AgentStream):
                        print(f"{ev.delta}", end="", flush=True)
                result = await handler
                print(f"\n✅ {agent_name} completed")
                return str(result)
            elif hasattr(agent, "aquery"):
                result = await agent.aquery(full_prompt)
                print(f"✅ {agent_name} completed")
                return str(result)
            else:
                raise AttributeError(f"Agent {agent_name} has no usable run/aquery methods")
        except Exception as e:
            error_msg = str(e)
            if "Max iterations" in error_msg or "WorkflowRuntimeError" in error_msg:
                print(f"⚠️  {agent_name} hit max iterations - falling back to direct LLM...")
                return await self._call_llm(self.llm, full_prompt)
            else:
                print(f"❌ {agent_name} error: {error_msg[:100]}")
                raise

    @step
    async def editorial_plan(self, ctx: Context, ev: StartEvent) -> EditorialPlanEvent:
        """Step 1: Create concise editorial strategy (not detailed content)."""
        print("\n" + "="*60)
        print("=== STEP 1: Editorial Strategy ===")
        print("="*60)

        topic = self.inputs.get("topic", "AI and Technology")
        num_posts = self.inputs.get("num_posts", 5)
        frequency = self.inputs.get("frequency", "weekly")

        prompt = f"""Create a comprehensive editorial strategy for {num_posts} LinkedIn posts about {topic}, published {frequency}.

        Provide a detailed strategic plan (400-600 words) covering:

        1. CONTENT THEMES ({num_posts} posts):
        - Post 1: [Specific angle/topic] - [Why this matters to audience]
        - Post 2: [Specific angle/topic] - [Why this matters to audience]
        - (Continue for all {num_posts} posts)

        2. TARGET AUDIENCE ANALYSIS:
        - Primary audience: [Role, industry, pain points]
        - Secondary audience: [Who else will benefit]
        - Audience interests and needs

        3. CONTENT STRATEGY:
        - Content mix breakdown (e.g., 40% educational, 30% thought leadership, 30% promotional)
        - Tone and voice guidelines
        - Key messages to emphasize
        - Engagement tactics

        4. HASHTAG STRATEGY:
        - 3-5 primary hashtags with reasoning
        - Post-specific hashtag suggestions
        
        5. SUCCESS METRICS:
        - What defines success for this campaign
        - Key engagement indicators to track

        Make it strategic and actionable."""

        result = await self._call_llm(self.manager_llm, prompt)

        self.workflow_data["editorial_plan"] = result
        print(f"\n✅ Editorial strategy created: {len(result)} characters")
        
        # Only save strategy, not detailed content
        save_text_utf8(self.output_dir / "1_editorial_strategy.md", result)
        
        return EditorialPlanEvent(result=result)

    @step
    async def create_posts_with_research(self, ctx: Context, ev: EditorialPlanEvent) -> LinkedInPostsEvent:
        """Step 2: Research AND write posts in one step - no intermediate files."""
        print("\n" + "="*60)
        print("=== STEP 2: Research & Write LinkedIn Posts ===")
        print("="*60)

        editorial_plan = ev.result
        topic = self.inputs.get("topic", "AI and Technology")
        num_posts = self.inputs.get("num_posts", 5)

        # Combine research and writing in ONE step
        task_context = f"""You MUST create EXACTLY {num_posts} LinkedIn posts. No more, no less.

Using this editorial strategy, research AND write {num_posts} complete LinkedIn posts about {topic}.

Editorial Strategy:
{editorial_plan}

CRITICAL REQUIREMENT: Create EXACTLY {num_posts} posts numbered 1 through {num_posts}.

YOUR TASK:
1. For each post angle in the strategy:
   - Research current info {"(use web_search tool)" if self.tools_working.get("web_search") else ""}{"(use search_knowledge tool)" if self.tools_working.get("knowledge_base") and self.expert_type == "prodotto" else ""}
   - Write a complete, ready-to-publish LinkedIn post
   
2. Each post should be:
   - 1800-2500 characters (aim for longer, more substantive posts)
   - Start with a scroll-stopping hook that makes people stop scrolling
   - Include 3-4 main body paragraphs with valuable insights
   - Short paragraphs (2-3 sentences max per paragraph)
   - Include 2-3 strategic emojis throughout
   - Add specific examples, stats, or stories to illustrate points
   - End with clear, actionable CTA
   - Include 3-5 relevant hashtags at the end
   - Use line breaks for readability

OUTPUT FORMAT (YOU MUST CREATE {num_posts} POSTS):
=== POST 1: [Title/Topic] ===
[Complete post text here including emojis and line breaks]

Hashtags: #hashtag1 #hashtag2 #hashtag3

RESEARCH SOURCES:
- Key insight 1: [specific data point or trend used]
- Key insight 2: [another source or statistic referenced]
---

=== POST 2: [Title/Topic] ===
[Complete post text here including emojis and line breaks]

Hashtags: #hashtag1 #hashtag2 #hashtag3

RESEARCH SOURCES:
- Key insight 1: [specific data point or trend used]
- Key insight 2: [another source or statistic referenced]
---

=== POST 3: [Title/Topic] ===
[Continue same format...]
---

REPEAT THIS FORMAT UNTIL YOU HAVE {num_posts} POSTS TOTAL.

VERIFICATION: After writing, count your posts. You should have EXACTLY {num_posts} posts numbered 1 through {num_posts}.

IMPORTANT: Include clear "RESEARCH SOURCES:" section for each post."""

        if self.expert_type == "generalista" and self.tools_working.get("web_search"):
            result = await self._ask_agent("generalist_expert", task_context)
        elif self.expert_type == "prodotto" and self.tools_working.get("knowledge_base"):
            result = await self._ask_agent("product_expert", task_context)
        else:
            result = await self._call_llm(self.llm, task_context)
        
        posts_clean = []
        research_notes = []

        sections = result.split("=== POST")
        for section in sections[1:]:
            if "RESEARCH SOURCES:" in section:
                parts = section.split("RESEARCH SOURCES:")
                posts_clean.append("=== POST" + parts[0].strip())
                if len(parts) > 1:
                    research_notes.append(parts[1].split("---")[0].strip())
            else:
                posts_clean.append("=== POST" + section.strip())

        # Validate we have the correct number of posts
        if len(posts_clean) != num_posts:
            print(f"⚠️  Warning: Expected {num_posts} posts but got {len(posts_clean)}. Adjusting...")
            
            if len(posts_clean) < num_posts:
                # Not enough posts - pad with placeholders
                for i in range(len(posts_clean), num_posts):
                    posts_clean.append(f"=== POST {i+1}: Additional Post Needed ===\n[Post content to be created]\n\nHashtags: #topic")
            elif len(posts_clean) > num_posts:
                # Too many posts - truncate
                posts_clean = posts_clean[:num_posts]
                research_notes = research_notes[:num_posts]

        posts_only = "\n\n".join(posts_clean)
        research_summary = "\n\n".join(f"POST {i+1} SOURCES:\n{note}" for i, note in enumerate(research_notes))

        self.workflow_data["posts"] = posts_only
        self.workflow_data["research"] = research_summary

        print(f"\n✅ Created {len(posts_clean)} posts with research")

        # Save posts cleanly
        save_text_utf8(self.output_dir / "2_linkedin_posts.md", posts_only)
        
        if research_summary:
            save_text_utf8(self.output_dir / "2_research_notes.txt", 
                          f"Research notes from post creation:\n\n{research_summary}")
        
        return LinkedInPostsEvent(posts=posts_only, research_notes=research_summary)

    @step
    async def create_visual_prompts(self, ctx: Context, ev: LinkedInPostsEvent) -> VisualsEvent:
        """Step 3: Generate actionable DALL-E prompts (not just concepts)."""
        print("\n" + "="*60)
        print("=== STEP 3: Generate Visual Prompts ===")
        print("="*60)

        posts = ev.posts
        num_posts = self.inputs.get("num_posts", 5)

        task_context = f"""Based on these LinkedIn posts, create EXACTLY {num_posts} READY-TO-USE DALL-E prompts for image generation.

        Posts:
        {posts[:2000]}...

        CRITICAL: You MUST create EXACTLY {num_posts} DALL-E prompts, one for each post.

        For each post, create a detailed DALL-E prompt that can be used immediately.

        DALLE PROMPT REQUIREMENTS:
        - Be specific and detailed (describe exactly what to show)
        - Include style (e.g., "minimalist flat design", "professional 3D render")
        - Specify colors (e.g., "blue and white color scheme")
        - No text in images
        - Professional and clean
        - LinkedIn-appropriate

        OUTPUT FORMAT (MUST HAVE {num_posts} PROMPTS):
        === POST 1 DALLE PROMPT ===
        [Complete, detailed DALL-E prompt ready to use]

        === POST 2 DALLE PROMPT ===
        [Complete, detailed DALL-E prompt ready to use]

        === POST 3 DALLE PROMPT ===
        [Complete, detailed DALL-E prompt ready to use]

        CONTINUE THIS FORMAT FOR ALL {num_posts} POSTS.

        VERIFICATION: Count your prompts. You should have EXACTLY {num_posts} prompts numbered 1 through {num_posts}."""

        result = await self._call_llm(self.llm, task_context)
        prompts = []
        sections = result.split("=== POST")
        for section in sections[1:]:
            prompt_text = section.split("DALLE PROMPT ===")[1].strip() if "DALLE PROMPT ===" in section else section.strip()
            prompt_text = prompt_text.split("===")[0].strip()
            if prompt_text:
                prompts.append(prompt_text)

        if len(prompts) != num_posts:
            print(f"⚠️  Warning: Expected {num_posts} prompts but got {len(prompts)}. Adjusting...")
            
            if len(prompts) < num_posts:
                for i in range(len(prompts), num_posts):
                    prompts.append(f"Professional LinkedIn post illustration for post {i+1}, minimalist design, blue and white color scheme, clean and modern")
            elif len(prompts) > num_posts:
                prompts = prompts[:num_posts]

        self.workflow_data["visual_prompts"] = prompts

        print(f"\n✅ Created {len(prompts)} visual prompts")

        prompts_text = "\n\n".join(f"POST {i+1} IMAGE PROMPT:\n{p}" for i, p in enumerate(prompts))
        save_text_utf8(self.output_dir / "3_dalle_prompts.txt", prompts_text)
        
        return VisualsEvent(prompts=prompts, concepts=result)

    @step
    async def create_content_calendar(self, ctx: Context, ev: VisualsEvent) -> StopEvent:
        """Step 4: Create publishing calendar ONLY (no content repetition)."""
        print("\n" + "="*60)
        print("=== STEP 4: Content Calendar ===")
        print("="*60)

        topic = self.inputs.get("topic", "AI and Technology")
        num_posts = self.inputs.get("num_posts", 5)
        frequency = self.inputs.get("frequency", "weekly")
        
        # Get the posts and prompts
        posts = self.workflow_data.get("posts", "")
        prompts = ev.prompts

        post_titles = []
        post_sections = posts.split("=== POST")
        for section in post_sections[1:]:  # Skip empty first section
            lines = section.strip().split("\n")
            if lines:
                title_line = lines[0].replace(":", "").strip()
                # Remove any numbering
                title_clean = title_line.split("===")[0].strip()
                post_titles.append(title_clean)

        # Ensure we have exactly num_posts titles
        while len(post_titles) < num_posts:
            post_titles.append(f"LinkedIn Post {len(post_titles) + 1}")
        if len(post_titles) > num_posts:
            post_titles = post_titles[:num_posts]

        titles_list = "\n".join([f"{i+1}. {title}" for i, title in enumerate(post_titles)])

        calendar_prompt = f"""Create a LinkedIn publishing calendar for EXACTLY {num_posts} posts about {topic}, published {frequency}.

        CRITICAL: You MUST create calendar entries for ALL {num_posts} posts. No more, no less.

        Post titles:
        {titles_list}

        Create a clear, readable calendar in this format:

        ## POST 1: {post_titles[0] if post_titles else "Post 1"}
        **Publishing Date:** [Calculate from next Monday, based on {frequency}]
        **Posting Time:** [Best time for LinkedIn, e.g., 8:00 AM or 12:00 PM]
        **Visual Asset:** DALL-E Prompt #1 (see 3_dalle_prompts.txt)
        **Target Metrics:** 
        - Impressions: [target number]
        - Engagement rate: [target %]
        - Comments: [target number]

        ---

        ## POST 2: {post_titles[1] if len(post_titles) > 1 else "Post 2"}
        **Publishing Date:** [Next date based on {frequency}]
        **Posting Time:** [Best time]
        **Visual Asset:** DALL-E Prompt #2 (see 3_dalle_prompts.txt)
        **Target Metrics:**
        - Impressions: [target number]
        - Engagement rate: [target %]
        - Comments: [target number]

        ---

        {''.join([f"""
        ## POST {i+1}: {post_titles[i] if i < len(post_titles) else f"Post {i+1}"}
        **Publishing Date:** [Next date based on {frequency}]
        **Posting Time:** [Best time]
        **Visual Asset:** DALL-E Prompt #{i+1} (see 3_dalle_prompts.txt)
        **Target Metrics:**
        - Impressions: [target number]
        - Engagement rate: [target %]
        - Comments: [target number]

        ---
        """ for i in range(2, num_posts)])}

        VERIFICATION: Count your calendar entries. You should have EXACTLY {num_posts} entries numbered 1 through {num_posts}.

        CRITICAL: Include ALL {num_posts} posts in the calendar!"""

        calendar = await self._call_llm(self.llm, calendar_prompt)

        final_output = f"""# LinkedIn Content Calendar - {topic}

## Publishing Schedule

{calendar}

---

## Files Generated

1. **1_editorial_strategy.md** - High-level content strategy
2. **2_linkedin_posts.md** - {num_posts} ready-to-publish posts
3. **2_research_notes.txt** - Research insights used
4. **3_dalle_prompts.txt** - {len(prompts)} image generation prompts
5. **4_content_calendar.md** - This publishing schedule

---

## Next Steps

1. Review and customize posts in `2_linkedin_posts.md`
2. Generate images using prompts in `3_dalle_prompts.txt`
3. Follow the publishing schedule above
4. Track metrics: engagement rate, reach, clicks, comments

---

## Quick Stats

- Total posts: {num_posts}
- Publishing frequency: {frequency}
- Content type: {"Technical/General" if self.expert_type == "generalista" else "Product-focused"}
- Tools used: {"Web search, " if self.tools_working.get("web_search") else ""}{"Knowledge base, " if self.tools_working.get("knowledge_base") else ""}LLM generation
"""

        save_text_utf8(self.output_dir / "4_content_calendar.md", final_output)
        
        print(f"\n✅ Content calendar created")
        print("\n" + "="*60)
        print("✅ Workflow Complete!")
        print("="*60)
        print(f"\nGenerated files:")
        print(f"  1. 1_editorial_strategy.md - Strategic overview")
        print(f"  2. 2_linkedin_posts.md - {num_posts} ready posts")
        print(f"  3. 2_research_notes.txt - Research references")
        print(f"  4. 3_dalle_prompts.txt - Image generation prompts")
        print(f"  5. 4_content_calendar.md - Publishing schedule")
        
        return StopEvent(
            result={
                "status": "success",
                "output_directory": str(self.output_dir),
                "files_generated": 5,
                "posts_created": num_posts,
                "prompts_created": len(prompts)
            }
        )