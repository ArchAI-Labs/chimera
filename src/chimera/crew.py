from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool, FileWriterTool, DallETool, ScrapeWebsiteTool
from crewai.tools import tool
from .utils.utils import print_output, check_memory_dir, LLM_Config
from .utils.storage_config import (
    get_long_term_memory,
    get_short_term_memory,
    get_entity_memory,
)
import os
from dotenv import load_dotenv
from .tools.duckduckgo_tool import MyCustomDuckDuckGoTool
from .tools.dalle_tool import download_image_tool
from .tools.qdrant_tool import search_knowledge, upsert_knowledge
from .utils.storage_qdrant import QdrantStorage

# -----------------------------------------------------------
# Environment loading
# Loads environment variables from the .env file to configure
# API keys, model settings, and Qdrant parameters.
# -----------------------------------------------------------
load_dotenv()


@CrewBase
class LinkedInCrew:
    """
    This class defines all agents, tasks, and tools required to
    build a fully automated LinkedIn content creation pipeline
    using CrewAI and LangChain integrations.
    """

    # -----------------------------------------------------------
    # Configuration paths
    # Define YAML configuration files for agents and tasks.
    # -----------------------------------------------------------
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    # Ensure memory directories exist before initializing
    check_memory_dir()

    # -----------------------------------------------------------
    # LLM configurations
    # -----------------------------------------------------------
    llm = LLM_Config(
        provider=os.getenv("PROVIDER"),
        model=os.getenv("MODEL"),
        base_url=os.getenv("BASE_URL"),
        temperature=float(os.getenv("TEMPERATURE")),
        max_tokens=int(os.getenv("MAX_TOKENS")),
        timeout=float(os.getenv("TIMEOUT")),
        callbacks=[print_output],
    )

    manager_llm = LLM_Config(
        provider=os.getenv("PROVIDER"),
        model=os.getenv("MANAGER_MODEL"),
        base_url=os.getenv("BASE_URL"),
        temperature=float(os.getenv("TEMPERATURE")),
        max_tokens=int(os.getenv("MAX_TOKENS")),
        timeout=float(os.getenv("TIMEOUT")),
        callbacks=[print_output],
    )

    # -----------------------------------------------------------
    # Tool initialization
    # Initialize core tools (web search, image generation, scraping, etc.)
    # -----------------------------------------------------------
    if os.environ.get("SERPER_API_KEY"):
        web_search_tool = SerperDevTool()
    else:
        web_search_tool = MyCustomDuckDuckGoTool()

    file_writer_tool = FileWriterTool()
    dalle_tool = DallETool(model="dall-e-3", size="1024x1024", quality="standard", n=1)
    scraper_tool = ScrapeWebsiteTool()

    # -----------------------------------------------------------
    # Agent definitions
    # Define all CrewAI agents used in the pipeline.
    # -----------------------------------------------------------
    @agent
    def manager(self) -> Agent:
        return Agent(
            config=self.agents_config["manager"],
            verbose=True,
            allow_delegation=False,
            tools=[self.web_search_tool],
            llm=self.manager_llm,
            max_iter=5,
        )

    @agent
    def generalist_expert(self) -> Agent:
        return Agent(
            config=self.agents_config["generalist_expert"],
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            tools=[self.web_search_tool],
            max_iter=5,
        )

    @agent
    def product_expert(self) -> Agent:
        print("Activated agent: Product Expert (uses RAG + Scraper)")
        return Agent(
            config=self.agents_config["product_expert"],
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            tools=[search_knowledge, upsert_knowledge, self.scraper_tool],
            max_iter=10,
        )

    @agent
    def copywriter(self) -> Agent:
        return Agent(
            config=self.agents_config["copywriter"],
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            max_iter=5,
        )

    @agent
    def designer(self) -> Agent:
        return Agent(
            config=self.agents_config["designer"],
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            tools=[self.dalle_tool, download_image_tool],
            max_iter=5,
        )

    @agent
    def planner(self) -> Agent:
        return Agent(
            config=self.agents_config["planner"],
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            tools=[self.file_writer_tool],
            max_iter=5,
        )

    # -----------------------------------------------------------
    # Task definitions
    # Each task corresponds to a specific part of the content pipeline.
    # -----------------------------------------------------------
    @task
    def editorial_plan_task(self) -> Task:
        return Task(config=self.tasks_config["editorial_plan"], agent=self.manager())

    @task
    def technical_content_task(self) -> Task:
        return Task(
            config=self.tasks_config["technical_content"],
            agent=self.generalist_expert(),
        )

    @task
    def product_content_task(self) -> Task:
        print(f"Building product content task with topic: {self.inputs.get('topic')}")
        return Task(
            config=self.tasks_config["product_content"], agent=self.product_expert()
        )

    @task
    def linkedin_post_task(self) -> Task:
        return Task(config=self.tasks_config["linkedin_post"], agent=self.copywriter())

    @task
    def visuals_task(self) -> Task:
        return Task(config=self.tasks_config["visuals"], agent=self.designer())

    @task
    def plan_posts_task(self) -> Task:
        return Task(config=self.tasks_config["plan_posts"], agent=self.planner())

    # -----------------------------------------------------------
    # Initialization
    # Handles input loading.
    # -----------------------------------------------------------
    def __init__(self, inputs=None):
        super().__init__()  # Initialize CrewBase
        self.inputs = inputs if inputs is not None else {}
        print(f"USER INPUTS RECEIVED: {self.inputs}")
        

    # -----------------------------------------------------------
    # Crew assembly
    # Combines agents and tasks into the final execution workflow.
    # -----------------------------------------------------------
    @crew
    def crew(self) -> Crew:
        expert_type = self.inputs.get("expert_type", "generalista")
        print(f"EXPERT TYPE: {expert_type}")

        if expert_type == "generalista":
            print("Building GENERALIST Crew...")
            return Crew(
                agents=[
                    self.manager(),
                    self.generalist_expert(),
                    self.copywriter(),
                    self.designer(),
                    self.planner(),
                ],
                tasks=[
                    self.editorial_plan_task(),
                    self.technical_content_task(),
                    self.linkedin_post_task(),
                    self.visuals_task(),
                    self.plan_posts_task(),
                ],
                process=Process.sequential,
                verbose=True,
                chat_llm=self.llm,
            )
        elif expert_type == "prodotto":
            print("Building PRODUCT Crew...")
            return Crew(
                agents=[
                    self.manager(),
                    self.product_expert(),
                    self.copywriter(),
                    self.designer(),
                    self.planner(),
                ],
                tasks=[
                    self.editorial_plan_task(),
                    self.product_content_task(),
                    self.linkedin_post_task(),
                    self.visuals_task(),
                    self.plan_posts_task(),
                ],
                process=Process.sequential,
                verbose=True,
                chat_llm=self.llm,
            )
        else:
            print("You are in else crew...")
            return Crew(
                agents=self.agents,
                tasks=self.tasks,
                process=Process.sequential,
                verbose=True,
                chat_llm=self.llm,
            )
