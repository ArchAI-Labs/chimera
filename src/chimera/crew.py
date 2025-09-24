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
import wget

@tool("Download Image Tool")
def download_image_tool(url: str, fname: str) -> str:
    """ Tool to download an image given its url and saves it using the passed fname. """
    ret = None
    try:
        ret = wget.download(url, fname)
    except Exception as e:
        print('Error during download:', e)
    if ret != fname: return f"There was an error during the download"
    return f"Image successfully saved as {fname}"

load_dotenv()


@CrewBase
class LinkedInCrew:
    """Crew with the goal of creating LinkedIn posts"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    check_memory_dir()

    llm = LLM_Config(
        provider=os.getenv("PROVIDER"),
        model=os.getenv("MODEL"),
        base_url=os.getenv("BASE_URL"),
        temperature=float(os.getenv("TEMPERATURE")),
        max_tokens=int(os.getenv("MAX_TOKENS")),
        timeout=float(os.getenv("TIMEOUT")),
        callbacks=[print_output],
    )

    # Gestione della memoria
    ltm = get_long_term_memory()
    stm = get_short_term_memory()
    entity = get_entity_memory()

    # Tool
    web_search_tool = SerperDevTool()
    file_writer_tool = FileWriterTool()
    dalle_tool = DallETool(model="dall-e-3", size="1024x1024", quality="standard", n=1)
    scraper_tool = ScrapeWebsiteTool()

    # =============== Agenti ===============
    @agent
    def manager(self) -> Agent:
        return Agent(
            config=self.agents_config["manager"],
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
        )

    @agent
    def expert(self) -> Agent:
        return Agent(
            config=self.agents_config["expert"],
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            tools=[self.web_search_tool],
        )
    
    @agent
    def product_expert(self) -> Agent:
        return Agent(
            config=self.agents_config["product_expert"],
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            tools=[self.scraper_tool],
        )

    @agent
    def copywriter(self) -> Agent:
        return Agent(
            config=self.agents_config["copywriter"],
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
        )
    
    @agent
    def designer(self) -> Agent:
        return Agent(
            config=self.agents_config["designer"],
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            tools=[self.dalle_tool, download_image_tool],
        )

    @agent
    def planner(self) -> Agent:
        return Agent(
            config=self.agents_config["planner"],
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            tools=[self.file_writer_tool],
        )

    # =============== Task ===============
    @task
    def editorial_plan_task(self) -> Task:
        return Task(config=self.tasks_config["editorial_plan"], agent=self.manager())

    @task
    def technical_content_task(self) -> Task:
        return Task(config=self.tasks_config["technical_content"], agent=self.expert())

    @task
    def product_content_task(self) -> Task:
        return Task(
            config=self.tasks_config["product_content"],
            agent=self.product_expert(),
        )

    @task
    def linkedin_post_task(self) -> Task:
        return Task(config=self.tasks_config["linkedin_post"], agent=self.copywriter())

    @task
    def visuals_task(self) -> Task:
        return Task(
            config=self.tasks_config["visuals"],
            agent=self.designer(),
        )

    @task
    def plan_posts_task(self) -> Task:
        return Task(config=self.tasks_config["plan_posts"], agent=self.planner())

    # --- Helper per scegliere i task in base all'esperto ---
    def build_tasks_for(self, expert_type: str):
        if (expert_type or "").lower() == "prodotto":
            return [
                self.editorial_plan_task(),
                self.product_content_task(),
                self.linkedin_post_task(),
                self.visuals_task(),
                self.plan_posts_task(),
            ]
        else:
            return [
                self.editorial_plan_task(),
                self.technical_content_task(),
                self.linkedin_post_task(),
                self.visuals_task(),
                self.plan_posts_task(),
            ]

    # =============== Crew ===============
    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[
                self.expert(),
                self.product_expert(),
                self.copywriter(),
                self.designer(),
                self.planner(),
            ],
            tasks=[],  # i task vengono decisi nel main
            process=Process.hierarchical,
            verbose=True,
            manager_agent=self.manager(),
        )



# memory=True,
            # long_term_memory=self.ltm,
            # short_term_memory=self.stm,
            # entity_memory=self.entity,
