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
from .utils.storage_qdrant import QdrantStorage
import os
from dotenv import load_dotenv
import wget
import time 
import uuid


######################tools###############

#TODO spostare in un file in tools
#####Dall-E tools####
@tool("Download Image Tool")
def download_image_tool(url: str, fname: str) -> str:
    """Tool to download an image given its url and save it using the passed fname."""
    ret = None
    try:
        ret = wget.download(url, fname)
    except Exception as e:
        print('Error during download:', e)
    if ret != fname:
        return f"There was an error during the download"
    return f"Image successfully saved as {fname}"



load_dotenv()

qdrant_client = QdrantStorage(type=os.environ.get("COLLECTION"))

#TODO: aggiungere tool

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

    manager_llm = LLM_Config(
        provider=os.getenv("PROVIDER"),
        model=os.getenv("MANGER_MODEL"),
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
            allow_delegation=True,
            llm=self.manager_llm,
            max_iter=5

        )

    @agent
    def expert(self) -> Agent:
        return Agent(
            config=self.agents_config["expert"],
            verbose=True,
            allow_delegation=True,
            llm=self.llm,
            tools=[self.web_search_tool],
            max_iter=5
        )

    @agent
    def product_expert(self) -> Agent:
        product_sites = os.getenv("PRODUCT_SITES")
        if product_sites:
            sites_list = [s.strip() for s in product_sites.split(",") if s.strip()]
            return Agent(
                config=self.agents_config["product_expert"],
                verbose=True,
                allow_delegation=False,
                llm=self.llm,
                tools=[self.scraper_tool],
                description=f"Product expert with knowledge from company sources: {sites_list}",
                max_iter=5
            )
        else:
            return None
    
    @agent
    def knowledge_manager(self) -> Agent:
        return Agent(
            config=self.agents_config["knowledge_manager"],
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            max_iter=5,
            # tools=[upsert_knowledge, search_knowledge],
            # description=f"Reads and Writes on knowledge"
        )

    @agent
    def copywriter(self) -> Agent:
        return Agent(
            config=self.agents_config["copywriter"],
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            max_iter=5
        )

    @agent
    def designer(self) -> Agent:
        return Agent(
            config=self.agents_config["designer"],
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            tools=[self.dalle_tool, download_image_tool],
            max_iter=5
        )

    @agent
    def planner(self) -> Agent:
        return Agent(
            config=self.agents_config["planner"],
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            tools=[self.file_writer_tool],
            max_iter=5
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
        return Task(config=self.tasks_config["product_content"], agent=self.product_expert())
    
    @task
    def knowledge_saving_task(self)-> Task:
        return Task(config=self.tasks_config["knowledge_saving"],agent=self.knowledge_manager() )

    @task
    def knowledge_search_task(self)-> Task:
        return Task(config=self.tasks_config["knowledge_searching"],agent=self.knowledge_manager() )
    
    @task
    def linkedin_post_task(self) -> Task:
        return Task(config=self.tasks_config["linkedin_post"], agent=self.copywriter())

    @task
    def visuals_task(self) -> Task:
        return Task(config=self.tasks_config["visuals"], agent=self.designer())

    @task
    def plan_posts_task(self) -> Task:
        return Task(config=self.tasks_config["plan_posts"], agent=self.planner())

    # =============== Crew ===============
    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[self.expert(), self.product_expert(), self.knowledge_manager(), 
                    self.copywriter(), self.designer(), self.planner()],
            tasks=self.tasks,
            process=Process.hierarchical,
            verbose=True,
            manager_agent=self.manager(),
            planning=True,
            manager_llm=self.manager_llm
        )




# memory=True,
            # long_term_memory=self.ltm,
            # short_term_memory=self.stm,
            # entity_memory=self.entity,
