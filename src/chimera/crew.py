from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool, FileWriterTool
from .utils.utils import print_output, check_memory_dir, LLM_Config
from .utils.storage_config import (
    get_long_term_memory,
    get_short_term_memory,
    get_entity_memory,
)
import os
from dotenv import load_dotenv

load_dotenv()

@CrewBase
class LinkedInCrew:
    """Crew with the goal of creating  post on  LinkedIn"""

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


#Gestione della memoria.
    ltm = get_long_term_memory()
    stm = get_short_term_memory()
    entity = get_entity_memory()

    # aggiunta dei tool
    web_search_tool = SerperDevTool()
    file_writer_tool = FileWriterTool()

    #Agenti 
    @agent
    def manager(self) -> Agent:
        return Agent(config=self.agents_config["manager"], verbose=True, allow_delegation=False, llm=self.llm)

    @agent
    def expert(self) -> Agent:
        return Agent(config=self.agents_config["expert"], verbose=True, allow_delegation=False, llm=self.llm)

    @agent
    def copywriter(self) -> Agent:
        return Agent(config=self.agents_config["copywriter"], verbose=True, allow_delegation=False, llm=self.llm)

    @agent
    def designer(self) -> Agent:
        return Agent(config=self.agents_config["designer"], verbose=True, allow_delegation=False, llm=self.llm)
    
    @agent
    def planner(self) -> Agent:
        return Agent(
            config=self.agents_config["planner"],
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            tools=[self.file_writer_tool]  #
        )


    # Task
    @task
    def editorial_plan_task(self) -> Task:
        return Task(config=self.tasks_config["editorial_plan"], agent=self.manager())

    @task
    def technical_content_task(self) -> Task:
        return Task(config=self.tasks_config["technical_content"], agent=self.expert())

    @task
    def linkedin_post_task(self) -> Task:
        return Task(config=self.tasks_config["linkedin_post"], agent=self.copywriter())

    @task
    def visuals_task(self) -> Task:
        return Task(config=self.tasks_config["visuals"], agent=self.designer())
    
    @task
    def plan_posts_task(self) -> Task:
        return Task(config=self.tasks_config["plan_posts"], agent=self.planner())

    # Definizione della Crew 
    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            memory=True,
            long_term_memory=self.ltm,
            short_term_memory=self.stm,
            entity_memory=self.entity,
        )


