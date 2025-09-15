from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task, before_kickoff
from crewai_tools import DirectoryReadTool, FileReadTool
from .tools.plantuml_tool import PlantUMLDiagramGeneratorTool
from .utils.utils import print_output, check_memory_dir, manage_output_dir, LLM_Config, ContextManager
from .utils.storage_config import (
    get_long_term_memory,
    get_short_term_memory,
    get_entity_memory,
)
import os
from pathlib import Path
from typing import Any, Dict, List
from dotenv import load_dotenv


load_dotenv()


@CrewBase
class CodeExplainer:
    """CodeExplainer crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    directory_read_tool = DirectoryReadTool(directory="./knowledge/")

    local_dir = os.getenv("LOCAL_DIR")
    output_dir = os.getenv("OUTPUT_DIR")
    check_memory_dir()
    # manage_output_dir(output_dir=output_dir)

    llm = LLM_Config(
        provider=os.getenv("PROVIDER"),
        model=os.getenv("MODEL"),
        base_url=os.getenv("BASE_URL"),
        temperature=float(os.getenv("TEMPERATURE")),
        max_tokens=int(os.getenv("MAX_TOKENS")),
        timeout=float(os.getenv("TIMEOUT")),
        callbacks=[print_output],
    )

    ltm = get_long_term_memory()
    stm = get_short_term_memory()
    entity = get_entity_memory()

    @agent
    def software_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["software_analyst"],
            verbose=True,
            allow_delegation=False,
            max_iter=10,
            memory=True,
            llm=self.llm,
        )
    
    @task
    def batch_analysis_task(self) -> Task:
        return Task(
            config=self.tasks_config["batch_analysis_task"], agent=self.batch_coordinator()
        )
    
    @task
    def analysis_task(self) -> Task:
        return Task(
            config=self.tasks_config["analysis_task"], agent=self.software_analyst()
        )

    @crew
    def crew(self) -> Crew:
        """Crea il crew CodeExplainer"""
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
