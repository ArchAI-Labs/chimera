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
from crewai.rag.qdrant.config import QdrantConfig
from qdrant_client import QdrantClient
import time 
import uuid


######################tools###############

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



#possibile metodo per la configurazione qdrant:

# def get_qdrant_config():
#     collection = os.getenv("QDRANT_COLLECTION", "crew_knowledge")
#     embedder = os.getenv("EMBEDDER", "jinaai/jina-embeddings-v2-base-en")
#     mode = os.getenv("QDRANT_MODE", "memory").lower()
#     if mode == "memory":
#         return QdrantConfig(
#            host=":memory:",
#            collection_name=collection,
#            embedding_model=embedder) 


# Crea client dalla configurazione che gli abbiamo passato.
#qdrant_config = get_qdrant_config()
#client = qdrant_config.create_client()
COLLECTION = "crew_knowledge"  #stringa con cui chiamiamo la colletion dell'informazione.
#COLLECTION = qdrant_config.collection_name
client = QdrantClient(":memory:") 



# definizione degli id. Ad ogni id è associato un vettore in qdrant.
#scriviamo questo metodo stable_id.

# - Normalizza il testo (spazi, minuscole).
# - Genera un **UUID deterministico** dal testo.
# - Così lo stesso testo avrà sempre lo stesso ID.

def stable_id(text: str) -> str:
    norm = " ".join(text.split()).strip().lower()
    return uuid.uuid5(uuid.NAMESPACE_URL, norm).hex


# Definizione della funzione di upsert.
# - Calcola un ID stabile per il testo.
# - Crea un payload (`metadata`) con testo e timestamp.
# - Inserisce (o aggiorna) nel DB il documento con:
#     - **embedding** del testo (calcolato automaticamente se il client supporta `documents=[text]`)
#     - **metadati**
#     - **id** unico
# - Restituisce un messaggio di conferma.



@tool("Write to knowledge base")
def upsert_knowledge(text: str) -> str:
    """Saves a text string inside the knowledge base"""
    _id = stable_id(text)
    metadata = {"text": text, "ts": int(time.time())}
    client.add(
        collection_name=COLLECTION,
        documents=[text],
        metadata=[metadata],
        ids=[_id]
    )
    return f"Salvato id={_id[:8]}"


@tool("Search in knowledge base")
def search_knowledge(query: str) -> str:
    """Search for something in the knowledge base."""
    top_k = 5
    res = client.query(
        collection_name=COLLECTION,
        query_text=query,  
        limit=int(top_k),
        query_filter=None
    )
    if not res:
        return "(nessun risultato)"
    lines = []
    for r in res:
        meta = r.metadata or {}
        lines.append(f"- {meta.get('text','<no text>')} (score={r.score:.3f}))")
    return "\n".join(lines)


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
            allow_delegation=True,
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
        product_sites = os.getenv("PRODUCT_SITES")
        sites_list = [s.strip() for s in product_sites.split(",") if s.strip()]
        return Agent(
            config=self.agents_config["product_expert"],
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            tools=[self.scraper_tool],
            description=f"Product expert with knowledge from company sources: {sites_list}"
        )
    
    @agent
    def knowledge_manager(self) -> Agent:
        return Agent(
            config=self.agents_config["knowledge_manager"],
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            tools=[upsert_knowledge, search_knowledge],
            description=f"Reads and Writes on knowledge"
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
        )




# memory=True,
            # long_term_memory=self.ltm,
            # short_term_memory=self.stm,
            # entity_memory=self.entity,
