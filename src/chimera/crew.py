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
#from crewai.memory.external.external_memory import ExternalMemory




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
    # ltm = get_long_term_memory()
    # stm = get_short_term_memory()
    # entity = get_entity_memory()

    # Tool
    if os.environ.get("SERPER_API_KEY"):
        web_search_tool = SerperDevTool()
    else:
        web_search_tool = MyCustomDuckDuckGoTool()
    
    file_writer_tool = FileWriterTool()
    dalle_tool = DallETool(model="dall-e-3", size="1024x1024", quality="standard", n=1)
    scraper_tool = ScrapeWebsiteTool()

    # # Tool RAG che utilizza la tua classe QdrantStorage personalizzata
    # # RagTool si occuperà di istanziare QdrantStorage internamente

    qdrant_client = QdrantStorage(type=os.getenv("COLLECTION", "default_collection")) #usa collection nell'env altrimenti default.
    #qdrant_rag_tool = RagTool(rag_storage=qdrant_client)
    #modifica
    # qdrant_rag_tool = RagTool(
    # rag_storage=qdrant_client,
    # name="qdrant_rag_tool",
    # description="Use this tool to search the internal company knowledge base stored in Qdrant to extract information about the topic.")
    
    def _scrape_and_load_data(self):
        print("🚀 Starting data ingestion from PRODUCT_SITES variable...")
        product_sites_str = os.getenv("PRODUCT_SITES")
        
        if not product_sites_str:
            print("❌ Errore: La variabile d'ambiente PRODUCT_SITES non è stata trovata.")
            return
        
        sites_list = [site.strip() for site in product_sites_str.split(',')]
        scraper = self.scraper_tool 

        # Usa LangChain per il chunking
        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
        except ImportError:
            print("LangChain non installato. Installa con: pip install langchain")
            return

        splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)

        for site_url in sites_list:
            if not site_url:
                continue
            
            print(f"Scraping data from: {site_url} ...")
            try:
                scraped_content = scraper.run(website_url=site_url)

                #  Pulizia da boilerplate
                if "The following text is scraped website content:" in scraped_content:
                    scraped_content = scraped_content.split("The following text is scraped website content:")[-1].strip()

                #  Preprocessing base
                import re
                scraped_content = re.sub(r'\s+', ' ', scraped_content).strip()

                # Skip contenuti inutili o vuoti
                if not scraped_content or "404" in scraped_content or len(scraped_content) < 200:
                    print(f"Contenuto vuoto o non valido da {site_url}, skip.")
                    continue

                # Chunking
                chunks = splitter.split_text(scraped_content)
                print(f"Suddiviso in {len(chunks)} chunk per l'indicizzazione.")

                # Salvataggio in Qdrant
                for i, chunk in enumerate(chunks):
                    self.qdrant_client.save(
                        value=chunk, 
                        metadata={
                            'source_url': site_url,
                            'chunk_index': i
                        }
                    )

                print(f"✅ Successfully scraped and indexed {len(chunks)} chunks from {site_url}")

            except Exception as e:
                print(f"❌ Failed to process {site_url}. Error: {e}")
        print("Data ingestion complete!")

        # =============== Agenti ===============
    
    @agent
    def manager(self) -> Agent:
        return Agent(
            config=self.agents_config["manager"],
            verbose=True,
            allow_delegation=False,
            tools=[self.web_search_tool],
            llm=self.manager_llm,
            max_iter=5)
    
    @agent
    def generalist_expert(self) -> Agent:
        return Agent(
            config=self.agents_config["generalist_expert"],
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            tools=[self.web_search_tool],
            max_iter=5
        )

    @agent
    def product_expert(self) -> Agent:
        #AGGIUNTA DI UN LOG PER VEDERE QUANDO SUBENTRA
        print("Attivato agente: Product Expert (usa RAG + Scraper)")
        return Agent(
            config=self.agents_config["product_expert"],
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            #memory=True,
            tools=[search_knowledge, upsert_knowledge, self.scraper_tool],  #[self.scraper_tool]
            max_iter=10
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
        return Task(config=self.tasks_config["technical_content"], agent=self.generalist_expert())

    @task
    def product_content_task(self) -> Task:
        #AGGIUNTA PRIMO LOG.
        print(f"Sto costruendo il task con topic: {self.inputs.get('topic')}")
        return Task(config=self.tasks_config["product_content"], agent=self.product_expert())
    
    @task
    def linkedin_post_task(self) -> Task:
        return Task(config=self.tasks_config["linkedin_post"], agent=self.copywriter())

    @task
    def visuals_task(self) -> Task:
        return Task(config=self.tasks_config["visuals"], agent=self.designer())

    @task
    def plan_posts_task(self) -> Task:
        return Task(config=self.tasks_config["plan_posts"], agent=self.planner())
    
    def __init__(self, inputs=None):
        # 1. Chiama il costruttore della classe genitore (CrewBase).
        #    QUESTA È LA RIGA FONDAMENTALE CHE RISOLVE TUTTO.
        super().__init__()

        # 2. Ora gestisci i tuoi input in modo sicuro.
        self.inputs = inputs if inputs is not None else {}
        #AGGIUNTA PER VEDERE L'INPUT- SECONDO LOG
        print(f"INPUT RICEVUTI DALL’UTENTE: {self.inputs}")


        # 3. Avvia il caricamento dei dati nella knowledge base.
        #    La logica è di nuovo incapsulata e si avvia alla creazione.
        # self._scrape_and_load_data()

    # =============== Crew ===============
    @crew
    def crew(self) -> Crew:
        expert_type = self.inputs.get("expert_type", "generalista")


        print(f"EXPERT TYPE {expert_type}")

        if expert_type == "generalista":
            print("YOU ARE IN GENERALIST CREW")
            return Crew(
                agents=[
                    self.manager(), 
                    self.generalist_expert(),
                    self.copywriter(), 
                    self.designer(), 
                    self.planner()
                    ],
                tasks=[
                    self.editorial_plan_task(), 
                    self.technical_content_task(), 
                    self.linkedin_post_task(), 
                    self.visuals_task(), 
                    self.plan_posts_task()
                       ],
                process=Process.sequential,
                verbose=True,
                chat_llm=self.llm,
                # memory=True,
                # long_term_memory=self.ltm,
                # short_term_memory=self.stm,
                # entity_memory=self.entity,
            )
        elif expert_type == "prodotto":
            print("YOU ARE IN PRODUCT CREW")
            return Crew(
                agents=[self.manager(), 
                        self.product_expert(),
                        self.copywriter(), 
                        self.designer(), 
                        self.planner()],
                tasks=[self.editorial_plan_task(), 
                       self.product_content_task(), 
                       self.linkedin_post_task(), 
                       self.visuals_task(), 
                       self.plan_posts_task()],
                process=Process.sequential,
                verbose=True,
                chat_llm=self.llm,
                #external_memory=ExternalMemory(storage=self.qdrant_client)
                # memory=True,
                # long_term_memory=self.ltm,
                # short_term_memory=self.stm,
                # entity_memory=self.entity,
            )
        else:
            print("YOU ARE IN ELSE CREW")
            return Crew(
                agents=self.agents,
                tasks=self.tasks,
                process=Process.sequential,
                verbose=True,
                chat_llm=self.llm,
                # memory=True,
                # long_term_memory=self.ltm,
                # short_term_memory=self.stm,
                # entity_memory=self.entity,
            )
