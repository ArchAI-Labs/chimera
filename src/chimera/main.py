import warnings
import os
from .crew import LinkedInCrew  # importa la crew
from datetime import datetime

# Ignora alcuni warning interni che non servono agli studenti
warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")


def run():
    """
    Entry point per CrewAI.
    """

    # Chiediamo all'utente se vuole contenuti generali o sul prodotto
    expert_type = input("Vuoi usare l'esperto 'generalista' o 'prodotto'? ").strip().lower()
    if expert_type not in ["generalista", "prodotto"]:
        print("Input non valido, userò 'generalista' come default.")
        expert_type = "generalista"

    # Se l'utente sceglie GENERALISTA → serve il topic
    topic = None
    # if expert_type == "generalista":
    topic = input("Inserisci l'argomento principale dei post (topic): ").strip()
        # if not topic:
        #     print("Nessun topic inserito, userò quello di default: Quantum Computing.")
        #     topic = "The latest news about quantum computing."

    # Numero di post
    try:
        num_posts = int(input("Quanti post vuoi generare? "))
    except ValueError:
        print("Input non valido, userò 3 post come default.")
        num_posts = 3

    # Frequenza
    frequency = input("Vuoi che le pubblicazioni siano weekly, monthly o altro?: ").strip().lower()
    if frequency not in ["weekly", "monthly"]:
        print("Frequenza non riconosciuta, userò 'weekly' come default.")
        frequency = "weekly"

    # Creiamo cartelle di output uniche per data/ora
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    text_dir = os.path.join("output", "texts", f"{expert_type}_{timestamp}")
    image_dir = os.path.join("output", "images", f"{expert_type}_{timestamp}")

    os.makedirs(text_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    product_sites_str = os.getenv("PRODUCT_SITES", "")
    sites_list = [s.strip() for s in product_sites_str.split(",") if s.strip()]

    # Inputs per la crew
    inputs = {
        "num_posts": num_posts,
        "frequency": frequency,
        "expert_type": expert_type,
        "text_dir": text_dir,
        "image_dir": image_dir,
        "topic": topic,
        "product_site": sites_list if sites_list else ""
    }
    print(inputs)


    # try:
    print("Initializing LinkedIn Crew...")
    #linkedin_crew = LinkedInCrew().crew()

    # 1. Crea un'istanza della classe
    crew_builder = LinkedInCrew(inputs=inputs)
    
    # 2. Assegna il dizionario 'inputs' all'attributo dell'istanza MODIFICA
    crew_builder.inputs = inputs

    # 3. Ora chiama il metodo .crew() sull'istanza che contiene gli input
    linkedin_crew = crew_builder.crew() 

    print("Starting content creation pipeline...")
    print("=" * 60)


    # Avvio della crew
    result = linkedin_crew.kickoff(inputs=inputs)

    # Output a video
    if isinstance(result, str):
        print("LinkedIn Crew Output:")
        print(result[:500] + "..." if len(result) > 500 else result)
    else:
        print("LinkedIn Crew ha prodotto un output non testuale.")
        print(result)

    # Salvataggio in file riassuntivo
    summary_file = os.path.join(text_dir, "linkedin_output.md")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(str(result))

    print(f"✅ Risultato salvato in {summary_file}")
    return result

    # except Exception as e:
    #     raise Exception(f"An error occurred while running the crew: {e}")
