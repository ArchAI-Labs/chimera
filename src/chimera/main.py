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
    # Chiediamo all'utente il topic
    topic = input("Inserisci l'argomento principale dei post (topic): ").strip()
    if not topic:
        print("Nessun topic inserito, userò quello di default: Quantum Computing.")
        topic = "The latest news about quantum computing."

    # Chiediamo all'utente quanti post vuole generare
    try:
        num_posts = int(input("Quanti post vuoi generare? "))
    except ValueError:
        print("Input non valido, userò 3 post come default.")
        num_posts = 3

    # Chiediamo la frequenza delle pubblicazioni
    frequency = input("Vuoi che le pubblicazioni siano weekly, monthly o altro?: ").strip().lower()
    if frequency not in ["weekly", "monthly"]:
        print("Frequenza non riconosciuta, userò 'weekly' come default.")
        frequency = "weekly"
    
    # Chiediamo all'utente se preferisce lavorare con l'esperto generalista o con l'esperto prodotto.
    expert_type = input("Vuoi usare l'esperto 'generalista' o 'prodotto'? ")

    # Creiamo le cartelle di output in automatico
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    text_dir = os.path.join("output", "texts", f"{topic}_{timestamp}")
    image_dir = os.path.join("output", "images", f"{topic}_{timestamp}")

    os.makedirs(text_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    # Inputs per la crew
    inputs = {
        "topic": topic,
        "num_posts": num_posts,
        "frequency": frequency,
        "expert_type": expert_type,
        "text_dir": text_dir,
        "image_dir": image_dir,
    }

    try:
        print("Initializing LinkedIn Crew...")
        linkedin_crew = LinkedInCrew()

        # Esecuzione della pipeline gerarchica
        print(" Starting content creation pipeline...")
        print("=" * 50)

        crew_instance = linkedin_crew.crew()
        crew_instance.tasks = linkedin_crew.build_tasks_for(expert_type)
        result = crew_instance.kickoff(inputs=inputs)

        # Stampa un riepilogo a video
        if isinstance(result, str):
            print("LinkedIn Crew Output:")
            print(result[:500] + "..." if len(result) > 500 else result)
        else:
            print("LinkedIn Crew ha prodotto un output non testuale.")
            print(result)

        # Salvataggio del riepilogo generale in un file
        summary_file = os.path.join(text_dir, "linkedin_output.md")
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write(str(result))

        print(f"Risultato salvato in {summary_file}")

        return result

    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")
