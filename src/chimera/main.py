import warnings
from .crew import LinkedInCrew  # importa la crew

# Ignora alcuni warning interni che non servono agli studenti
warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

def run():
    """
    Entry point per CrewAI.
    Questo viene richiamato dal comando `crewai run`.
    """
    # Chiediamo all'utente il topic
    topic = input("📌 Inserisci l'argomento principale dei post (topic): ").strip()
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
    frequency = input("📅 Vuoi che le pubblicazioni siano settimanali, mensili o con un'altra frequenza?: ").strip().lower()
    if frequency not in ["settimanali", "mensili"]:
        print("Frequenza non riconosciuta, userò 'settimanali' come default.")
        frequency = "settimanali"

    # Impostiamo gli input della crew
    inputs = {
        "topic": topic,
        "num_posts": num_posts,
        "frequency": frequency,
    }

    try:
        print("🚀 Initializing LinkedIn Crew...")
        linkedin_crew = LinkedInCrew()

        # Esecuzione sequenziale della pipeline
        print("📌 Starting content creation pipeline...")
        print("=" * 50)

        result = linkedin_crew.crew().kickoff(inputs=inputs)

        # Stampa un riepilogo a video
        if isinstance(result, str):
            print("✅ LinkedIn Crew Output:")
            print(result[:500] + "..." if len(result) > 500 else result)
        else:
            print("✅ LinkedIn Crew ha prodotto un output non testuale.")
            print(result)

        # Salvataggio su file markdown
        with open("linkedin_output.md", "w", encoding="utf-8") as f:
            f.write(str(result))

        print("📂 Risultato salvato in linkedin_output.md")

        return result

    except Exception as e:
        raise Exception(f"❌ An error occurred while running the crew: {e}")
