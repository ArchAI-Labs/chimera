import warnings
import os
from .crew import LinkedInCrew  # importa la crew
from datetime import datetime

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

#-----------------------------------------------------------
# This main file is intended to let you run your Crew locally.
# keep this file focused on running and testing your Crew configuration.
# You can modify the input parameters below to test different
# setups. The system will automatically interpolate any tasks
# and agents defined in the Crew.
#------------------------------------------------------------


def run():
    """
    Entry point per CrewAI.
    """

    expert_type = (
        input("Vuoi usare l'esperto 'generalista' o 'prodotto'? ").strip().lower()
    )
    if expert_type not in ["generalista", "prodotto"]:
        print("Input non valido, userò 'generalista' come default.")
        expert_type = "generalista"

    topic = None
    topic = input("Inserisci l'argomento principale dei post (topic): ").strip()

    try:
        num_posts = int(input("Quanti post vuoi generare? "))
    except ValueError:
        print("Input non valido, userò 3 post come default.")
        num_posts = 3

    frequency = (
        input("Vuoi che le pubblicazioni siano weekly, monthly o altro?: ")
        .strip()
        .lower()
    )
    if frequency not in ["weekly", "monthly"]:
        print("Frequenza non riconosciuta, userò 'weekly' come default.")
        frequency = "weekly"

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    text_dir = os.path.join("output", "texts", f"{expert_type}_{timestamp}")
    image_dir = os.path.join("output", "images", f"{expert_type}_{timestamp}")

    os.makedirs(text_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    product_sites_str = os.getenv("PRODUCT_SITES", "")
    sites_list = [s.strip() for s in product_sites_str.split(",") if s.strip()]

    inputs = {
        "num_posts": num_posts,
        "frequency": frequency,
        "expert_type": expert_type,
        "text_dir": text_dir,
        "image_dir": image_dir,
        "topic": topic,
        "product_site": sites_list if sites_list else "",
    }
    print(inputs)

    print("Initializing LinkedIn Crew...")

    crew_builder = LinkedInCrew(inputs=inputs)

    crew_builder.inputs = inputs

    linkedin_crew = crew_builder.crew()

    print("Starting content creation pipeline...")
    print("=" * 60)

    result = linkedin_crew.kickoff(inputs=inputs)

    if isinstance(result, str):
        print("LinkedIn Crew Output:")
        print(result[:500] + "..." if len(result) > 500 else result)
    else:
        print("LinkedIn Crew ha prodotto un output non testuale.")
        print(result)

    summary_file = os.path.join(text_dir, "linkedin_output.md")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(str(result))

    print(f"✅ Risultato salvato in {summary_file}")
    return result
