from crewai.tools import tool
from utils.storage_qdrant import QdrantStorage
import os
from dotenv import load_dotenv
import uuid
from time import time

load_dotenv()




# possibile metodo per la configurazione qdrant:

# def get_qdrant_config():
#     collection = os.getenv("QDRANT_COLLECTION", "crew_knowledge")
#     embedder = os.getenv("EMBEDDER", "jinaai/jina-embeddings-v2-base-en")
#     mode = os.getenv("QDRANT_MODE", "memory").lower()
#     if mode == "memory":
#         return QdrantConfig(
#            host=":memory:",
#            collection_name=collection,
#            embedding_model=embedder)


# definizione degli id. Ad ogni id è associato un vettore in qdrant.
# scriviamo questo metodo stable_id.

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
def upsert_knowledge(text: str, client: QdrantStorage) -> str:
    """Saves a text string inside the knowledge base"""
    _id = stable_id(text)
    metadata = {"text": text, "ts": int(time.time())}
    client.save(
        collection_name=os.environ.get("COLLECTION"),
        documents=[text],
        metadata=[metadata],
        ids=[_id],
    )
    return f"Salvato id={_id[:8]}, text={text[:20]}"


@tool("Search in knowledge base")
def search_knowledge(
    query: str,
    client: QdrantStorage,
    k: int = 5,
) -> str:
    """Search for something in the knowledge base."""
    res = client.search(
        query=query,
        limit=int(k),
    )
    if not res:
        return "(nessun risultato)"
    lines = []
    for r in res:
        meta = r.metadata or {}
        lines.append(f"- {meta.get('text','<no text>')} (score={r.score:.3f}))")
    return "\n".join(lines)
