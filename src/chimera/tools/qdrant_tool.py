from crewai.tools import tool
import os
from dotenv import load_dotenv
import uuid
import time
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct




load_dotenv()
from qdrant_client.qdrant_fastembed import TextEmbedding
embedder = TextEmbedding(model_name=os.getenv("EMBEDDER", "jinaai/jina-embeddings-v2-base-en"))

collection_name = "crew_knowledge"
client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
#COLLECTION = "crew_knowledge"
COLLECTION = os.getenv("COLLECTION")

import uuid

def stable_id(text: str) -> str:
    norm = " ".join(text.split()).strip().lower()
    # return hashlib.sha256(norm.encode("utf-8")).hexdigest()
    return uuid.uuid5(uuid.NAMESPACE_URL, norm).hex

@tool("Write to knowledge base")
def upsert_knowledge(text: str, url: str) -> str:
    """
    Saves a short fact into the unique in-memory collection.

    The argument 'text' must be a string containing the information to be saved.
    Returns a confirmation message with the ID assigned to the saved information.

    Example call: upsert_knowledge(text="Mario's favorite color is blue.")
    """
    import hashlib, uuid
    content_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
    deterministic_id = str(uuid.UUID(content_hash[:32]))
    text_embedding = next(iter(embedder.embed(text + " " + url)))
    client.upsert(
        collection_name=collection_name,
        points=[
            PointStruct(
                id=deterministic_id,
                vector={
                    "fast-jina-embeddings-v2-base-en": text_embedding,
                },
                payload={
                    'document':text,
                    'source_url':url
                }
            )
        ],
    )
    return f"Saved id={deterministic_id}"

@tool("Search knowledge base")
def search_knowledge(query: str) -> str:
    """
    Performs a semantic search in the in-memory collection.

    The argument 'query' must be a string containing the question or concept to search for.
    Returns a string with the results found, each with the text and the relevance score.
    If no results are found, it returns "(no results)".

    Example call: search_knowledge(query="What is Mario's favorite color?")
    """
    query_embedding = next(iter(embedder.embed(query)))

    search_results = client.query_points(
        collection_name=collection_name,
        query=query_embedding,
        using= 'fast-jina-embeddings-v2-base-en',
        limit=10
    ).points
    res = []
    for result in search_results:
        res.append(f"{result.payload['document']}\n{result.payload['source_url']}\n")
        print(f"ID: {result.id}, Score: {result.score} Payload: {result.payload}")
    if not res:
        return "(no results)"
    return "\n".join(res)




























# @tool("Cerca informazioni nella knowledge base Qdrant")
# def qdrant_search_tool(query: str) -> str:
#     """
#     Performs a semantic search in the Qdrant knowledge base and returns formatted results.
#     """
#     try:
#         top_k = 5
#         res = client.query(
#             collection_name=os.getenv("COLLECTION", "crew_knowledge"),
#             query_text=query,
#             limit=top_k
#         )

#         if not res:
#             return f"(Nessun risultato trovato per: {query})"

#         lines = []
#         for r in res:
#             meta = r.metadata or {}
#             src = meta.get("source_url", "unknown source")
#             text = meta.get("document", "<no text available>")
#             score = f"{r.score:.3f}"
#             lines.append(f"- **{src}** (score={score})\n{text[:250]}...\n")

#         return "\n".join(lines)

#     except Exception as e:
#         return f"Errore durante la ricerca in Qdrant: {e}"


# @tool("Write in the knowledge base (Qdrant in RAM)")
# def upsert_knowledge(text: str, tag: str = "") -> str:
#     """Save the results in qdrant."""
#     _id = stable_id(text)
#     metadata = {"text": text,"ts": int(time.time())}

#     client.add(
#         collection_name=COLLECTION,
#         documents=[text],   # usa embedding di default (fastembed integrato)
#         metadata=[metadata],
#         ids=[_id]
#     )
#     return f"Salvato id={_id[:8]}"

# @tool("Cerca nella knowledge base (Qdrant in RAM)")
# def search_knowledge(query: str) -> str:
#     """ erforms a semantic search in Qdrant and returns formatted results
#      """
#     top_k=5
#     res = client.query(
#         collection_name=COLLECTION,
#         query_text=query,   # embedding di default
#         limit=int(top_k),
#         query_filter=None
#     )
#     if not res:
#         return "(nessun risultato)"
#     lines = []
#     for r in res:
#         meta = r.metadata or {}
#         lines.append(f"- {meta.get('text','<no text>')} (score={r.score:.3f})")
#     return "\n".join(lines)


































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


# def stable_id(text: str) -> str:
#     norm = " ".join(text.split()).strip().lower()
#     return uuid.uuid5(uuid.NAMESPACE_URL, norm).hex


# Definizione della funzione di upsert.
# - Calcola un ID stabile per il testo.
# - Crea un payload (`metadata`) con testo e timestamp.
# - Inserisce (o aggiorna) nel DB il documento con:
#     - **embedding** del testo (calcolato automaticamente se il client supporta `documents=[text]`)
#     - **metadati**
#     - **id** unico
# - Restituisce un messaggio di conferma.


# @tool("Write to knowledge base")
# def upsert_knowledge(text: str, client: QdrantStorage) -> str:
#     """Saves a text string inside the knowledge base"""
#     _id = stable_id(text)
#     metadata = {"text": text, "ts": int(time.time())}
#     client.save(
#         collection_name=os.environ.get("COLLECTION"),
#         documents=[text],
#         metadata=[metadata],
#         ids=[_id],
#     )
#     return f"Salvato id={_id[:8]}, text={text[:20]}"


# @tool("Search in knowledge base")
# def search_knowledge(
#     query: str,
#     client: QdrantStorage,
#     k: int = 5,
# ) -> str:
#     """Search for something in the knowledge base."""
#     res = client.search(
#         query=query,
#         limit=int(k),
#     )
#     if not res:
#         return "(nessun risultato)"
#     lines = []
#     for r in res:
#         meta = r.metadata or {}
#         lines.append(f"- {meta.get('text','<no text>')} (score={r.score:.3f}))")
#     return "\n".join(lines)
