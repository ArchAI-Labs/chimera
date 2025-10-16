from crewai.tools import tool
import os
from dotenv import load_dotenv
import uuid
import time
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client.qdrant_fastembed import TextEmbedding

# chunk dei testi.
splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)

# carica le variabili d'ambiente
load_dotenv()

# embedder e variabili d'ambiente
embedder = TextEmbedding(model_name=os.getenv("EMBEDDER", "jinaai/jina-embeddings-v2-base-en"))
collection_name = os.getenv("COLLECTION", "crew_knowledge")
VECTOR_NAME = os.getenv("VECTOR", "fast-jina-embeddings-v2-base-en")

# modalità di utilizzo
mode = os.getenv("QDRANT_MODE", "memory")
print(f"Qdrant running in {mode} mode.")

if mode == "memory":
    client = QdrantClient(":memory:")
elif mode == "cloud":
    client = QdrantClient(host=os.getenv("QDRANT_HOST"), api_key=os.getenv("QDRANT_API_KEY"))
elif mode == "docker":
    client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
else:
    raise ValueError("Qdrant has 3 modes: memory, cloud or docker")

VECTOR_SIZE = getattr(embedder, "embedding_size", 768)  # uses embedder size if available
DISTANCE = Distance.COSINE
existing = {c.name for c in client.get_collections().collections}
if collection_name not in existing:
    client.create_collection(
        collection_name=collection_name,
        vectors_config={VECTOR_NAME: VectorParams(size=VECTOR_SIZE, distance=DISTANCE)},
    )
# ---------------------------------------------------

def stable_id(text: str) -> str:
    norm = " ".join(text.split()).strip().lower()
    return uuid.uuid5(uuid.NAMESPACE_URL, norm).hex

@tool("Write to knowledge base")
def upsert_knowledge(text: str, url: str) -> str:
    """
    Saves a short fact into the unique in-memory collection.

    The argument 'text' must be a string containing the information to be saved.
    Returns a confirmation message with the ID assigned to the saved information.

    Example call: upsert_knowledge(text="Mario's favorite color is blue.", url="http:/website.com")
    """
    import hashlib, uuid
    chunks = splitter.split_text(text)
    for chunk in chunks:
        content_hash = hashlib.sha256(chunk.encode('utf-8')).hexdigest()
        deterministic_id = str(uuid.UUID(content_hash[:32]))
        text_embedding = next(iter(embedder.embed(chunk)))
        client.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(
                    id=deterministic_id,
                    vector={VECTOR_NAME: text_embedding},  # use env VECTOR consistently
                    payload={
                        'document': chunk,
                        'source_url': url
                    }
                )
            ],
        )
    return f"Saved data from {url}"

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
        using=VECTOR_NAME,  # same VECTOR as in upsert
        limit=10
    ).points
    res = []
    for result in search_results:
        res.append(f"{result.payload['document']}\n{result.payload['source_url']}\n")
    if not res:
        return "(no results)"
    return "\n".join(res)
