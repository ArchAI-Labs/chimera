from crewai.tools import tool
import os
from dotenv import load_dotenv
import uuid
import time
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)

load_dotenv()
from qdrant_client.qdrant_fastembed import TextEmbedding
embedder = TextEmbedding(model_name=os.getenv("EMBEDDER", "jinaai/jina-embeddings-v2-base-en"))

collection_name = os.getenv("COLLECTION")
client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
#COLLECTION = "crew_knowledge"
# COLLECTION = os.getenv("COLLECTION")

if os.getenv("QDRANT_MODE") == "memory":
    client = QdrantClient(":memory:")
elif os.getenv("QDRANT_MODE") == "cloud":
    client = QdrantClient(host=os.getenv("QDRANT_HOST"), api_key=os.getenv("QDRANT_API_KEY"))
elif os.getenv("QDRANT_MODE") == "docker":
    client = QdrantClient(url=os.getenv("QDRANT_URL"))
else:
    raise ValueError("Qdrant has 3 mode: memory, cloud or docker")

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
                    vector={
                        "fast-jina-embeddings-v2-base-en": text_embedding,
                    },
                    payload={
                        'document':chunk,
                        'source_url':url
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
        using= os.getenv("VECTOR"),
        limit=10
    ).points
    res = []
    for result in search_results:
        res.append(f"{result.payload['document']}\n{result.payload['source_url']}\n")
        # print(f"ID: {result.id}, Score: {result.score} Payload: {result.payload}")
    if not res:
        return "(no results)"
    return "\n".join(res)