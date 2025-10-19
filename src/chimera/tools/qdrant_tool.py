from crewai.tools import tool
import os
from dotenv import load_dotenv
import uuid
import time
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client.qdrant_fastembed import TextEmbedding


# -----------------------------------------------------------
# Text chunking setup
# Splits large text inputs into smaller overlapping segments
# for better vectorization and semantic search accuracy.
# -----------------------------------------------------------
splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)


# -----------------------------------------------------------
# Environment variable loading
# This loads configuration values (e.g., API keys, modes, etc.)
# from a .env file using python-dotenv.
# -----------------------------------------------------------
load_dotenv()


# -----------------------------------------------------------
# Embedding and Qdrant configuration
# Retrieves the embedding model, collection name, and vector field
# from environment variables (with safe defaults).
# -----------------------------------------------------------
embedder = TextEmbedding(
    model_name=os.getenv("EMBEDDER", "jinaai/jina-embeddings-v2-base-en")
)
collection_name = os.getenv("COLLECTION", "crew_knowledge")
VECTOR_NAME = os.getenv("VECTOR", "fast-jina-embeddings-v2-base-en")


# -----------------------------------------------------------
# Qdrant mode selection
# Determines where the vector database will run:
# - "memory" → in-memory (RAM)
# - "cloud" → remote hosted Qdrant instance
# - "docker" → local containerized Qdrant service
# -----------------------------------------------------------
mode = os.getenv("QDRANT_MODE", "memory")
print(f"Qdrant running in {mode} mode.")

if mode == "memory":
    client = QdrantClient(":memory:")
elif mode == "cloud":
    client = QdrantClient(
        host=os.getenv("QDRANT_HOST"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )
elif mode == "docker":
    client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
else:
    raise ValueError("Qdrant has 3 modes: memory, cloud or docker")


# -----------------------------------------------------------
# Create collection if it doesn't exist
# Checks whether the specified collection already exists,
# and creates it if missing. The vector size is inferred
# from the embedder.
# -----------------------------------------------------------
VECTOR_SIZE = getattr(embedder, "embedding_size", 768)
DISTANCE = Distance.COSINE
existing = {c.name for c in client.get_collections().collections}

if collection_name not in existing:
    client.create_collection(
        collection_name=collection_name,
        vectors_config={VECTOR_NAME: VectorParams(size=VECTOR_SIZE, distance=DISTANCE)},
    )


# -----------------------------------------------------------
# Deterministic ID generator
# Produces a stable UUID based on normalized text content.
# Ensures that identical text always maps to the same ID.
# -----------------------------------------------------------
def stable_id(text: str) -> str:
    norm = " ".join(text.split()).strip().lower()
    return uuid.uuid5(uuid.NAMESPACE_URL, norm).hex


# -----------------------------------------------------------
# Tool: Write to knowledge base
# Inserts (or updates) information in the Qdrant collection.
# Automatically skips text chunks that already exist to
# prevent duplicates.
# -----------------------------------------------------------
@tool("Write to knowledge base")
def upsert_knowledge(text: str, url: str) -> str:
    """
    Saves a short fact into the unique in-memory collection.

    The argument 'text' must be a string containing the information to be saved.
    Returns a confirmation message with the ID assigned to the saved information.

    Example call: upsert_knowledge(text="Mario's favorite color is blue.", url="http:/website.com")
    """
    import hashlib, uuid

    # Split the input text into semantic chunks
    chunks = splitter.split_text(text)

    for chunk in chunks:
        # Create a deterministic ID from the text content
        content_hash = hashlib.sha256(chunk.encode("utf-8")).hexdigest()
        deterministic_id = str(uuid.UUID(content_hash[:32]))

        # Check if this chunk already exists in the collection
        try:
            existing = client.retrieve(
                collection_name=collection_name, ids=[deterministic_id]
            )

            # If an entry with this ID already exists, skip it
            if existing and existing[0].payload:
                print(f"Point already exists for {url}, skipping...")
                continue

        except Exception:
            # If the ID is not found, proceed with insertion
            pass

        # Generate an embedding and store the new data point
        text_embedding = next(iter(embedder.embed(chunk)))
        client.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(
                    id=deterministic_id,
                    vector={VECTOR_NAME: text_embedding},
                    payload={"document": chunk, "source_url": url},
                )
            ],
        )

    return f"Saved data from {url}"


# -----------------------------------------------------------
# Tool: Search knowledge base
# Performs a semantic vector search to retrieve the most
# relevant stored text segments related to a user query.
# -----------------------------------------------------------
@tool("Search knowledge base")
def search_knowledge(query: str) -> str:
    """
    Performs a semantic search in the in-memory collection.

    The argument 'query' must be a string containing the question or concept to search for.
    Returns a string with the results found, each with the text and the relevance score.
    If no results are found, it returns "(no results)".

    Example call: search_knowledge(query="What is Mario's favorite color?")
    """
    # Generate embedding for the input query
    query_embedding = next(iter(embedder.embed(query)))

    # Query Qdrant for semantically similar vectors
    search_results = client.query_points(
        collection_name=collection_name,
        query=query_embedding,
        using=VECTOR_NAME,
        limit=10,
    ).points

    # Collect retrieved chunks and their sources
    results = []
    for result in search_results:
        results.append(
            f"{result.payload['document']}\n{result.payload['source_url']}\n"
        )

    # Return formatted response or fallback message
    if not results:
        return "(no results)"
    return "\n".join(results)
