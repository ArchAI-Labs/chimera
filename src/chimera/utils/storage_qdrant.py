from typing import Any, Dict, List, Optional
import os, hashlib, uuid
from crewai.memory.storage.rag_storage import RAGStorage
from qdrant_client import QdrantClient
from qdrant_client.qdrant_fastembed import TextEmbedding
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning, module="qdrant_client")


class QdrantStorage(RAGStorage):
    """

    This class provides methods to store, search, and reset vector data
    using a Qdrant database instance. It allows both in-memory and
    persistent modes (local Docker or cloud).

    """

    def __init__(self, type, allow_reset=True, embedder_config=None, crew=None):
        """
        Args:
            type (str): Name of the Qdrant collection to use.
            allow_reset (bool): Whether collection reset is allowed.
            embedder_config (dict, optional): Custom embedder configuration.
            crew (Crew, optional): Optional Crew reference.
        """
        super().__init__(type, allow_reset, embedder_config, crew)
        self._initialize_app()

        # Always use a consistent embedding model for storage and retrieval
        self.embedder = TextEmbedding(
            model_name=os.getenv("EMBEDDER", "jinaai/jina-embeddings-v2-base-en")
        )

    # -----------------------------------------------------------
    # Semantic search
    # Performs similarity search over stored vectors in Qdrant.
    # -----------------------------------------------------------
    def search(
        self,
        query: str,
        limit: int = 3,
        filter: Optional[dict] = None,
        score_threshold: float = 0,
    ) -> List[Any]:
        """
        Perform a semantic search in the Qdrant collection.

        Args:
            query (str): The search query text.
            limit (int, optional): Maximum number of results to return. Defaults to 3.
            filter (dict, optional): Optional metadata filter.
            score_threshold (float, optional): Minimum relevance score threshold.

        Returns:
            List[dict]: List of search results with ID, metadata, and score.
        """
        print(f"[QdrantStorage] Received query: '{query}'")

        # Convert query to embedding vector
        query_vector = self.embedder.embed(query)[0]

        # Perform vector search in Qdrant
        response = self.client.search(
            collection_name=self.type,
            query_vector=query_vector,
            limit=limit,
            score_threshold=score_threshold,
        )

        # Format response into a list of result dictionaries
        results = [
            {
                "id": point.id,
                "metadata": point.payload,
                "context": point.payload.get("document", ""),
                "score": point.score,
            }
            for point in response
        ]

        print(f"[QdrantStorage] Results found: {len(results)}")
        return results

    # -----------------------------------------------------------
    # Reset collection
    # Deletes and resets the entire collection.
    # -----------------------------------------------------------
    def reset(self) -> None:
        self.client.delete_collection(self.type)

    # -----------------------------------------------------------
    # Qdrant initialization
    # Chooses connection mode (memory, docker, cloud) dynamically.
    # -----------------------------------------------------------
    def _initialize_app(self):
        """Initialize the Qdrant client based on environment configuration."""
        mode = os.getenv("QDRANT_MODE")
        if mode == "memory":
            self.client = QdrantClient(":memory:")
        elif mode == "cloud":
            self.client = QdrantClient(
                host=os.getenv("QDRANT_HOST"), api_key=os.getenv("QDRANT_API_KEY")
            )
        elif mode == "docker":
            self.client = QdrantClient(url=os.getenv("QDRANT_URL"))
        else:
            raise ValueError("Qdrant has 3 modes: memory, cloud or docker")

        # Set embedder info and ensure the collection exists
        self.client._embedding_model_name = os.getenv("EMBEDDER")
        if not self.client.collection_exists(self.type):
            self.client.create_collection(
                collection_name=self.type,
                vectors_config=self.client.get_fastembed_vector_params(),
                sparse_vectors_config=self.client.get_fastembed_sparse_vector_params(),
            )

    # -----------------------------------------------------------
    # Save document
    # Adds a new document vector and metadata to the Qdrant collection.
    # -----------------------------------------------------------
    def save(self, value: Any, metadata: Dict[str, Any]) -> None:
        """
        Save a document and its metadata into the Qdrant collection.

        Args:
            value (Any): Text or content to be embedded and stored.
            metadata (dict): Additional metadata (e.g., source URL, chunk index).
        """
        # Compute a deterministic hash to avoid duplicates
        content_hash = hashlib.sha256(value.encode("utf-8")).hexdigest()
        deterministic_id = str(uuid.UUID(content_hash[:32]))

        # Enrich metadata with document info
        metadata = metadata or {}
        metadata.update(
            {
                "content_hash": content_hash,
                "document": value,
            }
        )

        # Add new vector to Qdrant
        self.client.add(
            collection_name=self.type,
            documents=[value],
            metadata=[metadata],
            ids=[deterministic_id],
        )

        print(
            f"[QdrantStorage] Document saved in '{self.type}' "
            f"with ID {deterministic_id[:8]}..."
        )
