from typing import Any, Dict, List, Optional
import os, hashlib, uuid
from crewai.memory.storage.rag_storage import RAGStorage
from qdrant_client import QdrantClient
from qdrant_client.qdrant_fastembed import TextEmbedding
import warnings


warnings.filterwarnings("ignore", category=DeprecationWarning, module="qdrant_client")

class QdrantStorage(RAGStorage):
    def __init__(self, type, allow_reset=True, embedder_config=None, crew=None):
        super().__init__(type, allow_reset, embedder_config, crew)
        self._initialize_app()
        #  Usa sempre lo stesso modello di embedding
        self.embedder = TextEmbedding(model_name=os.getenv("EMBEDDER", "jinaai/jina-embeddings-v2-base-en"))

    def search(self, query: str, limit: int = 3, filter: Optional[dict] = None, score_threshold: float = 0) -> List[Any]:
        print(f"🔍 [QdrantStorage] Query ricevuta: '{query}'")
        query_vector = self.embedder.embed(query)[0]

        response = self.client.search(
            collection_name=self.type,
            query_vector=query_vector,
            limit=limit,
            score_threshold=score_threshold
        )

        results = [
            {
                "id": point.id,
                "metadata": point.payload,
                "context": point.payload.get("document", ""),
                "score": point.score,
            }
            for point in response
        ]

        print(f"📊 [QdrantStorage] Risultati trovati: {len(results)}")
        return results

    def reset(self) -> None:
        self.client.delete_collection(self.type)

    def _initialize_app(self):
        if os.getenv("QDRANT_MODE") == "memory":
            self.client = QdrantClient(":memory:")
        elif os.getenv("QDRANT_MODE") == "cloud":
            self.client = QdrantClient(host=os.getenv("QDRANT_HOST"), api_key=os.getenv("QDRANT_API_KEY"))
        elif os.getenv("QDRANT_MODE") == "docker":
            self.client = QdrantClient(url=os.getenv("QDRANT_URL"))
        else:
            raise ValueError("Qdrant has 3 mode: memory, cloud or docker")

        self.client._embedding_model_name = os.getenv("EMBEDDER")
        if not self.client.collection_exists(self.type):
            self.client.create_collection(
                collection_name=self.type,
                vectors_config=self.client.get_fastembed_vector_params(),
                sparse_vectors_config=self.client.get_fastembed_sparse_vector_params(),
            )

    def save(self, value: Any, metadata: Dict[str, Any]) -> None:
        content_hash = hashlib.sha256(value.encode('utf-8')).hexdigest()
        deterministic_id = str(uuid.UUID(content_hash[:32]))

        metadata = metadata or {}
        metadata.update({
            'content_hash': content_hash,
            'document': value  # 🔹 Necessario per il RAG
        })

        self.client.add(
            collection_name=self.type,
            documents=[value],
            metadata=[metadata],
            ids=[deterministic_id]
        )

        print(f"💾 [QdrantStorage] Documento salvato in {self.type} con ID {deterministic_id[:8]}...")
