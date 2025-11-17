"""
Unified memory and Qdrant configuration for LlamaIndex.

This class centralizes:
- Qdrant client initialization and hybrid vector store creation
- Embedding model selection with dynamic dimension detection
- Index initialization, chunking helpers, and CRUD operations
- Optional ChatMemoryBuffer helpers and per-scope memory indexes

Environment variables:
- QDRANT_MODE: memory | docker | cloud (default: memory)
- QDRANT_URL, QDRANT_HOST, QDRANT_PORT, QDRANT_API_KEY
- QDRANT_COLLECTION (default: linkedin_knowledge)
- QDRANT_SPARSE_MODEL (default: Qdrant/bm25)
- EMBEDDER, PROVIDER, OPENAI_API_KEY
- CHUNK_SIZE, CHUNK_OVERLAP
- LONG_TERM_TOKEN_LIMIT, SHORT_TERM_TOKEN_LIMIT, ENTITY_TOKEN_LIMIT
- LONG_TERM_COLLECTION, SHORT_TERM_COLLECTION, ENTITY_COLLECTION
"""

from typing import Optional, Tuple, List, Dict, Any
import os
import hashlib

from llama_index.core import VectorStoreIndex, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.tools import FunctionTool
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.vector_stores.types import (
    VectorStoreQuery,
    MetadataFilters,
    ExactMatchFilter,
)
from qdrant_client import QdrantClient


class MemoryConfig:
    """Central manager for Qdrant-backed hybrid retrieval and memory helpers."""

    def __init__(
        self,
        collection_name: Optional[str] = None,
        sparse_model: Optional[str] = None,
    ) -> None:
        self.collection_name = collection_name or os.getenv("QDRANT_COLLECTION", "linkedin_knowledge")
        self.sparse_model = sparse_model or os.getenv("QDRANT_SPARSE_MODEL", "Qdrant/bm25")

        # Lazy-initialized resources
        self._client: Optional[QdrantClient] = None
        self._vector_store: Optional[QdrantVectorStore] = None
        self._index: Optional[VectorStoreIndex] = None
        self._embed_model = None
        self._embed_dim = 384

    # -------- Env helpers --------
    @staticmethod
    def env_int(name: str, default: int) -> int:
        try:
            return int(os.getenv(name, str(default)))
        except Exception:
            return default

    # -------- Qdrant client & embeddings --------
    def get_qdrant_client(self) -> QdrantClient:
        """Initialize a Qdrant client from environment variables."""
        if self._client is not None:
            return self._client

        mode = os.getenv("QDRANT_MODE", "memory")
        url = os.getenv("QDRANT_URL")
        host = os.getenv("QDRANT_HOST", "localhost")
        port = int(os.getenv("QDRANT_PORT", "6333"))
        api_key = os.getenv("QDRANT_API_KEY")

        if mode == "memory":
            self._client = QdrantClient(location=":memory:")
        elif mode == "docker":
            self._client = QdrantClient(url=url or f"http://{host}:{port}")
        elif mode == "cloud":
            if url:
                self._client = QdrantClient(url=url, api_key=api_key)
            else:
                self._client = QdrantClient(host=host, port=port, api_key=api_key)
        else:
            self._client = QdrantClient(host=host, port=port)
        return self._client

    def _select_embedding_model(self) -> Tuple[object, int]:
        """Select embedding model and compute dimension dynamically."""
        if self._embed_model is not None:
            return self._embed_model, self._embed_dim

        embedder_name = (os.getenv("EMBEDDER") or "").strip()
        provider = (os.getenv("PROVIDER") or "ollama").lower()
        embed_model = None
        dim = 384

        try:
            if embedder_name and embedder_name.lower() != "none":
                try:
                    from llama_index.embeddings.fastembed import FastEmbedEmbedding
                    embed_model = FastEmbedEmbedding(model_name=embedder_name)
                except Exception:
                    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
                    embed_model = HuggingFaceEmbedding(model_name=embedder_name)
            else:
                if provider == "openai" and os.getenv("OPENAI_API_KEY"):
                    from llama_index.embeddings.openai import OpenAIEmbedding
                    embed_model = OpenAIEmbedding(
                        api_key=os.getenv("OPENAI_API_KEY"),
                        model="text-embedding-3-small",
                    )
                else:
                    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
                    model_name = (
                        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                        if provider == "gemini"
                        else "sentence-transformers/all-MiniLM-L6-v2"
                    )
                    embed_model = HuggingFaceEmbedding(model_name=model_name)

            if embed_model is not None:
                try:
                    vec = embed_model.get_text_embedding("hello world")
                    dim = len(vec) if hasattr(vec, "__len__") else dim
                except Exception:
                    dim = 1536 if provider == "openai" else 384
            else:
                dim = 384
        except Exception:
            embed_model = None
            dim = 384

        self._embed_model = embed_model
        self._embed_dim = dim
        return embed_model, dim

    # -------- Index initialization --------
    def initialize_index(self, collection_name: Optional[str] = None) -> Optional[VectorStoreIndex]:
        """Create a hybrid-enabled Qdrant vector store and VectorStoreIndex."""
        if self._index is not None and (collection_name is None or collection_name == self.collection_name):
            return self._index

        if collection_name is not None:
            self.collection_name = collection_name

        client = self.get_qdrant_client()
        self._vector_store = QdrantVectorStore(
            client=client,
            collection_name=self.collection_name,
            enable_hybrid=True,
            fastembed_sparse_model=self.sparse_model,
        )

        embed_model, _ = self._select_embedding_model()
        if embed_model:
            self._index = VectorStoreIndex.from_vector_store(
                vector_store=self._vector_store,
                embed_model=embed_model,
            )
        else:
            self._index = None
        return self._index

    # -------- Chunking helpers --------
    @staticmethod
    def get_chunking_params(default_size: int = 512, default_overlap: int = 50) -> Tuple[int, int]:
        try:
            size = int(os.getenv("CHUNK_SIZE", str(default_size)))
        except Exception:
            size = default_size
        try:
            overlap = int(os.getenv("CHUNK_OVERLAP", str(default_overlap)))
        except Exception:
            overlap = default_overlap
        size = max(1, size)
        overlap = max(0, min(overlap, size - 1))
        return size, overlap

    # -------- Operations --------
    def document_exists(self, text: str, collection_name: Optional[str] = None) -> bool:
        index = self.initialize_index(collection_name)
        if index is None:
            return False
        sha512 = hashlib.sha512(text.encode("utf-8")).hexdigest()
        filters = MetadataFilters(filters=[ExactMatchFilter(key="content_sha512", value=sha512)])
        try:
            result = index.vector_store.query(
                VectorStoreQuery(query_str="check duplicate", similarity_top_k=1, filters=filters)
            )
            return bool(getattr(result, "nodes", []) or getattr(result, "ids", []))
        except Exception:
            return False

    def upsert_knowledge(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        collection_name: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> str:
        index = self.initialize_index(collection_name)
        if index is None:
            return "Error: Qdrant index unavailable (embedding not configured)."

        # Skip if document already exists (by content hash)
        if self.document_exists(text, collection_name):
            return "Skipped: document already exists in knowledge base."

        if chunk_size is None or chunk_overlap is None:
            env_size, env_overlap = self.get_chunking_params()
            chunk_size = chunk_size if chunk_size is not None else env_size
            chunk_overlap = chunk_overlap if chunk_overlap is not None else env_overlap

        doc_metadata = metadata or {}
        doc_metadata["content_sha512"] = hashlib.sha512(text.encode("utf-8")).hexdigest()
        document = Document(text=text, metadata=doc_metadata)

        splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        nodes = splitter.get_nodes_from_documents([document])

        index.insert_nodes(nodes)
        return (
            f"Successfully added {len(nodes)} chunks to knowledge base. "
            f"Text length approximately: {len(text.split())} words."
        )

    def batch_upsert_knowledge(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        collection_name: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> str:
        index = self.initialize_index(collection_name)
        if index is None:
            return "Error: Qdrant index unavailable (embedding not configured)."

        if chunk_size is None or chunk_overlap is None:
            env_size, env_overlap = self.get_chunking_params()
            chunk_size = chunk_size if chunk_size is not None else env_size
            chunk_overlap = chunk_overlap if chunk_overlap is not None else env_overlap

        # Filter out duplicates by content hash before inserting
        documents = []
        skipped = 0
        for idx, text in enumerate(texts):
            if self.document_exists(text, collection_name):
                skipped += 1
                continue
            metadata = metadatas[idx] if metadatas and idx < len(metadatas) else {}
            metadata = dict(metadata) if metadata else {}
            metadata["content_sha512"] = hashlib.sha512(text.encode("utf-8")).hexdigest()
            documents.append(Document(text=text, metadata=metadata))

        splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        nodes = splitter.get_nodes_from_documents(documents)
        index.insert_nodes(nodes, show_progress=True)
        if not documents:
            return f"Skipped: all {len(texts)} documents already exist."
        return (
            f"Successfully added {len(documents)} new document(s) "
            f"({len(nodes)} chunks); skipped {skipped} duplicate(s)."
        )

    def search_knowledge(
        self,
        query: str,
        top_k: int = 5,
        collection_name: Optional[str] = None,
    ) -> str:
        index = self.initialize_index(collection_name)
        if index is None:
            return "Error: Qdrant vector index is not available (embedding setup failed)."

        # Use retriever-only path to avoid requiring an LLM when testing.
        # This returns top-k matches as plain text without synthesis.
        try:
            retriever = index.as_retriever(similarity_top_k=top_k)
            nodes = retriever.retrieve(query) or []
        except Exception as e:
            msg = str(e).lower()
            if "not found" in msg or "collection" in msg:
                return (
                    "Knowledge base is empty or not initialized yet. "
                    "Try adding content with upsert_knowledge first."
                )
            return f"Error querying knowledge base: {e}"

        lines = [f"Knowledge Base Search Results for: '{query}'", ""]
        if not nodes:
            lines.append("No relevant documents found.")
        else:
            lines.append("Top matches:")
            for i, n in enumerate(nodes, 1):
                score = getattr(n, "score", "N/A")
                text_preview = (n.get_text() if hasattr(n, "get_text") else getattr(n, "text", ""))[:300]
                lines.append(f"{i}. [Score: {score}]\n{text_preview}...\n")

        return "\n".join(lines)

    def delete_knowledge(self, doc_ids: List[str], collection_name: Optional[str] = None) -> str:
        index = self.initialize_index(collection_name)
        if index is None:
            return "Error: Qdrant index unavailable (embedding not configured)."
        for doc_id in doc_ids:
            index.delete_ref_doc(doc_id)
        return f"Successfully deleted {len(doc_ids)} documents from knowledge base."

    # -------- ChatMemoryBuffer helpers --------
    def get_long_term_memory(self, token_limit: Optional[int] = None) -> ChatMemoryBuffer:
        limit = token_limit or self.env_int("LONG_TERM_TOKEN_LIMIT", 8000)
        return ChatMemoryBuffer.from_defaults(token_limit=limit)

    def get_short_term_memory(self, token_limit: Optional[int] = None) -> ChatMemoryBuffer:
        limit = token_limit or self.env_int("SHORT_TERM_TOKEN_LIMIT", 2000)
        return ChatMemoryBuffer.from_defaults(token_limit=limit)

    def get_entity_memory(self, token_limit: Optional[int] = None) -> ChatMemoryBuffer:
        limit = token_limit or self.env_int("ENTITY_TOKEN_LIMIT", 4000)
        return ChatMemoryBuffer.from_defaults(token_limit=limit)

    # -------- Per-scope indexes --------
    def get_long_term_memory_index(self) -> Optional[VectorStoreIndex]:
        collection = os.getenv("LONG_TERM_COLLECTION", "memory_long_term")
        return self.initialize_index(collection)

    def get_short_term_memory_index(self) -> Optional[VectorStoreIndex]:
        collection = os.getenv("SHORT_TERM_COLLECTION", "memory_short_term")
        return self.initialize_index(collection)

    def get_entity_memory_index(self) -> Optional[VectorStoreIndex]:
        collection = os.getenv("ENTITY_COLLECTION", "memory_entity")
        return self.initialize_index(collection)

    # -------- Optional tool factory --------
    def create_tools(self) -> List[FunctionTool]:
        """Expose search/upsert as LlamaIndex FunctionTools."""
        search_tool = FunctionTool.from_defaults(
            fn=lambda query, top_k=5: self.search_knowledge(query, top_k=top_k, collection_name=self.collection_name),
            name="search_knowledge",
            description=(
                "Search the knowledge base for relevant information. "
                "Input should be a search query string. "
                "Returns relevant information from the knowledge base with sources."
            ),
        )
        upsert_tool = FunctionTool.from_defaults(
            fn=lambda text, metadata=None: self.upsert_knowledge(text, metadata=metadata, collection_name=self.collection_name),
            name="upsert_knowledge",
            description=(
                "Add or update information in the knowledge base. "
                "Input should be the text content to add. "
                "Use this to store important information for later retrieval."
            ),
        )
        return [search_tool, upsert_tool]