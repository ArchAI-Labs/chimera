"""
Qdrant knowledge base tools for LlamaIndex
Migrated from CrewAI implementation with RAG capabilities
"""
from typing import List, Dict, Any, Optional
import os
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.tools import FunctionTool
from llama_index.embeddings.openai import OpenAIEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams


# Initialize Qdrant client and vector store
def get_qdrant_client():
    """Get or create Qdrant client based on configuration"""
    mode = os.getenv("QDRANT_MODE", "memory")
    url = os.getenv("QDRANT_URL")
    host = os.getenv("QDRANT_HOST", "localhost")
    port = int(os.getenv("QDRANT_PORT", "6333"))
    api_key = os.getenv("QDRANT_API_KEY")

    if mode == "memory":
        return QdrantClient(location=":memory:")
    elif mode == "docker":
        return QdrantClient(url=url or f"http://{host}:{port}")
    elif mode == "cloud":
        if url:
            return QdrantClient(url=url, api_key=api_key)
        return QdrantClient(host=host, port=port, api_key=api_key)
    else:
        return QdrantClient(host=host, port=port)


# Global Qdrant client and vector store
_qdrant_client = None
_vector_store = None
_index = None


def _select_embedding_model():
    """Return (embed_model, dim) based on EMBEDDER or PROVIDER, computing dim from 'hello world'."""
    embedder_name = (os.getenv("EMBEDDER") or "").strip()
    provider = (os.getenv("PROVIDER") or "ollama").lower()
    embed_model = None
    dim = 384

    try:
        if embedder_name and embedder_name.lower() != "none":
            # Prefer FastEmbed for custom models
            try:
                from llama_index.embeddings.fastembed import FastEmbedEmbedding
                embed_model = FastEmbedEmbedding(model_name=embedder_name)
                print(f"✓ Using FastEmbed with model '{embedder_name}'")
            except Exception as fe_err:
                print(f"⚠️ FastEmbed unavailable ({fe_err}); falling back to HuggingFace '{embedder_name}'")
                from llama_index.embeddings.huggingface import HuggingFaceEmbedding
                embed_model = HuggingFaceEmbedding(model_name=embedder_name)
        else:
            if provider == "openai" and os.getenv("OPENAI_API_KEY"):
                from llama_index.embeddings.openai import OpenAIEmbedding
                embed_model = OpenAIEmbedding(
                    api_key=os.getenv("OPENAI_API_KEY"),
                    model="text-embedding-3-small"
                )
                print("✓ Using OpenAI embeddings (text-embedding-3-small)")
            else:
                # Provider fallback: multilingual for 'gemini', fast general for others
                from llama_index.embeddings.huggingface import HuggingFaceEmbedding
                model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" if provider == "gemini" else "sentence-transformers/all-MiniLM-L6-v2"
                embed_model = HuggingFaceEmbedding(model_name=model_name)
                print(f"✓ Using HuggingFace embeddings ({model_name})")

        # Compute dimension dynamically
        try:
            vec = embed_model.get_text_embedding("hello world")
            dim = len(vec) if hasattr(vec, "__len__") else 384
            print(f"✓ Computed embedding dimension: {dim}")
        except Exception as dim_err:
            print(f"⚠️ Could not compute embedding dimension: {dim_err}")
            # Reasonable defaults
            if provider == "openai":
                dim = 1536
            else:
                dim = 384

    except Exception as e:
        print(f"⚠️ Embedding setup failed: {e}")
        embed_model = None
        dim = 384

    return embed_model, dim

def initialize_qdrant_store(
    collection_name: str = os.getenv("QDRANT_COLLECTION", "linkedin_knowledge"),
    embedding_dim: int = 384
):
    """
    Initialize Qdrant vector store and index
    """
    global _qdrant_client, _vector_store, _index
    
    if _qdrant_client is None:
        _qdrant_client = get_qdrant_client()
        
        # Create vector store
        _vector_store = QdrantVectorStore(
            client=_qdrant_client,
            collection_name=collection_name
        )

        # Select embedding and compute dim from 'hello world'
        embed_model, computed_dim = _select_embedding_model()
        embedding_dim = computed_dim

        # Ensure collection exists with correct dim
        try:
            _qdrant_client.get_collection(collection_name)
        except Exception:
            _qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=embedding_dim,
                    distance=Distance.COSINE
                )
            )

        # Create index
        if embed_model:
            _index = VectorStoreIndex.from_vector_store(
                vector_store=_vector_store,
                embed_model=embed_model
            )
        else:
            print("⚠️  Vector store disabled - no embedding model available")
            _index = None
    
    return _index


def search_knowledge(
    query: str,
    top_k: int = 5,
    collection_name: str = os.getenv("QDRANT_COLLECTION", "linkedin_knowledge")
) -> str:
    """
    Search the knowledge base for relevant information
    
    Args:
        query: Search query string
        top_k: Number of top results to return
        collection_name: Qdrant collection name
        
    Returns:
        Formatted string with search results
    """
    try:
        # Initialize if needed
        index = initialize_qdrant_store(collection_name=collection_name)
        if index is None:
            return "Error: Qdrant vector index is not available (embedding setup failed)."
        
        # Create query engine
        query_engine = index.as_query_engine(
            similarity_top_k=top_k,
            response_mode="tree_summarize"
        )
        
        # Execute query
        response = query_engine.query(query)
        
        # Format results
        result_text = f"Knowledge Base Search Results for: '{query}'\n\n"
        result_text += f"Answer: {response.response}\n\n"
        
        if hasattr(response, 'source_nodes') and response.source_nodes:
            result_text += "Source Documents:\n"
            for idx, node in enumerate(response.source_nodes, 1):
                score = node.score if hasattr(node, 'score') else 'N/A'
                result_text += f"\n{idx}. [Score: {score}]\n{node.text[:300]}...\n"
        
        return result_text
        
    except Exception as e:
        return f"Error searching knowledge base: {str(e)}"


# Module-level helper for env-driven chunking
def _get_chunking_params(default_size: int = 512, default_overlap: int = 50) -> tuple[int, int]:
    """Read CHUNK_SIZE and CHUNK_OVERLAP from env with safe defaults and validation."""
    try:
        size = int(os.getenv("CHUNK_SIZE", str(default_size)))
    except Exception:
        size = default_size
    try:
        overlap = int(os.getenv("CHUNK_OVERLAP", str(default_overlap)))
    except Exception:
        overlap = default_overlap

    # ensure sensible values: size >= 1, 0 <= overlap < size
    size = max(1, size)
    overlap = max(0, min(overlap, size - 1))
    return size, overlap


def upsert_knowledge(
    text: str,
    metadata: Optional[Dict[str, Any]] = None,
    collection_name: str = os.getenv("QDRANT_COLLECTION", "linkedin_knowledge"),
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None
) -> str:
    """
    Add or update information in the knowledge base

    Args:
        text: Text content to add to the knowledge base
        metadata: Optional metadata dictionary
        collection_name: Qdrant collection name
        chunk_size: Size of text chunks (uses CHUNK_SIZE env if None)
        chunk_overlap: Overlap between chunks (uses CHUNK_OVERLAP env if None)

    Returns:
        Success message or error
    """
    try:
        # Initialize if needed
        index = initialize_qdrant_store(collection_name=collection_name)

        # Resolve chunking params from env if not provided
        if chunk_size is None or chunk_overlap is None:
            env_size, env_overlap = _get_chunking_params()
            chunk_size = chunk_size if chunk_size is not None else env_size
            chunk_overlap = chunk_overlap if chunk_overlap is not None else env_overlap

        # Create document
        doc_metadata = metadata or {}
        document = Document(text=text, metadata=doc_metadata)

        # Split into chunks
        splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        nodes = splitter.get_nodes_from_documents([document])

        # Insert into index
        index.insert_nodes(nodes)
        print(f"✓ Upserted {len(nodes)} chunks into Qdrant collection '{collection_name}'")
        return (
            f"Successfully added {len(nodes)} chunks to knowledge base. "
            f"Text length: {len(text)} characters."
        )

    except Exception as e:
        return f"Error adding to knowledge base: {str(e)}"


def batch_upsert_knowledge(
    texts: List[str],
    metadatas: Optional[List[Dict[str, Any]]] = None,
    collection_name: str = "linkedin_knowledge",
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None
) -> str:
    """
    Batch add documents to the knowledge base

    Args:
        texts: List of text contents
        metadatas: Optional list of metadata dictionaries
        collection_name: Qdrant collection name
        chunk_size: Size of text chunks (uses CHUNK_SIZE env if None)
        chunk_overlap: Overlap between chunks (uses CHUNK_OVERLAP env if None)

    Returns:
        Success message or error
    """
    try:
        # Initialize if needed
        index = initialize_qdrant_store(collection_name=collection_name)

        # Resolve chunking params from env if not provided
        if chunk_size is None or chunk_overlap is None:
            env_size, env_overlap = _get_chunking_params()
            chunk_size = chunk_size if chunk_size is not None else env_size
            chunk_overlap = chunk_overlap if chunk_overlap is not None else env_overlap

        # Create documents
        documents = []
        for idx, text in enumerate(texts):
            metadata = metadatas[idx] if metadatas and idx < len(metadatas) else {}
            documents.append(Document(text=text, metadata=metadata))

        # Split into chunks
        splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        nodes = splitter.get_nodes_from_documents(documents)

        # Insert into index
        index.insert_nodes(nodes)

        return (
            f"Successfully added {len(documents)} documents "
            f"({len(nodes)} chunks) to knowledge base."
        )

    except Exception as e:
        return f"Error batch adding to knowledge base: {str(e)}"


def delete_knowledge(
    doc_ids: List[str],
    collection_name: str = "linkedin_knowledge"
) -> str:
    """
    Delete documents from the knowledge base
    
    Args:
        doc_ids: List of document IDs to delete
        collection_name: Qdrant collection name
        
    Returns:
        Success message or error
    """
    try:
        index = initialize_qdrant_store(collection_name=collection_name)
        
        # Delete from vector store
        for doc_id in doc_ids:
            index.delete_ref_doc(doc_id)
        
        return f"Successfully deleted {len(doc_ids)} documents from knowledge base."
        
    except Exception as e:
        return f"Error deleting from knowledge base: {str(e)}"


# Create LlamaIndex tools
def create_qdrant_tools() -> List[FunctionTool]:
    """
    Create LlamaIndex function tools for Qdrant operations
    
    Returns:
        List of FunctionTool instances
    """
    search_tool = FunctionTool.from_defaults(
        fn=search_knowledge,
        name="search_knowledge",
        description=(
            "Search the knowledge base for relevant information. "
            "Input should be a search query string. "
            "Returns relevant information from the knowledge base with sources."
        )
    )
    
    upsert_tool = FunctionTool.from_defaults(
        fn=upsert_knowledge,
        name="upsert_knowledge",
        description=(
            "Add or update information in the knowledge base. "
            "Input should be the text content to add. "
            "Use this to store important information for later retrieval."
        )
    )
    
    return [search_tool, upsert_tool]