"""
Qdrant tool wrappers delegating to MemoryConfig.

This module now centralizes calls through `utils.memory_config.MemoryConfig`
to avoid duplication and ensure consistent hybrid search behavior.
It preserves the original function names for backward compatibility.
"""

from typing import List, Dict, Any, Optional
import os

from llama_index.core import VectorStoreIndex
from llama_index.core.tools import FunctionTool
from utils.memory_config import MemoryConfig


_mc = MemoryConfig()

def initialize_qdrant_store(
    collection_name: str = os.getenv("QDRANT_COLLECTION", "linkedin_knowledge"),
) -> Optional[VectorStoreIndex]:
    """Initialize and return the Qdrant-backed index via MemoryConfig."""
    return _mc.initialize_index(collection_name)


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
    return _mc.search_knowledge(query=query, top_k=top_k, collection_name=collection_name)


# Module-level helper for env-driven chunking
def _get_chunking_params(default_size: int = 512, default_overlap: int = 50) -> tuple[int, int]:
    """Compatibility shim; delegates to MemoryConfig."""
    return _mc.get_chunking_params(default_size, default_overlap)


def document_exists(
    text: str,
    collection_name: str = os.getenv("QDRANT_COLLECTION", "linkedin_knowledge")
) -> bool:
    return _mc.document_exists(text=text, collection_name=collection_name)

def upsert_knowledge(
    text: str,
    metadata: Optional[Dict[str, Any]] = None,
    collection_name: str = os.getenv("QDRANT_COLLECTION", "linkedin_knowledge"),
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None
) -> str:
    return _mc.upsert_knowledge(
        text=text,
        metadata=metadata,
        collection_name=collection_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def batch_upsert_knowledge(
    texts: List[str],
    metadatas: Optional[List[Dict[str, Any]]] = None,
    collection_name: str = "linkedin_knowledge",
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None
) -> str:
    return _mc.batch_upsert_knowledge(
        texts=texts,
        metadatas=metadatas,
        collection_name=collection_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def delete_knowledge(
    doc_ids: List[str],
    collection_name: str = "linkedin_knowledge"
) -> str:
    return _mc.delete_knowledge(doc_ids=doc_ids, collection_name=collection_name)


# Create LlamaIndex tools
def create_qdrant_tools() -> List[FunctionTool]:
    """Expose MemoryConfig tools (search and upsert) for agent usage."""
    return _mc.create_tools()