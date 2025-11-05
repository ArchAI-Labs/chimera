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
    
    if mode == "memory":
        return QdrantClient(location=":memory:")
    else:
        host = os.getenv("QDRANT_HOST", "localhost")
        port = int(os.getenv("QDRANT_PORT", "6333"))
        return QdrantClient(host=host, port=port)


# Global Qdrant client and vector store
_qdrant_client = None
_vector_store = None
_index = None


def initialize_qdrant_store(
    collection_name: str = "linkedin_knowledge",
    embedding_dim: int = 384  # Changed from 1536 (OpenAI) to 384 (sentence-transformers default)
):
    """
    Initialize Qdrant vector store and index
    
    Args:
        collection_name: Name of the Qdrant collection
        embedding_dim: Dimension of embeddings (384 for sentence-transformers, 1536 for OpenAI)
    """
    global _qdrant_client, _vector_store, _index
    
    if _qdrant_client is None:
        _qdrant_client = get_qdrant_client()
        
        # Determine embedding dimension based on provider
        provider = os.getenv("PROVIDER", "ollama").lower()
        if provider == "openai":
            embedding_dim = 1536
        else:
            embedding_dim = 384  # sentence-transformers default
        
        # Create collection if it doesn't exist
        try:
            _qdrant_client.get_collection(collection_name)
        except:
            _qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=embedding_dim,
                    distance=Distance.COSINE
                )
            )
        
        # Create vector store
        _vector_store = QdrantVectorStore(
            client=_qdrant_client,
            collection_name=collection_name
        )
        
        # Create embedding model based on provider
        if os.getenv("OPENAI_API_KEY") and provider == "openai":
            from llama_index.embeddings.openai import OpenAIEmbedding
            embed_model = OpenAIEmbedding(
                api_key=os.getenv("OPENAI_API_KEY"),
                model="text-embedding-ada-002"
            )
        else:
            # Use HuggingFace embeddings as fallback (no API key needed)
            try:
                from llama_index.embeddings.huggingface import HuggingFaceEmbedding
                embed_model = HuggingFaceEmbedding(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
                print("✓ Using HuggingFace embeddings (open-source)")
            except ImportError:
                print("⚠️  HuggingFace embeddings not available, install with:")
                print("   pip install llama-index-embeddings-huggingface sentence-transformers")
                # Use a basic embedding as last resort
                from llama_index.core.embeddings import BaseEmbedding
                embed_model = None
        
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
    collection_name: str = "linkedin_knowledge"
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


def upsert_knowledge(
    text: str,
    metadata: Optional[Dict[str, Any]] = None,
    collection_name: str = "linkedin_knowledge",
    chunk_size: int = 512,
    chunk_overlap: int = 50
) -> str:
    """
    Add or update information in the knowledge base
    
    Args:
        text: Text content to add to the knowledge base
        metadata: Optional metadata dictionary
        collection_name: Qdrant collection name
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        
    Returns:
        Success message or error
    """
    try:
        # Initialize if needed
        index = initialize_qdrant_store(collection_name=collection_name)
        
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
        
        return (
            f"Successfully added {len(nodes)} chunks to knowledge base. "
            f"Text length: {len(text)} characters."
        )
        
    except Exception as e:
        return f"Error adding to knowledge base: {str(e)}"


def batch_upsert_knowledge(
    texts: List[str],
    metadatas: Optional[List[Dict[str, Any]]] = None,
    collection_name: str = "linkedin_knowledge"
) -> str:
    """
    Batch add documents to the knowledge base
    
    Args:
        texts: List of text contents
        metadatas: Optional list of metadata dictionaries
        collection_name: Qdrant collection name
        
    Returns:
        Success message or error
    """
    try:
        # Initialize if needed
        index = initialize_qdrant_store(collection_name=collection_name)
        
        # Create documents
        documents = []
        for idx, text in enumerate(texts):
            metadata = metadatas[idx] if metadatas and idx < len(metadatas) else {}
            documents.append(Document(text=text, metadata=metadata))
        
        # Split into chunks
        splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
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