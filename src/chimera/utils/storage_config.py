"""
LlamaIndex-compatible memory configuration.

Provides simple helpers to construct memory buffers for:
- Long-term memory
- Short-term memory
- Entity memory

These return LlamaIndex ChatMemoryBuffer instances with configurable token limits.
"""

from typing import Optional
import os
from llama_index.core.memory import ChatMemoryBuffer

from .storage_qdrant import QdrantStorage


def _env_int(name: str, default: int) -> int:
    """Helper to read integer env values with defaults."""
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def get_long_term_memory(token_limit: Optional[int] = None) -> ChatMemoryBuffer:
    """
    Long-term memory as a larger ChatMemoryBuffer.

    Env override: LONG_TERM_TOKEN_LIMIT
    """
    limit = token_limit or _env_int("LONG_TERM_TOKEN_LIMIT", 8000)
    return ChatMemoryBuffer.from_defaults(token_limit=limit)


def get_short_term_memory(token_limit: Optional[int] = None) -> ChatMemoryBuffer:
    """
    Short-term memory as a smaller ChatMemoryBuffer.

    Env override: SHORT_TERM_TOKEN_LIMIT
    """
    limit = token_limit or _env_int("SHORT_TERM_TOKEN_LIMIT", 2000)
    return ChatMemoryBuffer.from_defaults(token_limit=limit)


def get_entity_memory(token_limit: Optional[int] = None) -> ChatMemoryBuffer:
    """
    Entity memory modeled as a medium ChatMemoryBuffer.

    Env override: ENTITY_TOKEN_LIMIT
    """
    limit = token_limit or _env_int("ENTITY_TOKEN_LIMIT", 4000)
    return ChatMemoryBuffer.from_defaults(token_limit=limit)