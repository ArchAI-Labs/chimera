"""
Lightweight wrappers around MemoryConfig for memory helpers.

This module retains the original function names while delegating
to `utils.memory_config.MemoryConfig` to avoid duplication.
"""

from typing import Optional
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core import VectorStoreIndex
from utils.memory_config import MemoryConfig


_mc = MemoryConfig()


def get_long_term_memory(token_limit: Optional[int] = None) -> ChatMemoryBuffer:
    return _mc.get_long_term_memory(token_limit)


def get_short_term_memory(token_limit: Optional[int] = None) -> ChatMemoryBuffer:
    return _mc.get_short_term_memory(token_limit)


def get_entity_memory(token_limit: Optional[int] = None) -> ChatMemoryBuffer:
    return _mc.get_entity_memory(token_limit)


def get_long_term_memory_index() -> Optional[VectorStoreIndex]:
    return _mc.get_long_term_memory_index()


def get_short_term_memory_index() -> Optional[VectorStoreIndex]:
    return _mc.get_short_term_memory_index()


def get_entity_memory_index() -> Optional[VectorStoreIndex]:
    return _mc.get_entity_memory_index()