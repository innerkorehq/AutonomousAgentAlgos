"""Long-term memory: one small protocol, adapters for the popular libraries, and event-native glue.

from aaa.memory import NativeMemory, MemoryHandler, memory_tools, Scope
"""

from aaa.memory.base import (
    Deletable,
    Forgettable,
    Listable,
    Memory,
    MissingMemoryCapability,
    Profiled,
    Updatable,
    require,
    supports,
)
from aaa.memory.callable import CallableMemory
from aaa.memory.fake import FakeMemory
from aaa.memory.handler import MemoryHandler, WriteBack
from aaa.memory.native import NativeMemory, llm_extractor
from aaa.memory.tools import memory_tools
from aaa.memory.types import AddResult, MemoryEntry, MemoryItem, Scope, render_memories

__all__ = [
    "AddResult",
    "CallableMemory",
    "Deletable",
    "FakeMemory",
    "Forgettable",
    "Listable",
    "Memory",
    "MemoryEntry",
    "MemoryHandler",
    "MemoryItem",
    "MissingMemoryCapability",
    "NativeMemory",
    "Profiled",
    "Scope",
    "Updatable",
    "WriteBack",
    "llm_extractor",
    "memory_tools",
    "render_memories",
    "require",
    "supports",
]
