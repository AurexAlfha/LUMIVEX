"""
LUMIVEX Memory System
Version: 0.1.0

Foundation for short-term and long-term AI memory.
"""

from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class Memory:
    content: str
    memory_type: str = "general"


class LumivexMemory:
    """Basic memory manager for LUMIVEX."""

    def __init__(self):
        self.memories: List[Memory] = []

    def add(self, content: str, memory_type: str = "general") -> None:
        if not content.strip():
            return

        self.memories.append(
            Memory(
                content=content.strip(),
                memory_type=memory_type,
            )
        )

    def get_all(self) -> List[Dict[str, str]]:
        return [
            {
                "content": memory.content,
                "type": memory.memory_type,
            }
            for memory in self.memories
        ]

    def search(self, query: str) -> List[Dict[str, str]]:
        query_words = query.lower().split()

        results = []

        for memory in self.memories:
            content = memory.content.lower()

            if any(word in content for word in query_words):
                results.append(
                    {
                        "content": memory.content,
                        "type": memory.memory_type,
                    }
                )

        return results


if __name__ == "__main__":
    memory = LumivexMemory()

    memory.add(
        "LUMIVEX is a continuously improving AI project.",
        "project",
    )

    print(memory.get_all())
    print(memory.search("improving AI"))
