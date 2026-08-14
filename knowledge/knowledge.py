"""
LUMIVEX Knowledge System
Version: 0.1.0

Foundation for storing and retrieving knowledge.
"""

from dataclasses import dataclass
from typing import List, Dict


@dataclass
class KnowledgeItem:
    title: str
    content: str
    source: str = "internal"


class LumivexKnowledge:
    """Basic knowledge manager for LUMIVEX."""

    def __init__(self):
        self.items: List[KnowledgeItem] = []

    def add(
        self,
        title: str,
        content: str,
        source: str = "internal",
    ) -> None:

        if not title.strip() or not content.strip():
            return

        self.items.append(
            KnowledgeItem(
                title=title.strip(),
                content=content.strip(),
                source=source,
            )
        )

    def get_all(self) -> List[Dict[str, str]]:
        return [
            {
                "title": item.title,
                "content": item.content,
                "source": item.source,
            }
            for item in self.items
        ]

    def search(self, query: str) -> List[Dict[str, str]]:
        query_words = query.lower().split()

        results = []

        for item in self.items:
            text = (
                item.title + " " + item.content
            ).lower()

            if any(word in text for word in query_words):
                results.append(
                    {
                        "title": item.title,
                        "content": item.content,
                        "source": item.source,
                    }
                )

        return results


if __name__ == "__main__":
    knowledge = LumivexKnowledge()

    knowledge.add(
        "LUMIVEX",
        "LUMIVEX is a continuously evolving AI platform.",
        "project",
    )

    print(knowledge.get_all())
    print(knowledge.search("AI platform"))
