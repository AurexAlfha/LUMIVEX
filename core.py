"""
LUMIVEX Core
Version: 0.1.0

Central orchestrator connecting the AI engine,
memory, knowledge and tools systems.
"""

from ai.engine import LumivexAI
from memory.memory import LumivexMemory
from knowledge.knowledge import LumivexKnowledge
from tools.tools import LumivexTools


class LumivexCore:
    """Central controller for the LUMIVEX AI system."""

    def __init__(self):
        self.ai = LumivexAI()
        self.memory = LumivexMemory()
        self.knowledge = LumivexKnowledge()
        self.tools = LumivexTools()

    def status(self):
        return {
            "system": "LUMIVEX",
            "version": "0.1.0",
            "ai": self.ai.status(),
            "memory": len(self.memory.memories),
            "knowledge": len(self.knowledge.items),
            "tools": self.tools.list_tools(),
        }

    def process(self, prompt: str):
        if not prompt.strip():
            return {
                "success": False,
                "error": "Prompt cannot be empty.",
            }

        self.memory.add(
            prompt,
            memory_type="conversation",
        )

        result = self.ai.generate(prompt)

        return {
            "success": result.success,
            "response": result.text,
            "model": result.model,
        }


if __name__ == "__main__":

    lumivex = LumivexCore()

    print(lumivex.status())

    response = lumivex.process(
        "Hello LUMIVEX!"
    )

    print(response)
