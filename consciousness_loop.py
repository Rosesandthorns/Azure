import logging
from datetime import datetime, timedelta
from typing import List

from model_interface import ModelInterface
from memory_service import MemoryEntry, MemoryService


class ConsciousnessLoop:
    def __init__(
        self,
        memory_service: MemoryService,
        model: ModelInterface,
        interval_seconds: int,
    ) -> None:
        self.memory_service = memory_service
        self.model = model
        self.interval = timedelta(seconds=interval_seconds)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.last_run = datetime.utcnow() - self.interval

    def should_run(self) -> bool:
        return datetime.utcnow() - self.last_run >= self.interval

    def run(self) -> List[MemoryEntry]:
        self.logger.info("Running consciousness loop")
        self.last_run = datetime.utcnow()
        recent = self.memory_service.list_memories(limit=10)
        if not recent:
            return []
        content = "\n".join([m.content for m in recent])
        messages = [
            {
                "role": "system",
                "content": (
                    "Generate a brief internal observation or idle thought. "
                    "Use uncertainty language and keep it short."
                ),
            },
            {"role": "user", "content": content},
        ]
        thought = self.model.generate(messages)
        entry = self.memory_service.create_memory(
            content=thought,
            memory_type="reflection",
            confidence=0.3,
            importance=0.2,
            user_id=recent[0].user_id,
            source_memory_ids=[m.id for m in recent],
        )
        return [entry]
