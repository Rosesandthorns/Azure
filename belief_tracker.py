import logging
from dataclasses import dataclass
from typing import List

from memory_service import MemoryEntry, MemoryService


@dataclass
class BeliefUpdate:
    memory_id: str
    new_confidence: float
    reason: str


class BeliefTracker:
    def __init__(self, memory_service: MemoryService) -> None:
        self.memory_service = memory_service
        self.logger = logging.getLogger(self.__class__.__name__)

    def detect_contradictions(
        self, new_memory: MemoryEntry, candidate_memories: List[MemoryEntry]
    ) -> List[BeliefUpdate]:
        updates: List[BeliefUpdate] = []
        new_tokens = set(new_memory.content.lower().split())
        for memory in candidate_memories:
            if memory.id == new_memory.id:
                continue
            memory_tokens = set(memory.content.lower().split())
            overlap = new_tokens.intersection(memory_tokens)
            if not overlap:
                continue
            if "not" in new_tokens or "no" in new_tokens:
                new_confidence = max(0.1, memory.confidence - 0.3)
                updates.append(
                    BeliefUpdate(
                        memory_id=memory.id,
                        new_confidence=new_confidence,
                        reason="Detected negation overlap",
                    )
                )
        return updates

    def apply_updates(self, updates: List[BeliefUpdate]) -> None:
        for update in updates:
            self.logger.info(
                "Updating belief %s confidence to %.2f due to %s",
                update.memory_id,
                update.new_confidence,
                update.reason,
            )
            self.memory_service.update_memory_confidence(
                update.memory_id, update.new_confidence
            )
