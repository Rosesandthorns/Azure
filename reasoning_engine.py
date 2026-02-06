import logging
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple

from belief_tracker import BeliefTracker
from memory_service import MemoryEntry, MemoryService


@dataclass
class ReasoningOutcome:
    new_memories: List[MemoryEntry]
    belief_updates: List[str]
    contradictions: List[str]


class ReasoningEngine:
    def __init__(self, memory_service: MemoryService, belief_tracker: BeliefTracker) -> None:
        self.memory_service = memory_service
        self.belief_tracker = belief_tracker
        self.logger = logging.getLogger(self.__class__.__name__)

    def connect_memories(
        self, new_memory: MemoryEntry, retrieved_memories: List[MemoryEntry]
    ) -> ReasoningOutcome:
        contradictions = []
        belief_updates = []
        updates = self.belief_tracker.detect_contradictions(new_memory, retrieved_memories)
        if updates:
            self.belief_tracker.apply_updates(updates)
            belief_updates = [u.memory_id for u in updates]
            contradictions = [u.reason for u in updates]

        new_memories: List[MemoryEntry] = []
        summary = self._maybe_generate_summary(retrieved_memories)
        if summary:
            summary_entry = self.memory_service.create_memory(
                content=summary,
                memory_type="summary",
                confidence=0.5,
                importance=0.4,
                user_id=new_memory.user_id,
                source_memory_ids=[m.id for m in retrieved_memories],
            )
            new_memories.append(summary_entry)
        if contradictions:
            aha_entry = self.memory_service.create_memory(
                content=f"Aha: potential contradiction spotted ({', '.join(contradictions)}).",
                memory_type="reflection",
                confidence=0.4,
                importance=0.3,
                user_id=new_memory.user_id,
                source_memory_ids=[m.id for m in retrieved_memories],
            )
            new_memories.append(aha_entry)

        return ReasoningOutcome(
            new_memories=new_memories,
            belief_updates=belief_updates,
            contradictions=contradictions,
        )

    def _maybe_generate_summary(self, memories: List[MemoryEntry]) -> str:
        if len(memories) < 3:
            return ""
        tokens: List[str] = []
        avg_conf = sum(m.confidence for m in memories) / len(memories)
        for memory in memories:
            tokens.extend([t for t in memory.content.lower().split() if len(t) > 3])
        if not tokens:
            return ""
        counts = Counter(tokens)
        top_terms = ", ".join([term for term, _ in counts.most_common(5)])
        hedge = "maybe " if avg_conf < 0.5 else ""
        return (
            "Summary of recurring topics from recent memories: "
            f"{hedge}{top_terms}."
        )
