import logging
from datetime import datetime, timedelta
from typing import List

from model_interface import ModelInterface
from memory_service import MemoryEntry, MemoryService


class ReflectionWorker:
    def __init__(
        self,
        memory_service: MemoryService,
        model: ModelInterface,
        interval_hours: int,
    ) -> None:
        self.memory_service = memory_service
        self.model = model
        self.interval = timedelta(hours=interval_hours)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.last_run = datetime.utcnow() - self.interval

    def should_run(self) -> bool:
        return datetime.utcnow() - self.last_run >= self.interval

    def run(self) -> List[MemoryEntry]:
        self.logger.info("Running reflection worker")
        self.last_run = datetime.utcnow()
        recent = self.memory_service.list_memories(limit=20)
        if not recent:
            return []
        summary = self._summarize(recent)
        created = []
        if summary:
            entry = self.memory_service.create_memory(
                content=summary,
                memory_type="summary",
                confidence=0.6,
                importance=0.5,
                user_id=recent[0].user_id,
                source_memory_ids=[m.id for m in recent],
            )
            created.append(entry)
        belief = self._derive_belief(recent)
        if belief:
            entry = self.memory_service.create_memory(
                content=belief,
                memory_type="belief",
                confidence=0.4,
                importance=0.4,
                user_id=recent[0].user_id,
                source_memory_ids=[m.id for m in recent],
            )
            created.append(entry)
        self._update_personality_state(recent)
        return created

    def _summarize(self, memories: List[MemoryEntry]) -> str:
        content = "\n".join([m.content for m in memories[:10]])
        messages = [
            {
                "role": "system",
                "content": "Summarize recurring themes. Be concise.",
            },
            {"role": "user", "content": content},
        ]
        return self.model.generate(messages)

    def _derive_belief(self, memories: List[MemoryEntry]) -> str:
        content = "\n".join([m.content for m in memories[:10]])
        messages = [
            {
                "role": "system",
                "content": "Derive a cautious belief with uncertainty words.",
            },
            {"role": "user", "content": content},
        ]
        return self.model.generate(messages)

    def _update_personality_state(self, memories: List[MemoryEntry]) -> None:
        state = self.memory_service.get_personality_state() or {
            "id": "azure",
            "mood": "curious",
            "energy": 0.5,
            "topic_interests": {},
            "relationship_strengths": {},
            "style_preferences": {"catchphrases": ["hmm", "oh!", "noted."]},
        }
        mood = self._compute_mood(memories)
        energy = min(1.0, state.get("energy", 0.5) + 0.05)
        topic_interests = state.get("topic_interests", {})
        relationship_strengths = state.get("relationship_strengths", {})
        for memory in memories:
            for token in memory.content.lower().split():
                if len(token) > 4:
                    topic_interests[token] = topic_interests.get(token, 0) + 1
            relationship_strengths[memory.user_id] = relationship_strengths.get(memory.user_id, 0) + 1
        state.update(
            {
                "mood": mood,
                "energy": energy,
                "topic_interests": topic_interests,
                "relationship_strengths": relationship_strengths,
                "style_preferences": state.get("style_preferences", {"catchphrases": ["hmm", "oh!", "noted."]}),
            }
        )
        self.memory_service.upsert_personality_state(state)

    def _compute_mood(self, memories: List[MemoryEntry]) -> str:
        positive = {"thanks", "great", "awesome", "love", "nice", "cool"}
        negative = {"stupid", "hate", "annoying", "bad", "ugh"}
        score = 0
        for memory in memories:
            text = memory.content.lower()
            if any(term in text for term in positive):
                score += 1
            if any(term in text for term in negative):
                score -= 1
        if score >= 2:
            return "excited"
        if score <= -2:
            return "guarded"
        return "curious"
