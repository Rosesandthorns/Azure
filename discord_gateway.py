import asyncio
import logging
from dataclasses import dataclass
from typing import Dict, List

import discord

from belief_tracker import BeliefTracker
from memory_service import MemoryEntry, MemoryService
from model_interface import ModelInterface
from persona import MEMORY_USAGE_GUIDANCE, PERSONA_PROMPT
from profile_service import ProfileService, UserProfile
from reasoning_engine import ReasoningEngine
from retrieval_engine import RetrievalEngine
from scheduler import Scheduler
from safety_filter import SafetyFilter


@dataclass
class DiscordConfig:
    token: str
    admin_user_ids: List[str]
    command_prefix: str
    rate_limit_seconds: int


@dataclass
class MemoryConfig:
    min_importance_to_store: float


class DiscordGateway(discord.Client):
    def __init__(
        self,
        config: DiscordConfig,
        memory_config: MemoryConfig,
        memory_service: MemoryService,
        profile_service: ProfileService,
        retrieval_engine: RetrievalEngine,
        model: ModelInterface,
        scheduler: Scheduler,
        safety_filter: SafetyFilter,
    ) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents)
        self.config = config
        self.memory_config = memory_config
        self.memory_service = memory_service
        self.profile_service = profile_service
        self.retrieval_engine = retrieval_engine
        self.model = model
        self.scheduler = scheduler
        self.safety_filter = safety_filter
        self.logger = logging.getLogger(self.__class__.__name__)
        self.belief_tracker = BeliefTracker(memory_service)
        self.reasoning_engine = ReasoningEngine(memory_service, self.belief_tracker)
        self.last_channel: discord.abc.Messageable | None = None
        self.last_response_at: Dict[int, float] = {}

    async def on_ready(self) -> None:
        self.logger.info("Logged in as %s", self.user)

    async def on_message(self, message: discord.Message) -> None:
        if message.author == self.user:
            return
        content = message.content.strip()
        if content.startswith(self.config.command_prefix):
            await self._handle_command(message)
            return
        mention_or_keyword = self._is_targeted_message(message)
        now = message.created_at.timestamp()
        last_time = self.last_response_at.get(message.author.id, 0.0)
        if not mention_or_keyword and now - last_time < self.config.rate_limit_seconds:
            return
        self.profile_service.update_style_from_message(str(message.author.id), content)
        self.profile_service.record_interaction(str(message.author.id))
        self.scheduler.update_on_message()
        self.last_channel = message.channel
        response = await self._process_message(message.author.id, content, mention_or_keyword)
        if response:
            await message.channel.send(response)
            self.last_response_at[message.author.id] = now

    async def send_proactive_message(self, content: str) -> None:
        if not self.last_channel:
            return
        await self.last_channel.send(content)

    async def send_dm(self, user_id: str, content: str) -> None:
        try:
            user = await self.fetch_user(int(user_id))
            if user:
                await user.send(content)
        except discord.HTTPException:
            self.logger.warning("Failed to DM user %s", user_id)

    async def _handle_command(self, message: discord.Message) -> None:
        content = message.content.strip()
        parts = content[len(self.config.command_prefix) :].split()
        if not parts:
            return
        command = parts[0].lower()
        args = parts[1:]
        is_admin = str(message.author.id) in self.config.admin_user_ids

        if command == "memory" and is_admin:
            limit = int(args[0]) if args else 10
            memories = self.memory_service.list_memories(limit=limit)
            formatted = "\n".join([f"{m.id[:8]}: {m.type} - {m.content[:80]}" for m in memories])
            await message.channel.send(formatted or "No memories yet.")
            return
        if command == "memory_delete" and is_admin:
            if not args:
                await message.channel.send("Provide a memory id.")
                return
            self.memory_service.delete_memory(args[0])
            self.retrieval_engine.delete_memory(args[0])
            await message.channel.send("Memory deleted.")
            return
        if command == "wipe_user" and is_admin:
            if not args:
                await message.channel.send("Provide a user id to wipe.")
                return
            self.memory_service.wipe_user(args[0])
            await message.channel.send("User data wiped.")
            return
        if command == "memory_help":
            await message.channel.send(
                "Commands: !memory [n], !memory_delete <id>, !wipe_user <id> (admin only)."
            )
            return
        await message.channel.send("Unknown command.")

    async def _process_message(
        self, user_id: int, content: str, mention_or_keyword: bool
    ) -> str:
        profile = self.profile_service.get_or_create_profile(str(user_id))
        entities, topics, claims = self._extract_signal(content)
        query = " ".join(topics + entities + claims)
        memory_ids = self.retrieval_engine.query(query or content, top_k=8)
        retrieved = [m for mid in memory_ids if (m := self.memory_service.get_memory(mid))]
        new_topics = [topic for topic in topics if topic not in profile.topics_seen]

        if self._is_mean_message(content):
            profile = self.profile_service.update_like_score(str(user_id), -0.2)
        if self._is_positive_message(content):
            profile = self.profile_service.update_like_score(str(user_id), 0.1)

        self.profile_service.update_interests(str(user_id), topics)
        self.profile_service.update_interests("self", topics)
        self.profile_service.mark_topics_seen(str(user_id), topics)

        if not self._should_reply(content, profile, new_topics, mention_or_keyword):
            self._maybe_store_memory(user_id, content, retrieved, force_store=True)
            return ""

        prompt = self._build_prompt(content, retrieved, profile, new_topics)
        reply = self.model.generate(prompt)
        reply = self.safety_filter.sanitize(reply)

        new_memory = self._maybe_store_memory(user_id, content, retrieved, force_store=True)
        if new_memory:
            self.retrieval_engine.upsert_memory(new_memory)
            outcome = self.reasoning_engine.connect_memories(new_memory, retrieved)
            for memory in outcome.new_memories:
                self.retrieval_engine.upsert_memory(memory)

        if "?" in content:
            self.scheduler.register_curiosity()

        return reply

    def _build_prompt(
        self,
        content: str,
        retrieved: List[MemoryEntry],
        profile: UserProfile,
        new_topics: List[str],
    ) -> List[Dict[str, str]]:
        memory_block = "\n".join(
            [
                f"[{m.type} | conf:{m.confidence:.2f}] {m.content}"
                for m in retrieved
            ]
        )
        style_hint = (
            f"Match user style: ~{profile.style.avg_word_count:.0f} words, "
            f"emoji ratio {profile.style.emoji_ratio:.2f}, "
            f"punctuation ratio {profile.style.punctuation_ratio:.2f}."
        )
        mode_hint = f"Communication mode: {self.scheduler.current_mode()}."
        tone_hint = "Be neutral and brief." if profile.like_score < -0.3 else "Be friendly and curious."
        curiosity_hint = "Ask a clarifying question about new topics." if new_topics else ""
        safety_hint = "Avoid flirty or intimate language."
        return [
            {"role": "system", "content": PERSONA_PROMPT},
            {"role": "system", "content": MEMORY_USAGE_GUIDANCE},
            {
                "role": "system",
                "content": f"Relevant memories:\n{memory_block}" if memory_block else "No relevant memories.",
            },
            {"role": "system", "content": style_hint},
            {"role": "system", "content": mode_hint},
            {"role": "system", "content": tone_hint},
            {"role": "system", "content": safety_hint},
            {"role": "system", "content": curiosity_hint},
            {"role": "user", "content": content},
        ]

    def _maybe_store_memory(
        self,
        user_id: int,
        content: str,
        retrieved: List[MemoryEntry],
        force_store: bool = False,
    ) -> MemoryEntry | None:
        importance = min(1.0, 0.2 + len(content) / 200)
        lowered = content.lower()
        if any(phrase in lowered for phrase in {"i like", "my favorite", "i am", "i'm"}):
            memory_type = "semantic"
            importance = max(importance, 0.5)
        elif "?" in content:
            memory_type = "curiosity"
            importance = max(importance, 0.4)
        else:
            memory_type = "episodic"
        if not force_store and importance < self.memory_config.min_importance_to_store:
            return None
        entry = self.memory_service.create_memory(
            content=content,
            memory_type=memory_type,
            confidence=0.6,
            importance=importance,
            user_id=str(user_id),
            source_memory_ids=[m.id for m in retrieved],
        )
        return entry

    def _extract_signal(self, content: str) -> tuple[List[str], List[str], List[str]]:
        tokens = [t.strip(".,!?") for t in content.split()]
        entities = [t for t in tokens if t.istitle()]
        topics = [t for t in tokens if len(t) > 5]
        claims = [t for t in tokens if t.lower() in {"is", "are", "was", "were"}]
        return entities, topics, claims

    def _is_targeted_message(self, message: discord.Message) -> bool:
        if self.user and self.user in message.mentions:
            return True
        return "azure" in message.content.lower()

    def _is_mean_message(self, content: str) -> bool:
        mean_terms = {"stupid", "idiot", "hate", "dumb", "trash", "shut up", "annoying"}
        lowered = content.lower()
        return any(term in lowered for term in mean_terms)

    def _is_positive_message(self, content: str) -> bool:
        positive_terms = {"thanks", "thank you", "appreciate", "helpful"}
        lowered = content.lower()
        return any(term in lowered for term in positive_terms)

    def _should_reply(
        self,
        content: str,
        profile: UserProfile,
        new_topics: List[str],
        mention_or_keyword: bool,
    ) -> bool:
        if mention_or_keyword:
            return True
        if "?" in content:
            return True
        if new_topics:
            return True
        if profile.like_score < -0.6:
            return False
        return True

    def pick_dm_target(self) -> UserProfile | None:
        profiles = sorted(
            self.profile_service.list_profiles(),
            key=lambda p: p.like_score,
            reverse=True,
        )
        if not profiles:
            return None
        return profiles[0]

    def build_boredom_prompt(self, profile: UserProfile) -> str:
        interest = profile.interests[0] if profile.interests else "something interesting"
        return f"Quick check-in about {interest}?"
