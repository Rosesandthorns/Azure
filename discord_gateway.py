import asyncio
import logging
from dataclasses import dataclass
from typing import Dict, List

import discord

from belief_tracker import BeliefTracker
from memory_service import MemoryEntry, MemoryService
from model_interface import ModelInterface
from persona import MEMORY_USAGE_GUIDANCE, PERSONA_PROMPT
from reasoning_engine import ReasoningEngine
from retrieval_engine import RetrievalEngine
from scheduler import Scheduler


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
        retrieval_engine: RetrievalEngine,
        model: ModelInterface,
        scheduler: Scheduler,
    ) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents)
        self.config = config
        self.memory_config = memory_config
        self.memory_service = memory_service
        self.retrieval_engine = retrieval_engine
        self.model = model
        self.scheduler = scheduler
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
        now = message.created_at.timestamp()
        last_time = self.last_response_at.get(message.author.id, 0.0)
        if now - last_time < self.config.rate_limit_seconds:
            return
        self.scheduler.update_on_message()
        self.last_channel = message.channel
        response = await self._process_message(message.author.id, content)
        if response:
            await message.channel.send(response)
            self.last_response_at[message.author.id] = now

    async def send_proactive_message(self, content: str) -> None:
        if not self.last_channel:
            return
        await self.last_channel.send(content)

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

    async def _process_message(self, user_id: int, content: str) -> str:
        entities, topics, claims = self._extract_signal(content)
        query = " ".join(topics + entities + claims)
        memory_ids = self.retrieval_engine.query(query or content, top_k=8)
        retrieved = [m for mid in memory_ids if (m := self.memory_service.get_memory(mid))]
        prompt = self._build_prompt(content, retrieved)
        reply = self.model.generate(prompt)

        new_memory = self._maybe_store_memory(user_id, content, retrieved)
        if new_memory:
            self.retrieval_engine.upsert_memory(new_memory)
            outcome = self.reasoning_engine.connect_memories(new_memory, retrieved)
            for memory in outcome.new_memories:
                self.retrieval_engine.upsert_memory(memory)

        if "?" in content:
            self.scheduler.register_curiosity()

        return reply

    def _build_prompt(self, content: str, retrieved: List[MemoryEntry]) -> List[Dict[str, str]]:
        memory_block = "\n".join(
            [
                f"[{m.type} | conf:{m.confidence:.2f}] {m.content}"
                for m in retrieved
            ]
        )
        return [
            {"role": "system", "content": PERSONA_PROMPT},
            {"role": "system", "content": MEMORY_USAGE_GUIDANCE},
            {
                "role": "system",
                "content": f"Relevant memories:\n{memory_block}" if memory_block else "No relevant memories.",
            },
            {"role": "user", "content": content},
        ]

    def _maybe_store_memory(
        self, user_id: int, content: str, retrieved: List[MemoryEntry]
    ) -> MemoryEntry | None:
        importance = min(1.0, 0.2 + len(content) / 200)
        if "?" in content:
            memory_type = "curiosity"
            importance = max(importance, 0.4)
        else:
            memory_type = "episodic"
        if importance < self.memory_config.min_importance_to_store:
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
