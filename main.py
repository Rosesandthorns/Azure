import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Dict

import yaml

from discord_gateway import DiscordConfig, DiscordGateway, MemoryConfig
from memory_service import MemoryService
from model_interface import build_model_interface
from reflection_worker import ReflectionWorker
from retrieval_engine import RetrievalEngine
from scheduler import Scheduler, SchedulerConfig


def load_config() -> Dict[str, Any]:
    config_path = Path(__file__).with_name("config.yaml")
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


async def background_loop(
    gateway: DiscordGateway,
    reflection_worker: ReflectionWorker,
    scheduler: Scheduler,
    model: Any,
    memory_service: MemoryService,
) -> None:
    while True:
        if reflection_worker.should_run():
            reflection_worker.run()
        if scheduler.should_proactively_speak():
            curiosity = memory_service.list_memories(limit=5)
            content = "\n".join([m.content for m in curiosity])
            messages = [
                {
                    "role": "system",
                    "content": "You may ask a short, polite question based on unresolved curiosities.",
                },
                {"role": "user", "content": content},
            ]
            reply = model.generate(messages)
            await gateway.send_proactive_message(reply)
            scheduler.record_proactive()
        await asyncio.sleep(10)


def main() -> None:
    config = load_config()
    setup_logging(config.get("app", {}).get("log_level", "INFO"))

    memory_config = MemoryConfig(
        min_importance_to_store=float(
            config.get("memory", {}).get("min_importance_to_store", 0.3)
        )
    )
    memory_service = MemoryService(
        config["storage"]["sqlite_path"],
        transparency_log_path=config.get("logging", {}).get("transparency_log_path"),
    )
    retrieval_engine = RetrievalEngine(
        config["storage"]["chroma_path"],
        config["storage"]["embedding_dim"],
    )
    model = build_model_interface(config.get("model", {}))
    scheduler = Scheduler(
        SchedulerConfig(
            inactivity_minutes=int(config["scheduler"]["inactivity_minutes"]),
            curiosity_trigger_count=int(config["scheduler"]["curiosity_trigger_count"]),
            max_proactive_per_day=int(config["scheduler"]["max_proactive_per_day"]),
            cooldown_minutes=int(config["memory"]["proactive_cooldown_minutes"]),
        )
    )
    reflection_worker = ReflectionWorker(
        memory_service,
        model,
        interval_hours=int(config["memory"]["reflection_interval_hours"]),
    )

    discord_config = DiscordConfig(
        token=config["discord"]["token"],
        admin_user_ids=[str(x) for x in config["discord"]["admin_user_ids"]],
        command_prefix=config["discord"]["command_prefix"],
        rate_limit_seconds=int(config["discord"]["rate_limit_seconds"]),
    )

    gateway = DiscordGateway(
        discord_config,
        memory_config,
        memory_service,
        retrieval_engine,
        model,
        scheduler,
    )

    async def runner() -> None:
        gateway.loop.create_task(
            background_loop(gateway, reflection_worker, scheduler, model, memory_service)
        )
        await gateway.start(discord_config.token)

    asyncio.run(runner())


if __name__ == "__main__":
    os.makedirs("./data", exist_ok=True)
    main()
