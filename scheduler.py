import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional
import random


@dataclass
class SchedulerConfig:
    inactivity_minutes: int
    curiosity_trigger_count: int
    cooldown_minutes: int
    boredom_decay_minutes: int
    boredom_increase_on_interaction: float


class Scheduler:
    def __init__(self, config: SchedulerConfig) -> None:
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.last_user_message = datetime.utcnow()
        self.last_proactive: Optional[datetime] = None
        self.curiosity_count = 0
        self.boredom = 10.0
        self.last_boredom_tick = datetime.utcnow()
        self.last_proactive_topic: Optional[str] = None
        self.modes = [
            "curious",
            "analytical",
            "concise",
            "structured",
            "cautious",
            "exploratory",
        ]

    def update_on_message(self) -> None:
        self.last_user_message = datetime.utcnow()
        self.boredom = min(10.0, self.boredom + self.config.boredom_increase_on_interaction)

    def register_curiosity(self) -> None:
        self.curiosity_count += 1

    def tick(self) -> None:
        now = datetime.utcnow()
        if now - self.last_boredom_tick < timedelta(minutes=self.config.boredom_decay_minutes):
            return
        self.last_boredom_tick = now
        self.boredom = max(0.0, self.boredom - 1.0)

    def should_proactively_speak(self, topic: str, probability: float) -> bool:
        now = datetime.utcnow()
        if self.last_proactive and now - self.last_proactive < timedelta(
            minutes=self.config.cooldown_minutes
        ):
            return False
        if topic == self.last_proactive_topic:
            probability *= 0.5
        probability = min(1.0, max(0.0, probability))
        return random.random() < probability

    def record_proactive(self, topic: str) -> None:
        self.last_proactive = datetime.utcnow()
        self.curiosity_count = 0
        self.last_proactive_topic = topic

    def should_dm(self) -> bool:
        return self.boredom <= 3.0

    def current_mode(self) -> str:
        index = int((10.0 - self.boredom) // 2)
        index = max(0, min(index, len(self.modes) - 1))
        return self.modes[index]
