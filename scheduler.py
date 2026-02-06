import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Optional, Set


@dataclass
class SchedulerConfig:
    inactivity_minutes: int
    curiosity_trigger_count: int
    max_proactive_per_day: int
    cooldown_minutes: int
    boredom_decay_minutes: int
    boredom_increase_on_interaction: float


class Scheduler:
    def __init__(self, config: SchedulerConfig) -> None:
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.last_user_message = datetime.utcnow()
        self.last_proactive: Optional[datetime] = None
        self.proactive_count = 0
        self.curiosity_count = 0
        self.boredom = 10.0
        self.last_boredom_tick = datetime.utcnow()
        self.last_proactive_day = date.today()
        self.proactive_topics: Set[str] = set()
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
        if date.today() != self.last_proactive_day:
            self.last_proactive_day = date.today()
            self.proactive_count = 0
            self.proactive_topics.clear()

    def should_proactively_speak(self, topic: str) -> bool:
        now = datetime.utcnow()
        if date.today() != self.last_proactive_day:
            self.last_proactive_day = date.today()
            self.proactive_count = 0
            self.proactive_topics.clear()
        if self.proactive_count >= self.config.max_proactive_per_day:
            return False
        if self.last_proactive and now - self.last_proactive < timedelta(
            minutes=self.config.cooldown_minutes
        ):
            return False
        inactive = now - self.last_user_message
        if inactive < timedelta(minutes=self.config.inactivity_minutes):
            return False
        if self.curiosity_count < self.config.curiosity_trigger_count:
            return False
        if topic in self.proactive_topics:
            return False
        return True

    def record_proactive(self, topic: str) -> None:
        self.last_proactive = datetime.utcnow()
        self.proactive_count += 1
        self.curiosity_count = 0
        self.proactive_topics.add(topic)

    def should_dm(self) -> bool:
        return self.boredom <= 3.0

    def current_mode(self) -> str:
        index = int((10.0 - self.boredom) // 2)
        index = max(0, min(index, len(self.modes) - 1))
        return self.modes[index]
