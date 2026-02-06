import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional


@dataclass
class SchedulerConfig:
    inactivity_minutes: int
    curiosity_trigger_count: int
    max_proactive_per_day: int
    cooldown_minutes: int


class Scheduler:
    def __init__(self, config: SchedulerConfig) -> None:
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.last_user_message = datetime.utcnow()
        self.last_proactive: Optional[datetime] = None
        self.proactive_count = 0
        self.curiosity_count = 0

    def update_on_message(self) -> None:
        self.last_user_message = datetime.utcnow()

    def register_curiosity(self) -> None:
        self.curiosity_count += 1

    def should_proactively_speak(self) -> bool:
        now = datetime.utcnow()
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
        return True

    def record_proactive(self) -> None:
        self.last_proactive = datetime.utcnow()
        self.proactive_count += 1
        self.curiosity_count = 0
