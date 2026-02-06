from dataclasses import dataclass
from typing import List


@dataclass
class SafetyFilter:
    banned_phrases: List[str]

    def sanitize(self, text: str) -> str:
        updated = text
        lowered = updated.lower()
        for phrase in self.banned_phrases:
            if phrase in lowered:
                updated = updated.replace(phrase, "friend")
                updated = updated.replace(phrase.capitalize(), "friend")
                lowered = updated.lower()
        return updated
