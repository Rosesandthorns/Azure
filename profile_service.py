import json
import logging
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class StyleProfile:
    avg_word_count: float
    emoji_ratio: float
    punctuation_ratio: float


@dataclass
class UserProfile:
    user_id: str
    like_score: float
    interests: List[str]
    topics_seen: List[str]
    style: StyleProfile
    last_interaction: Optional[str]


class ProfileService:
    def __init__(self, sqlite_path: str, default_self_interests: List[str]) -> None:
        self.sqlite_path = sqlite_path
        self.default_self_interests = default_self_interests
        self.logger = logging.getLogger(self.__class__.__name__)
        os.makedirs(os.path.dirname(sqlite_path), exist_ok=True)
        self.conn = sqlite3.connect(self.sqlite_path)
        self.conn.row_factory = sqlite3.Row
        self.ensure_schema()
        self._ensure_self_profile()

    def ensure_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id TEXT PRIMARY KEY,
                like_score REAL NOT NULL,
                interests TEXT NOT NULL,
                topics_seen TEXT NOT NULL,
                style TEXT NOT NULL,
                last_interaction TEXT
            );
            """
        )
        self.conn.commit()

    def _ensure_self_profile(self) -> None:
        profile = self.get_profile("self")
        if profile is None:
            self._create_profile("self", self.default_self_interests)

    def get_profile(self, user_id: str) -> Optional[UserProfile]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM user_profiles WHERE user_id = ?", (user_id,))
        row = cursor.fetchone()
        if not row:
            return None
        return self._row_to_profile(row)

    def get_or_create_profile(self, user_id: str) -> UserProfile:
        profile = self.get_profile(user_id)
        if profile:
            return profile
        self._create_profile(user_id, [])
        return self.get_profile(user_id) or self._default_profile(user_id)

    def update_like_score(self, user_id: str, delta: float) -> UserProfile:
        profile = self.get_or_create_profile(user_id)
        new_score = max(-1.0, min(1.0, profile.like_score + delta))
        self.conn.execute(
            "UPDATE user_profiles SET like_score = ? WHERE user_id = ?",
            (new_score, user_id),
        )
        self.conn.commit()
        return self.get_or_create_profile(user_id)

    def update_interests(self, user_id: str, topics: List[str]) -> UserProfile:
        profile = self.get_or_create_profile(user_id)
        updated = list({*profile.interests, *topics})
        self.conn.execute(
            "UPDATE user_profiles SET interests = ? WHERE user_id = ?",
            (json.dumps(updated), user_id),
        )
        self.conn.commit()
        return self.get_or_create_profile(user_id)

    def mark_topics_seen(self, user_id: str, topics: List[str]) -> UserProfile:
        profile = self.get_or_create_profile(user_id)
        updated = list({*profile.topics_seen, *topics})
        self.conn.execute(
            "UPDATE user_profiles SET topics_seen = ? WHERE user_id = ?",
            (json.dumps(updated), user_id),
        )
        self.conn.commit()
        return self.get_or_create_profile(user_id)

    def update_style_from_message(self, user_id: str, content: str) -> UserProfile:
        profile = self.get_or_create_profile(user_id)
        words = [w for w in content.split() if w]
        word_count = len(words)
        emoji_count = sum(1 for ch in content if ord(ch) > 10000)
        punctuation_count = sum(1 for ch in content if ch in "!?.,")
        total_chars = max(1, len(content))

        avg_word_count = (profile.style.avg_word_count * 0.8) + (word_count * 0.2)
        emoji_ratio = (profile.style.emoji_ratio * 0.8) + ((emoji_count / total_chars) * 0.2)
        punctuation_ratio = (profile.style.punctuation_ratio * 0.8) + (
            (punctuation_count / total_chars) * 0.2
        )
        style = StyleProfile(avg_word_count, emoji_ratio, punctuation_ratio)
        self.conn.execute(
            "UPDATE user_profiles SET style = ? WHERE user_id = ?",
            (json.dumps(style.__dict__), user_id),
        )
        self.conn.commit()
        return self.get_or_create_profile(user_id)

    def record_interaction(self, user_id: str) -> None:
        self.conn.execute(
            "UPDATE user_profiles SET last_interaction = ? WHERE user_id = ?",
            (datetime.utcnow().isoformat(), user_id),
        )
        self.conn.commit()

    def list_profiles(self) -> List[UserProfile]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM user_profiles WHERE user_id != 'self'")
        rows = cursor.fetchall()
        return [self._row_to_profile(row) for row in rows]

    def _create_profile(self, user_id: str, interests: List[str]) -> None:
        style = StyleProfile(avg_word_count=10.0, emoji_ratio=0.0, punctuation_ratio=0.1)
        self.conn.execute(
            """
            INSERT OR REPLACE INTO user_profiles
            (user_id, like_score, interests, topics_seen, style, last_interaction)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                0.0,
                json.dumps(interests),
                json.dumps([]),
                json.dumps(style.__dict__),
                None,
            ),
        )
        self.conn.commit()

    def _row_to_profile(self, row: sqlite3.Row) -> UserProfile:
        style_data = json.loads(row["style"])
        style = StyleProfile(
            avg_word_count=style_data.get("avg_word_count", 10.0),
            emoji_ratio=style_data.get("emoji_ratio", 0.0),
            punctuation_ratio=style_data.get("punctuation_ratio", 0.1),
        )
        return UserProfile(
            user_id=row["user_id"],
            like_score=row["like_score"],
            interests=json.loads(row["interests"]),
            topics_seen=json.loads(row["topics_seen"]),
            style=style,
            last_interaction=row["last_interaction"],
        )

    def _default_profile(self, user_id: str) -> UserProfile:
        return UserProfile(
            user_id=user_id,
            like_score=0.0,
            interests=[],
            topics_seen=[],
            style=StyleProfile(avg_word_count=10.0, emoji_ratio=0.0, punctuation_ratio=0.1),
            last_interaction=None,
        )
