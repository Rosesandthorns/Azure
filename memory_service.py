import json
import logging
import os
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class MemoryEntry:
    id: str
    content: str
    type: str
    confidence: float
    importance: float
    timestamp: str
    user_id: str
    source_memory_ids: List[str]


class MemoryService:
    def __init__(self, sqlite_path: str, transparency_log_path: str | None = None) -> None:
        self.sqlite_path = sqlite_path
        self.logger = logging.getLogger(self.__class__.__name__)
        self.transparency_log_path = transparency_log_path
        os.makedirs(os.path.dirname(sqlite_path), exist_ok=True)
        self.conn = sqlite3.connect(self.sqlite_path)
        self.conn.row_factory = sqlite3.Row
        self.ensure_schema()
        self._ensure_personality_state()

    def _ensure_personality_state(self) -> None:
        if self.get_personality_state():
            return
        self.upsert_personality_state(
            {
                "id": "azure",
                "mood": "curious",
                "energy": 0.5,
                "topic_interests": {},
                "relationship_strengths": {},
                "style_preferences": {"catchphrases": ["hmm", "oh!", "noted."]},
            }
        )

    def ensure_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                applied_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                type TEXT NOT NULL,
                confidence REAL NOT NULL,
                importance REAL NOT NULL,
                timestamp TEXT NOT NULL,
                user_id TEXT NOT NULL,
                source_memory_ids TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS transparency_logs (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                event TEXT NOT NULL,
                details TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS conversation_logs (
                id TEXT PRIMARY KEY,
                channel_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                mentioned INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS personality_state (
                id TEXT PRIMARY KEY,
                mood TEXT NOT NULL,
                energy REAL NOT NULL,
                topic_interests TEXT NOT NULL,
                relationship_strengths TEXT NOT NULL,
                style_preferences TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )
        self.conn.commit()

    def log_event(self, event: str, details: Dict[str, Any]) -> None:
        payload = json.dumps(details, ensure_ascii=False)
        self.conn.execute(
            "INSERT INTO transparency_logs (id, timestamp, event, details) VALUES (?, ?, ?, ?)",
            (str(uuid.uuid4()), datetime.utcnow().isoformat(), event, payload),
        )
        self.conn.commit()
        if self.transparency_log_path:
            with open(self.transparency_log_path, "a", encoding="utf-8") as file:
                file.write(f\"{datetime.utcnow().isoformat()} {event} {payload}\\n\")

    def create_memory(
        self,
        content: str,
        memory_type: str,
        confidence: float,
        importance: float,
        user_id: str,
        source_memory_ids: Optional[Iterable[str]] = None,
    ) -> MemoryEntry:
        memory_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()
        sources = list(source_memory_ids or [])
        entry = MemoryEntry(
            id=memory_id,
            content=content,
            type=memory_type,
            confidence=confidence,
            importance=importance,
            timestamp=timestamp,
            user_id=user_id,
            source_memory_ids=sources,
        )
        self.conn.execute(
            """
            INSERT INTO memories (id, content, type, confidence, importance, timestamp, user_id, source_memory_ids)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry.id,
                entry.content,
                entry.type,
                entry.confidence,
                entry.importance,
                entry.timestamp,
                entry.user_id,
                json.dumps(entry.source_memory_ids),
            ),
        )
        self.conn.commit()
        self.log_event("memory_created", {"memory_id": entry.id, "type": entry.type})
        return entry

    def log_conversation_message(
        self, channel_id: str, user_id: str, content: str, mentioned: bool
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO conversation_logs (id, channel_id, user_id, content, timestamp, mentioned)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                channel_id,
                user_id,
                content,
                datetime.utcnow().isoformat(),
                1 if mentioned else 0,
            ),
        )
        self.conn.commit()

    def list_recent_messages(self, channel_id: str, limit: int = 25) -> List[Dict[str, str]]:
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT user_id, content, timestamp, mentioned
            FROM conversation_logs
            WHERE channel_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (channel_id, limit),
        )
        rows = cursor.fetchall()
        return [
            {
                "user_id": row["user_id"],
                "content": row["content"],
                "timestamp": row["timestamp"],
                "mentioned": bool(row["mentioned"]),
            }
            for row in reversed(rows)
        ]

    def get_personality_state(self) -> Dict[str, object]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM personality_state LIMIT 1")
        row = cursor.fetchone()
        if not row:
            return {}
        return {
            "id": row["id"],
            "mood": row["mood"],
            "energy": row["energy"],
            "topic_interests": json.loads(row["topic_interests"]),
            "relationship_strengths": json.loads(row["relationship_strengths"]),
            "style_preferences": json.loads(row["style_preferences"]),
            "updated_at": row["updated_at"],
        }

    def upsert_personality_state(self, state: Dict[str, object]) -> None:
        payload = {
            "id": state.get("id", "azure"),
            "mood": state.get("mood", "curious"),
            "energy": float(state.get("energy", 0.5)),
            "topic_interests": json.dumps(state.get("topic_interests", {})),
            "relationship_strengths": json.dumps(state.get("relationship_strengths", {})),
            "style_preferences": json.dumps(state.get("style_preferences", {})),
            "updated_at": datetime.utcnow().isoformat(),
        }
        self.conn.execute(
            """
            INSERT OR REPLACE INTO personality_state
            (id, mood, energy, topic_interests, relationship_strengths, style_preferences, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload["id"],
                payload["mood"],
                payload["energy"],
                payload["topic_interests"],
                payload["relationship_strengths"],
                payload["style_preferences"],
                payload["updated_at"],
            ),
        )
        self.conn.commit()
        self.log_event("personality_state_updated", {"mood": payload["mood"], "energy": payload["energy"]})

    def list_memories(self, user_id: Optional[str] = None, limit: int = 20) -> List[MemoryEntry]:
        cursor = self.conn.cursor()
        if user_id:
            cursor.execute(
                "SELECT * FROM memories WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?",
                (user_id, limit),
            )
        else:
            cursor.execute("SELECT * FROM memories ORDER BY timestamp DESC LIMIT ?", (limit,))
        rows = cursor.fetchall()
        return [self._row_to_memory(row) for row in rows]

    def get_memory(self, memory_id: str) -> Optional[MemoryEntry]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM memories WHERE id = ?", (memory_id,))
        row = cursor.fetchone()
        if not row:
            return None
        return self._row_to_memory(row)

    def update_memory_confidence(self, memory_id: str, confidence: float) -> None:
        self.conn.execute(
            "UPDATE memories SET confidence = ? WHERE id = ?", (confidence, memory_id)
        )
        self.conn.commit()
        self.log_event("memory_confidence_updated", {"memory_id": memory_id, "confidence": confidence})

    def delete_memory(self, memory_id: str) -> None:
        self.conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        self.conn.commit()
        self.log_event("memory_deleted", {"memory_id": memory_id})

    def wipe_user(self, user_id: str) -> None:
        self.conn.execute("DELETE FROM memories WHERE user_id = ?", (user_id,))
        self.conn.commit()
        self.log_event("user_wiped", {"user_id": user_id})

    def _row_to_memory(self, row: sqlite3.Row) -> MemoryEntry:
        return MemoryEntry(
            id=row["id"],
            content=row["content"],
            type=row["type"],
            confidence=row["confidence"],
            importance=row["importance"],
            timestamp=row["timestamp"],
            user_id=row["user_id"],
            source_memory_ids=json.loads(row["source_memory_ids"]),
        )
