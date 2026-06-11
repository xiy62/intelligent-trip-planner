"""Anonymous profile and session memory for trip planning."""

from __future__ import annotations

import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..models.schemas import TripPlan, TripRequest

DEFAULT_MEMORY_DB_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "memory" / "trip_memory.sqlite3"
)
PROFILE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,128}$")


class MemoryService:
    """SQLite-backed anonymous memory store."""

    def __init__(self, db_path: Path | str = DEFAULT_MEMORY_DB_PATH):
        self.db_path = Path(db_path)
        self._initialized = False

    def build_memory_context(self, profile_id: Optional[str]) -> str:
        """Return a planner-ready profile summary, or an empty string."""
        profile = self.get_profile_snapshot(profile_id)
        return self.build_memory_context_from_profile(profile)

    def build_memory_context_from_profile(self, profile: Optional[Dict[str, Any]]) -> str:
        """Return a planner-ready profile summary from structured memory."""
        if not profile or int(profile.get("trip_count") or 0) <= 0:
            return ""

        parts = [f"过去成功生成过 {profile.get('trip_count')} 次行程"]
        recent_cities = list(profile.get("recent_cities") or [])
        preferences = list(profile.get("preferences") or [])
        if recent_cities:
            parts.append(f"最近目的地: {', '.join(recent_cities)}")
        if profile.get("transportation"):
            parts.append(f"常用交通方式: {profile['transportation']}")
        if profile.get("accommodation"):
            parts.append(f"常用住宿偏好: {profile['accommodation']}")
        if preferences:
            parts.append(f"历史偏好标签: {', '.join(preferences)}")

        return (
            "匿名历史偏好记忆（软约束，当前请求优先）: "
            + "；".join(parts)
            + "。如果历史偏好与本次请求冲突，必须以本次请求为准。"
        )

    def update_after_success(
        self,
        profile_id: Optional[str],
        conversation_id: str,
        request: TripRequest,
        plan: TripPlan,
        memory_applied: bool,
        memory_summary: str,
    ) -> None:
        """Persist profile/session memory after a validated plan is finalized."""
        if not profile_id or not self._is_valid_profile_id(profile_id):
            return

        now = time.time()
        existing = self.get_profile(profile_id)
        preferences = self._merge_unique(
            list(request.preferences),
            existing["preferences"] if existing else [],
            limit=12,
        )
        recent_cities = self._merge_recent(
            request.city,
            existing["recent_cities"] if existing else [],
            limit=8,
        )
        trip_count = (existing["trip_count"] if existing else 0) + 1
        last_summary = self._build_last_summary(request, preferences, recent_cities, trip_count)

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO profile_memory (
                    profile_id, transportation, accommodation, preferences_json,
                    recent_cities_json, trip_count, last_summary, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(profile_id) DO UPDATE SET
                    transportation = excluded.transportation,
                    accommodation = excluded.accommodation,
                    preferences_json = excluded.preferences_json,
                    recent_cities_json = excluded.recent_cities_json,
                    trip_count = excluded.trip_count,
                    last_summary = excluded.last_summary,
                    updated_at = excluded.updated_at
                """,
                (
                    profile_id,
                    request.transportation,
                    request.accommodation,
                    json.dumps(preferences, ensure_ascii=False),
                    json.dumps(recent_cities, ensure_ascii=False),
                    trip_count,
                    last_summary,
                    now,
                    now,
                ),
            )
            conn.execute(
                """
                INSERT OR REPLACE INTO conversation_memory (
                    conversation_id, profile_id, request_json, plan_json,
                    memory_applied, memory_summary, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    conversation_id,
                    profile_id,
                    request.model_dump_json(),
                    plan.model_dump_json(),
                    1 if memory_applied else 0,
                    memory_summary,
                    now,
                    now,
                ),
            )

    def clear_profile(self, profile_id: str) -> bool:
        """Delete profile and session memory for an anonymous profile id."""
        self._validate_profile_id(profile_id)
        with self._connect() as conn:
            profile_result = conn.execute(
                "DELETE FROM profile_memory WHERE profile_id = ?",
                (profile_id,),
            )
            conn.execute(
                "DELETE FROM conversation_memory WHERE profile_id = ?",
                (profile_id,),
            )
            return bool(profile_result.rowcount)

    def get_profile(self, profile_id: str) -> Optional[Dict[str, Any]]:
        """Return stored profile memory for debugging/API display."""
        self._validate_profile_id(profile_id)
        return self._get_profile(profile_id)

    def get_profile_snapshot(self, profile_id: Optional[str]) -> Optional[Dict[str, Any]]:
        """Return profile memory for optional request-time use without raising."""
        if not profile_id or not self._is_valid_profile_id(profile_id):
            return None
        return self._get_profile(profile_id)

    def _get_profile(self, profile_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT profile_id, transportation, accommodation, preferences_json,
                       recent_cities_json, trip_count, last_summary, created_at, updated_at
                FROM profile_memory
                WHERE profile_id = ?
                """,
                (profile_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "profile_id": row["profile_id"],
            "transportation": row["transportation"],
            "accommodation": row["accommodation"],
            "preferences": self._loads_list(row["preferences_json"]),
            "recent_cities": self._loads_list(row["recent_cities_json"]),
            "trip_count": int(row["trip_count"] or 0),
            "last_summary": row["last_summary"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def _connect(self) -> sqlite3.Connection:
        self._ensure_schema()
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        if self._initialized:
            return
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS profile_memory (
                    profile_id TEXT PRIMARY KEY,
                    transportation TEXT NOT NULL DEFAULT '',
                    accommodation TEXT NOT NULL DEFAULT '',
                    preferences_json TEXT NOT NULL DEFAULT '[]',
                    recent_cities_json TEXT NOT NULL DEFAULT '[]',
                    trip_count INTEGER NOT NULL DEFAULT 0,
                    last_summary TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_memory (
                    conversation_id TEXT PRIMARY KEY,
                    profile_id TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    plan_json TEXT NOT NULL,
                    memory_applied INTEGER NOT NULL DEFAULT 0,
                    memory_summary TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_conversation_memory_profile_id
                ON conversation_memory(profile_id)
                """
            )
        self._initialized = True

    def _validate_profile_id(self, profile_id: str) -> None:
        if not self._is_valid_profile_id(profile_id):
            raise ValueError("profile_id must be 8-128 characters of letters, numbers, '_' or '-'")

    def _is_valid_profile_id(self, profile_id: str) -> bool:
        return bool(PROFILE_ID_RE.match(profile_id))

    def _loads_list(self, raw: str) -> List[str]:
        try:
            data = json.loads(raw or "[]")
        except json.JSONDecodeError:
            return []
        if not isinstance(data, list):
            return []
        return [str(item) for item in data if str(item).strip()]

    def _merge_unique(self, current: List[str], existing: List[str], limit: int) -> List[str]:
        merged: List[str] = []
        for item in current + existing:
            value = str(item).strip()
            if value and value not in merged:
                merged.append(value)
        return merged[:limit]

    def _merge_recent(self, city: str, existing: List[str], limit: int) -> List[str]:
        merged = [city]
        for item in existing:
            value = str(item).strip()
            if value and value not in merged:
                merged.append(value)
        return merged[:limit]

    def _build_last_summary(
        self,
        request: TripRequest,
        preferences: List[str],
        recent_cities: List[str],
        trip_count: int,
    ) -> str:
        return (
            f"已记录 {trip_count} 次成功规划；最近目的地: {', '.join(recent_cities)}；"
            f"常用交通: {request.transportation}；常用住宿: {request.accommodation}；"
            f"偏好标签: {', '.join(preferences) if preferences else '无'}。"
        )


_memory_service: Optional[MemoryService] = None


def get_memory_service() -> MemoryService:
    """Return the process-wide anonymous memory service."""
    global _memory_service
    if _memory_service is None:
        _memory_service = MemoryService()
    return _memory_service
