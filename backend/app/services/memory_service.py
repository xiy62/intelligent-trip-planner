"""Anonymous profile and session memory for trip planning."""

from __future__ import annotations

import json
import re
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

    def build_memory_context_for_request(
        self,
        profile: Optional[Dict[str, Any]],
        request: TripRequest,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Return planner memory context plus current-request conflict explanations."""
        conflicts = self.detect_memory_conflicts(profile, request)
        summary = self.build_memory_context_from_profile(profile)
        if summary and conflicts:
            conflict_text = " ".join(item["explanation"] for item in conflicts)
            summary = f"{summary} Memory/current-request conflict note: {conflict_text}"
        return summary, conflicts

    def build_memory_context_from_profile(self, profile: Optional[Dict[str, Any]]) -> str:
        """Return a planner-ready profile summary from structured memory."""
        if not profile or int(profile.get("trip_count") or 0) <= 0:
            return ""

        parts = [f"previous successful trip plans: {profile.get('trip_count')}"]
        recent_cities = list(profile.get("recent_cities") or [])
        preferences = list(profile.get("preferences") or [])
        metadata = self._metadata_list_to_map(profile.get("preference_metadata"))
        if recent_cities:
            parts.append(f"recent destinations: {', '.join(recent_cities)}")
        if profile.get("transportation"):
            parts.append(
                "usual transportation: "
                + self._describe_metadata_value(
                    profile["transportation"],
                    metadata.get("transportation", {}),
                )
            )
        if profile.get("accommodation"):
            parts.append(
                "usual accommodation: "
                + self._describe_metadata_value(
                    profile["accommodation"],
                    metadata.get("accommodation", {}),
                )
            )
        if preferences:
            described_preferences = [
                self._describe_metadata_value(item, metadata.get("preferences", {}))
                for item in preferences
            ]
            parts.append(f"historical preference tags: {', '.join(described_preferences)}")

        return (
            "Anonymous historical preference memory (soft context only; current request fields have higher priority): "
            + "; ".join(parts)
            + ". Use this memory to personalize defaults, but do not overfit to older trips. "
            + "If historical memory conflicts with the current request, follow the current request."
        )

    def detect_memory_conflicts(
        self,
        profile: Optional[Dict[str, Any]],
        request: TripRequest,
    ) -> List[Dict[str, Any]]:
        """Return explicit explanations when current request overrides remembered fields."""
        if not profile or int(profile.get("trip_count") or 0) <= 0:
            return []

        metadata = self._metadata_list_to_map(profile.get("preference_metadata"))
        conflicts: List[Dict[str, Any]] = []
        for field, label in (
            ("transportation", "transportation"),
            ("accommodation", "accommodation"),
        ):
            remembered_value = str(profile.get(field) or "").strip()
            current_value = str(getattr(request, field) or "").strip()
            if not remembered_value or not current_value:
                continue
            if self._normalize_memory_value(remembered_value) == self._normalize_memory_value(current_value):
                continue
            record = metadata.get(field, {}).get(self._normalize_memory_value(remembered_value), {})
            conflicts.append(
                {
                    "field": field,
                    "remembered_value": remembered_value,
                    "current_value": current_value,
                    "resolution": "current_request_used",
                    "count": int(record.get("count") or 0),
                    "last_seen_at": record.get("last_seen_at"),
                    "source_type": record.get("source_type") or "explicit_request",
                    "explanation": (
                        f"Previous {label} was \"{remembered_value}\", but the current request asks for "
                        f"\"{current_value}\"; the current request is used."
                    ),
                }
            )
        return conflicts

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
        metadata = self._metadata_list_to_map(
            existing.get("preference_metadata") if existing else None
        )
        self._record_preference_metadata(metadata, "transportation", request.transportation, now)
        self._record_preference_metadata(metadata, "accommodation", request.accommodation, now)
        for preference in request.preferences:
            self._record_preference_metadata(metadata, "preferences", preference, now)
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
                    recent_cities_json, preference_metadata_json, trip_count,
                    last_summary, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(profile_id) DO UPDATE SET
                    transportation = excluded.transportation,
                    accommodation = excluded.accommodation,
                    preferences_json = excluded.preferences_json,
                    recent_cities_json = excluded.recent_cities_json,
                    preference_metadata_json = excluded.preference_metadata_json,
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
                    json.dumps(self._metadata_map_to_lists(metadata), ensure_ascii=False),
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
                       recent_cities_json, preference_metadata_json, trip_count,
                       last_summary, created_at, updated_at
                FROM profile_memory
                WHERE profile_id = ?
                """,
                (profile_id,),
            ).fetchone()
        if row is None:
            return None
        legacy_profile = {
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
        legacy_profile["preference_metadata"] = self._load_preference_metadata(
            row["preference_metadata_json"],
            legacy_profile,
        )
        return legacy_profile

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
                    preference_metadata_json TEXT NOT NULL DEFAULT '{}',
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
            self._ensure_profile_metadata_column(conn)
        self._initialized = True

    def _ensure_profile_metadata_column(self, conn: sqlite3.Connection) -> None:
        columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(profile_memory)").fetchall()
        }
        if "preference_metadata_json" not in columns:
            conn.execute(
                "ALTER TABLE profile_memory ADD COLUMN preference_metadata_json TEXT NOT NULL DEFAULT '{}'"
            )

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

    def _load_preference_metadata(
        self,
        raw: str,
        legacy_profile: Dict[str, Any],
    ) -> Dict[str, List[Dict[str, Any]]]:
        try:
            data = json.loads(raw or "{}")
        except json.JSONDecodeError:
            data = {}
        metadata = self._metadata_map_to_lists(self._metadata_list_to_map(data))
        if any(metadata.values()):
            return metadata

        timestamp = legacy_profile.get("updated_at") or legacy_profile.get("created_at")
        trip_count = max(1, int(legacy_profile.get("trip_count") or 1))
        fallback_map: Dict[str, Dict[str, Dict[str, Any]]] = {
            "transportation": {},
            "accommodation": {},
            "preferences": {},
        }
        if legacy_profile.get("transportation"):
            self._record_preference_metadata(
                fallback_map,
                "transportation",
                str(legacy_profile["transportation"]),
                float(timestamp or time.time()),
                count_increment=trip_count,
            )
        if legacy_profile.get("accommodation"):
            self._record_preference_metadata(
                fallback_map,
                "accommodation",
                str(legacy_profile["accommodation"]),
                float(timestamp or time.time()),
                count_increment=trip_count,
            )
        for preference in legacy_profile.get("preferences") or []:
            self._record_preference_metadata(
                fallback_map,
                "preferences",
                str(preference),
                float(timestamp or time.time()),
            )
        return self._metadata_map_to_lists(fallback_map)

    def _metadata_list_to_map(self, data: Any) -> Dict[str, Dict[str, Dict[str, Any]]]:
        metadata: Dict[str, Dict[str, Dict[str, Any]]] = {
            "transportation": {},
            "accommodation": {},
            "preferences": {},
        }
        if not isinstance(data, dict):
            return metadata
        for field in metadata:
            raw_records = data.get(field) or []
            if isinstance(raw_records, dict):
                raw_records = list(raw_records.values())
            if not isinstance(raw_records, list):
                continue
            for item in raw_records:
                if not isinstance(item, dict):
                    continue
                value = str(item.get("value") or "").strip()
                if not value:
                    continue
                key = self._normalize_memory_value(value)
                metadata[field][key] = {
                    "value": value,
                    "count": max(1, int(item.get("count") or 1)),
                    "last_seen_at": item.get("last_seen_at"),
                    "source_type": str(item.get("source_type") or "explicit_request"),
                }
        return metadata

    def _metadata_map_to_lists(
        self,
        metadata: Dict[str, Dict[str, Dict[str, Any]]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        result: Dict[str, List[Dict[str, Any]]] = {}
        for field in ("transportation", "accommodation", "preferences"):
            records = list((metadata.get(field) or {}).values())
            result[field] = sorted(
                records,
                key=lambda item: (
                    -int(item.get("count") or 0),
                    -(float(item.get("last_seen_at") or 0.0)),
                    str(item.get("value") or ""),
                ),
            )
        return result

    def _record_preference_metadata(
        self,
        metadata: Dict[str, Dict[str, Dict[str, Any]]],
        field: str,
        value: str,
        now: float,
        source_type: str = "explicit_request",
        count_increment: int = 1,
    ) -> None:
        normalized = self._normalize_memory_value(value)
        if not normalized:
            return
        field_records = metadata.setdefault(field, {})
        record = field_records.get(normalized)
        if record is None:
            field_records[normalized] = {
                "value": str(value).strip(),
                "count": max(1, count_increment),
                "last_seen_at": now,
                "source_type": source_type,
            }
            return
        record["value"] = str(value).strip()
        record["count"] = int(record.get("count") or 0) + max(1, count_increment)
        record["last_seen_at"] = now
        record["source_type"] = source_type

    def _describe_metadata_value(
        self,
        value: str,
        metadata: Dict[str, Dict[str, Any]],
    ) -> str:
        record = metadata.get(self._normalize_memory_value(value))
        if not record:
            return value
        details = [f"seen {int(record.get('count') or 0)} time(s)"]
        formatted_time = self._format_timestamp(record.get("last_seen_at"))
        if formatted_time:
            details.append(f"last seen {formatted_time}")
        if record.get("source_type"):
            details.append(f"source {record['source_type']}")
        return f"{record.get('value') or value} ({', '.join(details)})"

    def _format_timestamp(self, value: Any) -> str:
        try:
            timestamp = float(value)
        except (TypeError, ValueError):
            return ""
        if timestamp <= 0:
            return ""
        return datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y-%m-%d")

    def _normalize_memory_value(self, value: str) -> str:
        return re.sub(r"\s+", " ", str(value or "").strip()).casefold()

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
            f"Recorded {trip_count} successful trip plans; recent destinations: {', '.join(recent_cities)}; "
            f"usual transportation: {request.transportation}; usual accommodation: {request.accommodation}; "
            f"preference tags: {', '.join(preferences) if preferences else 'none'}."
        )


_memory_service: Optional[MemoryService] = None


def get_memory_service() -> MemoryService:
    """Return the process-wide anonymous memory service."""
    global _memory_service
    if _memory_service is None:
        _memory_service = MemoryService()
    return _memory_service
