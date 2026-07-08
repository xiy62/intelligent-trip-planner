"""Tests for anonymous trip memory persistence."""

from __future__ import annotations

import tempfile
import unittest
import sqlite3
import time
from pathlib import Path

from app.models.schemas import DayPlan, TripPlan, TripRequest
from app.services.memory_service import MemoryService


def build_request(profile_id: str) -> TripRequest:
    return TripRequest(
        city="北京",
        start_date="2026-06-01",
        end_date="2026-06-02",
        travel_days=2,
        transportation="公共交通",
        accommodation="经济型酒店",
        preferences=["历史文化", "美食"],
        free_text_input="希望行程不要太赶",
        profile_id=profile_id,
    )


def build_plan() -> TripPlan:
    return TripPlan(
        city="北京",
        start_date="2026-06-01",
        end_date="2026-06-02",
        days=[
            DayPlan(
                date="2026-06-01",
                day_index=0,
                description="第1天行程",
                transportation="公共交通",
                accommodation="经济型酒店",
                attractions=[],
                meals=[],
            )
        ],
        weather_info=[],
        overall_suggestions="舒适出行",
        budget=None,
    )


class MemoryServiceTests(unittest.TestCase):
    def test_profile_summary_and_clear(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_service = MemoryService(Path(tmpdir) / "memory.sqlite3")
            profile_id = "profile_memory_123"

            self.assertEqual(memory_service.build_memory_context(profile_id), "")
            memory_service.update_after_success(
                profile_id=profile_id,
                conversation_id="conversation-1",
                request=build_request(profile_id),
                plan=build_plan(),
                memory_applied=False,
                memory_summary="",
            )

            profile = memory_service.get_profile(profile_id)
            self.assertIsNotNone(profile)
            self.assertEqual(profile["trip_count"], 1)
            self.assertEqual(profile["transportation"], "公共交通")
            self.assertEqual(profile["accommodation"], "经济型酒店")
            self.assertEqual(profile["recent_cities"], ["北京"])
            self.assertIn("历史文化", profile["preferences"])
            self.assertIn("preference_metadata", profile)
            self.assertEqual(
                profile["preference_metadata"]["transportation"][0]["source_type"],
                "explicit_request",
            )

            summary = memory_service.build_memory_context(profile_id)
            self.assertIn("Anonymous historical preference memory", summary)
            self.assertIn("current request", summary)

            snapshot = memory_service.get_profile_snapshot(profile_id)
            self.assertEqual(snapshot["trip_count"], 1)
            self.assertEqual(snapshot["preferences"], ["历史文化", "美食"])
            self.assertIsNone(memory_service.get_profile_snapshot("bad id with spaces"))

            self.assertTrue(memory_service.clear_profile(profile_id))
            self.assertIsNone(memory_service.get_profile(profile_id))

    def test_repeated_preferences_increment_metadata_counts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_service = MemoryService(Path(tmpdir) / "memory.sqlite3")
            profile_id = "profile_counts_123"
            request = build_request(profile_id)
            plan = build_plan()

            memory_service.update_after_success(
                profile_id=profile_id,
                conversation_id="conversation-1",
                request=request,
                plan=plan,
                memory_applied=False,
                memory_summary="",
            )
            memory_service.update_after_success(
                profile_id=profile_id,
                conversation_id="conversation-2",
                request=request,
                plan=plan,
                memory_applied=True,
                memory_summary="memory",
            )

            profile = memory_service.get_profile(profile_id)
            metadata = profile["preference_metadata"]
            self.assertEqual(profile["trip_count"], 2)
            self.assertEqual(metadata["transportation"][0]["value"], "公共交通")
            self.assertEqual(metadata["transportation"][0]["count"], 2)
            self.assertEqual(metadata["accommodation"][0]["count"], 2)
            historical_tag = next(
                item for item in metadata["preferences"] if item["value"] == "历史文化"
            )
            self.assertEqual(historical_tag["count"], 2)
            self.assertIsNotNone(historical_tag["last_seen_at"])

    def test_legacy_profile_without_metadata_still_loads(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "legacy.sqlite3"
            now = time.time()
            with sqlite3.connect(db_path) as conn:
                conn.execute(
                    """
                    CREATE TABLE profile_memory (
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
                    CREATE TABLE conversation_memory (
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
                    INSERT INTO profile_memory (
                        profile_id, transportation, accommodation, preferences_json,
                        recent_cities_json, trip_count, last_summary, created_at, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "legacy_profile_123",
                        "Public transit",
                        "Budget hotel",
                        '["Museums"]',
                        '["New York"]',
                        3,
                        "legacy summary",
                        now,
                        now,
                    ),
                )

            memory_service = MemoryService(db_path)
            profile = memory_service.get_profile("legacy_profile_123")

            self.assertEqual(profile["transportation"], "Public transit")
            self.assertEqual(profile["accommodation"], "Budget hotel")
            self.assertEqual(profile["preference_metadata"]["transportation"][0]["count"], 3)
            self.assertEqual(profile["preference_metadata"]["accommodation"][0]["source_type"], "explicit_request")
            self.assertEqual(profile["preference_metadata"]["preferences"][0]["value"], "Museums")

    def test_memory_conflict_explanation_prefers_current_request(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_service = MemoryService(Path(tmpdir) / "memory.sqlite3")
            profile_id = "profile_conflict_123"
            memory_service.update_after_success(
                profile_id=profile_id,
                conversation_id="conversation-1",
                request=build_request(profile_id),
                plan=build_plan(),
                memory_applied=False,
                memory_summary="",
            )
            current_request = build_request(profile_id).model_copy(
                update={"accommodation": "豪华酒店"}
            )
            profile = memory_service.get_profile_snapshot(profile_id)
            summary, conflicts = memory_service.build_memory_context_for_request(
                profile,
                current_request,
            )

            self.assertEqual(len(conflicts), 1)
            self.assertEqual(conflicts[0]["field"], "accommodation")
            self.assertEqual(conflicts[0]["remembered_value"], "经济型酒店")
            self.assertEqual(conflicts[0]["current_value"], "豪华酒店")
            self.assertEqual(conflicts[0]["resolution"], "current_request_used")
            self.assertIn("current request is used", conflicts[0]["explanation"])
            self.assertIn("Memory/current-request conflict note", summary)

    def test_invalid_profile_id_rejected_for_explicit_access(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_service = MemoryService(Path(tmpdir) / "memory.sqlite3")
            with self.assertRaises(ValueError):
                memory_service.get_profile("bad id with spaces")


if __name__ == "__main__":
    unittest.main()
