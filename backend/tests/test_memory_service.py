"""Tests for anonymous trip memory persistence."""

from __future__ import annotations

import tempfile
import unittest
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

            summary = memory_service.build_memory_context(profile_id)
            self.assertIn("Anonymous historical preference memory", summary)
            self.assertIn("current request", summary)

            snapshot = memory_service.get_profile_snapshot(profile_id)
            self.assertEqual(snapshot["trip_count"], 1)
            self.assertEqual(snapshot["preferences"], ["历史文化", "美食"])
            self.assertIsNone(memory_service.get_profile_snapshot("bad id with spaces"))

            self.assertTrue(memory_service.clear_profile(profile_id))
            self.assertIsNone(memory_service.get_profile(profile_id))

    def test_invalid_profile_id_rejected_for_explicit_access(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_service = MemoryService(Path(tmpdir) / "memory.sqlite3")
            with self.assertRaises(ValueError):
                memory_service.get_profile("bad id with spaces")


if __name__ == "__main__":
    unittest.main()
