"""Tests for lightweight admin-token protection on local-admin API routes."""

from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from app.api.main import app
from app.api.routes import observability as observability_routes
from app.api.routes import rag_ingestion as rag_ingestion_routes
from app.api.routes import trip as trip_routes
from app.config import settings
from app.models.schemas import DayPlan, TripPlan


ADMIN_TOKEN = "test-admin-token"


def valid_trip_payload() -> dict:
    return {
        "city": "New York",
        "start_date": "2026-07-01",
        "end_date": "2026-07-01",
        "travel_days": 1,
        "transportation": "Public transit",
        "accommodation": "Mid-range hotel",
        "preferences": ["Museums"],
        "free_text_input": "Keep this relaxed.",
    }


def valid_trip_plan() -> TripPlan:
    return TripPlan(
        city="New York",
        start_date="2026-07-01",
        end_date="2026-07-01",
        days=[
            DayPlan(
                date="2026-07-01",
                day_index=0,
                description="Visit museum highlights.",
                transportation="Public transit",
                accommodation="Mid-range hotel",
                attractions=[],
                meals=[],
            )
        ],
        weather_info=[],
        overall_suggestions="Check opening hours before departure.",
        budget=None,
    )


class RoutePatchMixin:
    def setUp(self):
        self.client = TestClient(app)
        self.original_admin_api_token = settings.admin_api_token
        self.original_app_env = settings.app_env
        self._patches = []

    def tearDown(self):
        settings.admin_api_token = self.original_admin_api_token
        settings.app_env = self.original_app_env
        for module, name, original in reversed(self._patches):
            setattr(module, name, original)

    def patch_attr(self, module, name: str, value) -> None:
        self._patches.append((module, name, getattr(module, name)))
        setattr(module, name, value)

    def configure_admin(self, token: str = ADMIN_TOKEN, app_env: str = "local") -> None:
        settings.admin_api_token = token
        settings.app_env = app_env


class AdminRouteProtectionTests(RoutePatchMixin, unittest.TestCase):
    def test_rag_ingestion_rejects_missing_token_when_configured(self):
        self.configure_admin()

        response = self.client.get("/api/rag-ingestion/drafts")

        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.json()["detail"], "Invalid admin token")

    def test_rag_ingestion_rejects_wrong_token(self):
        self.configure_admin()

        response = self.client.get("/api/rag-ingestion/drafts", headers={"X-Admin-Token": "wrong-token"})

        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.json()["detail"], "Invalid admin token")

    def test_rag_ingestion_accepts_correct_token(self):
        self.configure_admin()
        self.patch_attr(rag_ingestion_routes, "list_draft_paths", lambda **kwargs: [])

        response = self.client.get("/api/rag-ingestion/drafts", headers={"X-Admin-Token": ADMIN_TOKEN})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"success": True, "data": []})

    def test_local_no_token_configured_remains_backward_compatible(self):
        self.configure_admin(token="", app_env="local")
        self.patch_attr(rag_ingestion_routes, "list_draft_paths", lambda **kwargs: [])

        response = self.client.get("/api/rag-ingestion/drafts")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"success": True, "data": []})

    def test_production_without_configured_token_rejects_admin_route(self):
        self.configure_admin(token="", app_env="production")

        response = self.client.get("/api/rag-ingestion/drafts")

        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.json()["detail"], "Admin API token is required")

    def test_observability_delete_requires_admin_token(self):
        self.configure_admin()

        response = self.client.delete("/api/observability/runs")

        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.json()["detail"], "Invalid admin token")

    def test_observability_delete_accepts_correct_token(self):
        self.configure_admin()

        class FakeObservabilityService:
            def delete_runs(self, source=None):
                self.source = source
                return 3

        self.patch_attr(observability_routes, "get_observability_service", lambda: FakeObservabilityService())

        response = self.client.delete(
            "/api/observability/runs",
            params={"source": "benchmark"},
            headers={"X-Admin-Token": ADMIN_TOKEN},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"success": True, "deleted": 3})

    def test_memory_inspect_requires_admin_token_when_configured(self):
        self.configure_admin()

        response = self.client.get("/api/trip/memory/profile_12345678")

        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.json()["detail"], "Invalid admin token")

    def test_memory_inspect_accepts_correct_token(self):
        self.configure_admin()

        class FakeMemoryService:
            def get_profile(self, profile_id: str):
                return {"profile_id": profile_id, "trip_count": 1}

            def build_memory_context(self, profile_id: str):
                return "memory summary"

        self.patch_attr(trip_routes, "get_memory_service", lambda: FakeMemoryService())

        response = self.client.get(
            "/api/trip/memory/profile_12345678",
            headers={"X-Admin-Token": ADMIN_TOKEN},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["data"], {"profile_id": "profile_12345678", "trip_count": 1})

    def test_public_trip_plan_remains_public_when_admin_token_configured(self):
        self.configure_admin()

        class FakePlanner:
            rag_mode = "test"

            def plan_trip_with_state(self, request):
                return {
                    "final_plan": valid_trip_plan(),
                    "conversation_id": "conversation-public",
                    "memory_applied": False,
                    "memory_summary": "",
                    "memory_profile": None,
                }

        class FakeObservabilityService:
            def persist_state(self, state, source: str, rag_mode: str):
                return None

        self.patch_attr(trip_routes, "get_trip_planner_agent", lambda: FakePlanner())
        self.patch_attr(trip_routes, "get_observability_service", lambda: FakeObservabilityService())

        response = self.client.post("/api/trip/plan", json=valid_trip_payload())

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["success"])
        self.assertEqual(response.json()["conversation_id"], "conversation-public")


if __name__ == "__main__":
    unittest.main()
