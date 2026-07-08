"""FastAPI boundary tests for trip-planning routes.

These tests patch route-level service factories so CI exercises request/response
contracts without creating the LangGraph planner or calling external services.
"""

from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from app.api.main import app
from app.api.routes import trip as trip_routes
from app.models.schemas import DayPlan, TripPlan


def valid_payload() -> dict:
    return {
        "city": "New York",
        "start_date": "2026-07-01",
        "end_date": "2026-07-02",
        "travel_days": 2,
        "transportation": "Public transit",
        "accommodation": "Mid-range hotel",
        "preferences": ["Museums", "Food"],
        "free_text_input": "Keep the pace relaxed.",
        "profile_id": "profile_12345678",
    }


def valid_trip_plan() -> TripPlan:
    return TripPlan(
        city="New York",
        start_date="2026-07-01",
        end_date="2026-07-02",
        days=[
            DayPlan(
                date="2026-07-01",
                day_index=0,
                description="Explore museum highlights.",
                transportation="Public transit",
                accommodation="Mid-range hotel",
                attractions=[],
                meals=[],
            ),
            DayPlan(
                date="2026-07-02",
                day_index=1,
                description="Visit parks and food stops.",
                transportation="Public transit",
                accommodation="Mid-range hotel",
                attractions=[],
                meals=[],
            ),
        ],
        weather_info=[],
        overall_suggestions="Verify opening hours before departure.",
        budget=None,
    )


class RoutePatchMixin:
    """Small helper for restoring route-level monkeypatches."""

    def setUp(self):
        self.client = TestClient(app)
        self._patches = []

    def tearDown(self):
        for name, original in reversed(self._patches):
            setattr(trip_routes, name, original)

    def patch_route_attr(self, name: str, value) -> None:
        self._patches.append((name, getattr(trip_routes, name)))
        setattr(trip_routes, name, value)


class TripPlanValidationApiTests(RoutePatchMixin, unittest.TestCase):
    def assert_validation_error(self, payload: dict) -> None:
        planner_called = False

        def fail_if_called():
            nonlocal planner_called
            planner_called = True
            raise AssertionError("planner must not be created for invalid requests")

        self.patch_route_attr("get_trip_planner_agent", fail_if_called)

        response = self.client.post("/api/trip/plan", json=payload)

        self.assertEqual(response.status_code, 422)
        self.assertIn("detail", response.json())
        self.assertFalse(planner_called)

    def test_missing_city_returns_validation_error(self):
        payload = valid_payload()
        payload.pop("city")

        self.assert_validation_error(payload)

    def test_blank_city_returns_validation_error(self):
        payload = valid_payload()
        payload["city"] = "   "

        self.assert_validation_error(payload)

    def test_invalid_date_format_returns_validation_error(self):
        payload = valid_payload()
        payload["start_date"] = "07/01/2026"

        self.assert_validation_error(payload)

    def test_end_date_before_start_date_returns_validation_error(self):
        payload = valid_payload()
        payload.update({"start_date": "2026-07-03", "end_date": "2026-07-02"})

        self.assert_validation_error(payload)

    def test_travel_days_mismatch_returns_validation_error(self):
        payload = valid_payload()
        payload["travel_days"] = 3

        self.assert_validation_error(payload)

    def test_too_many_preferences_returns_validation_error(self):
        payload = valid_payload()
        payload["preferences"] = [f"tag-{index}" for index in range(11)]

        self.assert_validation_error(payload)

    def test_invalid_preference_item_returns_validation_error(self):
        payload = valid_payload()
        payload["preferences"] = ["Museums", ""]

        self.assert_validation_error(payload)


class TripHealthApiTests(RoutePatchMixin, unittest.TestCase):
    def test_health_returns_workflow_metadata_and_nodes(self):
        class FakePlanner:
            def health_summary(self):
                return {
                    "planner_name": "FakePlanner",
                    "workflow": "langgraph_native",
                    "checkpointer": "MemorySaver",
                    "rag_mode": "test",
                    "parallel_retrieval_enabled": True,
                    "nodes": ["prepare_request", "plan_itinerary", "finalize_response"],
                }

        self.patch_route_attr("get_trip_planner_agent", lambda: FakePlanner())

        response = self.client.get("/api/trip/health")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "healthy")
        self.assertEqual(body["service"], "trip-planner")
        self.assertEqual(body["planner_name"], "FakePlanner")
        self.assertEqual(body["workflow"], "langgraph_native")
        self.assertEqual(body["nodes"], ["prepare_request", "plan_itinerary", "finalize_response"])


class TripPlanApiTests(RoutePatchMixin, unittest.TestCase):
    def test_plan_returns_success_response_shape(self):
        captured_request = None

        class FakePlanner:
            rag_mode = "test"

            def plan_trip_with_state(self, request):
                nonlocal captured_request
                captured_request = request
                return {
                    "final_plan": valid_trip_plan(),
                    "conversation_id": "conversation-123",
                    "memory_applied": True,
                    "memory_summary": "Historical memory was applied.",
                    "memory_profile": {
                        "profile_id": "profile_12345678",
                        "transportation": "Public transit",
                        "accommodation": "Mid-range hotel",
                        "preferences": ["Museums"],
                        "recent_cities": ["New York"],
                        "preference_metadata": {
                            "transportation": [
                                {
                                    "value": "Public transit",
                                    "count": 2,
                                    "last_seen_at": 2.0,
                                    "source_type": "explicit_request",
                                }
                            ],
                            "accommodation": [],
                            "preferences": [],
                        },
                        "trip_count": 1,
                        "last_summary": "Recorded one successful trip.",
                        "created_at": 1.0,
                        "updated_at": 2.0,
                    },
                    "memory_conflicts": [
                        {
                            "field": "accommodation",
                            "remembered_value": "Budget hotel",
                            "current_value": "Mid-range hotel",
                            "resolution": "current_request_used",
                            "count": 1,
                            "last_seen_at": 1.0,
                            "source_type": "explicit_request",
                            "explanation": "Previous accommodation was overridden by the current request.",
                        }
                    ],
                }

        class FakeObservabilityService:
            def __init__(self):
                self.persist_calls = []

            def persist_state(self, state, source: str, rag_mode: str):
                self.persist_calls.append((state, source, rag_mode))

        observability = FakeObservabilityService()
        self.patch_route_attr("get_trip_planner_agent", lambda: FakePlanner())
        self.patch_route_attr("get_observability_service", lambda: observability)

        response = self.client.post("/api/trip/plan", json=valid_payload())

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["success"])
        self.assertEqual(body["message"], "Trip plan generated successfully")
        self.assertEqual(body["conversation_id"], "conversation-123")
        self.assertTrue(body["memory_applied"])
        self.assertEqual(body["memory_summary"], "Historical memory was applied.")
        self.assertEqual(body["memory_profile"]["profile_id"], "profile_12345678")
        self.assertEqual(body["memory_profile"]["preference_metadata"]["transportation"][0]["count"], 2)
        self.assertEqual(body["memory_conflicts"][0]["field"], "accommodation")
        self.assertEqual(body["memory_conflicts"][0]["resolution"], "current_request_used")
        self.assertEqual(body["data"]["city"], "New York")
        self.assertEqual(len(body["data"]["days"]), 2)
        self.assertEqual(captured_request.city, "New York")
        self.assertEqual(len(observability.persist_calls), 1)
        self.assertEqual(observability.persist_calls[0][1], "runtime")
        self.assertEqual(observability.persist_calls[0][2], "test")

    def test_plan_returns_500_with_clear_error_when_planner_fails(self):
        class FailingPlanner:
            def plan_trip_with_state(self, request):
                raise RuntimeError("planner exploded")

        self.patch_route_attr("get_trip_planner_agent", lambda: FailingPlanner())

        response = self.client.post("/api/trip/plan", json=valid_payload())

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["detail"], "Trip planning failed: planner exploded")


class TripMemoryApiTests(RoutePatchMixin, unittest.TestCase):
    def test_clear_memory_missing_profile_id_returns_validation_error(self):
        service_called = False

        def fail_if_called():
            nonlocal service_called
            service_called = True
            raise AssertionError("memory service must not be created for invalid body")

        self.patch_route_attr("get_memory_service", fail_if_called)

        response = self.client.post("/api/trip/memory/clear", json={})

        self.assertEqual(response.status_code, 422)
        self.assertIn("detail", response.json())
        self.assertFalse(service_called)

    def test_clear_memory_invalid_profile_id_returns_400(self):
        class FakeMemoryService:
            def clear_profile(self, profile_id: str):
                raise ValueError("profile_id must be 8-128 characters of letters, numbers, '_' or '-'")

        self.patch_route_attr("get_memory_service", lambda: FakeMemoryService())

        response = self.client.post(
            "/api/trip/memory/clear",
            json={"profile_id": "bad id"},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("profile_id must be 8-128 characters", response.json()["detail"])

    def test_inspect_memory_returns_expected_shape_with_mocked_service(self):
        class FakeMemoryService:
            def get_profile(self, profile_id: str):
                return {
                    "profile_id": profile_id,
                    "transportation": "Public transit",
                    "accommodation": "Mid-range hotel",
                    "preferences": ["Museums", "Food"],
                    "recent_cities": ["New York"],
                    "trip_count": 2,
                    "last_summary": "Recorded two successful trips.",
                    "created_at": 1.0,
                    "updated_at": 2.0,
                }

            def build_memory_context(self, profile_id: str):
                return "Anonymous historical preference memory: museums and food."

        self.patch_route_attr("get_memory_service", lambda: FakeMemoryService())

        response = self.client.get("/api/trip/memory/profile_12345678")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["success"])
        self.assertEqual(body["profile_id"], "profile_12345678")
        self.assertEqual(body["data"]["trip_count"], 2)
        self.assertEqual(body["data"]["preferences"], ["Museums", "Food"])
        self.assertIn("Anonymous historical preference memory", body["memory_summary"])


if __name__ == "__main__":
    unittest.main()
