"""Tests for the LangGraph-native trip planner."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app.agents.langgraph_trip_planner import LangGraphTripPlanner
from app.config import settings
from app.models.langgraph_state import AttractionCandidate, HotelCandidate, RetryState
from app.models.schemas import TripPlan, TripRequest, WeatherInfo
from app.services.memory_service import MemoryService


def build_request() -> TripRequest:
    return TripRequest(
        city="New York",
        start_date="2026-06-01",
        end_date="2026-06-02",
        travel_days=2,
        transportation="Public transit",
        accommodation="Budget hotel",
        preferences=["Museums", "Food"],
        free_text_input="Keep the itinerary relaxed.",
    )


def build_valid_plan_json(
    attraction_name: str = "Metropolitan Museum of Art",
    hotel_name: str = "Pod Times Square",
    transportation: str = "Public transit",
    accommodation: str = "Budget hotel",
) -> str:
    return f"""```json
{{
  "city": "New York",
  "start_date": "2026-06-01",
  "end_date": "2026-06-02",
  "days": [
    {{
      "date": "2026-06-01",
      "day_index": 0,
      "description": "Museum and Central Park day",
      "transportation": "{transportation}",
      "accommodation": "{accommodation}",
      "hotel": {{
        "name": "{hotel_name}",
        "address": "400 W 42nd St, New York, NY",
        "estimated_cost": 220
      }},
      "attractions": [
        {{
          "name": "{attraction_name}",
          "address": "1000 5th Ave, New York, NY",
          "location": {{"longitude": -73.9632, "latitude": 40.7794}},
          "visit_duration": 180,
          "description": "Major museum and cultural landmark",
          "category": "Museum",
          "ticket_price": 30
        }},
        {{
          "name": "Central Park",
          "address": "New York, NY",
          "location": {{"longitude": -73.9654, "latitude": 40.7829}},
          "visit_duration": 120,
          "description": "Urban park with relaxed walking routes",
          "category": "Park",
          "ticket_price": 0
        }}
      ],
      "meals": [
        {{"type": "breakfast", "name": "Hotel breakfast", "estimated_cost": 20}},
        {{"type": "lunch", "name": "Museum cafe", "estimated_cost": 35}},
        {{"type": "dinner", "name": "Hell's Kitchen dinner", "estimated_cost": 60}}
      ]
    }},
    {{
      "date": "2026-06-02",
      "day_index": 1,
      "description": "Downtown food and waterfront day",
      "transportation": "{transportation}",
      "accommodation": "{accommodation}",
      "hotel": {{
        "name": "{hotel_name}",
        "address": "400 W 42nd St, New York, NY",
        "estimated_cost": 220
      }},
      "attractions": [
        {{
          "name": "Brooklyn Bridge",
          "address": "Brooklyn Bridge, New York, NY",
          "location": {{"longitude": -73.9969, "latitude": 40.7061}},
          "visit_duration": 90,
          "description": "Classic skyline walk",
          "category": "Landmark",
          "ticket_price": 0
        }},
        {{
          "name": "Chelsea Market",
          "address": "75 9th Ave, New York, NY",
          "location": {{"longitude": -74.006, "latitude": 40.7424}},
          "visit_duration": 120,
          "description": "Food hall and market",
          "category": "Food",
          "ticket_price": 0
        }}
      ],
      "meals": [
        {{"type": "breakfast", "name": "Bagel breakfast", "estimated_cost": 18}},
        {{"type": "lunch", "name": "Chelsea Market lunch", "estimated_cost": 40}},
        {{"type": "dinner", "name": "Downtown dinner", "estimated_cost": 70}}
      ]
    }}
  ],
  "weather_info": [
    {{
      "date": "2026-09-01",
      "day_weather": "Storm",
      "night_weather": "Storm",
      "day_temp": 99,
      "night_temp": 88,
      "wind_direction": "N",
      "wind_power": "40 km/h"
    }}
  ],
  "overall_suggestions": "Use transit and avoid overpacking each day.",
  "budget": {{
    "total_attractions": 30,
    "total_hotels": 440,
    "total_meals": 243,
    "total_transportation": 60,
    "total": 773
  }}
}}
```"""


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def invoke(self, prompt: str):
        self.calls.append(prompt)
        if len(self.responses) == 1:
            content = self.responses[0]
        else:
            content = self.responses.pop(0)
        return type("FakeMessage", (), {"content": content})()


def parse_candidate_response(response: str):
    items = []
    for index, line in enumerate(response.splitlines()):
        if "Address:" not in line:
            continue
        name, address = line.split("Address:", 1)
        name = name.split(".", 1)[-1].strip(" -")
        slug = name.lower().replace(" ", "-").replace("'", "")
        items.append(
            {
                "id": f"fake-{index}-{slug}",
                "name": name,
                "address": address.strip(),
                "image_url": f"/api/map/photo?photo_name=fake-{slug}",
                "maps_url": f"https://maps.example.com/{slug}",
                "website_url": f"https://www.example.com/{slug}",
                "rating": 4.7,
            }
        )
    return items


class FakeSearchTool:
    name = "map_search_poi"

    def __init__(self, attraction_responses, hotel_responses, meal_responses=None):
        self.attraction_rounds = [parse_candidate_response(item) for item in attraction_responses]
        self.hotel_items = parse_candidate_response(hotel_responses[0])
        self.meal_items = parse_candidate_response(meal_responses[0]) if meal_responses else []
        self.attraction_calls = 0
        self.calls = []

    def invoke(self, payload):
        self.calls.append(dict(payload))
        keyword = payload["keywords"].lower()
        if "hotel" in keyword or "inn" in keyword:
            return list(self.hotel_items)
        if any(term in keyword for term in ("restaurant", "dining", "food", "cafe", "lunch", "dinner", "breakfast")):
            return list(self.meal_items)
        round_index = min(self.attraction_calls // 3, len(self.attraction_rounds) - 1)
        self.attraction_calls += 1
        return list(self.attraction_rounds[round_index])


class FakeMapService:
    def __init__(self, attraction_responses, hotel_responses, meal_responses=None, route_responses=None):
        self.tool = FakeSearchTool(attraction_responses, hotel_responses, meal_responses)
        self.route_responses = list(route_responses or [])
        self.route_calls = []

    def get_langchain_tools(self):
        return [self.tool]

    def plan_route(self, **kwargs):
        self.route_calls.append(dict(kwargs))
        if self.route_responses:
            response = self.route_responses.pop(0)
            if isinstance(response, Exception):
                raise response
            return dict(response)
        return {"duration": 900, "distance": 1200}


class StaticSearchTool:
    name = "map_search_poi"

    def __init__(self, items):
        self.items = items

    def invoke(self, payload):
        return list(self.items)


class StaticMapService:
    def __init__(self, items, route_response=None):
        self.tool = StaticSearchTool(items)
        self.route_response = route_response or {"duration": 900, "distance": 1200}
        self.route_calls = []

    def get_langchain_tools(self):
        return [self.tool]

    def plan_route(self, **kwargs):
        self.route_calls.append(dict(kwargs))
        if isinstance(self.route_response, Exception):
            raise self.route_response
        return dict(self.route_response)


class FakeWeatherService:
    def __init__(self):
        self.results = [
            WeatherInfo(
                date="2026-06-01",
                day_weather="Clear",
                night_weather="Cloudy",
                day_temp=30,
                night_temp=20,
                wind_direction="S",
                wind_power="12 km/h",
            ),
            WeatherInfo(
                date="2026-06-02",
                day_weather="Overcast",
                night_weather="Light rain",
                day_temp=28,
                night_temp=18,
                wind_direction="E",
                wind_power="8 km/h",
            ),
        ]

    def get_weather_for_trip(self, city: str, start_date: str, travel_days: int):
        return list(self.results)

    def format_weather_for_planner(self, city: str, weather_info):
        lines = [f"Weather for {city}, aligned to trip dates:"]
        for item in weather_info:
            lines.append(f"- {item.date}: daytime {item.day_weather}, nighttime {item.night_weather}")
        return "\n".join(lines)


class FakeNativeRuntime:
    def __init__(
        self,
        attraction_responses,
        hotel_responses,
        planner_responses,
        meal_responses=None,
        route_responses=None,
    ):
        self.map_service = FakeMapService(
            attraction_responses,
            hotel_responses,
            meal_responses,
            route_responses,
        )
        self.llm = FakeLLM(planner_responses)
        self.weather_service = FakeWeatherService()

    def build_planner(self, **kwargs):
        return LangGraphTripPlanner(
            llm=self.llm,
            map_service=self.map_service,
            weather_service=self.weather_service,
            **kwargs,
        )


ATTRACTIONS_ALL = (
    "1. Metropolitan Museum of Art - Address: 1000 5th Ave, New York, NY\n"
    "2. Central Park - Address: New York, NY\n"
    "3. Brooklyn Bridge - Address: Brooklyn Bridge, New York, NY\n"
    "4. Chelsea Market - Address: 75 9th Ave, New York, NY"
)
HOTELS_ALL = "1. Pod Times Square - Address: 400 W 42nd St, New York, NY"
MEALS_ALL = "1. Katz's Delicatessen - Address: 205 E Houston St, New York, NY"


class LangGraphTripPlannerTests(unittest.TestCase):
    def build_overloaded_evaluation_state(self, planner: LangGraphTripPlanner, request: TripRequest):
        plan = TripPlan(
            **json.loads(build_valid_plan_json().split("```json\n", 1)[1].rsplit("\n```", 1)[0])
        )
        plan = planner._normalize_trip_plan(plan, request)
        plan.days[0].attractions.extend([plan.days[0].attractions[1].model_copy() for _ in range(3)])
        plan.budget.total_attractions = sum(
            attraction.ticket_price for day in plan.days for attraction in day.attractions
        )
        plan.budget.total = (
            plan.budget.total_attractions
            + plan.budget.total_hotels
            + plan.budget.total_meals
            + plan.budget.total_transportation
        )
        return {
            "request": request,
            "travel_dates": ["2026-06-01", "2026-06-02"],
            "draft_plan": plan,
            "candidate_attractions": [
                AttractionCandidate(name="Metropolitan Museum of Art", source_id="poi-met"),
                AttractionCandidate(name="Central Park", source_id="poi-park"),
                AttractionCandidate(name="Brooklyn Bridge", source_id="poi-bridge"),
                AttractionCandidate(name="Chelsea Market", source_id="poi-market"),
            ],
            "candidate_hotels": [HotelCandidate(name="Pod Times Square", source_id="poi-hotel")],
            "candidate_meals": [],
            "rag_chunks": [],
            "retry_counts": RetryState(plan_itinerary=1),
        }

    def test_weather_authority_and_checkpoint_snapshot(self):
        runtime = FakeNativeRuntime(
            attraction_responses=[ATTRACTIONS_ALL],
            hotel_responses=[HOTELS_ALL],
            planner_responses=[build_valid_plan_json()],
        )
        planner = runtime.build_planner()
        thread_id = "weather-authority-thread"
        state = planner.invoke_graph(build_request(), thread_id=thread_id)

        final_plan = state["final_plan"]
        self.assertEqual(final_plan.weather_info[0].date, "2026-06-01")
        self.assertEqual(final_plan.weather_info[0].day_weather, "Clear")
        self.assertEqual(final_plan.weather_info[1].night_weather, "Light rain")
        self.assertEqual(final_plan.days[0].attractions[0].poi_id, "fake-0-metropolitan-museum-of-art")
        self.assertEqual(final_plan.days[0].attractions[0].maps_url, "https://maps.example.com/metropolitan-museum-of-art")
        self.assertEqual(final_plan.days[0].attractions[0].image_url, "/api/map/photo?photo_name=fake-metropolitan-museum-of-art")
        self.assertEqual(final_plan.days[0].hotel.maps_url, "https://maps.example.com/pod-times-square")

        snapshot = planner.get_state_snapshot(thread_id)
        self.assertEqual(snapshot.values["final_plan"].city, "New York")
        self.assertEqual(snapshot.values["metrics"].evaluation_pass_count, 1)
        self.assertEqual(runtime.map_service.route_calls, [])

    def test_generate_uses_fresh_run_id_separate_from_conversation_id(self):
        runtime = FakeNativeRuntime(
            attraction_responses=[ATTRACTIONS_ALL],
            hotel_responses=[HOTELS_ALL],
            planner_responses=[build_valid_plan_json()],
        )
        planner = runtime.build_planner()
        request = build_request().model_copy(update={"conversation_id": "product-session"})
        first = planner.invoke_graph(request)
        second = planner.invoke_graph(request)
        self.assertEqual(first["conversation_id"], "product-session")
        self.assertEqual(second["conversation_id"], "product-session")
        self.assertNotEqual(first["run_id"], second["run_id"])

    def test_route_time_collection_uses_fake_map_provider_when_enabled(self):
        original_route_time_enabled = settings.route_time_evaluation_enabled
        original_segment_thresholds = settings.max_segment_minutes_by_mode
        original_daily_thresholds = settings.max_daily_transit_minutes_by_mode
        try:
            settings.route_time_evaluation_enabled = True
            settings.max_segment_minutes_by_mode = {
                "walking": 30,
                "transit": 45,
                "driving": 35,
                "bicycling": 30,
            }
            settings.max_daily_transit_minutes_by_mode = {
                "walking": 90,
                "transit": 150,
                "driving": 120,
                "bicycling": 120,
            }
            runtime = FakeNativeRuntime(
                attraction_responses=[ATTRACTIONS_ALL],
                hotel_responses=[HOTELS_ALL],
                planner_responses=[build_valid_plan_json()],
                route_responses=[
                    {"duration": 3600, "distance": 5000},
                    {"duration": 900, "distance": 1200},
                ],
            )
            planner = runtime.build_planner()
            state = planner.invoke_graph(build_request())
        finally:
            settings.route_time_evaluation_enabled = original_route_time_enabled
            settings.max_segment_minutes_by_mode = original_segment_thresholds
            settings.max_daily_transit_minutes_by_mode = original_daily_thresholds

        report = state["evaluation_report"]

        self.assertEqual(len(runtime.map_service.route_calls), 2)
        self.assertEqual(runtime.map_service.route_calls[0]["route_type"], "transit")
        self.assertEqual(state["route_time_estimates"][0].duration_minutes, 60)
        self.assertIn("route_day_0_long_transfer_60min", report.quality_warnings)
        self.assertIn("low_route_coherence_score", report.quality_warnings)

    def test_route_time_collection_respects_provider_call_cap(self):
        original_route_time_enabled = settings.route_time_evaluation_enabled
        original_max_route_calls = settings.max_route_time_evaluations_per_trip
        try:
            settings.route_time_evaluation_enabled = True
            settings.max_route_time_evaluations_per_trip = 1
            runtime = FakeNativeRuntime(
                attraction_responses=[ATTRACTIONS_ALL],
                hotel_responses=[HOTELS_ALL],
                planner_responses=[build_valid_plan_json()],
                route_responses=[
                    {"duration": 900, "distance": 1200},
                    {"duration": 900, "distance": 1200},
                ],
            )
            planner = runtime.build_planner()
            state = planner.invoke_graph(build_request())
        finally:
            settings.route_time_evaluation_enabled = original_route_time_enabled
            settings.max_route_time_evaluations_per_trip = original_max_route_calls

        self.assertEqual(len(runtime.map_service.route_calls), 1)
        self.assertEqual(len(state["route_time_estimates"]), 2)
        self.assertEqual(state["route_time_estimates"][1].fallback_reason, "route_time_call_cap_reached")
        self.assertIn("route_time_fallback_day_1_segment_0", state["evaluation_report"].quality_warnings)

    def test_meal_retrieval_grounding_and_metadata_enrichment(self):
        plan_json = build_valid_plan_json().replace(
            '{"type": "lunch", "name": "Museum cafe", "estimated_cost": 35}',
            (
                '{"type": "lunch", "name": "Katz\'s Delicatessen", '
                '"address": "205 E Houston St, New York, NY", "estimated_cost": 40}'
            ),
        )
        runtime = FakeNativeRuntime(
            attraction_responses=[ATTRACTIONS_ALL],
            hotel_responses=[HOTELS_ALL],
            meal_responses=[MEALS_ALL],
            planner_responses=[plan_json],
        )
        planner = runtime.build_planner()
        state = planner.invoke_graph(build_request())

        report = state["evaluation_report"]
        meal = state["final_plan"].days[0].meals[1]

        self.assertTrue(report.passed)
        self.assertIn("candidate_meals", state)
        self.assertEqual(state["candidate_meals"][0].name, "Katz's Delicatessen")
        self.assertEqual(meal.poi_id, "fake-0-katzs-delicatessen")
        self.assertEqual(meal.maps_url, "https://maps.example.com/katzs-delicatessen")
        self.assertTrue(any(link.entity_type == "meal" and link.evidence_type == "candidate_meal" for link in report.evidence_links))
        self.assertIn("Restaurant candidates", runtime.llm.calls[0])

    def test_default_quality_warnings_do_not_retry_langgraph_evaluation(self):
        original_quality_retry_enabled = settings.quality_retry_enabled
        try:
            settings.quality_retry_enabled = False
            planner = LangGraphTripPlanner(
                llm=FakeLLM([build_valid_plan_json()]),
                map_service=StaticMapService([]),
                weather_service=FakeWeatherService(),
            )
            state = planner.evaluate_itinerary(
                self.build_overloaded_evaluation_state(planner, build_request())
            )
        finally:
            settings.quality_retry_enabled = original_quality_retry_enabled

        report = state["evaluation_report"]
        self.assertTrue(report.passed)
        self.assertEqual(report.next_action, "finalize_response")
        self.assertIn("pacing_day_0_overloaded", report.quality_warnings)

    def test_strict_quality_warnings_retry_langgraph_evaluation(self):
        original_quality_retry_enabled = settings.quality_retry_enabled
        try:
            settings.quality_retry_enabled = True
            planner = LangGraphTripPlanner(
                llm=FakeLLM([build_valid_plan_json()]),
                map_service=StaticMapService([]),
                weather_service=FakeWeatherService(),
            )
            state = planner.evaluate_itinerary(
                self.build_overloaded_evaluation_state(planner, build_request())
            )
        finally:
            settings.quality_retry_enabled = original_quality_retry_enabled

        report = state["evaluation_report"]
        self.assertFalse(report.passed)
        self.assertEqual(report.hard_failures, [])
        self.assertEqual(report.next_action, "plan_itinerary")
        self.assertIn("strict_quality_retry_triggered", report.warnings)
        self.assertIn("warnings=strict_quality_retry_triggered", state["decision_trace"][0])

    def test_strict_quality_retry_exhaustion_falls_back_in_langgraph_evaluation(self):
        original_quality_retry_enabled = settings.quality_retry_enabled
        try:
            settings.quality_retry_enabled = True
            planner = LangGraphTripPlanner(
                max_retries=2,
                llm=FakeLLM([build_valid_plan_json()]),
                map_service=StaticMapService([]),
                weather_service=FakeWeatherService(),
            )
            graph_state = self.build_overloaded_evaluation_state(planner, build_request())
            graph_state["retry_counts"] = RetryState(plan_itinerary=3)
            state = planner.evaluate_itinerary(graph_state)
        finally:
            settings.quality_retry_enabled = original_quality_retry_enabled

        report = state["evaluation_report"]
        self.assertFalse(report.passed)
        self.assertEqual(report.hard_failures, [])
        self.assertEqual(report.next_action, "fallback_response")
        self.assertIn("strict_quality_retry_triggered", report.warnings)

    def test_planner_malformed_json_retries_then_succeeds(self):
        runtime = FakeNativeRuntime(
            attraction_responses=[ATTRACTIONS_ALL],
            hotel_responses=[HOTELS_ALL],
            planner_responses=["not-json", build_valid_plan_json()],
        )
        planner = runtime.build_planner(max_retries=2)
        state = planner.invoke_graph(build_request())

        self.assertEqual(state["final_plan"].city, "New York")
        self.assertEqual(state["retry_counts"].plan_itinerary, 2)
        self.assertEqual(state["metrics"].schema_failure_count, 1)
        self.assertTrue(state["evaluation_report"].passed)

    def test_grounding_failure_retries_attraction_retrieval(self):
        runtime = FakeNativeRuntime(
            attraction_responses=[
                "1. Statue of Liberty - Address: New York, NY",
                ATTRACTIONS_ALL,
            ],
            hotel_responses=[HOTELS_ALL],
            planner_responses=[build_valid_plan_json(), build_valid_plan_json()],
        )
        planner = runtime.build_planner(max_retries=2)
        state = planner.invoke_graph(build_request())

        self.assertEqual(state["final_plan"].city, "New York")
        self.assertTrue(state["evaluation_report"].passed)
        self.assertGreaterEqual(state["retry_counts"].retrieve_attractions, 1)
        self.assertEqual(state["metrics"].grounding_failure_count, 1)

    def test_attraction_retrieval_filters_travel_service_providers(self):
        planner = LangGraphTripPlanner(
            llm=FakeLLM([build_valid_plan_json()]),
            map_service=StaticMapService(
                [
                    {
                        "id": "service-1",
                        "name": "Fora Travel, Inc.",
                        "address": "New York, NY",
                        "type": "travel_agency, point_of_interest, establishment",
                        "raw": {"types": ["travel_agency", "point_of_interest", "establishment"]},
                    },
                    {
                        "id": "service-2",
                        "name": "Sidewalk Food Tours of New York",
                        "address": "New York, NY",
                        "type": "point_of_interest, establishment",
                        "raw": {"types": ["point_of_interest", "establishment"]},
                    },
                    {
                        "id": "service-3",
                        "name": "Solo New York",
                        "address": "400 Wireless Blvd #1, Hauppauge, NY 11788",
                        "type": "corporate_office, point_of_interest, establishment",
                        "raw": {"types": ["corporate_office", "point_of_interest", "establishment"]},
                    },
                    {
                        "id": "park-1",
                        "name": "Central Park",
                        "address": "New York, NY",
                        "type": "park, tourist_attraction, point_of_interest",
                        "raw": {"types": ["park", "tourist_attraction", "point_of_interest"]},
                    },
                    {
                        "id": "museum-1",
                        "name": "Metropolitan Museum of Art",
                        "address": "1000 5th Ave, New York, NY",
                        "type": "museum, tourist_attraction, point_of_interest",
                        "raw": {"types": ["museum", "tourist_attraction", "point_of_interest"]},
                    },
                ]
            ),
            weather_service=FakeWeatherService(),
        )
        request = build_request().model_copy(
            update={"preferences": ["solo travel", "women travelers", "safety"]}
        )

        state = planner.retrieve_attractions({"request": request})
        names = [candidate.name for candidate in state["candidate_attractions"]]

        self.assertNotIn("Fora Travel, Inc.", names)
        self.assertNotIn("Sidewalk Food Tours of New York", names)
        self.assertNotIn("Solo New York", names)
        self.assertIn("Central Park", names)
        self.assertIn("Metropolitan Museum of Art", names)

    def test_map_retrieval_passes_request_country_code(self):
        runtime = FakeNativeRuntime(
            attraction_responses=[ATTRACTIONS_ALL],
            hotel_responses=[HOTELS_ALL],
            planner_responses=[build_valid_plan_json()],
        )
        planner = runtime.build_planner()
        request = build_request().model_copy(update={"country_code": "JP"})

        planner.retrieve_attractions({"request": request})
        planner.retrieve_hotels({"request": request})
        planner.retrieve_meals({"request": request})

        self.assertTrue(runtime.map_service.tool.calls)
        self.assertTrue(
            all(call["country_code"] == "JP" for call in runtime.map_service.tool.calls)
        )

    def test_retry_exhaustion_falls_back(self):
        runtime = FakeNativeRuntime(
            attraction_responses=["1. Metropolitan Museum of Art - Address: 1000 5th Ave, New York, NY"],
            hotel_responses=[HOTELS_ALL],
            planner_responses=["bad-json", "still-bad-json"],
        )
        planner = runtime.build_planner(max_retries=1)
        state = planner.invoke_graph(build_request())

        self.assertTrue(state["final_plan"].overall_suggestions.startswith("This is a fallback"))
        self.assertEqual(state["metrics"].fallback_count, 1)
        self.assertEqual(state["retry_counts"].fallback_response, 1)

    def test_anonymous_profile_memory_is_injected_as_soft_context(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_service = MemoryService(Path(tmpdir) / "memory.sqlite3")
            profile_id = "profile_test_123"
            first_request = build_request().model_copy(update={"profile_id": profile_id})
            first_plan = TripPlan(
                **json.loads(build_valid_plan_json().split("```json\n", 1)[1].rsplit("\n```", 1)[0])
            )
            memory_service.update_after_success(
                profile_id=profile_id,
                conversation_id="first-conversation",
                request=first_request,
                plan=first_plan,
                memory_applied=False,
                memory_summary="",
            )

            runtime = FakeNativeRuntime(
                attraction_responses=[ATTRACTIONS_ALL],
                hotel_responses=[HOTELS_ALL],
                planner_responses=[build_valid_plan_json()],
            )
            planner = runtime.build_planner(memory_service=memory_service)
            current_request = build_request().model_copy(
                update={
                    "profile_id": profile_id,
                    "conversation_id": "second-conversation",
                    "accommodation": "Luxury hotel",
                }
            )
            state = planner.invoke_graph(current_request)

            self.assertTrue(state["memory_applied"])
            self.assertIn("Museums", state["memory_summary"])
            self.assertEqual(len(state["memory_conflicts"]), 1)
            self.assertEqual(state["memory_conflicts"][0]["field"], "accommodation")
            self.assertEqual(state["memory_conflicts"][0]["remembered_value"], "Budget hotel")
            self.assertEqual(state["memory_conflicts"][0]["current_value"], "Luxury hotel")
            self.assertIn("current request is used", state["memory_conflicts"][0]["explanation"])
            self.assertIn("Memory/current-request conflict note", state["memory_summary"])
            self.assertEqual(state["memory_profile"]["accommodation"], "Budget hotel")
            self.assertEqual(state["memory_profile"]["preference_metadata"]["accommodation"][0]["value"], "Budget hotel")
            self.assertIn("New York", state["memory_profile"]["recent_cities"])
            self.assertEqual(state["conversation_id"], "second-conversation")
            planner_prompt = runtime.llm.calls[0]
            self.assertIn("Anonymous preference memory", planner_prompt)
            self.assertIn("current request", planner_prompt)
            self.assertIn("highest priority", planner_prompt)
            self.assertIn("Accommodation preference: Luxury hotel", planner_prompt)

    def test_current_request_alignment_guardrail_overrides_memory_conflict(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_service = MemoryService(Path(tmpdir) / "memory.sqlite3")
            profile_id = "profile_conflict_123"
            first_request = build_request().model_copy(update={"profile_id": profile_id})
            first_plan = TripPlan(
                **json.loads(build_valid_plan_json().split("```json\n", 1)[1].rsplit("\n```", 1)[0])
            )
            memory_service.update_after_success(
                profile_id=profile_id,
                conversation_id="first-conversation",
                request=first_request,
                plan=first_plan,
                memory_applied=False,
                memory_summary="",
            )

            runtime = FakeNativeRuntime(
                attraction_responses=[ATTRACTIONS_ALL],
                hotel_responses=[HOTELS_ALL],
                planner_responses=[
                    build_valid_plan_json(accommodation="Budget hotel"),
                ],
            )
            planner = runtime.build_planner(max_retries=2, memory_service=memory_service)
            request = build_request().model_copy(
                update={
                    "profile_id": profile_id,
                    "accommodation": "Luxury hotel",
                }
            )
            state = planner.invoke_graph(request)

            self.assertTrue(state["evaluation_report"].passed)
            self.assertEqual(state["retry_counts"].plan_itinerary, 1)
            self.assertEqual(state["memory_conflicts"][0]["field"], "accommodation")
            self.assertEqual(state["memory_conflicts"][0]["resolution"], "current_request_used")
            self.assertEqual(state["final_plan"].days[0].accommodation, "Luxury hotel")
            self.assertEqual(state["final_plan"].days[1].accommodation, "Luxury hotel")
            self.assertEqual(memory_service.get_profile(profile_id)["accommodation"], "Luxury hotel")

    def test_fallback_does_not_write_anonymous_profile_memory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_service = MemoryService(Path(tmpdir) / "memory.sqlite3")
            profile_id = "profile_fallback_123"
            runtime = FakeNativeRuntime(
                attraction_responses=["1. Metropolitan Museum of Art - Address: 1000 5th Ave, New York, NY"],
                hotel_responses=[HOTELS_ALL],
                planner_responses=["bad-json", "still-bad-json"],
            )
            planner = runtime.build_planner(max_retries=1, memory_service=memory_service)
            request = build_request().model_copy(update={"profile_id": profile_id})
            state = planner.invoke_graph(request)

            self.assertTrue(state["final_plan"].overall_suggestions.startswith("This is a fallback"))
            self.assertIsNone(memory_service.get_profile(profile_id))


if __name__ == "__main__":
    unittest.main()
