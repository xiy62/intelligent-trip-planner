"""Tests for industrial-style itinerary quality diagnostics."""

from __future__ import annotations

import unittest

from app.agents.trip_plan_evaluation import evaluate_trip_plan, haversine_km
from app.models.langgraph_state import (
    AttractionCandidate,
    HotelCandidate,
    RAGChunk,
    RetryState,
)
from app.models.schemas import (
    Attraction,
    Budget,
    DayPlan,
    Hotel,
    Location,
    Meal,
    TripPlan,
    TripRequest,
    WeatherInfo,
)


def build_request(**overrides) -> TripRequest:
    data = {
        "city": "New York",
        "start_date": "2026-06-01",
        "end_date": "2026-06-02",
        "travel_days": 2,
        "transportation": "public transit",
        "accommodation": "mid-range hotel",
        "preferences": ["museums", "food"],
        "free_text_input": "Keep the museum and food route relaxed.",
    }
    data.update(overrides)
    return TripRequest(**data)


def build_plan(*, far_jump: bool = False, overloaded: bool = False) -> TripPlan:
    far_location = Location(longitude=-118.2437, latitude=34.0522)
    normal_attractions = [
        Attraction(
            name="The Metropolitan Museum of Art",
            address="1000 5th Ave, New York, NY",
            location=Location(longitude=-73.9632, latitude=40.7794),
            visit_duration=180,
            description="Major museum and cultural landmark",
            ticket_price=60,
        ),
        Attraction(
            name="Museum of Modern Art",
            address="11 W 53rd St, New York, NY",
            location=far_location if far_jump else Location(longitude=-73.9776, latitude=40.7614),
            visit_duration=120,
            description="Modern art museum and cultural stop",
            ticket_price=15,
        ),
    ]
    if overloaded:
        normal_attractions.extend(
            [
                Attraction(
                    name=f"Extra attraction {i}",
                    address="New York",
                    location=Location(longitude=-73.98 + i * 0.01, latitude=40.76),
                    visit_duration=120,
                    description="Additional crowded itinerary stop",
                    ticket_price=0,
                )
                for i in range(5)
            ]
        )
    days = [
        DayPlan(
            date="2026-06-01",
            day_index=0,
            description="Museum and food route with a relaxed pace",
            transportation="public transit",
            accommodation="mid-range hotel",
            hotel=Hotel(name="Midtown Hotel", address="New York", estimated_cost=300),
            attractions=normal_attractions,
            meals=[
                Meal(type="lunch", name="Museum cafe lunch", description="Food stop", estimated_cost=60),
                Meal(type="dinner", name="Neighborhood dinner", description="Food experience", estimated_cost=80),
            ],
        ),
        DayPlan(
            date="2026-06-02",
            day_index=1,
            description="Easy neighborhood walk",
            transportation="public transit",
            accommodation="mid-range hotel",
            hotel=Hotel(name="Midtown Hotel", address="New York", estimated_cost=300),
            attractions=[
                Attraction(
                    name="DUMBO",
                    address="Brooklyn, NY",
                    location=Location(longitude=-73.9887, latitude=40.7033),
                    visit_duration=120,
                    description="Food and neighborhood walk",
                    ticket_price=0,
                )
            ],
            meals=[
                Meal(type="lunch", name="Brooklyn lunch", description="Food experience", estimated_cost=60),
                Meal(type="dinner", name="Classic New York dinner", description="Food experience", estimated_cost=120),
            ],
        ),
    ]
    total_attractions = sum(attraction.ticket_price for day in days for attraction in day.attractions)
    total_hotels = sum(day.hotel.estimated_cost for day in days if day.hotel)
    total_meals = sum(meal.estimated_cost for day in days for meal in day.meals)
    return TripPlan(
        city="New York",
        start_date="2026-06-01",
        end_date="2026-06-02",
        days=days,
        weather_info=[
            WeatherInfo(date="2026-06-01", day_weather="sunny", night_weather="partly cloudy"),
            WeatherInfo(date="2026-06-02", day_weather="cloudy", night_weather="light rain"),
        ],
        overall_suggestions="Combine museums and food without overloading the route.",
        budget=Budget(
            total_attractions=total_attractions,
            total_hotels=total_hotels,
            total_meals=total_meals,
            total_transportation=120,
            total=total_attractions + total_hotels + total_meals + 120,
        ),
    )


def evaluate(plan: TripPlan):
    return evaluate_trip_plan(
        request=build_request(),
        travel_dates=["2026-06-01", "2026-06-02"],
        draft_plan=plan,
        candidate_attractions=[
            AttractionCandidate(name="The Metropolitan Museum of Art", source_id="poi-met"),
            AttractionCandidate(name="Museum of Modern Art", source_id="poi-moma"),
            AttractionCandidate(name="DUMBO", source_id="poi-dumbo"),
            *[
                AttractionCandidate(name=f"Extra attraction {i}", source_id=f"poi-extra-{i}")
                for i in range(5)
            ],
        ],
        candidate_hotels=[HotelCandidate(name="Midtown Hotel", source_id="poi-hotel")],
        rag_chunks=[
            RAGChunk(
                chunk_id="nyc-museums-landmarks-001-overview",
                source="knowledge",
                title="New York museums and landmarks",
                content="The Metropolitan Museum of Art, MoMA, and nearby neighborhoods fit a museum and food route.",
                metadata={
                    "doc_id": "nyc-museums-landmarks-001",
                    "source_url": "https://www.nyctourism.com/",
                },
            )
        ],
        retry_counts=RetryState(),
        max_retries=2,
    )


class TripPlanEvaluationQualityTests(unittest.TestCase):
    def test_haversine_distance_is_reasonable(self):
        distance = haversine_km(
            Location(longitude=-73.9632, latitude=40.7794),
            Location(longitude=-73.9776, latitude=40.7614),
        )

        self.assertGreater(distance, 2.0)
        self.assertLess(distance, 3.0)

    def test_quality_scores_are_recorded_without_blocking_hard_pass(self):
        report = evaluate(build_plan())

        self.assertTrue(report.passed)
        self.assertEqual(report.next_action, "finalize_response")
        self.assertGreaterEqual(report.scores.attribution_coverage_score, 1.0)
        self.assertGreaterEqual(report.scores.route_coherence_score, 0.9)
        self.assertGreaterEqual(report.scores.pacing_score, 0.8)
        self.assertGreaterEqual(report.scores.preference_match_score, 0.5)
        self.assertTrue(report.evidence_links)
        self.assertEqual(report.evidence_links[0].evidence_type, "candidate_attraction")

    def test_route_coherence_flags_long_same_day_jump(self):
        report = evaluate(build_plan(far_jump=True))

        self.assertTrue(report.passed)
        self.assertLess(report.scores.route_coherence_score, 0.75)
        self.assertIn("low_route_coherence_score", report.quality_warnings)

    def test_pacing_flags_overloaded_day_as_soft_warning(self):
        report = evaluate(build_plan(overloaded=True))

        self.assertTrue(report.passed)
        self.assertLess(report.scores.pacing_score, 0.75)
        self.assertIn("low_pacing_score", report.quality_warnings)

    def test_unsupported_entity_lowers_attribution_and_still_uses_hard_grounding(self):
        plan = build_plan()
        plan.days[0].attractions[0].name = "Unsupported Attraction"

        report = evaluate(plan)

        self.assertFalse(report.passed)
        self.assertIn("retrieval_grounding_attractions", report.hard_failures)
        self.assertLess(report.scores.attribution_coverage_score, 1.0)
        self.assertTrue(any(link.evidence_type == "none" for link in report.evidence_links))

    def test_empty_day_attractions_fail_content_completeness_and_retry_planner(self):
        plan = build_plan()
        plan.days[0].attractions = []
        plan.budget.total_attractions = sum(
            attraction.ticket_price for day in plan.days for attraction in day.attractions
        )
        plan.budget.total = (
            plan.budget.total_attractions
            + plan.budget.total_hotels
            + plan.budget.total_meals
            + plan.budget.total_transportation
        )

        report = evaluate(plan)

        self.assertFalse(report.passed)
        self.assertIn("content_completeness_attractions", report.hard_failures)
        self.assertIn("pacing_day_0_no_attractions", report.quality_warnings)
        self.assertEqual(report.next_action, "plan_itinerary")

    def test_empty_day_attractions_retry_retrieval_when_no_candidates_exist(self):
        plan = build_plan()
        plan.days[0].attractions = []
        plan.budget.total_attractions = sum(
            attraction.ticket_price for day in plan.days for attraction in day.attractions
        )
        plan.budget.total = (
            plan.budget.total_attractions
            + plan.budget.total_hotels
            + plan.budget.total_meals
            + plan.budget.total_transportation
        )

        report = evaluate_trip_plan(
            request=build_request(),
            travel_dates=["2026-06-01", "2026-06-02"],
            draft_plan=plan,
            candidate_attractions=[],
            candidate_hotels=[HotelCandidate(name="Midtown Hotel", source_id="poi-hotel")],
            rag_chunks=[],
            retry_counts=RetryState(),
            max_retries=2,
        )

        self.assertFalse(report.passed)
        self.assertIn("content_completeness_attractions", report.hard_failures)
        self.assertEqual(report.next_action, "retrieve_attractions")


if __name__ == "__main__":
    unittest.main()
