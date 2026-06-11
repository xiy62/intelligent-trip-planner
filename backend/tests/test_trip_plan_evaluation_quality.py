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
        "city": "北京",
        "start_date": "2026-06-01",
        "end_date": "2026-06-02",
        "travel_days": 2,
        "transportation": "公共交通",
        "accommodation": "经济型酒店",
        "preferences": ["历史文化", "美食"],
        "free_text_input": "希望行程不要太赶",
    }
    data.update(overrides)
    return TripRequest(**data)


def build_plan(*, far_jump: bool = False, overloaded: bool = False) -> TripPlan:
    far_location = Location(longitude=121.47, latitude=31.23)
    normal_attractions = [
        Attraction(
            name="故宫",
            address="东城区景山前街4号",
            location=Location(longitude=116.397, latitude=39.917),
            visit_duration=180,
            description="历史文化景点",
            ticket_price=60,
        ),
        Attraction(
            name="天坛",
            address="东城区天坛路",
            location=far_location if far_jump else Location(longitude=116.41, latitude=39.88),
            visit_duration=120,
            description="世界文化遗产和历史文化景点",
            ticket_price=15,
        ),
    ]
    if overloaded:
        normal_attractions.extend(
            [
                Attraction(
                    name=f"景点{i}",
                    address="北京",
                    location=Location(longitude=116.42 + i * 0.01, latitude=39.88),
                    visit_duration=120,
                    description="加塞景点",
                    ticket_price=0,
                )
                for i in range(5)
            ]
        )
    days = [
        DayPlan(
            date="2026-06-01",
            day_index=0,
            description="历史文化和美食路线，节奏舒适",
            transportation="公共交通",
            accommodation="经济型酒店",
            hotel=Hotel(name="如家酒店", address="北京", estimated_cost=300),
            attractions=normal_attractions,
            meals=[
                Meal(type="lunch", name="北京小吃", description="美食体验", estimated_cost=60),
                Meal(type="dinner", name="胡同晚餐", description="美食体验", estimated_cost=80),
            ],
        ),
        DayPlan(
            date="2026-06-02",
            day_index=1,
            description="轻松城市漫步",
            transportation="公共交通",
            accommodation="经济型酒店",
            hotel=Hotel(name="如家酒店", address="北京", estimated_cost=300),
            attractions=[
                Attraction(
                    name="南锣鼓巷",
                    address="东城区南锣鼓巷",
                    location=Location(longitude=116.403, latitude=39.94),
                    visit_duration=120,
                    description="美食和胡同文化",
                    ticket_price=0,
                )
            ],
            meals=[
                Meal(type="lunch", name="胡同小吃", description="美食体验", estimated_cost=60),
                Meal(type="dinner", name="北京烤鸭", description="美食体验", estimated_cost=120),
            ],
        ),
    ]
    total_attractions = sum(attraction.ticket_price for day in days for attraction in day.attractions)
    total_hotels = sum(day.hotel.estimated_cost for day in days if day.hotel)
    total_meals = sum(meal.estimated_cost for day in days for meal in day.meals)
    return TripPlan(
        city="北京",
        start_date="2026-06-01",
        end_date="2026-06-02",
        days=days,
        weather_info=[
            WeatherInfo(date="2026-06-01", day_weather="晴", night_weather="多云"),
            WeatherInfo(date="2026-06-02", day_weather="阴", night_weather="小雨"),
        ],
        overall_suggestions="历史文化和美食结合，行程不要太赶。",
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
            AttractionCandidate(name="故宫", source_id="poi-palace"),
            AttractionCandidate(name="天坛", source_id="poi-temple"),
            AttractionCandidate(name="南锣鼓巷", source_id="poi-hutong"),
            *[
                AttractionCandidate(name=f"景点{i}", source_id=f"poi-extra-{i}")
                for i in range(5)
            ],
        ],
        candidate_hotels=[HotelCandidate(name="如家酒店", source_id="poi-hotel")],
        rag_chunks=[
            RAGChunk(
                chunk_id="beijing-history-core-001-overview",
                source="knowledge",
                title="北京历史文化核心线路",
                content="故宫、天坛和南锣鼓巷适合历史文化和美食路线。",
                metadata={
                    "doc_id": "beijing-history-core-001",
                    "source_url": "https://example.com/beijing",
                },
            )
        ],
        retry_counts=RetryState(),
        max_retries=2,
    )


class TripPlanEvaluationQualityTests(unittest.TestCase):
    def test_haversine_distance_is_reasonable(self):
        distance = haversine_km(
            Location(longitude=116.397, latitude=39.917),
            Location(longitude=116.41, latitude=39.88),
        )

        self.assertGreater(distance, 3.0)
        self.assertLess(distance, 5.0)

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
        plan.days[0].attractions[0].name = "不存在景点"

        report = evaluate(plan)

        self.assertFalse(report.passed)
        self.assertIn("retrieval_grounding_attractions", report.hard_failures)
        self.assertLess(report.scores.attribution_coverage_score, 1.0)
        self.assertTrue(any(link.evidence_type == "none" for link in report.evidence_links))


if __name__ == "__main__":
    unittest.main()
