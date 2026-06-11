"""Tests for the custom planner observability store."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.models.langgraph_state import (
    EvaluationReport,
    EvaluationScores,
    RAGChunk,
    RetryState,
    RunMetrics,
    UnsupportedEntity,
)
from app.models.schemas import (
    Attraction,
    Budget,
    DayPlan,
    Hotel,
    Location,
    TripPlan,
    TripRequest,
    WeatherInfo,
)
from app.services.observability_service import ObservabilityService


def build_request() -> TripRequest:
    return TripRequest(
        city="北京",
        start_date="2026-06-01",
        end_date="2026-06-02",
        travel_days=2,
        transportation="公共交通",
        accommodation="经济型酒店",
        preferences=["历史文化"],
        free_text_input="希望看故宫",
        profile_id="profile_123456",
        conversation_id="conversation-1",
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
                description="历史文化路线",
                transportation="公共交通",
                accommodation="经济型酒店",
                hotel=Hotel(name="如家酒店", address="北京", estimated_cost=300),
                attractions=[
                    Attraction(
                        name="故宫",
                        address="北京",
                        location=Location(longitude=116.397, latitude=39.917),
                        visit_duration=180,
                        description="历史文化景点",
                        ticket_price=60,
                    )
                ],
                meals=[],
            )
        ],
        weather_info=[
            WeatherInfo(
                date="2026-06-01",
                day_weather="晴",
                night_weather="多云",
                day_temp=28,
                night_temp=20,
                wind_direction="南风",
                wind_power="2级",
            )
        ],
        overall_suggestions="注意防晒",
        budget=Budget(
            total_attractions=60,
            total_hotels=300,
            total_meals=0,
            total_transportation=50,
            total=410,
        ),
    )


def build_metrics(*, passed: bool, recovered: bool = False, fallback: bool = False) -> RunMetrics:
    return RunMetrics(
        started_at=100.0,
        ended_at=101.5,
        end_to_end_ms=1500.0,
        node_latency_ms={
            "prepare_request": 1.5,
            "plan_itinerary": 20.0,
            "evaluate_itinerary": 2.5,
            "finalize_response": 1.0,
        },
        node_attempts={
            "prepare_request": 1,
            "plan_itinerary": 2 if recovered else 1,
            "evaluate_itinerary": 2 if recovered else 1,
            "finalize_response": 1,
        },
        evaluation_pass_count=1 if passed else 0,
        evaluation_attempt_count=2 if recovered else 1,
        first_evaluation_pass=False if recovered else passed,
        final_evaluation_pass=passed,
        recovered_after_retry=recovered,
        fallback_count=1 if fallback else 0,
    )


class ObservabilityServiceTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.service = ObservabilityService(Path(self.tmpdir.name) / "observability.sqlite3")

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_persist_passed_run_writes_summary_and_events(self):
        report = EvaluationReport(
            passed=True,
            hard_failures=[],
            scores=EvaluationScores(
                schema_score=1.0,
                date_coverage_score=1.0,
                budget_consistency_score=1.0,
                grounding_score=1.0,
                pacing_score=0.9,
                route_coherence_score=0.8,
                preference_match_score=0.75,
                attribution_coverage_score=1.0,
            ),
            quality_warnings=["route_day_0_long_jump_20km"],
            next_action="finalize_response",
        )
        state = {
            "request": build_request(),
            "conversation_id": "conversation-1",
            "metrics": build_metrics(passed=True),
            "evaluation_report": report,
            "retry_counts": RetryState(plan_itinerary=1, evaluate_itinerary=1),
            "decision_trace": [
                "prepare_request: normalized request and travel dates",
                "evaluate_itinerary: next_action=finalize_response passed=True",
            ],
            "rag_chunks": [
                RAGChunk(
                    chunk_id="beijing-history-core-001-overview",
                    source="knowledge",
                    title="北京历史文化核心线路建议",
                    content="故宫适合历史文化路线",
                    metadata={
                        "doc_id": "beijing-history-core-001",
                        "source_url": "https://example.com",
                        "section": "overview",
                        "city": "北京",
                        "theme": "历史文化",
                        "rag_backend": "chroma_retrieval",
                    },
                )
            ],
            "final_plan": build_plan(),
        }

        run_id = self.service.persist_state(state, source="runtime", rag_mode="rag_chroma")
        detail = self.service.get_run_detail(run_id)

        self.assertIsNotNone(detail)
        self.assertEqual(detail["city"], "北京")
        self.assertTrue(detail["passed"])
        self.assertEqual(detail["scores"]["grounding_score"], 1.0)
        self.assertEqual(detail["scores"]["attribution_coverage_score"], 1.0)
        self.assertEqual(detail["retrieved_rag_sources"][0]["doc_id"], "beijing-history-core-001")
        self.assertGreaterEqual(len(detail["events"]), 3)

        summary = self.service.summary()
        self.assertEqual(summary["avg_attribution_coverage_score"], 1.0)
        self.assertEqual(summary["quality_warning_rate"], 1.0)

    def test_failed_run_summary_reports_categorization_and_recovery(self):
        failed_report = EvaluationReport(
            passed=False,
            hard_failures=["retrieval_grounding_attractions"],
            scores=EvaluationScores(schema_score=1.0, grounding_score=0.5),
            unsupported_entities=[
                UnsupportedEntity(
                    entity_type="attraction",
                    name="不存在景点",
                    reason="not found in retrieved candidates",
                )
            ],
            next_action="retrieve_attractions",
        )
        recovered_report = EvaluationReport(
            passed=True,
            hard_failures=[],
            scores=EvaluationScores(schema_score=1.0, grounding_score=1.0),
            next_action="finalize_response",
        )
        self.service.persist_state(
            {
                "request": build_request(),
                "conversation_id": "failed-run",
                "metrics": build_metrics(passed=False),
                "evaluation_report": failed_report,
                "retry_counts": RetryState(evaluate_itinerary=1),
                "decision_trace": ["evaluate_itinerary: next_action=retrieve_attractions passed=False"],
                "final_plan": None,
            },
            source="benchmark",
        )
        self.service.persist_state(
            {
                "request": build_request(),
                "conversation_id": "recovered-run",
                "metrics": build_metrics(passed=True, recovered=True),
                "evaluation_report": recovered_report,
                "retry_counts": RetryState(plan_itinerary=2, evaluate_itinerary=2),
                "decision_trace": ["evaluate_itinerary: next_action=finalize_response passed=True"],
                "final_plan": build_plan(),
            },
            source="benchmark",
        )

        summary = self.service.summary()

        self.assertEqual(summary["total_runs"], 2)
        self.assertEqual(summary["failure_category_counts"]["retrieval_grounding_attractions"], 1)
        self.assertEqual(summary["failure_categorization_coverage"], 1.0)
        self.assertEqual(summary["recovery_rate"], 0.5)

    def test_missing_metrics_and_report_does_not_crash(self):
        run_id = self.service.persist_state(
            {
                "request": build_request(),
                "conversation_id": "partial-run",
                "decision_trace": ["partial: no evaluator output"],
            },
            source="runtime",
        )

        detail = self.service.get_run_detail(run_id)

        self.assertIsNotNone(detail)
        self.assertIsNone(detail["passed"])
        self.assertEqual(detail["evaluation_report"], {})

    def test_list_filters_and_cleanup_by_source(self):
        report = EvaluationReport(
            passed=False,
            hard_failures=["budget_consistency"],
            scores=EvaluationScores(schema_score=1.0),
            next_action="plan_itinerary",
        )
        self.service.persist_state(
            {
                "request": build_request(),
                "conversation_id": "runtime-run",
                "metrics": build_metrics(passed=False),
                "evaluation_report": report,
                "decision_trace": ["evaluate_itinerary: next_action=plan_itinerary passed=False"],
            },
            source="runtime",
        )
        self.service.persist_state(
            {
                "request": build_request(),
                "conversation_id": "benchmark-run",
                "metrics": build_metrics(passed=False),
                "evaluation_report": report,
                "decision_trace": ["evaluate_itinerary: next_action=plan_itinerary passed=False"],
            },
            source="benchmark",
        )

        filtered = self.service.list_runs(source="runtime", failure_type="budget_consistency")
        deleted = self.service.delete_runs(source="benchmark")

        self.assertEqual(len(filtered), 1)
        self.assertEqual(deleted, 1)
        self.assertEqual(len(self.service.list_runs()), 1)


if __name__ == "__main__":
    unittest.main()
