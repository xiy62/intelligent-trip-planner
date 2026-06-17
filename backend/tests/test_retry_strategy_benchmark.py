"""Tests for targeted retry vs full-rerun benchmark helpers."""

from __future__ import annotations

import unittest

from app.models.langgraph_state import RunMetrics
from scripts.benchmark_retry_strategies import (
    estimate_full_rerun_baseline,
    summarize,
    targeted_retry_metrics,
)


class RetryStrategyBenchmarkTests(unittest.TestCase):
    def test_estimates_full_rerun_work_from_evaluation_attempts(self):
        metrics = RunMetrics(
            evaluation_attempt_count=2,
            node_attempts={
                "prepare_request": 1,
                "retrieve_attractions": 2,
                "retrieve_hotels": 1,
                "retrieve_weather": 1,
                "retrieve_rag_context": 2,
                "plan_itinerary": 2,
                "evaluate_itinerary": 2,
            },
            node_total_latency_ms={
                "prepare_request": 10.0,
                "retrieve_attractions": 200.0,
                "retrieve_hotels": 50.0,
                "retrieve_weather": 40.0,
                "retrieve_rag_context": 100.0,
                "plan_itinerary": 1000.0,
                "evaluate_itinerary": 20.0,
            },
        )

        estimate = estimate_full_rerun_baseline({"metrics": metrics})

        self.assertEqual(estimate["evaluation_attempts"], 2)
        self.assertEqual(estimate["estimated_full_rerun_external_service_node_calls"], 8)
        self.assertEqual(estimate["estimated_full_rerun_llm_calls"], 2)
        self.assertGreater(estimate["estimated_full_rerun_work_ms"], 0)

    def test_targeted_metrics_count_actual_retry_work(self):
        metrics = RunMetrics(
            node_attempts={
                "retrieve_attractions": 2,
                "retrieve_hotels": 1,
                "retrieve_weather": 1,
                "retrieve_rag_context": 2,
                "plan_itinerary": 2,
            },
            node_total_latency_ms={
                "retrieve_attractions": 200.0,
                "retrieve_hotels": 50.0,
                "retrieve_weather": 40.0,
                "retrieve_rag_context": 100.0,
                "plan_itinerary": 1000.0,
            },
        )

        targeted = targeted_retry_metrics({"metrics": metrics})

        self.assertEqual(targeted["targeted_external_service_node_calls"], 6)
        self.assertEqual(targeted["targeted_llm_calls"], 2)
        self.assertEqual(targeted["targeted_work_ms"], 1390.0)

    def test_summary_reports_reduction_rates(self):
        entries = [
            {
                "first_evaluation_pass": False,
                "recovered_after_retry": True,
                "external_service_node_call_savings": 2,
                "llm_call_savings": 0,
                "estimated_work_ms_savings": 100.0,
                "targeted_external_service_node_calls": 6,
                "estimated_full_rerun_external_service_node_calls": 8,
                "targeted_work_ms": 900.0,
                "estimated_full_rerun_work_ms": 1000.0,
            }
        ]

        result = summarize(entries)

        self.assertEqual(result["recovery_rate"], 1.0)
        self.assertEqual(result["external_service_node_call_reduction_rate"], 0.25)
        self.assertEqual(result["estimated_work_reduction_rate"], 0.1)


if __name__ == "__main__":
    unittest.main()
