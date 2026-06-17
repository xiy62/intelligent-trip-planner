"""Tests for adversarial evaluator calibration benchmark helpers."""

from __future__ import annotations

import unittest

from scripts.benchmark_adversarial_evaluation import (
    precision_recall_f1,
    run_case,
    summarize,
)


class AdversarialEvaluationBenchmarkTests(unittest.TestCase):
    def test_precision_recall_f1_counts_validator_outcomes(self):
        results = [
            {
                "expected_failures": ["budget_consistency"],
                "actual_failures": ["budget_consistency"],
            },
            {
                "expected_failures": ["budget_consistency"],
                "actual_failures": [],
            },
            {
                "expected_failures": [],
                "actual_failures": ["budget_consistency"],
            },
            {
                "expected_failures": [],
                "actual_failures": [],
            },
        ]

        metrics = precision_recall_f1(results, "budget_consistency")

        self.assertEqual(metrics["tp"], 1)
        self.assertEqual(metrics["fp"], 1)
        self.assertEqual(metrics["fn"], 1)
        self.assertEqual(metrics["precision"], 0.5)
        self.assertEqual(metrics["recall"], 0.5)
        self.assertEqual(metrics["f1"], 0.5)

    def test_run_case_detects_hallucinated_attraction(self):
        result = run_case(
            {
                "case_id": "hallucinated_attraction_test",
                "mutation": "hallucinated_attraction",
                "expected_failures": ["retrieval_grounding_attractions"],
                "expected_next_action": "retrieve_attractions",
            }
        )

        self.assertIn("retrieval_grounding_attractions", result["actual_failures"])
        self.assertEqual(result["actual_next_action"], "retrieve_attractions")
        self.assertTrue(result["detected_expected_failure"])

    def test_invalid_date_is_request_validation_failure(self):
        result = run_case(
            {
                "case_id": "invalid_date",
                "mutation": "request_validation",
                "request_overrides": {
                    "start_date": "2026-07-03",
                    "end_date": "2026-07-01",
                    "travel_days": 3,
                },
                "expected_failures": ["request_validation"],
                "expected_next_action": "reject_request",
            }
        )

        self.assertEqual(result["actual_failures"], ["request_validation"])
        self.assertEqual(result["actual_next_action"], "reject_request")

    def test_summary_reports_detection_escape_and_routing(self):
        results = [
            {
                "expected_failures": ["schema_correctness"],
                "actual_failures": ["schema_correctness"],
                "expected_next_action": "plan_itinerary",
                "actual_next_action": "plan_itinerary",
                "detected_expected_failure": True,
            },
            {
                "expected_failures": ["budget_consistency"],
                "actual_failures": [],
                "expected_next_action": "plan_itinerary",
                "actual_next_action": "finalize_response",
                "detected_expected_failure": False,
            },
        ]

        summary = summarize(results)

        self.assertEqual(summary["failure_detection_rate"], 0.5)
        self.assertEqual(summary["unsafe_plan_escape_rate"], 0.5)
        self.assertEqual(summary["routing_action_accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()
