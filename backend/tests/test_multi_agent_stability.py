import unittest

from app.agents.experience_agent import ExperienceAgent
from app.models.schemas import TripRequest
from scripts.benchmark_multi_agent_stability import compare_with_reference, summarize


def benchmark_row(*, case_index=1, repeat=1, passed=True, attraction=None, hotel=None, meal=None):
    return {
        "case_index": case_index,
        "repeat": repeat,
        "passed": passed,
        "fallback": not passed,
        "agent_error": None,
        "unsupported_entities": 0,
        "materialization_failures": [],
        "retrieval_recall": 1.0,
        "forbidden_retrieval_ids": [],
        "budget_usage": {},
        "token_usage": {"total_tokens": 0},
        "latency_ms": 0,
        "layers": {},
        "final_ids": {
            "attraction": attraction or ["a1"],
            "hotel": hotel or ["h1"],
            "meal": meal or ["m1"],
            "day_assignment": ["0:a1"],
            "route_order": [["a1"]],
        },
    }


class MultiAgentStabilityTests(unittest.TestCase):
    def test_attraction_queries_are_city_scoped_normalized_and_deterministic(self):
        request = TripRequest(
            city="New York",
            start_date="2026-06-01",
            end_date="2026-06-02",
            travel_days=2,
            transportation="Public transit",
            accommodation="Mid-range hotel",
            preferences=["Museums", "culture", "Museums"],
        )
        first = ExperienceAgent._attraction_queries(request, ["art galleries", "historic museums"])
        second = ExperienceAgent._attraction_queries(request, ["historic museums", "art galleries"])
        expected = [
            ("new york culture museums attractions", "base_anchor"),
            ("new york culture attractions", "supplemental"),
            ("new york museums attractions", "supplemental"),
        ]
        self.assertEqual(first, expected)
        self.assertEqual(second, expected)

    def test_single_repeat_summary_uses_saved_validated_reference_rows(self):
        current = [benchmark_row(attraction=["a1", "a2"])]
        reference = [
            benchmark_row(repeat=1, attraction=["a1", "a2"]),
            benchmark_row(repeat=2, passed=False, attraction=[]),
        ]
        self.assertEqual(summarize(current)["validated_pair_count"], 0)
        overlap = compare_with_reference(current, reference)
        self.assertEqual(overlap["validated_pair_count"], 1)
        self.assertEqual(overlap["overall_jaccard"], 1.0)
        self.assertEqual(overlap["attraction_jaccard"], 1.0)


if __name__ == "__main__":
    unittest.main()
