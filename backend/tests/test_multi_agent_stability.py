import threading
import time
import unittest

from app.agents.experience_agent import ExperienceAgent
from app.models.schemas import TripRequest
from scripts.benchmark_multi_agent_stability import (
    compare_with_reference,
    execute_order,
    prewarm_parallel_rag,
    summarize,
)


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

    def test_parallel_execution_uses_isolated_workers_and_returns_schedule_order(self):
        barrier = threading.Barrier(2)
        created = []
        completion_order = []
        callback_threads = []

        def worker_factory():
            token = len(created) + 1
            created.append(token)
            return {"token": token}, {"token": token}

        def runner(planner, llm, case, case_index, repeat):
            self.assertEqual(planner["token"], llm["token"])
            barrier.wait(timeout=1)
            if case_index == 1:
                time.sleep(0.03)
            return {"case_index": case_index, "repeat": repeat, "status": "completed",
                    "passed": True, "worker_token": planner["token"]}

        def on_result(rows, row):
            completion_order.append(row["case_index"])
            callback_threads.append(threading.current_thread().name)

        rows = execute_order([{}, {}], [(1, 1), (2, 1)], max_workers=2,
                             worker_factory=worker_factory, runner=runner, on_result=on_result)
        self.assertEqual([row["case_index"] for row in rows], [1, 2])
        self.assertEqual(len({row["worker_token"] for row in rows}), 2)
        self.assertEqual(completion_order, [2, 1])
        self.assertEqual(callback_threads, [threading.current_thread().name] * 2)

    def test_sequential_execution_reuses_one_worker_and_records_exceptions(self):
        created = []

        def worker_factory():
            created.append(len(created) + 1)
            return object(), object()

        def runner(planner, llm, case, case_index, repeat):
            if case_index == 2:
                raise RuntimeError("expected failure")
            return {"case_index": case_index, "repeat": repeat, "status": "completed", "passed": True}

        rows = execute_order([{}, {}], [(1, 1), (2, 1)], max_workers=1,
                             worker_factory=worker_factory, runner=runner)
        self.assertEqual(len(created), 1)
        self.assertEqual(rows[0]["status"], "completed")
        self.assertEqual(rows[1]["status"], "runtime_error")
        self.assertEqual(rows[1]["error_type"], "RuntimeError")

    def test_parallel_execution_rejects_non_positive_worker_count(self):
        with self.assertRaises(ValueError):
            execute_order([], [], max_workers=0, worker_factory=lambda: (object(), object()))

    def test_parallel_chroma_is_prewarmed_once_before_workers(self):
        class FakeRAGService:
            def __init__(self):
                self.calls = 0

            def _get_vectorstore(self):
                self.calls += 1

        service = FakeRAGService()
        prewarm_parallel_rag(service, "chroma_retrieval", 3)
        self.assertEqual(service.calls, 1)
        prewarm_parallel_rag(service, "chroma_retrieval", 1)
        prewarm_parallel_rag(service, "local_lightweight", 3)
        self.assertEqual(service.calls, 1)


if __name__ == "__main__":
    unittest.main()
