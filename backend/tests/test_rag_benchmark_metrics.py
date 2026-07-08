"""Tests for RAG benchmark recall and source capture helpers."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app.models.langgraph_state import RAGChunk
from app.models.langgraph_state import EvaluationReport, EvaluationScores, RetryState, RunMetrics
from app.models.schemas import TripRequest
from app.services.rag_service import TravelRAGService
from scripts import benchmark_trip_planners
from scripts.benchmark_rag_ranking_ablation import normalize_claim_text
from scripts.benchmark_trip_planners import (
    aggregate_results,
    compact_rag_sources,
    forbidden_doc_hit_count_for_entry,
    forbidden_doc_hits_for_entry,
    load_benchmark_cases,
    plan_with_langgraph,
    recall_for_entry,
    retrieval_stage_latency_ms,
)


class RAGBenchmarkMetricTests(unittest.TestCase):
    def test_recall_for_entry_handles_full_partial_none_and_unlabeled(self):
        full = {
            "expected_rag_doc_ids": ["doc-a", "doc-b"],
            "retrieved_rag_sources": [{"doc_id": "doc-a"}, {"doc_id": "doc-b"}],
        }
        partial = {
            "expected_rag_doc_ids": ["doc-a", "doc-b"],
            "retrieved_rag_sources": [{"doc_id": "doc-a"}, {"doc_id": "doc-c"}],
        }
        miss = {
            "expected_rag_doc_ids": ["doc-a"],
            "retrieved_rag_sources": [{"doc_id": "doc-c"}],
        }
        unlabeled = {
            "expected_rag_doc_ids": [],
            "retrieved_rag_sources": [{"doc_id": "doc-a"}],
        }

        self.assertEqual(recall_for_entry(full), 1.0)
        self.assertEqual(recall_for_entry(partial), 0.5)
        self.assertEqual(recall_for_entry(miss), 0.0)
        self.assertIsNone(recall_for_entry(unlabeled))

    def test_forbidden_doc_hits_only_checks_top_four_sources(self):
        entry = {
            "forbidden_rag_doc_ids": ["doc-bad", "doc-too-deep"],
            "retrieved_rag_sources": [
                {"doc_id": "doc-good-1"},
                {"doc_id": "doc-bad"},
                {"doc_id": "doc-good-2"},
                {"doc_id": "doc-good-3"},
                {"doc_id": "doc-too-deep"},
            ],
        }

        self.assertEqual(forbidden_doc_hits_for_entry(entry), ["doc-bad"])
        self.assertEqual(forbidden_doc_hit_count_for_entry(entry), 1)
        self.assertIsNone(forbidden_doc_hits_for_entry({"retrieved_rag_sources": []}))
        self.assertIsNone(forbidden_doc_hit_count_for_entry({"retrieved_rag_sources": []}))

    def test_forbidden_doc_hit_count_counts_repeated_top_four_slots(self):
        entry = {
            "forbidden_rag_doc_ids": ["doc-bad"],
            "retrieved_rag_sources": [
                {"doc_id": "doc-bad"},
                {"doc_id": "doc-good"},
                {"doc_id": "doc-bad"},
            ],
        }

        self.assertEqual(forbidden_doc_hits_for_entry(entry), ["doc-bad"])
        self.assertEqual(forbidden_doc_hit_count_for_entry(entry), 2)

    def test_aggregate_results_reports_recall_only_for_labeled_requests(self):
        entries = [
            {
                "latency_ms": 10.0,
                "parallel_retrieval_enabled": True,
                "retrieval_stage_latency_ms": 4.0,
                "first_evaluation_pass": True,
                "recovered_after_retry": False,
                "fallback": False,
                "report": {
                    "passed": True,
                    "hard_failures": [],
                    "scores": {"grounding_score": 1.0},
                    "quality_warnings": ["pacing_day_0_overloaded"],
                },
                "expected_rag_doc_ids": ["doc-a", "doc-b"],
                "retrieved_rag_sources": [{"doc_id": "doc-a"}, {"doc_id": "doc-b"}],
            },
            {
                "latency_ms": 20.0,
                "parallel_retrieval_enabled": True,
                "retrieval_stage_latency_ms": 8.0,
                "report": {
                    "passed": False,
                    "hard_failures": [],
                    "scores": {"grounding_score": 0.5},
                    "quality_warnings": ["route_day_1_long_jump_25.0km"],
                },
                "first_evaluation_pass": False,
                "recovered_after_retry": False,
                "fallback": False,
                "expected_rag_doc_ids": ["doc-a", "doc-b"],
                "retrieved_rag_sources": [{"doc_id": "doc-a"}],
            },
            {
                "latency_ms": 30.0,
                "parallel_retrieval_enabled": True,
                "retrieval_stage_latency_ms": 12.0,
                "report": {
                    "passed": True,
                    "hard_failures": [],
                    "scores": {"grounding_score": 0.5},
                    "quality_warnings": ["preference_terms_missing:夜景"],
                },
                "first_evaluation_pass": True,
                "recovered_after_retry": False,
                "fallback": False,
                "expected_rag_doc_ids": [],
                "retrieved_rag_sources": [{"doc_id": "doc-z"}],
            },
        ]

        summary = aggregate_results(entries)

        self.assertEqual(summary["recall_labeled_request_count"], 2)
        self.assertEqual(summary["forbidden_retrieval_count"], 0)
        self.assertIsNone(summary["negative_precision"])
        self.assertIsNone(summary["forbidden_doc_hit_rate"])
        self.assertTrue(summary["parallel_retrieval_enabled"])
        self.assertEqual(summary["avg_retrieval_stage_latency_ms"], 8.0)
        self.assertEqual(summary["retrieval_hit_rate"], 1.0)
        self.assertEqual(summary["retrieval_recall_at_4"], 0.75)
        self.assertIn("hard_validation_pass_rate", summary)
        self.assertIn("avg_pacing_score", summary)
        self.assertIn("avg_route_coherence_score", summary)
        self.assertIn("avg_preference_match_score", summary)
        self.assertIn("avg_attribution_coverage_score", summary)
        self.assertIn("content_completeness_failure_rate", summary)
        self.assertIn("retrieved_unique_doc_count_avg", summary)
        self.assertIn("duplicate_doc_rate", summary)
        self.assertIn("avg_rerank_score", summary)
        self.assertEqual(summary["quality_warning_rate"], 1.0)
        self.assertEqual(summary["pacing_warning_rate"], 0.3333)
        self.assertEqual(summary["route_warning_rate"], 0.3333)
        self.assertEqual(summary["preference_warning_rate"], 0.3333)

    def test_aggregate_results_reports_negative_retrieval_metrics(self):
        entries = [
            {
                "latency_ms": 10.0,
                "parallel_retrieval_enabled": True,
                "retrieval_stage_latency_ms": 4.0,
                "first_evaluation_pass": True,
                "recovered_after_retry": False,
                "fallback": False,
                "report": {"passed": True, "hard_failures": [], "scores": {}, "quality_warnings": []},
                "expected_rag_doc_ids": ["doc-good"],
                "forbidden_rag_doc_ids": ["doc-bad"],
                "retrieved_rag_sources": [
                    {"doc_id": "doc-good"},
                    {"doc_id": "doc-bad"},
                ],
            },
            {
                "latency_ms": 12.0,
                "parallel_retrieval_enabled": True,
                "retrieval_stage_latency_ms": 5.0,
                "first_evaluation_pass": True,
                "recovered_after_retry": False,
                "fallback": False,
                "report": {"passed": True, "hard_failures": [], "scores": {}, "quality_warnings": []},
                "expected_rag_doc_ids": ["doc-other"],
                "forbidden_rag_doc_ids": ["doc-forbidden"],
                "retrieved_rag_sources": [
                    {"doc_id": "doc-other"},
                    {"doc_id": "doc-safe"},
                ],
            },
            {
                "latency_ms": 15.0,
                "parallel_retrieval_enabled": True,
                "retrieval_stage_latency_ms": 6.0,
                "first_evaluation_pass": True,
                "recovered_after_retry": False,
                "fallback": False,
                "report": {"passed": True, "hard_failures": [], "scores": {}, "quality_warnings": []},
                "expected_rag_doc_ids": [],
                "retrieved_rag_sources": [{"doc_id": "unlabeled"}],
            },
        ]

        summary = aggregate_results(entries)

        self.assertEqual(summary["forbidden_retrieval_count"], 1)
        self.assertEqual(summary["negative_precision"], 0.75)
        self.assertEqual(summary["forbidden_doc_hit_rate"], 0.5)

    def test_loader_strips_benchmark_metadata_before_trip_request(self):
        payload = [
            {
                "city": "北京",
                "start_date": "2026-06-01",
                "end_date": "2026-06-03",
                "travel_days": 3,
                "transportation": "公共交通",
                "accommodation": "经济型酒店",
                "preferences": ["历史文化"],
                "free_text_input": "希望以故宫为主",
                "expected_rag_doc_ids": ["beijing-history-core-001"],
                "expected_rag_themes": ["历史文化"],
                "forbidden_rag_doc_ids": ["shanghai-food-night-002"],
                "benchmark_note": "metadata should not be passed into TripRequest",
            }
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = Path(tmpdir) / "dataset.json"
            dataset.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            cases = load_benchmark_cases(dataset)

        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].request.city, "北京")
        self.assertEqual(cases[0].expected_rag_doc_ids, ["beijing-history-core-001"])
        self.assertEqual(cases[0].expected_rag_sections, [])
        self.assertEqual(cases[0].expected_rag_claims, [])
        self.assertEqual(cases[0].forbidden_rag_doc_ids, ["shanghai-food-night-002"])
        self.assertFalse(hasattr(cases[0].request, "expected_rag_doc_ids"))
        self.assertFalse(hasattr(cases[0].request, "forbidden_rag_doc_ids"))

    def test_loader_defaults_missing_forbidden_doc_ids_for_backward_compatibility(self):
        payload = [
            {
                "city": "北京",
                "start_date": "2026-06-01",
                "end_date": "2026-06-03",
                "travel_days": 3,
                "transportation": "公共交通",
                "accommodation": "经济型酒店",
                "preferences": ["历史文化"],
                "free_text_input": "希望以故宫为主",
                "expected_rag_doc_ids": ["beijing-history-core-001"],
            }
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = Path(tmpdir) / "dataset.json"
            dataset.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            cases = load_benchmark_cases(dataset)

        self.assertEqual(cases[0].forbidden_rag_doc_ids, [])

    def test_loader_parses_section_and_claim_labels(self):
        payload = [
            {
                "city": "Chicago",
                "start_date": "2026-06-01",
                "end_date": "2026-06-02",
                "travel_days": 2,
                "transportation": "public transit",
                "accommodation": "mid-range hotel",
                "preferences": ["museums"],
                "free_text_input": "Museum Campus by water taxi",
                "expected_rag_doc_ids": ["chicago-chicago-001"],
                "expected_rag_sections": [
                    {"doc_id": "chicago-chicago-001", "section": "transport"}
                ],
                "expected_rag_claims": [
                    {
                        "claim_id": "water-taxi",
                        "doc_id": "chicago-chicago-001",
                        "section": "transport",
                        "category": "transport",
                        "evidence_quote": "You can get to the Chicago Museum Campus by Shoreline Sightseeing Water Taxi.",
                    }
                ],
            }
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = Path(tmpdir) / "dataset.json"
            dataset.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            cases = load_benchmark_cases(dataset)

        self.assertEqual(cases[0].expected_rag_sections[0]["section"], "transport")
        self.assertEqual(cases[0].expected_rag_claims[0]["claim_id"], "water-taxi")
        self.assertFalse(hasattr(cases[0].request, "expected_rag_claims"))

    def test_us_rag_benchmark_labels_reference_existing_corpus_docs(self):
        repo_root = Path(__file__).resolve().parents[1]
        dataset_path = repo_root / "benchmarks" / "trip_requests.us_rag_benchmark.json"
        corpus_dir = repo_root / "data" / "knowledge" / "us"
        corpus_doc_ids = set()
        for corpus_file in corpus_dir.glob("*.json"):
            for document in json.loads(corpus_file.read_text(encoding="utf-8")):
                corpus_doc_ids.add(document["doc_id"])

        cases = load_benchmark_cases(dataset_path)
        expected_doc_ids = {
            doc_id for case in cases for doc_id in case.expected_rag_doc_ids
        }

        self.assertEqual(len(cases), 15)
        self.assertTrue(expected_doc_ids)
        self.assertTrue(expected_doc_ids.issubset(corpus_doc_ids))

    def test_us_hard_rag_benchmark_labels_reference_existing_sections_and_quotes(self):
        repo_root = Path(__file__).resolve().parents[1]
        dataset_path = repo_root / "benchmarks" / "trip_requests.us_rag_hard_benchmark.json"
        service = TravelRAGService(knowledge_root=repo_root / "data" / "knowledge")
        section_texts = {
            (doc.doc_id, section): text
            for doc in service.load_knowledge_docs()
            if doc.country == "US"
            for section, text in service._iter_document_sections(doc)
        }
        valid_sections = {"overview", "planning_tips", "transport", "seasonality"}

        cases = load_benchmark_cases(dataset_path)
        expected_doc_ids = {
            doc_id for case in cases for doc_id in case.expected_rag_doc_ids
        }

        self.assertEqual(len(cases), 9)
        self.assertTrue(expected_doc_ids)
        self.assertTrue(all(case.expected_rag_sections for case in cases))
        self.assertTrue(all(case.expected_rag_claims for case in cases))

        for case in cases:
            for label in case.expected_rag_sections:
                key = (label["doc_id"], label["section"])
                self.assertIn(label["section"], valid_sections)
                self.assertIn(key, section_texts)
            for claim in case.expected_rag_claims:
                key = (claim["doc_id"], claim["section"])
                self.assertIn(key, section_texts)
                self.assertIn(
                    normalize_claim_text(claim["evidence_quote"]),
                    normalize_claim_text(section_texts[key]),
                )

    def test_compact_rag_sources_serializes_chunk_metadata(self):
        state = {
            "rag_chunks": [
                RAGChunk(
                    chunk_id="doc-a-overview",
                    source="official_tourism_portal",
                    title="Doc A",
                    content="content",
                    metadata={
                        "doc_id": "doc-a",
                        "source_url": "https://example.com/a",
                        "section": "overview",
                        "city": "北京",
                        "theme": "历史文化",
                        "rag_backend": "chroma_retrieval",
                        "vector_rank": 1,
                        "rerank_score": 1.25,
                        "rerank_reasons": ["vector_rank:1", "theme_overlap:history"],
                        "dedup_rank": 1,
                    },
                )
            ]
        }

        sources = compact_rag_sources(state)

        self.assertEqual(sources[0]["doc_id"], "doc-a")
        self.assertEqual(sources[0]["source_url"], "https://example.com/a")
        self.assertEqual(sources[0]["rag_backend"], "chroma_retrieval")
        self.assertEqual(sources[0]["vector_rank"], 1)
        self.assertEqual(sources[0]["rerank_score"], 1.25)
        self.assertEqual(sources[0]["dedup_rank"], 1)

    def test_retrieval_stage_latency_uses_parallel_critical_path(self):
        latency = retrieval_stage_latency_ms(
            {
                "retrieve_attractions": 120.0,
                "retrieve_rag_context": 80.0,
                "retrieve_hotels": 350.0,
                "retrieve_weather": 40.0,
            }
        )

        self.assertEqual(latency, 350.0)

    def test_plan_with_langgraph_persists_observability_when_enabled(self):
        request = TripRequest(
            city="北京",
            start_date="2026-06-01",
            end_date="2026-06-02",
            travel_days=2,
            transportation="公共交通",
            accommodation="经济型酒店",
            preferences=["历史文化"],
            free_text_input="希望看故宫",
        )
        report = EvaluationReport(
            passed=True,
            hard_failures=[],
            scores=EvaluationScores(schema_score=1.0, grounding_score=1.0),
            next_action="finalize_response",
        )
        metrics = RunMetrics(
            started_at=1.0,
            ended_at=2.0,
            end_to_end_ms=1000.0,
            node_latency_ms={"evaluate_itinerary": 2.0},
            node_attempts={"evaluate_itinerary": 1},
            evaluation_attempt_count=1,
            first_evaluation_pass=True,
            final_evaluation_pass=True,
        )
        state = {
            "request": request,
            "conversation_id": "bench-test",
            "evaluation_report": report,
            "metrics": metrics,
            "retry_counts": RetryState(evaluate_itinerary=1),
            "decision_trace": ["evaluate_itinerary: next_action=finalize_response passed=True"],
            "rag_chunks": [],
            "final_plan": None,
        }

        class FakePlanner:
            rag_mode = "chroma_retrieval"
            parallel_retrieval_enabled = True

            def invoke_graph(self, request, thread_id=None):
                return state

        class FakeObservabilityService:
            def __init__(self):
                self.calls = []

            def persist_state(self, persisted_state, **kwargs):
                self.calls.append((persisted_state, kwargs))
                return "obs-run-1"

        fake_service = FakeObservabilityService()
        original_getter = benchmark_trip_planners.get_observability_service
        benchmark_trip_planners.get_observability_service = lambda: fake_service
        try:
            result = plan_with_langgraph(
                FakePlanner(),
                request,
                benchmark_metadata={"expected_rag_doc_ids": ["doc-a"]},
                persist_observability=True,
            )
        finally:
            benchmark_trip_planners.get_observability_service = original_getter

        self.assertEqual(result["observability_run_id"], "obs-run-1")
        self.assertTrue(result["parallel_retrieval_enabled"])
        self.assertEqual(result["retrieval_stage_latency_ms"], 0.0)
        self.assertEqual(fake_service.calls[0][1]["source"], "benchmark")
        self.assertEqual(fake_service.calls[0][1]["rag_mode"], "chroma_retrieval")

    def test_plan_with_langgraph_records_runtime_error_without_crashing(self):
        request = TripRequest(
            city="杭州",
            start_date="2026-06-01",
            end_date="2026-06-02",
            travel_days=2,
            transportation="公共交通",
            accommodation="经济型酒店",
            preferences=["自然风光"],
            free_text_input="希望看西湖",
        )

        class FailingPlanner:
            rag_mode = "chroma_retrieval"
            parallel_retrieval_enabled = True

            def invoke_graph(self, request, thread_id=None):
                raise TimeoutError("external service timed out")

        result = plan_with_langgraph(
            FailingPlanner(),
            request,
            benchmark_metadata={"expected_rag_doc_ids": ["hangzhou-westlake-001"]},
            persist_observability=True,
        )

        self.assertFalse(result["report"]["passed"])
        self.assertEqual(result["report"]["hard_failures"], ["benchmark_runtime_error"])
        self.assertTrue(result["fallback"])
        self.assertEqual(result["error"]["type"], "TimeoutError")


if __name__ == "__main__":
    unittest.main()
