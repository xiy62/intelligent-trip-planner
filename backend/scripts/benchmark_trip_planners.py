"""Benchmark local-lightweight RAG vs. Chroma-backed RAG on the same request set."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agents.langgraph_trip_planner import LangGraphTripPlanner
from app.models.schemas import TripPlan, TripRequest
from app.services.observability_service import get_observability_service
from app.services.rag_service import get_rag_service


REQUEST_FIELDS = {
    "city",
    "start_date",
    "end_date",
    "travel_days",
    "transportation",
    "accommodation",
    "preferences",
    "free_text_input",
}


@dataclass
class BenchmarkCase:
    request: TripRequest
    expected_rag_doc_ids: List[str] = field(default_factory=list)
    expected_rag_themes: List[str] = field(default_factory=list)
    benchmark_note: str = ""

    def metadata_dump(self) -> Dict[str, Any]:
        return {
            "expected_rag_doc_ids": self.expected_rag_doc_ids,
            "expected_rag_themes": self.expected_rag_themes,
            "benchmark_note": self.benchmark_note,
        }


def load_benchmark_cases(dataset_path: Path) -> List[BenchmarkCase]:
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    cases: List[BenchmarkCase] = []
    for item in data:
        request_payload = {key: item[key] for key in REQUEST_FIELDS if key in item}
        cases.append(
            BenchmarkCase(
                request=TripRequest(**request_payload),
                expected_rag_doc_ids=list(item.get("expected_rag_doc_ids", [])),
                expected_rag_themes=list(item.get("expected_rag_themes", [])),
                benchmark_note=str(item.get("benchmark_note", "")),
            )
        )
    return cases


def load_requests(dataset_path: Path) -> List[TripRequest]:
    """Backward-compatible helper for tests or ad-hoc scripts."""
    return [case.request for case in load_benchmark_cases(dataset_path)]


def percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def compact_rag_sources(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    sources: List[Dict[str, Any]] = []
    for chunk in state.get("rag_chunks", []):
        metadata = dict(chunk.metadata)
        sources.append(
            {
                "chunk_id": chunk.chunk_id,
                "doc_id": metadata.get("doc_id", ""),
                "title": chunk.title,
                "source_url": metadata.get("source_url", ""),
                "section": metadata.get("section", ""),
                "city": metadata.get("city", ""),
                "theme": metadata.get("theme", ""),
                "rag_backend": metadata.get("rag_backend", ""),
                "vector_rank": metadata.get("vector_rank"),
                "rerank_score": metadata.get("rerank_score"),
                "rerank_reasons": metadata.get("rerank_reasons", []),
                "dedup_rank": metadata.get("dedup_rank"),
            }
        )
    return sources


def recall_for_entry(entry: Dict[str, Any]) -> Optional[float]:
    expected = set(entry.get("expected_rag_doc_ids", []))
    if not expected:
        return None
    retrieved = {
        source.get("doc_id")
        for source in entry.get("retrieved_rag_sources", [])[:4]
        if source.get("doc_id")
    }
    return len(expected & retrieved) / len(expected)


def quality_warning_category(warning: str) -> str:
    """Group granular quality warnings into benchmark-friendly categories."""
    if warning.startswith("pacing_") or warning == "low_pacing_score":
        return "pacing"
    if warning.startswith("route_") or warning == "low_route_coherence_score":
        return "route_coherence"
    if warning.startswith("preference_terms_missing") or warning == "low_preference_match_score":
        return "preference_match"
    if warning == "low_attribution_coverage":
        return "attribution"
    return "other"


def retrieval_stage_latency_ms(node_latency_ms: Dict[str, float]) -> float:
    """Estimate retrieval critical-path latency for the current graph topology."""
    attraction_and_rag = (
        float(node_latency_ms.get("retrieve_attractions", 0.0))
        + float(node_latency_ms.get("retrieve_rag_context", 0.0))
    )
    return round(
        max(
            attraction_and_rag,
            float(node_latency_ms.get("retrieve_hotels", 0.0)),
            float(node_latency_ms.get("retrieve_weather", 0.0)),
        ),
        3,
    )


def plan_with_langgraph(
    planner: LangGraphTripPlanner,
    request: TripRequest,
    *,
    benchmark_metadata: Optional[Dict[str, Any]] = None,
    persist_observability: bool = False,
) -> Dict[str, Any]:
    thread_id = f"bench-{int(time.time() * 1000)}-{request.city}-{request.travel_days}"
    benchmark_metadata = benchmark_metadata or {}
    started_at = time.perf_counter()
    try:
        state = planner.invoke_graph(request, thread_id=thread_id)
    except Exception as exc:
        return {
            "planner": planner.rag_mode,
            "parallel_retrieval_enabled": bool(getattr(planner, "parallel_retrieval_enabled", False)),
            "request": request.model_dump(),
            "expected_rag_doc_ids": list(benchmark_metadata.get("expected_rag_doc_ids", [])),
            "expected_rag_themes": list(benchmark_metadata.get("expected_rag_themes", [])),
            "benchmark_note": benchmark_metadata.get("benchmark_note", ""),
            "retrieved_rag_sources": [],
            "latency_ms": round((time.perf_counter() - started_at) * 1000.0, 3),
            "report": {
                "passed": False,
                "hard_failures": ["benchmark_runtime_error"],
                "warnings": [f"{exc.__class__.__name__}: {exc}"],
                "quality_warnings": [],
                "scores": {},
                "unsupported_entities": [],
                "unsupported_claims": [],
                "evidence_links": [],
                "next_action": "fallback_response",
            },
            "first_evaluation_pass": False,
            "final_evaluation_pass": False,
            "recovered_after_retry": False,
            "fallback": True,
            "retry_counts": {},
            "decision_trace": [f"benchmark_runtime_error: {exc.__class__.__name__}"],
            "node_latency_ms": {},
            "retrieval_stage_latency_ms": 0.0,
            "final_plan_summary": {"days": 0, "city": request.city, "attractions": 0, "hotels": 0},
            "error": {
                "type": exc.__class__.__name__,
                "message": str(exc),
            },
        }
    final_plan = state.get("final_plan")
    report = state.get("evaluation_report")
    metrics = state.get("metrics")
    result = {
        "planner": planner.rag_mode,
        "parallel_retrieval_enabled": bool(getattr(planner, "parallel_retrieval_enabled", False)),
        "request": request.model_dump(),
        "expected_rag_doc_ids": list(benchmark_metadata.get("expected_rag_doc_ids", [])),
        "expected_rag_themes": list(benchmark_metadata.get("expected_rag_themes", [])),
        "benchmark_note": benchmark_metadata.get("benchmark_note", ""),
        "retrieved_rag_sources": compact_rag_sources(state),
        "latency_ms": round((metrics.end_to_end_ms if metrics else 0.0), 3),
        "report": report.model_dump() if report is not None else {},
        "first_evaluation_pass": metrics.first_evaluation_pass if metrics is not None else None,
        "final_evaluation_pass": metrics.final_evaluation_pass if metrics is not None else None,
        "recovered_after_retry": metrics.recovered_after_retry if metrics is not None else False,
        "fallback": bool(metrics and metrics.fallback_count > 0),
        "retry_counts": state.get("retry_counts").model_dump() if state.get("retry_counts") else {},
        "decision_trace": list(state.get("decision_trace", [])),
        "node_latency_ms": metrics.node_latency_ms if metrics is not None else {},
        "retrieval_stage_latency_ms": retrieval_stage_latency_ms(metrics.node_latency_ms)
        if metrics is not None
        else 0.0,
        "final_plan_summary": summarize_trip_plan(final_plan),
    }
    if persist_observability:
        result["observability_run_id"] = get_observability_service().persist_state(
            state,
            source="benchmark",
            rag_mode=planner.rag_mode,
            benchmark_metadata={
                **benchmark_metadata,
                "retrieved_rag_sources": result["retrieved_rag_sources"],
            },
        )
    return result


def summarize_trip_plan(plan: TripPlan | None) -> Dict[str, Any]:
    if plan is None:
        return {"days": 0, "city": "", "attractions": 0}
    return {
        "city": plan.city,
        "days": len(plan.days),
        "attractions": sum(len(day.attractions) for day in plan.days),
        "hotels": sum(1 for day in plan.days if day.hotel is not None),
    }


def is_likely_fallback_plan(plan: TripPlan, request: TripRequest) -> bool:
    if not plan.days:
        return True
    if plan.overall_suggestions.startswith(f"这是为您规划的{request.city}{request.travel_days}日游行程"):
        return True
    first_day = plan.days[0]
    if first_day.attractions and first_day.attractions[0].name == f"{request.city}景点1":
        return True
    return False


def aggregate_results(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    latencies = [entry["latency_ms"] for entry in entries]
    retrieval_stage_latencies = [entry.get("retrieval_stage_latency_ms", 0.0) for entry in entries]
    reports = [entry.get("report", {}) for entry in entries]
    hard_failures = [failure for report in reports for failure in report.get("hard_failures", [])]
    grounded_scores = [report.get("scores", {}).get("grounding_score", 0.0) for report in reports]
    pacing_scores = [report.get("scores", {}).get("pacing_score", 0.0) for report in reports]
    route_scores = [report.get("scores", {}).get("route_coherence_score", 0.0) for report in reports]
    preference_scores = [report.get("scores", {}).get("preference_match_score", 0.0) for report in reports]
    attribution_scores = [
        report.get("scores", {}).get("attribution_coverage_score", 0.0) for report in reports
    ]
    quality_warning_runs = [
        report for report in reports if report.get("quality_warnings", [])
    ]
    quality_warning_counts: Dict[str, int] = {}
    quality_warning_category_counts: Dict[str, int] = {}
    quality_warning_category_runs: Dict[str, int] = {
        "pacing": 0,
        "route_coherence": 0,
        "preference_match": 0,
        "attribution": 0,
        "other": 0,
    }
    for report in reports:
        categories_for_run = set()
        for warning in report.get("quality_warnings", []):
            quality_warning_counts[warning] = quality_warning_counts.get(warning, 0) + 1
            category = quality_warning_category(str(warning))
            quality_warning_category_counts[category] = (
                quality_warning_category_counts.get(category, 0) + 1
            )
            categories_for_run.add(category)
        for category in categories_for_run:
            quality_warning_category_runs[category] = (
                quality_warning_category_runs.get(category, 0) + 1
            )
    initially_failed_runs = [entry for entry in entries if entry.get("first_evaluation_pass") is False]
    recovered_runs = [entry for entry in entries if entry.get("recovered_after_retry")]
    recall_values = [value for entry in entries if (value := recall_for_entry(entry)) is not None]
    hit_values = [1.0 if value > 0 else 0.0 for value in recall_values]
    retrieved_doc_id_lists = [
        [
            source.get("doc_id")
            for source in entry.get("retrieved_rag_sources", [])
            if source.get("doc_id")
        ]
        for entry in entries
    ]
    unique_doc_counts = [len(set(doc_ids)) for doc_ids in retrieved_doc_id_lists if doc_ids]
    duplicate_doc_rates = [
        (len(doc_ids) - len(set(doc_ids))) / len(doc_ids)
        for doc_ids in retrieved_doc_id_lists
        if doc_ids
    ]
    rerank_scores = [
        float(source["rerank_score"])
        for entry in entries
        for source in entry.get("retrieved_rag_sources", [])
        if source.get("rerank_score") is not None
    ]
    return {
        "request_count": len(entries),
        "parallel_retrieval_enabled": bool(entries and entries[0].get("parallel_retrieval_enabled")),
        "avg_latency_ms": round(statistics.fmean(latencies), 3) if latencies else 0.0,
        "avg_retrieval_stage_latency_ms": round(statistics.fmean(retrieval_stage_latencies), 3)
        if retrieval_stage_latencies
        else 0.0,
        "p50_latency_ms": round(percentile(latencies, 0.5), 3),
        "p95_latency_ms": round(percentile(latencies, 0.95), 3),
        "initial_failure_rate": round(len(initially_failed_runs) / len(entries), 4) if entries else 0.0,
        "recovery_rate": round(len(recovered_runs) / len(initially_failed_runs), 4)
        if initially_failed_runs
        else 0.0,
        "recovered_runs": len(recovered_runs),
        "initially_failed_runs": len(initially_failed_runs),
        "valid_itinerary_pass_rate": round(
            sum(1 for report in reports if report.get("passed")) / len(reports), 4
        )
        if reports
        else 0.0,
        "hard_validation_pass_rate": round(
            sum(1 for report in reports if report.get("passed")) / len(reports), 4
        )
        if reports
        else 0.0,
        "fallback_rate": round(sum(1 for entry in entries if entry.get("fallback")) / len(entries), 4)
        if entries
        else 0.0,
        "schema_failure_rate": round(hard_failures.count("schema_correctness") / len(entries), 4)
        if entries
        else 0.0,
        "date_coverage_failure_rate": round(hard_failures.count("date_coverage") / len(entries), 4)
        if entries
        else 0.0,
        "budget_consistency_failure_rate": round(hard_failures.count("budget_consistency") / len(entries), 4)
        if entries
        else 0.0,
        "content_completeness_failure_rate": round(
            hard_failures.count("content_completeness_attractions") / len(entries), 4
        )
        if entries
        else 0.0,
        "grounding_failure_rate": round(
            sum(1 for report in reports if any(item.startswith("retrieval_grounding") for item in report.get("hard_failures", [])))
            / len(entries),
            4,
        )
        if entries
        else 0.0,
        "avg_grounding_score": round(statistics.fmean(grounded_scores), 4) if grounded_scores else 0.0,
        "quality_warning_rate": round(len(quality_warning_runs) / len(reports), 4)
        if reports
        else 0.0,
        "quality_warning_counts": quality_warning_counts,
        "quality_warning_category_counts": quality_warning_category_counts,
        "pacing_warning_rate": round(
            quality_warning_category_runs.get("pacing", 0) / len(reports), 4
        )
        if reports
        else 0.0,
        "route_warning_rate": round(
            quality_warning_category_runs.get("route_coherence", 0) / len(reports), 4
        )
        if reports
        else 0.0,
        "preference_warning_rate": round(
            quality_warning_category_runs.get("preference_match", 0) / len(reports), 4
        )
        if reports
        else 0.0,
        "attribution_warning_rate": round(
            quality_warning_category_runs.get("attribution", 0) / len(reports), 4
        )
        if reports
        else 0.0,
        "avg_pacing_score": round(statistics.fmean(pacing_scores), 4) if pacing_scores else 0.0,
        "avg_route_coherence_score": round(statistics.fmean(route_scores), 4) if route_scores else 0.0,
        "avg_preference_match_score": round(statistics.fmean(preference_scores), 4)
        if preference_scores
        else 0.0,
        "avg_attribution_coverage_score": round(statistics.fmean(attribution_scores), 4)
        if attribution_scores
        else 0.0,
        "attribution_coverage_rate": round(
            sum(1 for value in attribution_scores if value >= 0.8) / len(attribution_scores), 4
        )
        if attribution_scores
        else 0.0,
        "recall_labeled_request_count": len(recall_values),
        "retrieval_hit_rate": round(statistics.fmean(hit_values), 4) if hit_values else None,
        "retrieval_recall_at_4": round(statistics.fmean(recall_values), 4) if recall_values else None,
        "retrieved_unique_doc_count_avg": round(statistics.fmean(unique_doc_counts), 4)
        if unique_doc_counts
        else 0.0,
        "duplicate_doc_rate": round(statistics.fmean(duplicate_doc_rates), 4)
        if duplicate_doc_rates
        else 0.0,
        "avg_rerank_score": round(statistics.fmean(rerank_scores), 4)
        if rerank_scores
        else None,
    }


def build_summary(baseline_entries: List[Dict[str, Any]], rag_entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    baseline_summary = aggregate_results(baseline_entries)
    rag_summary = aggregate_results(rag_entries)
    return {
        "baseline_local_rag": baseline_summary,
        "rag_chroma": rag_summary,
        "delta": {
            "pass_rate_delta": round(
                rag_summary["valid_itinerary_pass_rate"] - baseline_summary["valid_itinerary_pass_rate"], 4
            ),
            "initial_failure_rate_delta": round(
                rag_summary["initial_failure_rate"] - baseline_summary["initial_failure_rate"], 4
            ),
            "recovery_rate_delta": round(
                rag_summary["recovery_rate"] - baseline_summary["recovery_rate"], 4
            ),
            "fallback_rate_delta": round(
                rag_summary["fallback_rate"] - baseline_summary["fallback_rate"], 4
            ),
            "avg_latency_ms_delta": round(
                rag_summary["avg_latency_ms"] - baseline_summary["avg_latency_ms"], 3
            ),
            "grounding_score_delta": round(
                rag_summary["avg_grounding_score"] - baseline_summary["avg_grounding_score"], 4
            ),
            "attribution_coverage_score_delta": round(
                rag_summary["avg_attribution_coverage_score"]
                - baseline_summary["avg_attribution_coverage_score"],
                4,
            ),
            "route_coherence_score_delta": round(
                rag_summary["avg_route_coherence_score"]
                - baseline_summary["avg_route_coherence_score"],
                4,
            ),
            "pacing_score_delta": round(
                rag_summary["avg_pacing_score"] - baseline_summary["avg_pacing_score"], 4
            ),
            "preference_match_score_delta": round(
                rag_summary["avg_preference_match_score"]
                - baseline_summary["avg_preference_match_score"],
                4,
            ),
            "quality_warning_rate_delta": round(
                rag_summary["quality_warning_rate"] - baseline_summary["quality_warning_rate"],
                4,
            ),
            "preference_warning_rate_delta": round(
                rag_summary["preference_warning_rate"]
                - baseline_summary["preference_warning_rate"],
                4,
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark local-lightweight RAG vs Chroma-backed RAG.")
    parser.add_argument(
        "--dataset",
        default="benchmarks/trip_requests.rag_benchmark.json",
        help="Path to JSON request dataset relative to backend/ or absolute path.",
    )
    parser.add_argument(
        "--output",
        default="benchmarks/results/trip_planner_rag_benchmark.json",
        help="Path to JSON output file relative to backend/ or absolute path.",
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Rebuild the Chroma index before running the benchmark.",
    )
    parser.add_argument(
        "--persist-observability",
        action="store_true",
        help="Persist each benchmark run into the local SQLite observability store.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    output_path = Path(args.output)
    if not dataset_path.is_absolute():
        dataset_path = ROOT / dataset_path
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cases = load_benchmark_cases(dataset_path)
    rag_service = get_rag_service()
    rag_service.ensure_index(force_rebuild=args.rebuild_index)
    baseline_planner = LangGraphTripPlanner(rag_mode="local_lightweight", rag_service=rag_service)
    rag_planner = LangGraphTripPlanner(rag_mode="chroma_retrieval", rag_service=rag_service)

    baseline_entries: List[Dict[str, Any]] = []
    rag_entries: List[Dict[str, Any]] = []
    for case in cases:
        baseline_entries.append(
            plan_with_langgraph(
                baseline_planner,
                case.request,
                benchmark_metadata=case.metadata_dump(),
                persist_observability=args.persist_observability,
            )
        )
        rag_entries.append(
            plan_with_langgraph(
                rag_planner,
                case.request,
                benchmark_metadata=case.metadata_dump(),
                persist_observability=args.persist_observability,
            )
        )

    output = {
        "dataset": str(dataset_path),
        "summary": build_summary(baseline_entries, rag_entries),
        "results": {
            "baseline_local_rag": baseline_entries,
            "rag_chroma": rag_entries,
        },
    }
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2))
    print(f"\nSaved benchmark results to {output_path}")


if __name__ == "__main__":
    main()
