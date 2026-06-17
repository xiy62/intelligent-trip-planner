"""Compare LangGraph targeted retry against a simulated full-rerun baseline."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agents.langgraph_trip_planner import LangGraphTripPlanner
from app.models.schemas import TripRequest
from scripts.benchmark_trip_planners import load_benchmark_cases


WORKFLOW_NODES = [
    "prepare_request",
    "retrieve_attractions",
    "retrieve_hotels",
    "retrieve_weather",
    "retrieve_rag_context",
    "plan_itinerary",
    "evaluate_itinerary",
]
EXTERNAL_RETRIEVAL_NODES = [
    "retrieve_attractions",
    "retrieve_hotels",
    "retrieve_weather",
    "retrieve_rag_context",
]


def average_node_latency(node: str, node_attempts: Dict[str, int], node_total_latency: Dict[str, float]) -> float:
    attempts = max(1, int(node_attempts.get(node, 0) or 0))
    return float(node_total_latency.get(node, 0.0)) / attempts


def estimate_full_rerun_baseline(state: Dict[str, Any]) -> Dict[str, Any]:
    """Estimate work if every failed evaluation reran the complete workflow."""
    metrics = state.get("metrics")
    if metrics is None:
        return {}
    attempts = max(1, int(metrics.evaluation_attempt_count or 1))
    node_attempts = dict(metrics.node_attempts)
    node_total_latency = dict(metrics.node_total_latency_ms)

    avg_latency_by_node = {
        node: average_node_latency(node, node_attempts, node_total_latency)
        for node in WORKFLOW_NODES
    }
    full_node_attempts = {node: attempts for node in WORKFLOW_NODES}
    full_work_ms = sum(avg_latency_by_node.values()) * attempts
    full_external_calls = attempts * len(EXTERNAL_RETRIEVAL_NODES)

    return {
        "evaluation_attempts": attempts,
        "estimated_full_rerun_node_attempts": full_node_attempts,
        "estimated_full_rerun_external_service_node_calls": full_external_calls,
        "estimated_full_rerun_llm_calls": attempts,
        "estimated_full_rerun_work_ms": round(full_work_ms, 3),
    }


def targeted_retry_metrics(state: Dict[str, Any]) -> Dict[str, Any]:
    metrics = state.get("metrics")
    if metrics is None:
        return {}
    node_attempts = dict(metrics.node_attempts)
    node_total_latency = dict(metrics.node_total_latency_ms)
    external_calls = sum(node_attempts.get(node, 0) for node in EXTERNAL_RETRIEVAL_NODES)
    llm_calls = node_attempts.get("plan_itinerary", 0)
    work_ms = sum(float(value) for value in node_total_latency.values())
    return {
        "targeted_node_attempts": node_attempts,
        "targeted_external_service_node_calls": external_calls,
        "targeted_llm_calls": llm_calls,
        "targeted_work_ms": round(work_ms, 3),
    }


def compare_retry_strategy_for_request(planner: LangGraphTripPlanner, request: TripRequest) -> Dict[str, Any]:
    thread_id = f"retry-bench-{int(time.time() * 1000)}-{request.city}-{request.travel_days}"
    state = planner.invoke_graph(request, thread_id=thread_id)
    report = state.get("evaluation_report")
    metrics = state.get("metrics")
    targeted = targeted_retry_metrics(state)
    full = estimate_full_rerun_baseline(state)

    external_call_savings = (
        full.get("estimated_full_rerun_external_service_node_calls", 0)
        - targeted.get("targeted_external_service_node_calls", 0)
    )
    llm_call_savings = (
        full.get("estimated_full_rerun_llm_calls", 0)
        - targeted.get("targeted_llm_calls", 0)
    )
    work_ms_savings = full.get("estimated_full_rerun_work_ms", 0.0) - targeted.get("targeted_work_ms", 0.0)

    return {
        "request": request.model_dump(),
        "passed": bool(report and report.passed),
        "hard_failures": list(report.hard_failures) if report else [],
        "first_evaluation_pass": metrics.first_evaluation_pass if metrics else None,
        "recovered_after_retry": metrics.recovered_after_retry if metrics else False,
        "fallback": bool(metrics and metrics.fallback_count > 0),
        "decision_trace": list(state.get("decision_trace", [])),
        **targeted,
        **full,
        "external_service_node_call_savings": external_call_savings,
        "llm_call_savings": llm_call_savings,
        "estimated_work_ms_savings": round(work_ms_savings, 3),
    }


def summarize(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    savings = [entry["external_service_node_call_savings"] for entry in entries]
    llm_savings = [entry["llm_call_savings"] for entry in entries]
    work_savings = [entry["estimated_work_ms_savings"] for entry in entries]
    initially_failed = [entry for entry in entries if entry.get("first_evaluation_pass") is False]
    recovered = [entry for entry in entries if entry.get("recovered_after_retry")]
    targeted_external = [entry["targeted_external_service_node_calls"] for entry in entries]
    full_external = [entry["estimated_full_rerun_external_service_node_calls"] for entry in entries]
    targeted_work = [entry["targeted_work_ms"] for entry in entries]
    full_work = [entry["estimated_full_rerun_work_ms"] for entry in entries]

    total_targeted_external = sum(targeted_external)
    total_full_external = sum(full_external)
    total_targeted_work = sum(targeted_work)
    total_full_work = sum(full_work)
    return {
        "request_count": len(entries),
        "initially_failed_runs": len(initially_failed),
        "recovered_runs": len(recovered),
        "recovery_rate": round(len(recovered) / len(initially_failed), 4) if initially_failed else 0.0,
        "avg_external_service_node_call_savings": round(statistics.fmean(savings), 4) if savings else 0.0,
        "avg_llm_call_savings": round(statistics.fmean(llm_savings), 4) if llm_savings else 0.0,
        "avg_estimated_work_ms_savings": round(statistics.fmean(work_savings), 3) if work_savings else 0.0,
        "targeted_external_service_node_calls_total": total_targeted_external,
        "estimated_full_rerun_external_service_node_calls_total": total_full_external,
        "external_service_node_call_reduction_rate": round(
            (total_full_external - total_targeted_external) / total_full_external,
            4,
        )
        if total_full_external
        else 0.0,
        "targeted_work_ms_total": round(total_targeted_work, 3),
        "estimated_full_rerun_work_ms_total": round(total_full_work, 3),
        "estimated_work_reduction_rate": round((total_full_work - total_targeted_work) / total_full_work, 4)
        if total_full_work
        else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark targeted retry vs simulated full rerun.")
    parser.add_argument("--dataset", default="benchmarks/trip_requests.us_rag_benchmark.json")
    parser.add_argument("--output", default="benchmarks/results/retry_strategy_benchmark.json")
    parser.add_argument("--rag-mode", default="chroma_retrieval")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    output_path = Path(args.output)
    if not dataset_path.is_absolute():
        dataset_path = ROOT / dataset_path
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    planner = LangGraphTripPlanner(rag_mode=args.rag_mode)
    entries = [
        compare_retry_strategy_for_request(planner, case.request)
        for case in load_benchmark_cases(dataset_path)
    ]
    output = {
        "dataset": str(dataset_path),
        "rag_mode": args.rag_mode,
        "summary": summarize(entries),
        "results": entries,
    }
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2))
    print(f"\nSaved retry strategy benchmark results to {output_path}")


if __name__ == "__main__":
    main()
