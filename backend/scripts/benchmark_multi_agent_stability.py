"""Run the fixed 12-case, two-repeat Multi-only stability benchmark."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.tools import StructuredTool

from app.agents.canonical_audit import audit_canonical_fields
from app.agents.evidence_snapshot import (
    AgentEvidenceSnapshot,
    EvidenceSnapshotFile,
    EvidenceSnapshotMismatch,
    ExperienceEvidenceSnapshot,
    ReplayProviderEscape,
    WorkflowEvidenceSnapshot,
    request_fingerprint,
    request_snapshot,
)
from app.agents.multi_agent_trip_planner import MultiAgentTripPlanner
from app.config import get_settings
from app.models.schemas import TripRequest
from app.services.llm_service import get_llm
from app.services.rag_service import get_rag_service

REQUEST_FIELDS = ("city", "start_date", "end_date", "travel_days", "transportation",
                  "accommodation", "preferences", "free_text_input")


class ReplayGuardMapService:
    def _escape(self, keywords: str, city: str, citylimit: bool = True, page_size: int = 10,
                country_code: str = ""):
        raise ReplayProviderEscape(f"replay attempted map search: {keywords} in {city}")

    def get_langchain_tools(self):
        return [StructuredTool.from_function(
            func=self._escape,
            name="map_search_poi",
            description="Replay guard that rejects live map searches.",
        )]

    def get_poi_detail(self, poi_id: str):
        raise ReplayProviderEscape(f"replay attempted place detail: {poi_id}")

    def plan_route(self, *args, **kwargs):
        raise ReplayProviderEscape("replay attempted route provider access")


class ReplayGuardRAGService:
    def retrieve_chunks(self, *args, **kwargs):
        raise ReplayProviderEscape("replay attempted RAG provider access")

    def retrieve_local_chunks(self, *args, **kwargs):
        raise ReplayProviderEscape("replay attempted local RAG access")


class CountingRunnable:
    def __init__(self, runnable, owner):
        self.runnable, self.owner = runnable, owner

    def invoke(self, value, config=None, **kwargs):
        config = dict(config or {})
        config["callbacks"] = list(config.get("callbacks", [])) + [self.owner.handler]
        return self.runnable.invoke(value, config=config, **kwargs)


class CountingLLM:
    def __init__(self, llm):
        self.llm = llm
        self.handler = UsageMetadataCallbackHandler()

    def reset(self):
        self.handler = UsageMetadataCallbackHandler()

    def invoke(self, value, config=None, **kwargs):
        config = dict(config or {})
        config["callbacks"] = list(config.get("callbacks", [])) + [self.handler]
        return self.llm.invoke(value, config=config, **kwargs)

    def with_structured_output(self, *args, **kwargs):
        return CountingRunnable(self.llm.with_structured_output(*args, **kwargs), self)

    def usage(self):
        totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        for value in self.handler.usage_metadata.values():
            for key in totals:
                totals[key] += int(value.get(key, 0) or 0)
        return totals


def ids_by_type(plan):
    result = {"attraction": [], "hotel": [], "meal": [], "day_assignment": [], "route_order": []}
    if plan is None:
        return result
    for day in plan.days:
        attractions = [item.poi_id for item in day.attractions if item.poi_id]
        result["attraction"].extend(attractions)
        result["route_order"].append(attractions)
        result["day_assignment"].extend([f"{day.day_index}:{item}" for item in attractions])
        result["meal"].extend(item.poi_id for item in day.meals if item.poi_id)
        if day.hotel and day.hotel.poi_id:
            result["hotel"].append(day.hotel.poi_id)
    for key in ("attraction", "hotel", "meal", "day_assignment"):
        result[key] = sorted(set(result[key]))
    return result


def jaccard(left, right):
    a, b = set(left), set(right)
    return len(a & b) / len(a | b) if a or b else 1.0


def route_similarity(left, right):
    left_pairs = {(day, a, b) for day, values in enumerate(left) for a, b in zip(values, values[1:])}
    right_pairs = {(day, a, b) for day, values in enumerate(right) for a, b in zip(values, values[1:])}
    return jaccard(left_pairs, right_pairs)


def layer_ids(row, layer, key):
    return row.get("layers", {}).get(layer, {}).get(key, [])


def budget_violations(row):
    budget = row.get("budget_usage", {})
    violations = []
    for role, limits in budget.get("role_limits", {}).items():
        used = budget.get("role_used", {}).get(role, {})
        for resource, limit in limits.items():
            if used.get(resource, 0) > limit:
                violations.append(f"{role}.{resource}:{used.get(resource, 0)}>{limit}")
    for resource, limit in budget.get("global_limits", {}).items():
        used = budget.get("global_used", {}).get(resource, 0)
        if used > limit:
            violations.append(f"global.{resource}:{used}>{limit}")
    return violations


def run_one(planner, llm, case, case_index, repeat, *, evidence_mode="live", evidence_snapshot=None):
    request = TripRequest(**{key: case[key] for key in REQUEST_FIELDS})
    if evidence_mode == "replay":
        if evidence_snapshot is None:
            raise EvidenceSnapshotMismatch(f"missing replay snapshot for case {case_index}")
        if evidence_snapshot.request_fingerprint != request_fingerprint(request):
            raise EvidenceSnapshotMismatch(f"request fingerprint mismatch for case {case_index}")
        if evidence_snapshot.request != request_snapshot(request):
            raise EvidenceSnapshotMismatch(f"request payload mismatch for case {case_index}")
        planner.evidence_snapshot = evidence_snapshot.model_copy(deep=True)
    else:
        planner.evidence_snapshot = None
    llm.reset()
    started = time.perf_counter()
    state = planner.invoke_graph(request, thread_id=f"stability-{case_index}-{repeat}-{uuid4()}")
    elapsed = (time.perf_counter() - started) * 1000
    report, metrics = state.get("evaluation_report"), state.get("agent_metrics")
    plan = state.get("final_plan")
    rag_ids = [chunk.metadata.get("doc_id", "") for chunk in state.get("rag_chunks", [])]
    expected, forbidden = set(case.get("expected_rag_doc_ids", [])), set(case.get("forbidden_rag_doc_ids", []))
    budget = metrics.budget_usage if metrics else {}
    canonical_audit = audit_canonical_fields(plan, state["candidate_registry"])
    row = {"case_index": case_index, "repeat": repeat, "city": request.city,
            "status": "completed", "passed": bool(report and report.passed),
            "fallback": bool(getattr(state.get("metrics"), "fallback_count", 0)),
            "agent_error": state.get("agent_error") or None,
            "unsupported_entities": len(report.unsupported_entities if report else []),
            "materialization_failures": list(state.get("materialization_failures", [])),
            "canonical_field_hallucination_count": canonical_audit.mismatch_count,
            "canonical_field_mismatches": [item.model_dump(mode="json") for item in canonical_audit.mismatches],
            "canonical_audited_entity_count": canonical_audit.audited_entity_count,
            "skipped_generic_meal_count": canonical_audit.skipped_generic_meal_count,
            "retrieval_recall": len(expected & set(rag_ids)) / max(1, len(expected)),
            "forbidden_retrieval_ids": sorted(forbidden & set(rag_ids)),
            "latency_ms": round(elapsed, 3), "token_usage": llm.usage(),
            "budget_usage": budget, "early_stop_reasons": metrics.early_stop_reasons if metrics else {},
            "targeted_retries": metrics.targeted_retries if metrics else [],
            "layers": metrics.stability_trace if metrics else {},
            "final_ids": ids_by_type(plan), "evidence_mode": evidence_mode,
            "replay_provider_escape_count": (
                int(evidence_mode == "replay" and budget.get("global_used", {}).get("maps", 0) > 0)
                + int(evidence_mode == "replay" and budget.get("global_used", {}).get("rag", 0) > 0)
            )}
    if evidence_mode == "record" and repeat == 1:
        registry = state["candidate_registry"]
        row["_evidence_snapshot"] = WorkflowEvidenceSnapshot(
            request_fingerprint=request_fingerprint(request),
            request=request_snapshot(request),
            experience=ExperienceEvidenceSnapshot(
                entities=[item.model_copy(deep=True) for item in registry.entities.values()
                          if item.entity_type == "attraction"],
                rag_chunks=[item.model_copy(deep=True) for item in state.get("rag_chunks", [])],
            ),
            logistics=AgentEvidenceSnapshot(
                entities=[item.model_copy(deep=True) for item in registry.entities.values()
                          if item.entity_type in {"hotel", "meal"}],
            ),
            weather_info=[item.model_copy(deep=True) for item in state.get("weather_info", [])],
        )
    return row


def prewarm_parallel_rag(rag_service, rag_mode, max_workers):
    """Initialize the shared read-only Chroma handle before worker threads can race."""
    if max_workers > 1 and rag_mode == "chroma_retrieval":
        rag_service._get_vectorstore()


def build_worker(rag_mode, llm_backend=None, rag_service=None, *, evidence_mode="live"):
    llm = CountingLLM(llm_backend or get_llm())
    if evidence_mode == "replay":
        return MultiAgentTripPlanner(llm=llm, rag_mode=rag_mode,
                                     rag_service=ReplayGuardRAGService(),
                                     map_service=ReplayGuardMapService()), llm
    return MultiAgentTripPlanner(llm=llm, rag_mode=rag_mode, rag_service=rag_service), llm


def safe_run_one(planner, llm, case, case_index, repeat, *, runner=run_one):
    try:
        return runner(planner, llm, case, case_index, repeat)
    except EvidenceSnapshotMismatch as exc:
        return {"case_index": case_index, "repeat": repeat, "status": "runtime_error",
                "passed": False, "error_type": "snapshot_mismatch", "error": str(exc)[:500]}
    except ReplayProviderEscape as exc:
        return {"case_index": case_index, "repeat": repeat, "status": "runtime_error",
                "passed": False, "error_type": "replay_provider_escape", "error": str(exc)[:500]}
    except Exception as exc:
        return {"case_index": case_index, "repeat": repeat, "status": "runtime_error",
                "passed": False, "error_type": type(exc).__name__, "error": str(exc)[:500]}


def execute_order(cases, order, *, max_workers, worker_factory, runner=run_one, on_result=None):
    """Execute workflows with isolated parallel workers and deterministic result ordering."""
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1")
    results_by_position = {}

    def record(position, row):
        results_by_position[position] = row
        ordered_rows = [results_by_position[index] for index in sorted(results_by_position)]
        if on_result is not None:
            on_result(ordered_rows, row)

    if max_workers == 1:
        planner, llm = worker_factory()
        for position, (case_index, repeat) in enumerate(order):
            row = safe_run_one(planner, llm, cases[case_index - 1], case_index, repeat, runner=runner)
            record(position, row)
        return [results_by_position[index] for index in range(len(order))]

    def isolated_workflow(position, case_index, repeat):
        planner, llm = worker_factory()
        row = safe_run_one(planner, llm, cases[case_index - 1], case_index, repeat, runner=runner)
        return position, row

    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="multi-benchmark") as executor:
        futures = [executor.submit(isolated_workflow, position, case_index, repeat)
                   for position, (case_index, repeat) in enumerate(order)]
        for future in as_completed(futures):
            position, row = future.result()
            record(position, row)
    return [results_by_position[index] for index in range(len(order))]


def compare_rows(first, second):
    all_first = sum((first["final_ids"][key] for key in ("attraction", "hotel", "meal")), [])
    all_second = sum((second["final_ids"][key] for key in ("attraction", "hotel", "meal")), [])
    first_pool = (layer_ids(first, "experience", "candidate_pool_ids")
                  + layer_ids(first, "logistics", "hotel_pool_ids")
                  + layer_ids(first, "logistics", "meal_pool_ids"))
    second_pool = (layer_ids(second, "experience", "candidate_pool_ids")
                   + layer_ids(second, "logistics", "hotel_pool_ids")
                   + layer_ids(second, "logistics", "meal_pool_ids"))
    first_shortlist = (layer_ids(first, "experience", "shortlist_ids")
                       + layer_ids(first, "logistics", "hotel_shortlist_ids")
                       + layer_ids(first, "logistics", "meal_shortlist_ids"))
    second_shortlist = (layer_ids(second, "experience", "shortlist_ids")
                        + layer_ids(second, "logistics", "hotel_shortlist_ids")
                        + layer_ids(second, "logistics", "meal_shortlist_ids"))
    first_proposal = (layer_ids(first, "experience", "core_ids")
                      + layer_ids(first, "experience", "optional_ids")
                      + [value for value in [layer_ids(first, "logistics", "primary_hotel_id")] if value]
                      + layer_ids(first, "logistics", "selected_meal_ids"))
    second_proposal = (layer_ids(second, "experience", "core_ids")
                       + layer_ids(second, "experience", "optional_ids")
                       + [value for value in [layer_ids(second, "logistics", "primary_hotel_id")] if value]
                       + layer_ids(second, "logistics", "selected_meal_ids"))
    return {"case_index": first["case_index"], "candidate_pool": jaccard(first_pool, second_pool),
            "ranked_shortlist": jaccard(first_shortlist, second_shortlist),
            "proposal": jaccard(first_proposal, second_proposal),
            "overall": jaccard(all_first, all_second),
            "attraction": jaccard(first["final_ids"]["attraction"], second["final_ids"]["attraction"]),
            "hotel_exact": first["final_ids"]["hotel"] == second["final_ids"]["hotel"],
            "meal": jaccard(first["final_ids"]["meal"], second["final_ids"]["meal"]),
            "day_assignment": jaccard(first["final_ids"]["day_assignment"], second["final_ids"]["day_assignment"]),
            "route_order": route_similarity(first["final_ids"]["route_order"], second["final_ids"]["route_order"])}


def overlap_summary(pairs):
    def average(key):
        return statistics.mean(item[key] for item in pairs) if pairs else 0.0
    return {"validated_pair_count": len(pairs),
            "candidate_pool_jaccard": average("candidate_pool"),
            "ranked_shortlist_jaccard": average("ranked_shortlist"),
            "proposal_jaccard": average("proposal"), "overall_jaccard": average("overall"),
            "attraction_jaccard": average("attraction"), "primary_hotel_exact_match": average("hotel_exact"),
            "meal_jaccard": average("meal"), "day_assignment_jaccard": average("day_assignment"),
            "route_order_similarity": average("route_order"), "pairs": pairs}


def summarize(rows):
    pairs = []
    for case_index in range(1, 13):
        values = sorted([row for row in rows if row["case_index"] == case_index], key=lambda row: row["repeat"])
        if len(values) != 2 or not all(row["passed"] for row in values):
            continue
        pairs.append(compare_rows(*values))
    llm_calls = [row.get("budget_usage", {}).get("global_used", {}).get("llm", 0) for row in rows]
    map_calls = [row.get("budget_usage", {}).get("global_used", {}).get("maps", 0) for row in rows]
    rag_calls = [row.get("budget_usage", {}).get("global_used", {}).get("rag", 0) for row in rows]
    violations = [{"case_index": row["case_index"], "repeat": row["repeat"],
                   "violations": budget_violations(row)} for row in rows if budget_violations(row)]
    summary = {"completed_runs": len(rows), "pass_rate": sum(row["passed"] for row in rows) / max(1, len(rows)),
            "fallback_rate": sum(row["fallback"] for row in rows) / max(1, len(rows)),
            "agent_error_count": sum(bool(row["agent_error"]) for row in rows),
            "unsupported_entity_count": sum(row["unsupported_entities"] for row in rows),
            "materialization_failure_count": sum(len(row["materialization_failures"]) for row in rows),
            "canonical_field_hallucination_count": sum(
                row.get("canonical_field_hallucination_count", 0) for row in rows
            ),
            "replay_provider_escape_count": sum(row.get("replay_provider_escape_count", 0) for row in rows),
            "retrieval_recall_at_4": statistics.mean(row["retrieval_recall"] for row in rows) if rows else 0,
            "forbidden_retrieval_count": sum(len(row["forbidden_retrieval_ids"]) for row in rows),
            "avg_llm_calls": statistics.mean(llm_calls) if llm_calls else 0,
            "avg_maps_calls": statistics.mean(map_calls) if map_calls else 0,
            "max_llm_calls": max(llm_calls, default=0), "max_maps_calls": max(map_calls, default=0),
            "max_rag_calls": max(rag_calls, default=0),
            "per_workflow_budget_violation_count": len(violations),
            "per_workflow_budget_violations": violations,
            "avg_tokens": statistics.mean(row["token_usage"]["total_tokens"] for row in rows) if rows else 0,
            "avg_latency_ms": statistics.mean(row["latency_ms"] for row in rows) if rows else 0}
    summary.update(overlap_summary(pairs))
    return summary


def compare_with_reference(rows, reference_rows):
    pairs = []
    for current in rows:
        if not current.get("passed"):
            continue
        for reference in reference_rows:
            if reference.get("case_index") == current.get("case_index") and reference.get("passed"):
                pair = compare_rows(current, reference)
                pair["current_repeat"] = current.get("repeat")
                pair["reference_repeat"] = reference.get("repeat")
                pairs.append(pair)
    return overlap_summary(pairs)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="benchmarks/trip_requests.rag_benchmark.json")
    parser.add_argument("--output", default="benchmarks/results/multi_agent_12case_stability_2x.json")
    parser.add_argument("--repeat-count", type=int, choices=(1, 2), default=2)
    parser.add_argument("--reference-result", default="")
    parser.add_argument("--max-workers", type=int, default=2,
                        help="Parallel workflow workers; use 1 for a sequential control")
    parser.add_argument("--evidence-mode", choices=("live", "record", "replay"), default="live")
    parser.add_argument("--evidence-snapshot",
                        default="benchmarks/results/multi_agent_evidence_snapshot.json")
    return parser


def file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def configuration_fingerprint(settings, dataset, args):
    try:
        git_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT.parent, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except Exception:
        git_sha = "unknown"
    shared_model = os.getenv("LLM_MODEL_ID") or settings.openai_model
    return {
        "dataset_sha256": file_sha256(dataset),
        "git_commit": git_sha,
        "models": {
            "experience": settings.experience_model or shared_model,
            "logistics": settings.logistics_model or shared_model,
            "composer": settings.composer_model or shared_model,
        },
        "temperature": 0,
        "rag_mode": settings.rag_mode,
        "route_time_evaluation_enabled": settings.route_time_evaluation_enabled,
        "max_route_time_evaluations_per_trip": settings.max_route_time_evaluations_per_trip,
        "max_workers": args.max_workers,
        "repeat_count": args.repeat_count,
        "evidence_mode": args.evidence_mode,
    }


def main():
    parser = build_parser()
    args = parser.parse_args()
    dataset, output = ROOT / args.dataset, ROOT / args.output
    snapshot_path = ROOT / args.evidence_snapshot
    cases = json.loads(dataset.read_text())
    if args.max_workers < 1:
        parser.error("--max-workers must be at least 1")
    settings = get_settings()
    rag_mode = settings.rag_mode
    config = configuration_fingerprint(settings, dataset, args)
    replay_cases = {}
    snapshot_file = None
    if args.evidence_mode == "replay":
        if not snapshot_path.exists():
            parser.error(f"evidence snapshot does not exist: {snapshot_path}")
        try:
            snapshot_file = EvidenceSnapshotFile.model_validate_json(snapshot_path.read_text())
            snapshot_file.validate_integrity()
        except (ValueError, EvidenceSnapshotMismatch) as exc:
            parser.error(str(exc))
        if snapshot_file.metadata.get("dataset_sha256") != config["dataset_sha256"]:
            parser.error("snapshot dataset hash does not match current dataset")
        replay_cases = snapshot_file.cases
    llm_backend = get_llm()
    rag_service = get_rag_service() if args.evidence_mode != "replay" else None
    if args.evidence_mode != "replay":
        prewarm_parallel_rag(rag_service, rag_mode, args.max_workers)
    order = [(index, 1) for index in range(1, 13)]
    if args.repeat_count == 2:
        order += [(index, 2) for index in range(12, 0, -1)]
    total_runs = len(order)
    reference_rows = []
    if args.reference_result:
        reference_rows = json.loads((ROOT / args.reference_result).read_text()).get("results", [])
    benchmark_started = time.perf_counter()
    completion_count = 0
    completion_order = []
    payload = {}
    recorded_cases = {}

    def persist(rows, row):
        nonlocal completion_count, payload
        evidence = row.pop("_evidence_snapshot", None)
        if evidence is not None:
            recorded_cases[evidence.request_fingerprint] = evidence
        completion_count += 1
        completion_order.append({"case_index": row["case_index"], "repeat": row["repeat"]})
        print(f"[{completion_count}/{total_runs}] case={row['case_index']} repeat={row['repeat']} "
              f"status={row['status']} passed={row.get('passed')}", flush=True)
        completed = [item for item in rows if item["status"] == "completed"]
        summary = summarize(completed)
        elapsed_ms = (time.perf_counter() - benchmark_started) * 1000
        summed_latency_ms = sum(item.get("latency_ms", 0) for item in completed)
        summary["benchmark_wall_time_ms"] = round(elapsed_ms, 3)
        summary["summed_workflow_latency_ms"] = round(summed_latency_ms, 3)
        summary["effective_parallelism"] = round(summed_latency_ms / elapsed_ms, 4) if elapsed_ms else 0.0
        summary["throughput_workflows_per_minute"] = (
            round(len(rows) * 60_000 / elapsed_ms, 4) if elapsed_ms else 0.0
        )
        if reference_rows:
            summary["reference_overlap"] = compare_with_reference(completed, reference_rows)
            all_validated = len(completed) == total_runs and summary["pass_rate"] == 1.0
            overall_above_half = summary["reference_overlap"]["overall_jaccard"] > 0.5
            summary["acceptance"] = {"all_workflows_validated": all_validated,
                                     "overall_jaccard_gt_0_5": overall_above_half,
                                     "passed": all_validated and overall_above_half}
        payload = {"benchmark": f"multi_agent_12case_stability_{args.repeat_count}x",
                   "reference_result": args.reference_result or None,
                   "configuration": config,
                   "evidence": {"mode": args.evidence_mode,
                                "snapshot_path": args.evidence_snapshot if args.evidence_mode != "live" else None,
                                "snapshot_hash": None},
                   "execution": {"mode": "parallel" if args.max_workers > 1 else "sequential",
                                 "max_workers": args.max_workers,
                                 "completion_order": list(completion_order)},
                   "summary": summary, "results": rows}
        if args.evidence_mode == "record":
            recorded = EvidenceSnapshotFile(
                metadata={"dataset_sha256": config["dataset_sha256"],
                          "recorded_at": datetime.now(timezone.utc).isoformat(),
                          "configuration": config},
                cases=recorded_cases,
            ).with_hash()
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            snapshot_path.write_text(recorded.model_dump_json(indent=2) + "\n")
            payload["evidence"]["snapshot_hash"] = recorded.snapshot_hash
        elif snapshot_file is not None:
            payload["evidence"]["snapshot_hash"] = snapshot_file.snapshot_hash
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2) + "\n")

    def benchmark_runner(planner, llm, case, case_index, repeat):
        snapshot = None
        if args.evidence_mode == "replay":
            request = TripRequest(**{key: case[key] for key in REQUEST_FIELDS})
            snapshot = replay_cases.get(request_fingerprint(request))
        return run_one(planner, llm, case, case_index, repeat,
                       evidence_mode=args.evidence_mode, evidence_snapshot=snapshot)

    execute_order(cases, order, max_workers=args.max_workers,
                  worker_factory=lambda: build_worker(rag_mode, llm_backend, rag_service,
                                                      evidence_mode=args.evidence_mode),
                  runner=benchmark_runner, on_result=persist)
    print(json.dumps(payload["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
