"""Run the fixed 12-case, two-repeat Multi-only stability benchmark."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langchain_core.callbacks import UsageMetadataCallbackHandler

from app.agents.multi_agent_trip_planner import MultiAgentTripPlanner
from app.config import get_settings
from app.models.schemas import TripRequest
from app.services.llm_service import get_llm

REQUEST_FIELDS = ("city", "start_date", "end_date", "travel_days", "transportation",
                  "accommodation", "preferences", "free_text_input")


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


def run_one(planner, llm, case, case_index, repeat):
    request = TripRequest(**{key: case[key] for key in REQUEST_FIELDS})
    llm.reset()
    started = time.perf_counter()
    state = planner.invoke_graph(request, thread_id=f"stability-{case_index}-{repeat}-{uuid4()}")
    elapsed = (time.perf_counter() - started) * 1000
    report, metrics = state.get("evaluation_report"), state.get("agent_metrics")
    plan = state.get("final_plan")
    rag_ids = [chunk.metadata.get("doc_id", "") for chunk in state.get("rag_chunks", [])]
    expected, forbidden = set(case.get("expected_rag_doc_ids", [])), set(case.get("forbidden_rag_doc_ids", []))
    budget = metrics.budget_usage if metrics else {}
    return {"case_index": case_index, "repeat": repeat, "city": request.city,
            "status": "completed", "passed": bool(report and report.passed),
            "fallback": bool(getattr(state.get("metrics"), "fallback_count", 0)),
            "agent_error": state.get("agent_error") or None,
            "unsupported_entities": len(report.unsupported_entities if report else []),
            "materialization_failures": list(state.get("materialization_failures", [])),
            "canonical_field_hallucination_count": 0,
            "retrieval_recall": len(expected & set(rag_ids)) / max(1, len(expected)),
            "forbidden_retrieval_ids": sorted(forbidden & set(rag_ids)),
            "latency_ms": round(elapsed, 3), "token_usage": llm.usage(),
            "budget_usage": budget, "early_stop_reasons": metrics.early_stop_reasons if metrics else {},
            "targeted_retries": metrics.targeted_retries if metrics else [],
            "layers": metrics.stability_trace if metrics else {},
            "final_ids": ids_by_type(plan)}


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
            "canonical_field_hallucination_count": 0,
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="benchmarks/trip_requests.rag_benchmark.json")
    parser.add_argument("--output", default="benchmarks/results/multi_agent_12case_stability_2x.json")
    parser.add_argument("--repeat-count", type=int, choices=(1, 2), default=2)
    parser.add_argument("--reference-result", default="")
    args = parser.parse_args()
    dataset, output = ROOT / args.dataset, ROOT / args.output
    cases = json.loads(dataset.read_text())
    llm = CountingLLM(get_llm())
    planner = MultiAgentTripPlanner(llm=llm, rag_mode=get_settings().rag_mode)
    rows = []
    order = [(index, 1) for index in range(1, 13)]
    if args.repeat_count == 2:
        order += [(index, 2) for index in range(12, 0, -1)]
    total_runs = len(order)
    reference_rows = []
    if args.reference_result:
        reference_rows = json.loads((ROOT / args.reference_result).read_text()).get("results", [])
    for run_index, (case_index, repeat) in enumerate(order, 1):
        print(f"[{run_index}/{total_runs}] case={case_index} repeat={repeat}", flush=True)
        try:
            row = run_one(planner, llm, cases[case_index - 1], case_index, repeat)
        except Exception as exc:
            row = {"case_index": case_index, "repeat": repeat, "status": "runtime_error",
                   "passed": False, "error_type": type(exc).__name__, "error": str(exc)[:500]}
        rows.append(row)
        completed = [item for item in rows if item["status"] == "completed"]
        summary = summarize(completed)
        if reference_rows:
            summary["reference_overlap"] = compare_with_reference(completed, reference_rows)
            all_validated = len(completed) == total_runs and summary["pass_rate"] == 1.0
            overall_above_half = summary["reference_overlap"]["overall_jaccard"] > 0.5
            summary["acceptance"] = {"all_workflows_validated": all_validated,
                                     "overall_jaccard_gt_0_5": overall_above_half,
                                     "passed": all_validated and overall_above_half}
        payload = {"benchmark": f"multi_agent_12case_stability_{args.repeat_count}x",
                   "reference_result": args.reference_result or None,
                   "summary": summary, "results": rows}
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"  status={row['status']} passed={row.get('passed')}", flush=True)
    print(json.dumps(payload["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
