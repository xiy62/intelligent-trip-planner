"""Run the fixed three-case single-baseline vs active multi-agent smoke comparison."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.agents.langgraph_trip_planner import LangGraphTripPlanner
from app.agents.multi_agent_trip_planner import MultiAgentTripPlanner
from app.config import get_settings
from app.models.schemas import TripRequest

CASES_PATH = BACKEND_ROOT / "benchmarks" / "multi_agent_live_smoke.json"
DEFAULT_OUTPUT = BACKEND_ROOT / "benchmarks" / "results" / "multi_agent_live_smoke_latest.json"


def credentials_available() -> tuple[bool, list[str]]:
    settings = get_settings()
    missing = []
    if not settings.google_maps_api_key:
        missing.append("GOOGLE_MAPS_API_KEY")
    if not (os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or settings.openai_api_key):
        missing.append("LLM_API_KEY_OR_OPENAI_API_KEY")
    return not missing, missing


def summarize(workflow: str, case_id: str, elapsed_ms: float, state: dict) -> dict:
    report = state.get("evaluation_report")
    metrics = state.get("agent_metrics")
    registry = state.get("candidate_registry")
    return {
        "case_id": case_id,
        "workflow": workflow,
        "status": "completed",
        "latency_ms": round(elapsed_ms, 3),
        "passed": bool(report and report.passed),
        "fallback": bool(getattr(state.get("metrics"), "fallback_count", 0)),
        "hard_failures": list(getattr(report, "hard_failures", [])),
        "source_ids": sorted(registry.entities) if registry is not None else [],
        "rag_chunk_ids": [chunk.chunk_id for chunk in state.get("rag_chunks", [])],
        "agent_metrics": metrics.model_dump() if metrics is not None else {},
        "proposal_versions": {
            "experience": getattr(state.get("experience_proposal"), "version", None),
            "logistics": getattr(state.get("logistics_proposal"), "version", None),
            "composer": getattr(state.get("id_draft"), "version", None),
        },
        "agent_error": state.get("agent_error") or None,
        "agent_retry_state": (state.get("agent_retry_state").model_dump()
                              if state.get("agent_retry_state") is not None else {}),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workflow", choices=["all", "single_benchmark_baseline", "multi_agent_active"],
                        default="all")
    parser.add_argument("--case-id", default="")
    args = parser.parse_args()
    cases = json.loads(CASES_PATH.read_text())
    if args.case_id:
        cases = [case for case in cases if case["case_id"] == args.case_id]
        if not cases:
            raise SystemExit(f"unknown case id: {args.case_id}")
    available, missing = credentials_available()
    selected_workflows = (["single_benchmark_baseline", "multi_agent_active"]
                          if args.workflow == "all" else [args.workflow])
    existing_rows = []
    if args.output.exists() and args.workflow != "all":
        existing_rows = json.loads(args.output.read_text()).get("results", [])
        existing_rows = [row for row in existing_rows
                         if not (row.get("workflow") in selected_workflows and
                                 (not args.case_id or row.get("case_id") == args.case_id))]
    rows = []
    if not available:
        for case in cases:
            for workflow in selected_workflows:
                rows.append({"case_id": case["case_id"], "workflow": workflow,
                             "status": "blocked_external_credentials", "missing": missing})
    else:
        all_planners = {
            "single_benchmark_baseline": LangGraphTripPlanner(rag_mode=get_settings().rag_mode),
            "multi_agent_active": MultiAgentTripPlanner(rag_mode=get_settings().rag_mode),
        }
        planners = {name: all_planners[name] for name in selected_workflows}
        for case in cases:
            case_id = case.pop("case_id")
            request = TripRequest(**case)
            for workflow, planner in planners.items():
                started = time.perf_counter()
                try:
                    state = planner.plan_trip_with_state(request)
                    rows.append(summarize(workflow, case_id, (time.perf_counter() - started) * 1000, state))
                except Exception as exc:
                    rows.append({"case_id": case_id, "workflow": workflow,
                                 "status": "provider_or_runtime_failure", "error_type": type(exc).__name__,
                                 "error": str(exc)[:500]})
    rows = existing_rows + rows
    rows.sort(key=lambda row: (row.get("case_id", ""), row.get("workflow", "")))
    payload = {"benchmark": "three_case_live_smoke", "temperature": 0,
               "statistical_claim_allowed": False, "results": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
