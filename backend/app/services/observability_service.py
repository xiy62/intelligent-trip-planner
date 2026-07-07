"""SQLite-backed observability for planner runs and evaluation traces."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from uuid import uuid4

from ..models.langgraph_state import EvaluationReport, RAGChunk, RunMetrics, RetryState, TripGraphState
from ..models.schemas import TripPlan, TripRequest

DEFAULT_OBSERVABILITY_DB_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "observability"
    / "trip_observability.sqlite3"
)


class ObservabilityService:
    """Persist compact LangGraph run traces for debugging and evaluation analysis."""

    def __init__(self, db_path: Path | str = DEFAULT_OBSERVABILITY_DB_PATH):
        self.db_path = Path(db_path)
        self._initialized = False

    def persist_state(
        self,
        state: TripGraphState,
        *,
        source: str = "runtime",
        rag_mode: str = "",
        benchmark_metadata: Optional[Dict[str, Any]] = None,
        run_id: Optional[str] = None,
    ) -> str:
        """Persist a completed or partial graph state and return the stored run id."""
        benchmark_metadata = benchmark_metadata or {}
        request = state.get("request")
        metrics = state.get("metrics")
        report = state.get("evaluation_report")
        evaluation_history = list(state.get("evaluation_history", []))
        if report is not None and not evaluation_history:
            evaluation_history = [report]
        retry_counts = state.get("retry_counts")
        final_plan = state.get("final_plan")
        now = time.time()
        stored_run_id = run_id or f"{source}-{uuid4()}"
        conversation_id = str(state.get("conversation_id") or getattr(request, "conversation_id", "") or "")
        profile_id = str(getattr(request, "profile_id", "") or "")
        hard_failures = self._aggregate_hard_failures(evaluation_history)
        scores = report.scores.model_dump() if report is not None else {}
        fallback = bool(metrics and metrics.fallback_count > 0)

        started_at = metrics.started_at if metrics is not None else now
        ended_at = metrics.ended_at if metrics is not None and metrics.ended_at else now
        end_to_end_ms = metrics.end_to_end_ms if metrics is not None else 0.0

        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO agent_runs (
                    run_id, conversation_id, profile_id, source, city, travel_days,
                    rag_mode, started_at, ended_at, end_to_end_ms, passed,
                    first_evaluation_pass, final_evaluation_pass, recovered_after_retry,
                    fallback, evaluation_attempt_count, hard_failures_json, scores_json,
                    request_json, final_plan_summary_json, evaluation_report_json, evaluation_history_json,
                    retry_counts_json, retrieved_rag_sources_json, benchmark_metadata_json,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    stored_run_id,
                    conversation_id,
                    profile_id,
                    source,
                    getattr(request, "city", ""),
                    int(getattr(request, "travel_days", 0) or 0),
                    rag_mode,
                    float(started_at or now),
                    float(ended_at or now),
                    float(end_to_end_ms or 0.0),
                    self._bool_to_int(report.passed if report is not None else None),
                    self._bool_to_int(metrics.first_evaluation_pass if metrics is not None else None),
                    self._bool_to_int(metrics.final_evaluation_pass if metrics is not None else None),
                    self._bool_to_int(metrics.recovered_after_retry if metrics is not None else None),
                    self._bool_to_int(fallback),
                    int(metrics.evaluation_attempt_count if metrics is not None else 0),
                    self._dumps(hard_failures),
                    self._dumps(scores),
                    self._request_json(request),
                    self._dumps(self._summarize_trip_plan(final_plan)),
                    self._dumps(report.model_dump() if report is not None else {}),
                    self._dumps([item.model_dump() for item in evaluation_history]),
                    self._dumps(retry_counts.model_dump() if retry_counts is not None else {}),
                    self._dumps(self.compact_rag_sources(state.get("rag_chunks", []))),
                    self._dumps(benchmark_metadata),
                    now,
                ),
            )
            conn.execute("DELETE FROM agent_run_events WHERE run_id = ?", (stored_run_id,))
            conn.executemany(
                """
                INSERT INTO agent_run_events (
                    event_id, run_id, node_name, attempt, latency_ms,
                    event_type, message, payload_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                self._build_events(stored_run_id, state, metrics, report, now),
            )
        return stored_run_id

    def list_runs(
        self,
        *,
        limit: int = 50,
        source: Optional[str] = None,
        city: Optional[str] = None,
        passed: Optional[bool] = None,
        failure_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return compact recent runs with optional filters."""
        clauses: List[str] = []
        params: List[Any] = []
        if source:
            clauses.append("source = ?")
            params.append(source)
        if city:
            clauses.append("city = ?")
            params.append(city)
        if passed is not None:
            clauses.append("passed = ?")
            params.append(self._bool_to_int(passed))
        if failure_type:
            clauses.append("hard_failures_json LIKE ?")
            params.append(f"%{failure_type}%")
        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        safe_limit = max(1, min(int(limit), 500))
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT *
                FROM agent_runs
                {where_sql}
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (*params, safe_limit),
            ).fetchall()
        return [self._row_to_run(row) for row in rows]

    def get_run_detail(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Return one run with ordered trace events."""
        with self._connect() as conn:
            run = conn.execute("SELECT * FROM agent_runs WHERE run_id = ?", (run_id,)).fetchone()
            if run is None:
                return None
            events = conn.execute(
                """
                SELECT *
                FROM agent_run_events
                WHERE run_id = ?
                ORDER BY created_at ASC, event_id ASC
                """,
                (run_id,),
            ).fetchall()
        detail = self._row_to_run(run)
        detail["events"] = [self._row_to_event(row) for row in events]
        return detail

    def summary(self) -> Dict[str, Any]:
        """Return aggregate observability metrics across persisted runs."""
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM agent_runs").fetchall()
            event_rows = conn.execute(
                """
                SELECT run_id,
                       SUM(CASE WHEN event_type = 'node' THEN 1 ELSE 0 END) AS node_events,
                       SUM(CASE WHEN event_type = 'routing' THEN 1 ELSE 0 END) AS routing_events
                FROM agent_run_events
                GROUP BY run_id
                """
            ).fetchall()
        runs = [self._row_to_run(row) for row in rows]
        event_counts = {
            row["run_id"]: {
                "node_events": int(row["node_events"] or 0),
                "routing_events": int(row["routing_events"] or 0),
            }
            for row in event_rows
        }
        total = len(runs)
        failed_or_initially_failed = [
            run for run in runs if run["passed"] is False or bool(run["hard_failures"])
        ]
        categorized_failed = [run for run in failed_or_initially_failed if run["hard_failures"]]
        trace_complete = [
            run
            for run in runs
            if event_counts.get(run["run_id"], {}).get("node_events", 0) > 0
            and event_counts.get(run["run_id"], {}).get("routing_events", 0) > 0
            and bool(run["evaluation_report"])
        ]
        grounding_scores = [
            float(run["scores"].get("grounding_score", 0.0))
            for run in runs
            if isinstance(run["scores"], dict)
        ]
        pacing_scores = self._score_values(runs, "pacing_score")
        route_scores = self._score_values(runs, "route_coherence_score")
        preference_scores = self._score_values(runs, "preference_match_score")
        attribution_scores = self._score_values(runs, "attribution_coverage_score")
        quality_warning_runs = [
            run
            for run in runs
            if isinstance(run["evaluation_report"], dict)
            and bool(run["evaluation_report"].get("quality_warnings", []))
        ]
        attributed_runs = [
            run
            for run in runs
            if isinstance(run["scores"], dict)
            and float(run["scores"].get("attribution_coverage_score", 0.0)) >= 0.8
        ]
        failure_counts: Dict[str, int] = {}
        for run in runs:
            for failure in run["hard_failures"]:
                failure_counts[failure] = failure_counts.get(failure, 0) + 1

        initially_failed = [run for run in runs if run["first_evaluation_pass"] is False]
        recovered = [run for run in runs if run["recovered_after_retry"] is True]
        return {
            "total_runs": total,
            "pass_rate": self._ratio(sum(1 for run in runs if run["passed"] is True), total),
            "fallback_rate": self._ratio(sum(1 for run in runs if run["fallback"] is True), total),
            "recovery_rate": self._ratio(len(recovered), len(initially_failed)),
            "avg_latency_ms": round(
                sum(float(run["end_to_end_ms"] or 0.0) for run in runs) / total, 3
            )
            if total
            else 0.0,
            "avg_evaluation_attempts": round(
                sum(int(run["evaluation_attempt_count"] or 0) for run in runs) / total, 3
            )
            if total
            else 0.0,
            "failure_category_counts": failure_counts,
            "avg_grounding_score": round(sum(grounding_scores) / len(grounding_scores), 4)
            if grounding_scores
            else 0.0,
            "avg_pacing_score": self._average(pacing_scores),
            "avg_route_coherence_score": self._average(route_scores),
            "avg_preference_match_score": self._average(preference_scores),
            "avg_attribution_coverage_score": self._average(attribution_scores),
            "quality_warning_rate": self._ratio(len(quality_warning_runs), total),
            "attribution_coverage_rate": self._ratio(len(attributed_runs), total),
            "trace_coverage": self._ratio(len(trace_complete), total),
            "failed_or_initially_failed_runs": len(failed_or_initially_failed),
            "failure_categorization_coverage": self._ratio(
                len(categorized_failed), len(failed_or_initially_failed)
            ),
        }

    def delete_runs(self, *, source: Optional[str] = None) -> int:
        """Delete persisted runs, optionally filtered by source."""
        with self._connect() as conn:
            if source:
                rows = conn.execute("SELECT run_id FROM agent_runs WHERE source = ?", (source,)).fetchall()
            else:
                rows = conn.execute("SELECT run_id FROM agent_runs").fetchall()
            run_ids = [row["run_id"] for row in rows]
            if not run_ids:
                return 0
            placeholders = ",".join("?" for _ in run_ids)
            conn.execute(f"DELETE FROM agent_run_events WHERE run_id IN ({placeholders})", run_ids)
            conn.execute(f"DELETE FROM agent_runs WHERE run_id IN ({placeholders})", run_ids)
        return len(run_ids)

    def _connect(self) -> sqlite3.Connection:
        self._ensure_schema()
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        if self._initialized:
            return
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_runs (
                    run_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL DEFAULT '',
                    profile_id TEXT NOT NULL DEFAULT '',
                    source TEXT NOT NULL DEFAULT 'runtime',
                    city TEXT NOT NULL DEFAULT '',
                    travel_days INTEGER NOT NULL DEFAULT 0,
                    rag_mode TEXT NOT NULL DEFAULT '',
                    started_at REAL NOT NULL DEFAULT 0,
                    ended_at REAL NOT NULL DEFAULT 0,
                    end_to_end_ms REAL NOT NULL DEFAULT 0,
                    passed INTEGER,
                    first_evaluation_pass INTEGER,
                    final_evaluation_pass INTEGER,
                    recovered_after_retry INTEGER,
                    fallback INTEGER NOT NULL DEFAULT 0,
                    evaluation_attempt_count INTEGER NOT NULL DEFAULT 0,
                    hard_failures_json TEXT NOT NULL DEFAULT '[]',
                    scores_json TEXT NOT NULL DEFAULT '{}',
                    request_json TEXT NOT NULL DEFAULT '{}',
                    final_plan_summary_json TEXT NOT NULL DEFAULT '{}',
                    evaluation_report_json TEXT NOT NULL DEFAULT '{}',
                    evaluation_history_json TEXT NOT NULL DEFAULT '[]',
                    retry_counts_json TEXT NOT NULL DEFAULT '{}',
                    retrieved_rag_sources_json TEXT NOT NULL DEFAULT '[]',
                    benchmark_metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_run_events (
                    event_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    node_name TEXT NOT NULL DEFAULT '',
                    attempt INTEGER NOT NULL DEFAULT 0,
                    latency_ms REAL NOT NULL DEFAULT 0,
                    event_type TEXT NOT NULL DEFAULT 'node',
                    message TEXT NOT NULL DEFAULT '',
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    created_at REAL NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES agent_runs(run_id) ON DELETE CASCADE
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_agent_runs_created_at ON agent_runs(created_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_agent_runs_source ON agent_runs(source)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_agent_runs_city ON agent_runs(city)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_agent_run_events_run_id ON agent_run_events(run_id)")
            self._ensure_column(
                conn,
                "agent_runs",
                "evaluation_history_json",
                "TEXT NOT NULL DEFAULT '[]'",
            )
        self._initialized = True

    def _build_events(
        self,
        run_id: str,
        state: TripGraphState,
        metrics: Optional[RunMetrics],
        report: Optional[EvaluationReport],
        created_at: float,
    ) -> List[tuple[Any, ...]]:
        events: List[tuple[Any, ...]] = []
        if metrics is not None:
            for node_name, latency_ms in metrics.node_latency_ms.items():
                attempt = int(metrics.node_attempts.get(node_name, 0))
                events.append(
                    (
                        f"{run_id}-node-{node_name}",
                        run_id,
                        node_name,
                        attempt,
                        float(latency_ms or 0.0),
                        "node",
                        f"{node_name}: latency={latency_ms}ms attempts={attempt}",
                        self._dumps({"node_attempts": metrics.node_attempts}),
                        created_at + len(events) * 0.0001,
                    )
                )
        for index, message in enumerate(state.get("decision_trace", [])):
            node_name = str(message).split(":", 1)[0] if ":" in str(message) else ""
            events.append(
                (
                    f"{run_id}-trace-{index}",
                    run_id,
                    node_name,
                    index + 1,
                    0.0,
                    "routing",
                    str(message),
                    "{}",
                    created_at + len(events) * 0.0001,
                )
            )
        evaluation_history = list(state.get("evaluation_history", []))
        if report is not None and not evaluation_history:
            evaluation_history = [report]
        for index, item in enumerate(evaluation_history):
            events.append(
                (
                    f"{run_id}-evaluation-{index}",
                    run_id,
                    "evaluate_itinerary",
                    index + 1,
                    float((metrics.node_latency_ms.get("evaluate_itinerary", 0.0) if metrics else 0.0)),
                    "evaluation",
                    (
                        f"evaluation[{index + 1}]: passed={item.passed} next_action={item.next_action} "
                        f"hard_failures={','.join(item.hard_failures) if item.hard_failures else 'none'}"
                    ),
                    self._dumps(item.model_dump()),
                    created_at + len(events) * 0.0001,
                )
            )
        if metrics is not None and metrics.fallback_count:
            events.append(
                (
                    f"{run_id}-fallback",
                    run_id,
                    "fallback_response",
                    int(metrics.node_attempts.get("fallback_response", 0)),
                    float(metrics.node_latency_ms.get("fallback_response", 0.0)),
                    "fallback",
                    "fallback_response: fallback plan returned",
                    "{}",
                    created_at + len(events) * 0.0001,
                )
            )
        elif state.get("final_plan") is not None:
            events.append(
                (
                    f"{run_id}-finalize",
                    run_id,
                    "finalize_response",
                    int(metrics.node_attempts.get("finalize_response", 0)) if metrics else 0,
                    float(metrics.node_latency_ms.get("finalize_response", 0.0)) if metrics else 0.0,
                    "finalize",
                    "finalize_response: final plan available",
                    "{}",
                    created_at + len(events) * 0.0001,
                )
            )
        return events

    def _row_to_run(self, row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "run_id": row["run_id"],
            "conversation_id": row["conversation_id"],
            "profile_id": row["profile_id"],
            "source": row["source"],
            "city": row["city"],
            "travel_days": row["travel_days"],
            "rag_mode": row["rag_mode"],
            "started_at": row["started_at"],
            "ended_at": row["ended_at"],
            "end_to_end_ms": row["end_to_end_ms"],
            "passed": self._int_to_bool(row["passed"]),
            "first_evaluation_pass": self._int_to_bool(row["first_evaluation_pass"]),
            "final_evaluation_pass": self._int_to_bool(row["final_evaluation_pass"]),
            "recovered_after_retry": self._int_to_bool(row["recovered_after_retry"]),
            "fallback": self._int_to_bool(row["fallback"]),
            "evaluation_attempt_count": row["evaluation_attempt_count"],
            "hard_failures": self._loads(row["hard_failures_json"], []),
            "scores": self._loads(row["scores_json"], {}),
            "request": self._loads(row["request_json"], {}),
            "final_plan_summary": self._loads(row["final_plan_summary_json"], {}),
            "evaluation_report": self._loads(row["evaluation_report_json"], {}),
            "evaluation_history": self._loads(row["evaluation_history_json"], []),
            "retry_counts": self._loads(row["retry_counts_json"], {}),
            "retrieved_rag_sources": self._loads(row["retrieved_rag_sources_json"], []),
            "benchmark_metadata": self._loads(row["benchmark_metadata_json"], {}),
            "created_at": row["created_at"],
        }

    def _row_to_event(self, row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "event_id": row["event_id"],
            "run_id": row["run_id"],
            "node_name": row["node_name"],
            "attempt": row["attempt"],
            "latency_ms": row["latency_ms"],
            "event_type": row["event_type"],
            "message": row["message"],
            "payload": self._loads(row["payload_json"], {}),
            "created_at": row["created_at"],
        }

    def _request_json(self, request: Optional[TripRequest]) -> str:
        if request is None:
            return "{}"
        return request.model_dump_json()

    def _summarize_trip_plan(self, plan: Optional[TripPlan]) -> Dict[str, Any]:
        if plan is None:
            return {"city": "", "days": 0, "attractions": 0, "hotels": 0}
        return {
            "city": plan.city,
            "days": len(plan.days),
            "attractions": sum(len(day.attractions) for day in plan.days),
            "hotels": sum(1 for day in plan.days if day.hotel is not None),
        }

    def compact_rag_sources(self, chunks: Iterable[RAGChunk]) -> List[Dict[str, Any]]:
        """Serialize retrieved chunks into a compact source-evidence shape."""
        sources: List[Dict[str, Any]] = []
        for chunk in chunks:
            metadata = dict(chunk.metadata)
            sources.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "doc_id": metadata.get("doc_id", ""),
                    "title": chunk.title,
                    "source_url": metadata.get("source_url", ""),
                    "section": metadata.get("section", ""),
                    "sections": metadata.get("sections", []),
                    "packed_section_count": metadata.get("packed_section_count", 1),
                    "city": metadata.get("city", ""),
                    "theme": metadata.get("theme", ""),
                    "rag_backend": metadata.get("rag_backend", ""),
                }
            )
        return sources

    def _aggregate_hard_failures(self, reports: Iterable[EvaluationReport]) -> List[str]:
        failures: List[str] = []
        for report in reports:
            for failure in report.hard_failures:
                if failure not in failures:
                    failures.append(failure)
        return failures

    def _ensure_column(
        self,
        conn: sqlite3.Connection,
        table_name: str,
        column_name: str,
        column_definition: str,
    ) -> None:
        columns = [row[1] for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()]
        if column_name not in columns:
            conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_definition}")

    def _dumps(self, value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, default=str)

    def _loads(self, raw: str, fallback: Any) -> Any:
        try:
            return json.loads(raw or "")
        except json.JSONDecodeError:
            return fallback

    def _bool_to_int(self, value: Optional[bool]) -> Optional[int]:
        if value is None:
            return None
        return 1 if value else 0

    def _int_to_bool(self, value: Optional[int]) -> Optional[bool]:
        if value is None:
            return None
        return bool(value)

    def _ratio(self, numerator: int, denominator: int) -> float:
        return round(numerator / denominator, 4) if denominator else 0.0

    def _score_values(self, runs: Iterable[Dict[str, Any]], key: str) -> List[float]:
        return [
            float(run["scores"].get(key, 0.0))
            for run in runs
            if isinstance(run.get("scores"), dict)
        ]

    def _average(self, values: List[float]) -> float:
        return round(sum(values) / len(values), 4) if values else 0.0


_observability_service: Optional[ObservabilityService] = None


def get_observability_service() -> ObservabilityService:
    """Return the process-wide observability service."""
    global _observability_service
    if _observability_service is None:
        _observability_service = ObservabilityService()
    return _observability_service
