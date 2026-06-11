"""LangGraph-centered trip planner with LangChain-native active runtime."""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from uuid import uuid4

from langchain_core.output_parsers import PydanticOutputParser
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph

from ..models.langgraph_state import (
    AttractionCandidate,
    EvaluationReport,
    HotelCandidate,
    PlannerInputBundle,
    RAGChunk,
    RequestContext,
    RetryState,
    RunMetrics,
    TripGraphState,
)
from ..models.schemas import Budget, TripPlan, TripRequest, WeatherInfo
from ..prompts.attraction import build_attraction_search_terms
from ..prompts.hotel import build_hotel_search_terms
from ..prompts.planner import build_planner_prompt, build_retry_feedback
from ..services.amap_service import get_amap_service
from ..services.llm_service import get_llm
from ..services.memory_service import MemoryService, get_memory_service
from ..services.rag_service import TravelRAGService, get_rag_service
from ..services.weather_service import get_weather_service
from .trip_plan_evaluation import evaluate_trip_plan, normalize_entity_name

ALLOWED_MSGPACK_MODULES = [
    ("app.models.schemas", "TripRequest"),
    ("app.models.schemas", "TripPlan"),
    ("app.models.schemas", "WeatherInfo"),
    ("app.models.langgraph_state", "RequestContext"),
    ("app.models.langgraph_state", "AttractionCandidate"),
    ("app.models.langgraph_state", "HotelCandidate"),
    ("app.models.langgraph_state", "RAGChunk"),
    ("app.models.langgraph_state", "PlannerInputBundle"),
    ("app.models.langgraph_state", "EvaluationReport"),
    ("app.models.langgraph_state", "EvaluationScores"),
    ("app.models.langgraph_state", "EvidenceLink"),
    ("app.models.langgraph_state", "UnsupportedEntity"),
    ("app.models.langgraph_state", "RetryState"),
    ("app.models.langgraph_state", "RunMetrics"),
]


class LangGraphTripPlanner:
    """LangGraph orchestrator backed by LangChain-native runtime services."""

    def __init__(
        self,
        max_retries: int = 2,
        rag_mode: str = "local_lightweight",
        rag_service: Optional[TravelRAGService] = None,
        memory_service: Optional[MemoryService] = None,
        llm: Optional[Any] = None,
        amap_service: Optional[Any] = None,
        weather_service: Optional[Any] = None,
    ):
        self.weather_service = weather_service or get_weather_service()
        self.llm = llm or get_llm()
        self.amap_service = amap_service or get_amap_service()
        tools = {tool.name: tool for tool in self.amap_service.get_langchain_tools()}
        self.search_poi_tool = tools.get("amap_search_poi")
        if self.search_poi_tool is None:
            raise ValueError("AMap service must provide the amap_search_poi LangChain tool")
        self.max_retries = max_retries
        self.rag_mode = rag_mode
        self.rag_service = rag_service or get_rag_service()
        self.memory_service = memory_service or get_memory_service()
        self.checkpointer = MemorySaver(
            serde=JsonPlusSerializer(allowed_msgpack_modules=ALLOWED_MSGPACK_MODULES)
        )
        self.graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(TripGraphState)
        builder.add_node("prepare_request", self.prepare_request)
        builder.add_node("retrieve_attractions", self.retrieve_attractions)
        builder.add_node("retrieve_hotels", self.retrieve_hotels)
        builder.add_node("retrieve_weather", self.retrieve_weather)
        builder.add_node("retrieve_rag_context", self.retrieve_rag_context)
        builder.add_node("plan_itinerary", self.plan_itinerary)
        builder.add_node("evaluate_itinerary", self.evaluate_itinerary)
        builder.add_node("finalize_response", self.finalize_response)
        builder.add_node("fallback_response", self.fallback_response)

        builder.add_edge(START, "prepare_request")
        builder.add_edge("prepare_request", "retrieve_attractions")
        builder.add_edge("retrieve_attractions", "retrieve_hotels")
        builder.add_edge("retrieve_hotels", "retrieve_weather")
        builder.add_edge("retrieve_weather", "retrieve_rag_context")
        builder.add_edge("retrieve_rag_context", "plan_itinerary")
        builder.add_edge("plan_itinerary", "evaluate_itinerary")
        builder.add_conditional_edges(
            "evaluate_itinerary",
            self._route_after_evaluation,
            {
                "finalize_response": "finalize_response",
                "plan_itinerary": "plan_itinerary",
                "retrieve_attractions": "retrieve_attractions",
                "retrieve_hotels": "retrieve_hotels",
                "fallback_response": "fallback_response",
            },
        )
        builder.add_edge("finalize_response", END)
        builder.add_edge("fallback_response", END)
        return builder.compile(checkpointer=self.checkpointer)

    def plan_trip(self, request: TripRequest) -> TripPlan:
        """Run the graph and return the final plan."""
        state = self.plan_trip_with_state(request)
        return state["final_plan"]

    def plan_trip_with_state(self, request: TripRequest) -> TripGraphState:
        """Run the graph and return final state plus response metadata."""
        state = self.invoke_graph(request)
        final_plan = state.get("final_plan")
        if final_plan is None:
            state["final_plan"] = self._create_fallback_plan(request)
        return state

    def invoke_graph(self, request: TripRequest, thread_id: Optional[str] = None) -> TripGraphState:
        """Run the graph and return the final state for debugging/tests."""
        conversation_id = thread_id or request.conversation_id or str(uuid4())
        initial_state: TripGraphState = {
            "request": request,
            "conversation_id": conversation_id,
            "memory_applied": False,
            "memory_summary": "",
            "memory_profile": {},
        }
        config = {"configurable": {"thread_id": conversation_id}}
        return self.graph.invoke(initial_state, config=config)

    def get_state_snapshot(self, thread_id: str):
        """Return the persisted graph snapshot for a thread id."""
        config = {"configurable": {"thread_id": thread_id}}
        return self.graph.get_state(config)

    def health_summary(self) -> Dict[str, object]:
        """Expose a small health summary for the API."""
        return {
            "planner_name": self.__class__.__name__,
            "workflow": "langgraph_native",
            "checkpointer": self.checkpointer.__class__.__name__,
            "rag_mode": self.rag_mode,
            "nodes": [
                "prepare_request",
                "retrieve_attractions",
                "retrieve_hotels",
                "retrieve_weather",
                "retrieve_rag_context",
                "plan_itinerary",
                "evaluate_itinerary",
                "finalize_response",
                "fallback_response",
            ],
        }

    def prepare_request(self, state: TripGraphState) -> TripGraphState:
        start = time.perf_counter()
        request = state["request"]
        travel_dates = self._get_travel_dates(request.start_date, request.travel_days)
        memory_profile = self.memory_service.get_profile_snapshot(request.profile_id)
        memory_summary = self.memory_service.build_memory_context_from_profile(memory_profile)
        memory_applied = bool(memory_summary)
        request_context = RequestContext(
            city=request.city,
            start_date=request.start_date,
            end_date=request.end_date,
            travel_days=request.travel_days,
            transportation=request.transportation,
            accommodation=request.accommodation,
            preferences=list(request.preferences),
            free_text_input=request.free_text_input or "",
            memory_context=memory_summary,
            summary=self._build_request_summary(request),
        )
        metrics = self._record_node_metrics(state, "prepare_request", start)
        trace_message = "prepare_request: normalized request and travel dates"
        if memory_applied:
            trace_message += " with anonymous profile memory"
        trace = self._append_trace(state, trace_message)
        return {
            "travel_dates": travel_dates,
            "request_context": request_context,
            "memory_applied": memory_applied,
            "memory_summary": memory_summary,
            "memory_profile": memory_profile or {},
            "retry_counts": self._increment_retry_count(state, "prepare_request"),
            "metrics": metrics,
            "decision_trace": trace,
        }

    def retrieve_attractions(self, state: TripGraphState) -> TripGraphState:
        start = time.perf_counter()
        request = state["request"]
        retry_counts = state.get("retry_counts") or RetryState()
        candidates: List[AttractionCandidate] = []
        seen = set()
        for term in build_attraction_search_terms(request, retry_counts.retrieve_attractions):
            for item in self.search_poi_tool.invoke(
                {"keywords": term, "city": request.city, "citylimit": True, "page_size": 8}
            ):
                candidate = AttractionCandidate(
                    name=str(item.get("name", "")),
                    address=str(item.get("address", "")),
                    source="amap_http",
                    source_id=str(item.get("id", "")),
                    raw_text=json.dumps(item, ensure_ascii=False),
                )
                normalized = self._normalize_entity_name(candidate.name)
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                candidates.append(candidate)
        candidates = candidates[:8]
        metrics = self._record_node_metrics(state, "retrieve_attractions", start)
        trace = self._append_trace(
            state,
            f"retrieve_attractions: collected {len(candidates)} candidates",
        )
        return {
            "candidate_attractions": candidates,
            "retry_counts": self._increment_retry_count(state, "retrieve_attractions"),
            "metrics": metrics,
            "decision_trace": trace,
        }

    def retrieve_hotels(self, state: TripGraphState) -> TripGraphState:
        start = time.perf_counter()
        request = state["request"]
        candidates: List[HotelCandidate] = []
        seen = set()
        for term in build_hotel_search_terms(request):
            for item in self.search_poi_tool.invoke(
                {"keywords": term, "city": request.city, "citylimit": True, "page_size": 8}
            ):
                candidate = HotelCandidate(
                    name=str(item.get("name", "")),
                    address=str(item.get("address", "")),
                    source="amap_http",
                    source_id=str(item.get("id", "")),
                    raw_text=json.dumps(item, ensure_ascii=False),
                )
                normalized = self._normalize_entity_name(candidate.name)
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                candidates.append(candidate)
        candidates = candidates[:8]
        metrics = self._record_node_metrics(state, "retrieve_hotels", start)
        trace = self._append_trace(
            state,
            f"retrieve_hotels: collected {len(candidates)} candidates",
        )
        return {
            "candidate_hotels": candidates,
            "retry_counts": self._increment_retry_count(state, "retrieve_hotels"),
            "metrics": metrics,
            "decision_trace": trace,
        }

    def retrieve_weather(self, state: TripGraphState) -> TripGraphState:
        start = time.perf_counter()
        request = state["request"]
        weather_info = self.weather_service.get_weather_for_trip(
            city=request.city,
            start_date=request.start_date,
            travel_days=request.travel_days,
        )
        metrics = self._record_node_metrics(state, "retrieve_weather", start)
        trace = self._append_trace(
            state,
            f"retrieve_weather: retrieved {len(weather_info)} weather entries",
        )
        return {
            "weather_info": weather_info,
            "retry_counts": self._increment_retry_count(state, "retrieve_weather"),
            "metrics": metrics,
            "decision_trace": trace,
        }

    def retrieve_rag_context(self, state: TripGraphState) -> TripGraphState:
        start = time.perf_counter()
        chunks = self.rag_service.retrieve_chunks(
            state["request"],
            rag_mode=self.rag_mode,
            attraction_candidates=list(state.get("candidate_attractions", [])),
        )
        metrics = self._record_node_metrics(state, "retrieve_rag_context", start)
        trace = self._append_trace(
            state,
            f"retrieve_rag_context: selected {len(chunks)} local knowledge chunks",
        )
        return {
            "rag_chunks": chunks,
            "retry_counts": self._increment_retry_count(state, "retrieve_rag_context"),
            "metrics": metrics,
            "decision_trace": trace,
        }

    def retrieve_local_rag_chunks(self, request: TripRequest) -> List[RAGChunk]:
        """Return lightweight local knowledge chunks for a request."""
        return self.rag_service.retrieve_local_chunks(request)

    def plan_itinerary(self, state: TripGraphState) -> TripGraphState:
        start = time.perf_counter()
        request = state["request"]
        planner_inputs = PlannerInputBundle(
            request_context=state["request_context"],
            attraction_candidates=list(state.get("candidate_attractions", [])),
            hotel_candidates=list(state.get("candidate_hotels", [])),
            weather_info=list(state.get("weather_info", [])),
            rag_chunks=list(state.get("rag_chunks", [])),
        )
        planner_query = self._build_langgraph_planner_query(state, planner_inputs)
        raw_response, draft_plan = self._run_native_planner(planner_query, request)
        if draft_plan is not None:
            draft_plan = self._normalize_trip_plan(draft_plan, request)
            draft_plan = self._apply_authoritative_weather(
                draft_plan, list(state.get("weather_info", [])), request
            )
        metrics = self._record_node_metrics(state, "plan_itinerary", start)
        trace = self._append_trace(
            state,
            "plan_itinerary: generated draft plan"
            if draft_plan is not None
            else "plan_itinerary: failed to parse planner output",
        )
        return {
            "planner_inputs": planner_inputs,
            "draft_plan_raw": raw_response,
            "draft_plan": draft_plan,
            "retry_counts": self._increment_retry_count(state, "plan_itinerary"),
            "metrics": metrics,
            "decision_trace": trace,
        }

    def evaluate_itinerary(self, state: TripGraphState) -> TripGraphState:
        start = time.perf_counter()
        report = self._evaluate_plan(state)
        metrics = self._record_node_metrics(state, "evaluate_itinerary", start)
        metrics.evaluation_attempt_count += 1
        if metrics.first_evaluation_pass is None:
            metrics.first_evaluation_pass = report.passed
        metrics.final_evaluation_pass = report.passed
        self._apply_evaluation_metrics(metrics, report)
        trace = self._append_trace(
            state,
            (
                f"evaluate_itinerary: next_action={report.next_action} passed={report.passed}"
                f" hard_failures={','.join(report.hard_failures) if report.hard_failures else 'none'}"
            ),
        )
        history = list(state.get("evaluation_history", []))
        history.append(report)
        return {
            "evaluation_report": report,
            "evaluation_history": history,
            "retry_counts": self._increment_retry_count(state, "evaluate_itinerary"),
            "metrics": metrics,
            "decision_trace": trace,
        }

    def finalize_response(self, state: TripGraphState) -> TripGraphState:
        start = time.perf_counter()
        metrics = self._record_node_metrics(state, "finalize_response", start)
        metrics.evaluation_pass_count += 1
        metrics.final_evaluation_pass = True
        metrics.recovered_after_retry = bool(
            metrics.first_evaluation_pass is False and metrics.evaluation_pass_count > 0
        )
        metrics.ended_at = time.time()
        metrics.end_to_end_ms = max(0.0, (metrics.ended_at - metrics.started_at) * 1000.0)
        trace = self._append_trace(state, "finalize_response: finalized validated trip plan")
        final_plan = state.get("draft_plan")
        if final_plan is not None:
            trace = self._persist_memory_after_success(state, final_plan, trace)
        return {
            "final_plan": final_plan,
            "retry_counts": self._increment_retry_count(state, "finalize_response"),
            "metrics": metrics,
            "decision_trace": trace,
        }

    def fallback_response(self, state: TripGraphState) -> TripGraphState:
        start = time.perf_counter()
        metrics = self._record_node_metrics(state, "fallback_response", start)
        metrics.fallback_count += 1
        metrics.final_evaluation_pass = False
        metrics.recovered_after_retry = False
        metrics.ended_at = time.time()
        metrics.end_to_end_ms = max(0.0, (metrics.ended_at - metrics.started_at) * 1000.0)
        trace = self._append_trace(state, "fallback_response: returned fallback plan")
        fallback = self._create_fallback_plan(state["request"])
        fallback = self._apply_authoritative_weather(
            fallback, list(state.get("weather_info", [])), state["request"]
        )
        return {
            "final_plan": fallback,
            "retry_counts": self._increment_retry_count(state, "fallback_response"),
            "metrics": metrics,
            "decision_trace": trace,
        }

    def _route_after_evaluation(self, state: TripGraphState) -> str:
        report = state.get("evaluation_report")
        if report is None:
            return "fallback_response"
        return report.next_action

    def _evaluate_plan(self, state: TripGraphState) -> EvaluationReport:
        return evaluate_trip_plan(
            request=state["request"],
            travel_dates=list(state.get("travel_dates", [])),
            draft_plan=state.get("draft_plan"),
            candidate_attractions=list(state.get("candidate_attractions", [])),
            candidate_hotels=list(state.get("candidate_hotels", [])),
            rag_chunks=list(state.get("rag_chunks", [])),
            retry_counts=state.get("retry_counts") or RetryState(),
            max_retries=self.max_retries,
        )

    def _next_action_with_retry_budget(self, retry_counts: RetryState, action: str) -> str:
        attempts = getattr(retry_counts, action, 0)
        if attempts >= self.max_retries + 1:
            return "fallback_response"
        return action

    def _apply_evaluation_metrics(self, metrics: RunMetrics, report: EvaluationReport) -> None:
        if "schema_correctness" in report.hard_failures:
            metrics.schema_failure_count += 1
        if "date_coverage" in report.hard_failures:
            metrics.date_coverage_failure_count += 1
        if "budget_consistency" in report.hard_failures:
            metrics.budget_consistency_failure_count += 1
        if any(failure.startswith("retrieval_grounding") for failure in report.hard_failures):
            metrics.grounding_failure_count += 1

    def _persist_memory_after_success(
        self, state: TripGraphState, final_plan: TripPlan, trace: List[str]
    ) -> List[str]:
        request = state["request"]
        if not request.profile_id:
            return trace
        try:
            self.memory_service.update_after_success(
                profile_id=request.profile_id,
                conversation_id=state.get("conversation_id") or request.conversation_id or str(uuid4()),
                request=request,
                plan=final_plan,
                memory_applied=bool(state.get("memory_applied")),
                memory_summary=state.get("memory_summary", ""),
            )
            return trace + ["memory: stored anonymous profile/session memory"]
        except Exception as exc:
            return trace + [f"memory: skipped persistence due to {exc.__class__.__name__}"]

    def _build_langgraph_planner_query(
        self, state: TripGraphState, planner_inputs: PlannerInputBundle
    ) -> str:
        request = state["request"]
        attractions_text = self._serialize_attraction_candidates(planner_inputs.attraction_candidates)
        hotels_text = self._serialize_hotel_candidates(planner_inputs.hotel_candidates)
        weather_text = self.weather_service.format_weather_for_planner(
            request.city, planner_inputs.weather_info
        )
        rag_text = self._serialize_rag_chunks(planner_inputs.rag_chunks)
        retry_feedback = build_retry_feedback(
            state.get("evaluation_report"),
            planner_inputs,
            list(state.get("travel_dates", [])),
        )
        return build_planner_prompt(
            request=request,
            attractions_text=attractions_text,
            weather_text=weather_text,
            hotels_text=hotels_text,
            rag_text=rag_text,
            retry_feedback=retry_feedback,
            memory_context=state.get("memory_summary", ""),
        )

    def _run_native_planner(self, prompt: str, request: TripRequest) -> tuple[str, Optional[TripPlan]]:
        parser = PydanticOutputParser(pydantic_object=TripPlan)
        prompt_with_format = f"{prompt}\n\n{parser.get_format_instructions()}"
        response = self.llm.invoke(prompt_with_format)
        raw_response = self._message_to_text(response)
        draft_plan: Optional[TripPlan]
        try:
            draft_plan = parser.parse(raw_response)
        except Exception:
            draft_plan = self._parse_planner_response_strict(raw_response, request)
        return raw_response, draft_plan

    def _message_to_text(self, response) -> str:
        content = getattr(response, "content", response)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(str(item))
            return "\n".join(part for part in parts if part)
        return str(content)

    def _parse_planner_response_strict(self, response: str, request: TripRequest) -> Optional[TripPlan]:
        json_str = self._extract_json_string(response)
        if not json_str:
            return None
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            return None
        try:
            return TripPlan(**data)
        except Exception:
            return None

    def _extract_json_string(self, response: str) -> str:
        if "```json" in response:
            json_start = response.find("```json") + 7
            json_end = response.find("```", json_start)
            return response[json_start:json_end].strip()
        if "```" in response:
            json_start = response.find("```") + 3
            json_end = response.find("```", json_start)
            return response[json_start:json_end].strip()
        if "{" in response and "}" in response:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            return response[json_start:json_end]
        return ""

    def _serialize_attraction_candidates(self, candidates: List[AttractionCandidate]) -> str:
        if not candidates:
            return "暂无景点候选，请根据用户偏好谨慎规划并标记信息不完整。"
        return "\n".join(
            f"- {candidate.name} | 地址: {candidate.address or '未知'}"
            for candidate in candidates
        )

    def _serialize_hotel_candidates(self, candidates: List[HotelCandidate]) -> str:
        if not candidates:
            return "暂无酒店候选，请优先推荐交通便利的住宿区域。"
        return "\n".join(
            f"- {candidate.name} | 地址: {candidate.address or '未知'}"
            for candidate in candidates
        )

    def _serialize_rag_chunks(self, chunks: List[RAGChunk]) -> str:
        return "\n".join(f"- {chunk.title}: {chunk.content}" for chunk in chunks)

    def _build_request_summary(self, request: TripRequest) -> str:
        preferences = "、".join(request.preferences) if request.preferences else "无明确偏好"
        free_text = request.free_text_input or "无"
        return (
            f"{request.city} {request.travel_days}天行程，交通方式为{request.transportation}，"
            f"住宿偏好为{request.accommodation}，偏好包括{preferences}，额外要求：{free_text}。"
        )

    def _get_travel_dates(self, start_date: str, travel_days: int) -> List[str]:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        return [(start + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(travel_days)]

    def _record_node_metrics(self, state: TripGraphState, node_name: str, started_at: float) -> RunMetrics:
        metrics = state.get("metrics")
        if metrics is None:
            metrics = RunMetrics(started_at=time.time())
        else:
            metrics = metrics.model_copy(deep=True)
            if metrics.started_at == 0.0:
                metrics.started_at = time.time()
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        metrics.node_latency_ms[node_name] = round(elapsed_ms, 3)
        metrics.node_attempts[node_name] = metrics.node_attempts.get(node_name, 0) + 1
        return metrics

    def _increment_retry_count(self, state: TripGraphState, node_name: str) -> RetryState:
        retry_counts = state.get("retry_counts")
        if retry_counts is None:
            retry_counts = RetryState()
        else:
            retry_counts = retry_counts.model_copy(deep=True)
        current = getattr(retry_counts, node_name, 0)
        setattr(retry_counts, node_name, current + 1)
        return retry_counts

    def _append_trace(self, state: TripGraphState, message: str) -> List[str]:
        trace = list(state.get("decision_trace", []))
        trace.append(message)
        return trace

    def _normalize_entity_name(self, value: str) -> str:
        return normalize_entity_name(value)

    def _normalize_trip_plan(self, trip_plan: TripPlan, request: TripRequest) -> TripPlan:
        travel_dates = self._get_travel_dates(request.start_date, request.travel_days)
        raw_weather = [w for w in trip_plan.weather_info if w.date]
        weather_by_date = {w.date: w for w in raw_weather}

        def copy_weather(src: WeatherInfo, target_date: str) -> WeatherInfo:
            return WeatherInfo(
                date=target_date,
                day_weather=src.day_weather,
                night_weather=src.night_weather,
                day_temp=src.day_temp,
                night_temp=src.night_temp,
                wind_direction=src.wind_direction,
                wind_power=src.wind_power,
            )

        normalized_weather: List[WeatherInfo] = []
        missing_indices: List[int] = []
        has_exact_match = False

        for current_date in travel_dates:
            if current_date in weather_by_date:
                has_exact_match = True
                normalized_weather.append(copy_weather(weather_by_date[current_date], current_date))
            else:
                normalized_weather.append(
                    WeatherInfo(
                        date=current_date,
                        day_weather="未知",
                        night_weather="未知",
                        day_temp=0,
                        night_temp=0,
                        wind_direction="",
                        wind_power="",
                    )
                )
                missing_indices.append(len(normalized_weather) - 1)

        if missing_indices and raw_weather and not has_exact_match:
            for index, missing_index in enumerate(missing_indices):
                if index >= len(raw_weather):
                    break
                normalized_weather[missing_index] = copy_weather(raw_weather[index], normalized_weather[missing_index].date)

        trip_plan.city = request.city
        trip_plan.start_date = request.start_date
        trip_plan.end_date = request.end_date
        trip_plan.weather_info = normalized_weather
        for day in trip_plan.days:
            day.transportation = request.transportation
            day.accommodation = request.accommodation
        trip_plan.budget = self._normalize_budget(trip_plan)
        return trip_plan

    def _normalize_budget(self, trip_plan: TripPlan) -> Budget:
        """Make budget totals internally consistent after model generation."""
        total_attractions = sum(
            (attraction.ticket_price or 0)
            for day in trip_plan.days
            for attraction in day.attractions
        )
        total_hotels = sum(
            ((day.hotel.estimated_cost or 0) if day.hotel is not None else 0)
            for day in trip_plan.days
        )
        total_meals = sum(
            (meal.estimated_cost or 0)
            for day in trip_plan.days
            for meal in day.meals
        )
        total_transportation = (
            trip_plan.budget.total_transportation
            if trip_plan.budget is not None
            else 0
        )
        return Budget(
            total_attractions=total_attractions,
            total_hotels=total_hotels,
            total_meals=total_meals,
            total_transportation=total_transportation,
            total=total_attractions + total_hotels + total_meals + total_transportation,
        )

    def _apply_authoritative_weather(
        self, trip_plan: TripPlan, weather_info: List[WeatherInfo], request: TripRequest
    ) -> TripPlan:
        travel_dates = self._get_travel_dates(request.start_date, request.travel_days)
        weather_by_date = {w.date: w for w in weather_info if w.date}

        aligned: List[WeatherInfo] = []
        missing = False
        for current_date in travel_dates:
            if current_date in weather_by_date:
                aligned.append(weather_by_date[current_date])
            else:
                missing = True
                aligned.append(
                    WeatherInfo(
                        date=current_date,
                        day_weather="未知",
                        night_weather="未知",
                        day_temp=0,
                        night_temp=0,
                        wind_direction="",
                        wind_power="",
                    )
                )
        trip_plan.weather_info = aligned
        if missing:
            note = "部分日期天气数据不可用，已按行程日期补齐为“未知”，请出发前再次确认实时天气。"
            if note not in trip_plan.overall_suggestions:
                trip_plan.overall_suggestions = f"{trip_plan.overall_suggestions}\n{note}".strip()
        return trip_plan

    def _create_fallback_plan(self, request: TripRequest) -> TripPlan:
        start_date = datetime.strptime(request.start_date, "%Y-%m-%d")
        days = []
        for index in range(request.travel_days):
            current_date = start_date + timedelta(days=index)
            days.append(
                {
                    "date": current_date.strftime("%Y-%m-%d"),
                    "day_index": index,
                    "description": f"第{index + 1}天行程",
                    "transportation": request.transportation,
                    "accommodation": request.accommodation,
                    "attractions": [],
                    "meals": [],
                }
            )
        return TripPlan(
            city=request.city,
            start_date=request.start_date,
            end_date=request.end_date,
            days=days,
            weather_info=[],
            overall_suggestions=f"这是为您规划的{request.city}{request.travel_days}日游行程,建议提前查看各景点的开放时间。",
            budget=None,
        )


_langgraph_trip_planner: Optional[LangGraphTripPlanner] = None


def get_trip_planner_agent() -> LangGraphTripPlanner:
    """Return a singleton LangGraph planner."""
    global _langgraph_trip_planner
    if _langgraph_trip_planner is None:
        _langgraph_trip_planner = LangGraphTripPlanner()
    return _langgraph_trip_planner
