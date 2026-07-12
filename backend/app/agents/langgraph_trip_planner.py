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
    MealCandidate,
    PlannerInputBundle,
    RAGChunk,
    RequestContext,
    RetryState,
    RouteTimeEstimate,
    RunMetrics,
    TripGraphState,
)
from ..models.schemas import Budget, TripPlan, TripRequest, WeatherInfo
from ..prompts.attraction import build_attraction_search_terms
from ..prompts.hotel import build_hotel_search_terms
from ..prompts.planner import build_planner_prompt, build_retry_feedback
from ..config import get_settings
from ..services.llm_service import get_llm
from ..services.map_service import get_map_service
from ..services.memory_service import MemoryService, get_memory_service
from ..services.rag_service import TravelRAGService, get_rag_service
from ..services.weather_service import get_weather_service
from .trip_plan_evaluation import evaluate_trip_plan, normalize_entity_name, route_type_for_transportation

NON_ATTRACTION_PLACE_TYPES = {
    "car_rental",
    "consultant",
    "corporate_office",
    "finance",
    "insurance_agency",
    "lodging",
    "real_estate_agency",
    "service",
    "tour_agency",
    "tourist_information_center",
    "travel_agency",
}

NON_ATTRACTION_NAME_TERMS = {
    "agency",
    "concierge",
    "tour operator",
    "tours of",
    "travel agency",
    "travel boutique",
    "travel inc",
    "travel, inc",
}

MEAL_PREFERENCE_HINTS = {
    "food",
    "restaurant",
    "restaurants",
    "cafe",
    "coffee",
    "dining",
    "dinner",
    "lunch",
    "breakfast",
    "street food",
    "food hall",
    "market",
}

ATTRACTION_PLACE_TYPE_HINTS = {
    "amusement_park",
    "aquarium",
    "art_gallery",
    "botanical_garden",
    "historical_landmark",
    "museum",
    "national_park",
    "park",
    "performing_arts_theater",
    "tourist_attraction",
    "visitor_center",
    "zoo",
}

ALLOWED_MSGPACK_MODULES = [
    ("app.models.schemas", "TripRequest"),
    ("app.models.schemas", "TripPlan"),
    ("app.models.schemas", "WeatherInfo"),
    ("app.models.langgraph_state", "RequestContext"),
    ("app.models.langgraph_state", "AttractionCandidate"),
    ("app.models.langgraph_state", "HotelCandidate"),
    ("app.models.langgraph_state", "MealCandidate"),
    ("app.models.langgraph_state", "RouteTimeEstimate"),
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
        map_service: Optional[Any] = None,
        weather_service: Optional[Any] = None,
    ):
        self.weather_service = weather_service or get_weather_service()
        self.llm = llm or get_llm()
        self.map_service = map_service or get_map_service()
        tools = {tool.name: tool for tool in self.map_service.get_langchain_tools()}
        self.search_poi_tool = tools.get("map_search_poi")
        if self.search_poi_tool is None:
            raise ValueError("Map service must provide the map_search_poi LangChain tool")
        self.max_retries = max_retries
        self.rag_mode = rag_mode
        self.rag_service = rag_service or get_rag_service()
        self.memory_service = memory_service or get_memory_service()
        self.parallel_retrieval_enabled = True
        self.checkpointer = MemorySaver(
            serde=JsonPlusSerializer(allowed_msgpack_modules=ALLOWED_MSGPACK_MODULES)
        )
        self.graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(TripGraphState)
        builder.add_node("prepare_request", self.prepare_request)
        builder.add_node("retrieve_attractions", self.retrieve_attractions)
        builder.add_node("retrieve_hotels", self.retrieve_hotels)
        builder.add_node("retrieve_meals", self.retrieve_meals)
        builder.add_node("retrieve_weather", self.retrieve_weather)
        builder.add_node("retrieve_rag_context", self.retrieve_rag_context)
        builder.add_node("retry_retrieve_attractions", self.retrieve_attractions)
        builder.add_node("retry_retrieve_hotels", self.retrieve_hotels)
        builder.add_node("retry_retrieve_meals", self.retrieve_meals)
        builder.add_node("retry_retrieve_rag_context", self.retrieve_rag_context)
        builder.add_node("plan_itinerary", self.plan_itinerary)
        builder.add_node("collect_route_times", self.collect_route_times)
        builder.add_node("evaluate_itinerary", self.evaluate_itinerary)
        builder.add_node("finalize_response", self.finalize_response)
        builder.add_node("fallback_response", self.fallback_response)

        builder.add_edge(START, "prepare_request")
        builder.add_edge("prepare_request", "retrieve_attractions")
        builder.add_edge("prepare_request", "retrieve_hotels")
        builder.add_edge("prepare_request", "retrieve_meals")
        builder.add_edge("prepare_request", "retrieve_weather")
        builder.add_edge("retrieve_attractions", "retrieve_rag_context")
        builder.add_edge(
            ["retrieve_hotels", "retrieve_meals", "retrieve_weather", "retrieve_rag_context"],
            "plan_itinerary",
        )
        builder.add_edge("plan_itinerary", "collect_route_times")
        builder.add_edge("collect_route_times", "evaluate_itinerary")
        builder.add_conditional_edges(
            "evaluate_itinerary",
            self._route_after_evaluation,
            {
                "finalize_response": "finalize_response",
                "plan_itinerary": "plan_itinerary",
                "retrieve_attractions": "retry_retrieve_attractions",
                "retrieve_hotels": "retry_retrieve_hotels",
                "retrieve_meals": "retry_retrieve_meals",
                "fallback_response": "fallback_response",
            },
        )
        builder.add_edge("retry_retrieve_attractions", "retry_retrieve_rag_context")
        builder.add_edge("retry_retrieve_rag_context", "plan_itinerary")
        builder.add_edge("retry_retrieve_hotels", "plan_itinerary")
        builder.add_edge("retry_retrieve_meals", "plan_itinerary")
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
        run_id = thread_id or str(uuid4())
        conversation_id = request.conversation_id or str(uuid4())
        initial_state: TripGraphState = {
            "request": request,
            "run_id": run_id,
            "conversation_id": conversation_id,
            "memory_applied": False,
            "memory_summary": "",
            "memory_profile": {},
            "memory_conflicts": [],
        }
        config = {"configurable": {"thread_id": run_id}}
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
            "parallel_retrieval_enabled": self.parallel_retrieval_enabled,
            "nodes": [
                "prepare_request",
                "retrieve_attractions",
                "retrieve_hotels",
                "retrieve_meals",
                "retrieve_weather",
                "retrieve_rag_context",
                "retry_retrieve_attractions",
                "retry_retrieve_hotels",
                "retry_retrieve_meals",
                "retry_retrieve_rag_context",
                "plan_itinerary",
                "collect_route_times",
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
        memory_summary, memory_conflicts = self.memory_service.build_memory_context_for_request(
            memory_profile,
            request,
        )
        memory_applied = bool(memory_summary)
        request_context = RequestContext(
            city=request.city,
            start_date=request.start_date,
            end_date=request.end_date,
            travel_days=request.travel_days,
            transportation=request.transportation,
            accommodation=request.accommodation,
            country_code=request.country_code,
            preferences=list(request.preferences),
            free_text_input=request.free_text_input or "",
            memory_context=memory_summary,
            summary=self._build_request_summary(request),
        )
        metrics = self._record_node_metrics(state, "prepare_request", start)
        trace_message = "prepare_request: normalized request and travel dates"
        if memory_applied:
            trace_message += " with anonymous profile memory"
        if memory_conflicts:
            trace_message += f" and {len(memory_conflicts)} memory conflict(s) resolved by current request"
        trace = self._append_trace(state, trace_message)
        return {
            "travel_dates": travel_dates,
            "request_context": request_context,
            "memory_applied": memory_applied,
            "memory_summary": memory_summary,
            "memory_profile": memory_profile or {},
            "memory_conflicts": memory_conflicts,
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
                {
                    "keywords": term,
                    "city": request.city,
                    "citylimit": True,
                    "page_size": 8,
                    "country_code": request.country_code,
                }
            ):
                if self._is_non_attraction_poi(item, request.city):
                    continue
                candidate = AttractionCandidate(
                    name=str(item.get("name", "")),
                    address=str(item.get("address", "")),
                    location=item.get("location") or None,
                    source="google_maps",
                    source_id=str(item.get("id", "")),
                    raw_text=json.dumps(item, ensure_ascii=False),
                    rating=item.get("rating"),
                    maps_url=item.get("maps_url"),
                    website_url=item.get("website_url"),
                    image_url=item.get("image_url"),
                    photo_names=list(item.get("photo_names") or []),
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
                {
                    "keywords": term,
                    "city": request.city,
                    "citylimit": True,
                    "page_size": 8,
                    "country_code": request.country_code,
                }
            ):
                candidate = HotelCandidate(
                    name=str(item.get("name", "")),
                    address=str(item.get("address", "")),
                    location=item.get("location") or None,
                    source="google_maps",
                    source_id=str(item.get("id", "")),
                    raw_text=json.dumps(item, ensure_ascii=False),
                    rating=item.get("rating"),
                    maps_url=item.get("maps_url"),
                    website_url=item.get("website_url"),
                    image_url=item.get("image_url"),
                    photo_names=list(item.get("photo_names") or []),
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

    def retrieve_meals(self, state: TripGraphState) -> TripGraphState:
        start = time.perf_counter()
        request = state["request"]
        candidates: List[MealCandidate] = []
        seen = set()
        for term in self._build_meal_search_terms(request):
            for item in self.search_poi_tool.invoke(
                {
                    "keywords": term,
                    "city": request.city,
                    "citylimit": True,
                    "page_size": 5,
                    "country_code": request.country_code,
                }
            ):
                candidate = MealCandidate(
                    name=str(item.get("name", "")),
                    address=str(item.get("address", "")),
                    location=item.get("location") or None,
                    source="google_maps",
                    source_id=str(item.get("id", "")),
                    raw_text=json.dumps(item, ensure_ascii=False),
                    rating=item.get("rating"),
                    maps_url=item.get("maps_url"),
                    website_url=item.get("website_url"),
                    image_url=item.get("image_url"),
                    photo_names=list(item.get("photo_names") or []),
                )
                normalized = self._normalize_entity_name(candidate.name)
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                candidates.append(candidate)
        candidates = candidates[:6]
        metrics = self._record_node_metrics(state, "retrieve_meals", start)
        trace = self._append_trace(
            state,
            f"retrieve_meals: collected {len(candidates)} candidates",
        )
        return {
            "candidate_meals": candidates,
            "retry_counts": self._increment_retry_count(state, "retrieve_meals"),
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
            meal_candidates=list(state.get("candidate_meals", [])),
            weather_info=list(state.get("weather_info", [])),
            rag_chunks=list(state.get("rag_chunks", [])),
        )
        planner_query = self._build_langgraph_planner_query(state, planner_inputs)
        raw_response, draft_plan = self._run_native_planner(planner_query, request)
        if draft_plan is not None:
            draft_plan = self._normalize_trip_plan(draft_plan, request)
            draft_plan = self._enrich_trip_plan_with_place_metadata(draft_plan, state)
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

    def collect_route_times(self, state: TripGraphState) -> TripGraphState:
        start = time.perf_counter()
        settings = get_settings()
        estimates: List[RouteTimeEstimate] = []
        if settings.route_time_evaluation_enabled and state.get("draft_plan") is not None:
            estimates = self._collect_route_time_estimates(
                state["request"],
                state["draft_plan"],
                max_calls=max(0, settings.max_route_time_evaluations_per_trip),
            )
        metrics = self._record_node_metrics(state, "collect_route_times", start)
        trace = self._append_trace(
            state,
            (
                f"collect_route_times: collected {len(estimates)} estimates"
                if settings.route_time_evaluation_enabled
                else "collect_route_times: disabled"
            ),
        )
        return {
            "route_time_estimates": estimates,
            "retry_counts": self._increment_retry_count(state, "collect_route_times"),
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
                f" warnings={','.join(report.warnings) if report.warnings else 'none'}"
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
        settings = get_settings()
        return evaluate_trip_plan(
            request=state["request"],
            travel_dates=list(state.get("travel_dates", [])),
            draft_plan=state.get("draft_plan"),
            candidate_attractions=list(state.get("candidate_attractions", [])),
            candidate_hotels=list(state.get("candidate_hotels", [])),
            candidate_meals=list(state.get("candidate_meals", [])),
            route_time_estimates=list(state.get("route_time_estimates", [])),
            route_time_evaluation_enabled=settings.route_time_evaluation_enabled,
            max_segment_minutes_by_mode=settings.max_segment_minutes_by_mode,
            max_daily_transit_minutes_by_mode=settings.max_daily_transit_minutes_by_mode,
            rag_chunks=list(state.get("rag_chunks", [])),
            retry_counts=state.get("retry_counts") or RetryState(),
            max_retries=self.max_retries,
            quality_retry_enabled=settings.quality_retry_enabled,
            min_pacing_score=settings.min_pacing_score,
            min_route_coherence_score=settings.min_route_coherence_score,
            min_preference_match_score=settings.min_preference_match_score,
        )

    def _next_action_with_retry_budget(self, retry_counts: RetryState, action: str) -> str:
        attempts = getattr(retry_counts, action, 0)
        if attempts >= self.max_retries + 1:
            return "fallback_response"
        return action

    def _collect_route_time_estimates(
        self,
        request: TripRequest,
        draft_plan: TripPlan,
        max_calls: int,
    ) -> List[RouteTimeEstimate]:
        estimates: List[RouteTimeEstimate] = []
        calls_made = 0
        for day in draft_plan.days:
            if len(day.attractions) <= 1:
                continue
            route_type = route_type_for_transportation(day.transportation or request.transportation)
            for index in range(len(day.attractions) - 1):
                origin = day.attractions[index]
                destination = day.attractions[index + 1]
                base = {
                    "day_index": day.day_index,
                    "segment_index": index,
                    "from_name": origin.name,
                    "to_name": destination.name,
                    "route_type": route_type,
                }
                if calls_made >= max_calls:
                    estimates.append(
                        RouteTimeEstimate(
                            **base,
                            source="not_called",
                            fallback_reason="route_time_call_cap_reached",
                        )
                    )
                    continue
                calls_made += 1
                try:
                    route = self.map_service.plan_route(
                        origin_address=origin.address or origin.name,
                        destination_address=destination.address or destination.name,
                        origin_city=request.city,
                        destination_city=request.city,
                        route_type=route_type,
                    )
                    if not route:
                        estimates.append(
                            RouteTimeEstimate(
                                **base,
                                fallback_reason="empty_route_response",
                            )
                        )
                        continue
                    duration_seconds = float(route.get("duration") or 0)
                    duration_minutes = round(duration_seconds / 60.0, 2) if duration_seconds > 0 else None
                    estimates.append(
                        RouteTimeEstimate(
                            **base,
                            duration_minutes=duration_minutes,
                            distance_meters=(
                                float(route.get("distance"))
                                if route.get("distance") is not None
                                else None
                            ),
                            source=str(route.get("source") or "map_provider"),
                            fallback_reason="" if duration_minutes else "missing_duration",
                        )
                    )
                except Exception as exc:
                    estimates.append(
                        RouteTimeEstimate(
                            **base,
                            error=exc.__class__.__name__,
                            fallback_reason="provider_error",
                        )
                    )
        return estimates

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
        meals_text = self._serialize_meal_candidates(planner_inputs.meal_candidates)
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
            meals_text=meals_text,
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
            return "No attraction candidates were retrieved. Plan conservatively and mark information as incomplete."
        return "\n".join(
            f"- {candidate.name} | Address: {candidate.address or 'unknown'}"
            for candidate in candidates
        )

    def _serialize_hotel_candidates(self, candidates: List[HotelCandidate]) -> str:
        if not candidates:
            return "No hotel candidates were retrieved. Prefer a convenient neighborhood instead of inventing a hotel."
        return "\n".join(
            f"- {candidate.name} | Address: {candidate.address or 'unknown'}"
            for candidate in candidates
        )

    def _serialize_meal_candidates(self, candidates: List[MealCandidate]) -> str:
        if not candidates:
            return "No restaurant candidates were retrieved. Use generic meal suggestions instead of inventing named restaurants."
        return "\n".join(
            f"- {candidate.name} | Address: {candidate.address or 'unknown'}"
            for candidate in candidates
        )

    def _build_meal_search_terms(self, request: TripRequest) -> List[str]:
        terms: List[str] = []
        normalized_preferences = [str(preference).strip().lower() for preference in request.preferences]
        for preference in normalized_preferences:
            if any(hint in preference for hint in MEAL_PREFERENCE_HINTS):
                terms.append(f"{preference} restaurants")
                break
        terms.extend(["restaurants", "local dining"])
        deduped: List[str] = []
        for term in terms:
            normalized = term.strip().lower()
            if normalized and normalized not in deduped:
                deduped.append(normalized)
        return deduped[:3]

    def _serialize_rag_chunks(self, chunks: List[RAGChunk]) -> str:
        return "\n".join(f"- {chunk.title}: {chunk.content}" for chunk in chunks)

    def _build_request_summary(self, request: TripRequest) -> str:
        preferences = ", ".join(request.preferences) if request.preferences else "no explicit preferences"
        free_text = request.free_text_input or "none"
        return (
            f"{request.travel_days}-day trip to {request.city}; transportation={request.transportation}; "
            f"accommodation={request.accommodation}; preferences={preferences}; extra requirements={free_text}."
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
        metrics.node_total_latency_ms[node_name] = round(
            metrics.node_total_latency_ms.get(node_name, 0.0) + elapsed_ms,
            3,
        )
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
        return [message]

    def _normalize_entity_name(self, value: str) -> str:
        return normalize_entity_name(value)

    def _is_non_attraction_poi(self, item: Dict[str, Any], city: str = "") -> bool:
        """Filter map-provider service businesses before planner grounding."""
        name = str(item.get("name", "")).lower()
        address = str(item.get("address", "")).lower()
        raw = item.get("raw") if isinstance(item.get("raw"), dict) else {}
        raw_types = raw.get("types") if isinstance(raw, dict) else []
        type_values = []
        if item.get("type"):
            type_values.extend(str(item.get("type", "")).lower().replace(",", " ").split())
        if isinstance(raw_types, list):
            type_values.extend(str(value).lower() for value in raw_types)
        type_set = set(type_values)

        if self._is_outside_requested_city(address, city):
            return True
        if type_set & ATTRACTION_PLACE_TYPE_HINTS:
            return False
        if type_set & NON_ATTRACTION_PLACE_TYPES:
            return True
        return any(term in name for term in NON_ATTRACTION_NAME_TERMS)

    def _is_outside_requested_city(self, address: str, city: str) -> bool:
        """Catch broad Google text-search results that are outside the requested city."""
        normalized_city = self._normalize_entity_name(city)
        if not address or not normalized_city:
            return False
        if normalized_city == "newyork":
            nyc_aliases = (
                "new york, ny",
                "manhattan, ny",
                "brooklyn, ny",
                "queens, ny",
                "bronx, ny",
                "staten island, ny",
            )
            return ", ny" in address and not any(alias in address for alias in nyc_aliases)
        return city.lower() not in address and ", " in address

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
                        day_weather="Unknown",
                        night_weather="Unknown",
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

    def _enrich_trip_plan_with_place_metadata(self, trip_plan: TripPlan, state: TripGraphState) -> TripPlan:
        attraction_metadata = self._candidate_metadata_by_name(
            list(state.get("candidate_attractions", []))
        )
        hotel_metadata = self._candidate_metadata_by_name(
            list(state.get("candidate_hotels", []))
        )
        meal_metadata = self._candidate_metadata_by_name(
            list(state.get("candidate_meals", []))
        )

        for day in trip_plan.days:
            for attraction in day.attractions:
                metadata = attraction_metadata.get(self._normalize_entity_name(attraction.name))
                if not metadata:
                    continue
                attraction.poi_id = attraction.poi_id or str(metadata.get("id") or "")
                attraction.maps_url = attraction.maps_url or metadata.get("maps_url")
                attraction.website_url = attraction.website_url or metadata.get("website_url")
                attraction.image_url = attraction.image_url or metadata.get("image_url")
                if not attraction.photos and metadata.get("image_url"):
                    attraction.photos = [metadata["image_url"]]
                if attraction.rating is None and metadata.get("rating") is not None:
                    attraction.rating = metadata.get("rating")

            if day.hotel:
                metadata = hotel_metadata.get(self._normalize_entity_name(day.hotel.name))
                if metadata:
                    day.hotel.poi_id = day.hotel.poi_id or str(metadata.get("id") or "")
                    day.hotel.maps_url = day.hotel.maps_url or metadata.get("maps_url")
                    day.hotel.website_url = day.hotel.website_url or metadata.get("website_url")
                    day.hotel.image_url = day.hotel.image_url or metadata.get("image_url")
                    if not day.hotel.rating and metadata.get("rating") is not None:
                        day.hotel.rating = str(metadata.get("rating"))
            for meal in day.meals:
                metadata = meal_metadata.get(self._normalize_entity_name(meal.name))
                if not metadata:
                    continue
                meal.poi_id = meal.poi_id or str(metadata.get("id") or "")
                meal.maps_url = meal.maps_url or metadata.get("maps_url")
                meal.website_url = meal.website_url or metadata.get("website_url")
                meal.image_url = meal.image_url or metadata.get("image_url")
                if not meal.address and metadata.get("address"):
                    meal.address = metadata.get("address")
        return trip_plan

    def _candidate_metadata_by_name(self, candidates: List[Any]) -> Dict[str, Dict[str, Any]]:
        metadata_by_name: Dict[str, Dict[str, Any]] = {}
        for candidate in candidates:
            normalized = self._normalize_entity_name(getattr(candidate, "name", ""))
            if not normalized:
                continue
            metadata: Dict[str, Any] = {
                "id": getattr(candidate, "source_id", ""),
                "name": getattr(candidate, "name", ""),
                "address": getattr(candidate, "address", ""),
            }
            raw_text = getattr(candidate, "raw_text", "")
            if raw_text:
                try:
                    raw = json.loads(raw_text)
                    if isinstance(raw, dict):
                        metadata.update(raw)
                except json.JSONDecodeError:
                    pass
            metadata_by_name[normalized] = metadata
        return metadata_by_name

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
                        day_weather="Unknown",
                        night_weather="Unknown",
                        day_temp=0,
                        night_temp=0,
                        wind_direction="",
                        wind_power="",
                    )
                )
        trip_plan.weather_info = aligned
        if missing:
            note = "Some weather data was unavailable and was filled as Unknown. Please verify real-time weather before departure."
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
                    "description": f"Day {index + 1} itinerary",
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
            overall_suggestions=(
                f"This is a fallback {request.travel_days}-day itinerary shell for {request.city}. "
                "Please verify attraction opening hours before departure."
            ),
            budget=None,
        )


_langgraph_trip_planner: Optional[LangGraphTripPlanner] = None


def get_trip_planner_agent() -> LangGraphTripPlanner:
    """Return a singleton LangGraph planner."""
    global _langgraph_trip_planner
    if _langgraph_trip_planner is None:
        _langgraph_trip_planner = LangGraphTripPlanner(rag_mode=get_settings().rag_mode)
    return _langgraph_trip_planner
