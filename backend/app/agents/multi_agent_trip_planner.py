"""Deterministic orchestration for three bounded trip-planning specialists."""

from __future__ import annotations

import time
from typing import Any, Dict, List

from langgraph.graph import END, START, StateGraph

from ..models.langgraph_state import (
    AttractionCandidate,
    HotelCandidate,
    MealCandidate,
    RouteFailureDetail,
    TripGraphState,
)
from ..models.multi_agent import AgentFeedback, AgentMetric, AgentMetrics, AgentRetryState, CandidateRegistry
from ..models.schemas import TripRequest
from ..services.llm_service import get_role_llm
from .composer_agent import ComposerAgent
from .experience_agent import ExperienceAgent
from .itinerary_materializer import ItineraryMaterializer
from .langgraph_trip_planner import LangGraphTripPlanner
from .logistics_agent import LogisticsAgent
from .tool_gateway import ToolGateway


class MultiAgentTripPlanner(LangGraphTripPlanner):
    """Active multi-agent graph with deterministic handoffs and materialization."""

    EXPERIENCE_BUDGETS = {"attraction_search": 3, "rag_search": 1, "place_detail": 2}

    def __init__(self, *args, **kwargs):
        injected_llm = kwargs.get("llm")
        super().__init__(*args, **kwargs)
        self.experience_llm = injected_llm or get_role_llm("experience")
        self.logistics_llm = injected_llm or get_role_llm("logistics")
        self.composer_llm = injected_llm or get_role_llm("composer")

    def _build_graph(self):
        builder = StateGraph(TripGraphState)
        builder.add_node("prepare_request", self.prepare_multi_agent_request)
        builder.add_node("authoritative_weather", self.retrieve_weather)
        builder.add_node("experience_agent", self.run_experience_agent)
        builder.add_node("experience_retry", self.run_experience_agent)
        builder.add_node("research_join", lambda state: {})
        builder.add_node("logistics_agent", self.run_logistics_agent)
        builder.add_node("logistics_retry", self.run_logistics_agent)
        builder.add_node("composer_agent", self.run_composer_agent)
        builder.add_node("composer_retry", self.run_composer_agent)
        builder.add_node("canonical_materializer", self.materialize_draft)
        builder.add_node("collect_route_times", self.collect_route_times)
        builder.add_node("evaluate_itinerary", self.evaluate_itinerary)
        builder.add_node("finalize_response", self.finalize_response)
        builder.add_node("fallback_response", self.fallback_response)
        builder.add_edge(START, "prepare_request")
        builder.add_edge("prepare_request", "authoritative_weather")
        builder.add_edge("prepare_request", "experience_agent")
        builder.add_edge(["authoritative_weather", "experience_agent"], "research_join")
        builder.add_conditional_edges("research_join", self._route_after_agent,
                                      {"continue": "logistics_agent", "fallback_response": "fallback_response"})
        builder.add_conditional_edges("logistics_agent", self._route_after_agent,
                                      {"continue": "composer_agent", "fallback_response": "fallback_response"})
        builder.add_conditional_edges("composer_agent", self._route_after_agent,
                                      {"continue": "canonical_materializer", "fallback_response": "fallback_response"})
        builder.add_edge("canonical_materializer", "collect_route_times")
        builder.add_edge("collect_route_times", "evaluate_itinerary")
        builder.add_conditional_edges("evaluate_itinerary", self._route_multi_result,
                                      {"finalize_response": "finalize_response",
                                       "experience_retry": "experience_retry",
                                       "logistics_retry": "logistics_retry",
                                       "composer_retry": "composer_retry",
                                       "fallback_response": "fallback_response"})
        builder.add_conditional_edges("experience_retry", self._route_after_agent,
                                      {"continue": "logistics_agent", "fallback_response": "fallback_response"})
        builder.add_conditional_edges("logistics_retry", self._route_after_agent,
                                      {"continue": "composer_agent", "fallback_response": "fallback_response"})
        builder.add_conditional_edges("composer_retry", self._route_after_agent,
                                      {"continue": "canonical_materializer", "fallback_response": "fallback_response"})
        builder.add_edge("finalize_response", END)
        builder.add_edge("fallback_response", END)
        return builder.compile(checkpointer=self.checkpointer)

    def prepare_multi_agent_request(self, state: TripGraphState) -> TripGraphState:
        update = self.prepare_request(state)
        update.update({"candidate_registry": CandidateRegistry(run_id=state["run_id"]),
                       "agent_retry_state": AgentRetryState(), "agent_metrics": AgentMetrics(),
                       "materialization_failures": [], "route_time_estimates": []})
        return update

    def run_experience_agent(self, state: TripGraphState) -> TripGraphState:
        started = time.perf_counter()
        registry = state["candidate_registry"].model_copy(deep=True)
        gateway = ToolGateway(registry=registry, tools={
            "attraction_search": self._gateway_map_search,
            "rag_search": self._gateway_rag_search,
            "place_detail": self._gateway_place_detail,
        }, budgets=self.EXPERIENCE_BUDGETS)
        retries = (state.get("agent_retry_state") or AgentRetryState()).model_copy(deep=True)
        attempt = retries.experience_attempts + 1
        feedback = self._feedback(state, "experience")
        try:
            result = ExperienceAgent(llm=getattr(self, "experience_llm", self.llm), gateway=gateway,
                                     deterministic_fallback=self._experience_fallback).run(
                                         request=state["request"], feedback=feedback,
                                         previous=state.get("experience_proposal"), attempt=attempt)
        except Exception as exc:
            return self._agent_error_update(state, "experience", exc)
        retries.experience_attempts = attempt
        if attempt > 1:
            retries.global_revisions += 1
        metrics = (state.get("agent_metrics") or AgentMetrics()).model_copy(deep=True)
        metrics.by_agent["experience"] = AgentMetric(attempts=attempt,
                                                       latency_ms=(time.perf_counter() - started) * 1000,
                                                       tool_calls=dict(gateway.call_counts))
        metrics.handoff_trace.append({"from": "experience", "to": "logistics",
                                      "proposal_version": result.proposal.version})
        if feedback is not None:
            metrics.targeted_retries.append("experience")
        return {"candidate_registry": registry, "experience_proposal": result.proposal,
                "rag_chunks": result.rag_chunks, "agent_metrics": metrics,
                "agent_retry_state": retries, "agent_error": {}, "logistics_proposal": None, "id_draft": None,
                "draft_plan": None, "route_time_estimates": [], "materialization_failures": [],
                "decision_trace": self._append_trace(state, "agent_handoff: experience -> logistics")}

    def run_logistics_agent(self, state: TripGraphState) -> TripGraphState:
        started = time.perf_counter()
        registry = state["candidate_registry"].model_copy(deep=True)
        cluster_count = len(state["experience_proposal"].clusters)
        hotel_limit = min(3, 1 + cluster_count)
        meal_limit = min(4, max(2, cluster_count))
        gateway = ToolGateway(registry=registry,
                              tools={"hotel_search": self._gateway_map_search,
                                     "meal_search": self._gateway_map_search},
                              budgets={"hotel_search": hotel_limit, "meal_search": meal_limit})
        retries = (state.get("agent_retry_state") or AgentRetryState()).model_copy(deep=True)
        attempt = retries.logistics_attempts + 1
        feedback = self._feedback(state, "logistics")
        try:
            proposal = LogisticsAgent(llm=getattr(self, "logistics_llm", self.llm), gateway=gateway).run(
                request=state["request"], experience=state["experience_proposal"],
                feedback=feedback, previous=state.get("logistics_proposal"),
                attempt=attempt)
        except Exception as exc:
            return self._agent_error_update(state, "logistics", exc)
        retries.logistics_attempts = attempt
        if attempt > 1:
            retries.global_revisions += 1
        metrics = (state.get("agent_metrics") or AgentMetrics()).model_copy(deep=True)
        metrics.by_agent["logistics"] = AgentMetric(attempts=attempt,
                                                     latency_ms=(time.perf_counter() - started) * 1000,
                                                     tool_calls=dict(gateway.call_counts))
        metrics.handoff_trace.append({"from": "logistics", "to": "composer",
                                      "proposal_version": proposal.version})
        if feedback is not None:
            metrics.targeted_retries.append("logistics")
        return {"candidate_registry": registry, "logistics_proposal": proposal,
                "agent_metrics": metrics,
                "agent_retry_state": retries, "agent_error": {}, "id_draft": None, "draft_plan": None,
                "route_time_estimates": [], "materialization_failures": [],
                "decision_trace": self._append_trace(state, "agent_handoff: logistics -> composer")}

    def run_composer_agent(self, state: TripGraphState) -> TripGraphState:
        started = time.perf_counter()
        retries = (state.get("agent_retry_state") or AgentRetryState()).model_copy(deep=True)
        attempt = retries.composer_attempts + 1
        feedback = self._feedback(state, "composer")
        try:
            draft = ComposerAgent(llm=getattr(self, "composer_llm", self.llm)).run(
                request=state["request"], experience=state["experience_proposal"],
                logistics=state["logistics_proposal"], weather_info=list(state.get("weather_info", [])),
                feedback=feedback, previous=state.get("id_draft"), attempt=attempt)
        except Exception as exc:
            return self._agent_error_update(state, "composer", exc)
        retries.composer_attempts = attempt
        if attempt > 1:
            retries.global_revisions += 1
        metrics = (state.get("agent_metrics") or AgentMetrics()).model_copy(deep=True)
        metrics.by_agent["composer"] = AgentMetric(attempts=attempt,
                                                    latency_ms=(time.perf_counter() - started) * 1000)
        metrics.handoff_trace.append({"from": "composer", "to": "canonical_materializer",
                                      "proposal_version": draft.version})
        if feedback is not None:
            metrics.targeted_retries.append("composer")
        return {"id_draft": draft, "agent_metrics": metrics, "agent_retry_state": retries,
                "agent_error": {},
                "draft_plan": None, "route_time_estimates": [], "materialization_failures": [],
                "decision_trace": self._append_trace(state, "agent_handoff: composer -> materializer")}

    def materialize_draft(self, state: TripGraphState) -> TripGraphState:
        result = ItineraryMaterializer().materialize(request=state["request"],
                                                       registry=state["candidate_registry"],
                                                       experience=state["experience_proposal"],
                                                       logistics=state["logistics_proposal"],
                                                       draft=state["id_draft"],
                                                       weather_info=list(state.get("weather_info", [])))
        registry = state["candidate_registry"]
        candidates = self._evaluation_candidates(registry)
        metrics = (state.get("agent_metrics") or AgentMetrics()).model_copy(deep=True)
        metrics.invalid_source_ids.extend(
            failure.source_id for failure in result.failures
            if failure.code in {"unknown_id", "unapproved_id", "wrong_entity_type"} and failure.source_id
        )
        return {"draft_plan": result.plan,
                "materialization_failures": [failure.model_dump() for failure in result.failures],
                "agent_metrics": metrics,
                **candidates,
                "decision_trace": self._append_trace(
                    state, f"materialization: {'success' if result.succeeded else 'failed'}")}

    def _route_multi_result(self, state: TripGraphState) -> str:
        report = state.get("evaluation_report")
        if report is not None and report.passed:
            return "finalize_response"
        retries = state.get("agent_retry_state") or AgentRetryState()
        if retries.global_revisions >= 3 or report is None or report.failure_owner is None:
            return "fallback_response"
        attempts = {"experience": retries.experience_attempts,
                    "logistics": retries.logistics_attempts,
                    "composer": retries.composer_attempts}
        owner = report.failure_owner
        return f"{owner}_retry" if attempts[owner] < 2 else "fallback_response"

    @staticmethod
    def _route_after_agent(state: TripGraphState) -> str:
        return "fallback_response" if state.get("agent_error") else "continue"

    def _agent_error_update(self, state: TripGraphState, owner: str, exc: Exception) -> TripGraphState:
        metrics = (state.get("agent_metrics") or AgentMetrics()).model_copy(deep=True)
        metric = metrics.by_agent.get(owner, AgentMetric())
        metric.attempts += 1
        metrics.by_agent[owner] = metric
        return {"agent_error": {"owner": owner, "code": getattr(exc, "code", type(exc).__name__),
                                "message": str(exc)[:300]},
                "agent_metrics": metrics,
                "decision_trace": self._append_trace(state, f"agent_error: {owner} labeled fallback")}

    def evaluate_itinerary(self, state: TripGraphState) -> TripGraphState:
        update = super().evaluate_itinerary(state)
        report = update["evaluation_report"]
        report.materialization_failures = list(state.get("materialization_failures", []))
        report.route_failure_details = self._route_failure_details(state, report)
        owner = self._failure_owner(state, report)
        report.failure_owner = owner
        report.revision_target = owner
        update["evaluation_report"] = report
        history = list(state.get("evaluation_history", []))
        history.append(report)
        update["evaluation_history"] = history
        return update

    def _failure_owner(self, state: TripGraphState, report) -> Any:
        if report.passed:
            return None
        failures = list(state.get("materialization_failures", []))
        registry = state.get("candidate_registry")
        for failure in failures:
            source_id = failure.get("source_id", "")
            entity = registry.entities.get(source_id) if registry and source_id else None
            if entity and entity.entity_type == "attraction":
                return "experience"
            if entity and entity.entity_type in {"hotel", "meal"}:
                return "logistics"
            if failure.get("code") in {"unknown_id", "unapproved_id"}:
                path = failure.get("path", "")
                if "attraction" in path:
                    return "experience"
                if "hotel" in path or "meal" in path:
                    return "logistics"
            return "composer"
        hard = set(report.hard_failures)
        if hard & {"retrieval_grounding_attractions", "content_completeness_attractions"}:
            return "experience"
        if hard & {"retrieval_grounding_hotels", "retrieval_grounding_meals"}:
            return "logistics"
        route_problem = any(detail.kind != "missing_route_data" for detail in report.route_failure_details)
        if route_problem and "low_route_coherence_score" in report.quality_warnings:
            retries = state.get("agent_retry_state") or AgentRetryState()
            return "composer" if retries.composer_attempts < 2 else "logistics"
        if hard:
            return "composer"
        return None

    @staticmethod
    def _route_failure_details(state: TripGraphState, report) -> List[RouteFailureDetail]:
        details = []
        retries = state.get("agent_retry_state") or AgentRetryState()
        for warning in report.quality_warnings:
            if warning.startswith("route_time_fallback_day_"):
                parts = warning.split("_")
                details.append(RouteFailureDetail(day_index=int(parts[4]), segment_indices=[int(parts[6])],
                                                  kind="missing_route_data"))
            elif warning.startswith("route_day_") and ("long_transfer" in warning or "long_jump" in warning):
                parts = warning.split("_")
                details.append(RouteFailureDetail(day_index=int(parts[2]), segment_indices=[],
                                                  kind=("candidate_set_problem"
                                                        if retries.composer_attempts >= 2
                                                        else "ordering_problem")))
        return details

    @staticmethod
    def _feedback(state: TripGraphState, owner: str):
        report = state.get("evaluation_report")
        if report is None or report.failure_owner != owner:
            return None
        return AgentFeedback(owner=owner, codes=list(report.hard_failures) + list(report.quality_warnings),
                             details=list(report.materialization_failures) +
                                     [item.model_dump() for item in report.route_failure_details])

    def health_summary(self) -> Dict[str, object]:
        return {"planner_name": self.__class__.__name__, "workflow": "langgraph_multi_agent",
                "checkpointer": self.checkpointer.__class__.__name__, "rag_mode": self.rag_mode,
                "parallel_retrieval_enabled": True,
                "agent_roles": ["experience", "logistics", "composer"],
                "tool_budgets": {"experience": self.EXPERIENCE_BUDGETS,
                                 "logistics": "dynamic_by_cluster", "composer": {}},
                "retry_budgets": {"per_agent_revisions": 1, "global_revisions": 3},
                "nodes": ["prepare_request", "authoritative_weather", "experience_agent",
                          "logistics_agent", "composer_agent", "canonical_materializer",
                          "collect_route_times", "evaluate_itinerary", "finalize_response", "fallback_response"]}

    def _gateway_map_search(self, *, query: str, city: str, country_code: str, **_: Any):
        return self.search_poi_tool.invoke({"keywords": query, "city": city, "citylimit": True,
                                            "page_size": 8, "country_code": country_code})

    def _gateway_rag_search(self, *, request: TripRequest, **_: Any):
        return self.rag_service.retrieve_chunks(request, rag_mode=self.rag_mode, attraction_candidates=[])

    def _gateway_place_detail(self, *, source_id: str, **_: Any):
        if hasattr(self.map_service, "get_poi_detail"):
            return self.map_service.get_poi_detail(source_id)
        return {}

    def _experience_fallback(self, request: TripRequest):
        items = self._gateway_map_search(query=f"top attractions {request.preferences[0] if request.preferences else ''}",
                                         city=request.city, country_code=request.country_code)
        chunks = self.rag_service.retrieve_chunks(request, rag_mode=self.rag_mode, attraction_candidates=[])
        return items, chunks

    @staticmethod
    def _evaluation_candidates(registry: CandidateRegistry) -> Dict[str, List[Any]]:
        attractions, hotels, meals = [], [], []
        for entity in registry.entities.values():
            common = dict(name=entity.name, address=entity.address, location=entity.location,
                          source=entity.provider, source_id=entity.source_id, rating=entity.rating,
                          maps_url=entity.maps_url, website_url=entity.website_url,
                          image_url=entity.image_url, photo_names=entity.photo_names)
            if entity.entity_type == "attraction":
                attractions.append(AttractionCandidate(**common))
            elif entity.entity_type == "hotel":
                hotels.append(HotelCandidate(**common))
            else:
                meals.append(MealCandidate(**common))
        return {"candidate_attractions": attractions, "candidate_hotels": hotels,
                "candidate_meals": meals}
