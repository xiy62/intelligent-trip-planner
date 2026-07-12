"""Deterministic orchestration for three bounded trip-planning specialists."""

from __future__ import annotations

import time
from typing import Any, Dict, List

from langgraph.graph import END, START, StateGraph

from ..models.langgraph_state import AttractionCandidate, HotelCandidate, MealCandidate, TripGraphState
from ..models.multi_agent import AgentMetric, AgentMetrics, AgentRetryState, CandidateRegistry
from ..models.schemas import TripRequest
from .composer_agent import ComposerAgent
from .experience_agent import ExperienceAgent
from .itinerary_materializer import ItineraryMaterializer
from .langgraph_trip_planner import LangGraphTripPlanner
from .logistics_agent import LogisticsAgent
from .tool_gateway import ToolGateway


class MultiAgentTripPlanner(LangGraphTripPlanner):
    """Active multi-agent graph with deterministic handoffs and materialization."""

    EXPERIENCE_BUDGETS = {"attraction_search": 3, "rag_search": 1, "place_detail": 2}

    def _build_graph(self):
        builder = StateGraph(TripGraphState)
        builder.add_node("prepare_request", self.prepare_multi_agent_request)
        builder.add_node("authoritative_weather", self.retrieve_weather)
        builder.add_node("experience_agent", self.run_experience_agent)
        builder.add_node("logistics_agent", self.run_logistics_agent)
        builder.add_node("composer_agent", self.run_composer_agent)
        builder.add_node("canonical_materializer", self.materialize_draft)
        builder.add_node("collect_route_times", self.collect_route_times)
        builder.add_node("evaluate_itinerary", self.evaluate_itinerary)
        builder.add_node("finalize_response", self.finalize_response)
        builder.add_node("fallback_response", self.fallback_response)
        builder.add_edge(START, "prepare_request")
        builder.add_edge("prepare_request", "authoritative_weather")
        builder.add_edge("prepare_request", "experience_agent")
        builder.add_edge(["authoritative_weather", "experience_agent"], "logistics_agent")
        builder.add_edge("logistics_agent", "composer_agent")
        builder.add_edge("composer_agent", "canonical_materializer")
        builder.add_edge("canonical_materializer", "collect_route_times")
        builder.add_edge("collect_route_times", "evaluate_itinerary")
        builder.add_conditional_edges("evaluate_itinerary", self._route_multi_result,
                                      {"finalize_response": "finalize_response",
                                       "fallback_response": "fallback_response"})
        builder.add_edge("finalize_response", END)
        builder.add_edge("fallback_response", END)
        return builder.compile(checkpointer=self.checkpointer)

    def prepare_multi_agent_request(self, state: TripGraphState) -> TripGraphState:
        update = self.prepare_request(state)
        update.update({"candidate_registry": CandidateRegistry(run_id=state["run_id"]),
                       "agent_retry_state": AgentRetryState(), "agent_metrics": AgentMetrics()})
        return update

    def run_experience_agent(self, state: TripGraphState) -> TripGraphState:
        started = time.perf_counter()
        registry = state["candidate_registry"].model_copy(deep=True)
        gateway = ToolGateway(registry=registry, tools={
            "attraction_search": self._gateway_map_search,
            "rag_search": self._gateway_rag_search,
            "place_detail": self._gateway_place_detail,
        }, budgets=self.EXPERIENCE_BUDGETS)
        result = ExperienceAgent(llm=self.llm, gateway=gateway,
                                 deterministic_fallback=self._experience_fallback).run(
                                     request=state["request"], attempt=1)
        metrics = (state.get("agent_metrics") or AgentMetrics()).model_copy(deep=True)
        metrics.by_agent["experience"] = AgentMetric(attempts=1,
                                                       latency_ms=(time.perf_counter() - started) * 1000,
                                                       tool_calls=dict(gateway.call_counts))
        metrics.handoff_trace.append({"from": "experience", "to": "logistics",
                                      "proposal_version": result.proposal.version})
        return {"candidate_registry": registry, "experience_proposal": result.proposal,
                "rag_chunks": result.rag_chunks, "agent_metrics": metrics,
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
        proposal = LogisticsAgent(llm=self.llm, gateway=gateway).run(
            request=state["request"], experience=state["experience_proposal"], attempt=1)
        metrics = (state.get("agent_metrics") or AgentMetrics()).model_copy(deep=True)
        metrics.by_agent["logistics"] = AgentMetric(attempts=1,
                                                     latency_ms=(time.perf_counter() - started) * 1000,
                                                     tool_calls=dict(gateway.call_counts))
        metrics.handoff_trace.append({"from": "logistics", "to": "composer",
                                      "proposal_version": proposal.version})
        return {"candidate_registry": registry, "logistics_proposal": proposal,
                "agent_metrics": metrics,
                "decision_trace": self._append_trace(state, "agent_handoff: logistics -> composer")}

    def run_composer_agent(self, state: TripGraphState) -> TripGraphState:
        started = time.perf_counter()
        draft = ComposerAgent(llm=self.llm).run(request=state["request"],
                                                experience=state["experience_proposal"],
                                                logistics=state["logistics_proposal"],
                                                weather_info=list(state.get("weather_info", [])), attempt=1)
        metrics = (state.get("agent_metrics") or AgentMetrics()).model_copy(deep=True)
        metrics.by_agent["composer"] = AgentMetric(attempts=1,
                                                    latency_ms=(time.perf_counter() - started) * 1000)
        metrics.handoff_trace.append({"from": "composer", "to": "canonical_materializer",
                                      "proposal_version": draft.version})
        return {"id_draft": draft, "agent_metrics": metrics,
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
        return {"draft_plan": result.plan,
                "materialization_failures": [failure.model_dump() for failure in result.failures],
                **candidates,
                "decision_trace": self._append_trace(
                    state, f"materialization: {'success' if result.succeeded else 'failed'}")}

    def _route_multi_result(self, state: TripGraphState) -> str:
        report = state.get("evaluation_report")
        return "finalize_response" if report is not None and report.passed else "fallback_response"

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
