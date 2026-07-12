"""Deterministic candidate retrieval plus one structured logistics selection."""

from __future__ import annotations

from math import asin, cos, radians, sin, sqrt
from typing import Any, List, Optional

from ..models.multi_agent import (
    AgentFeedback,
    ExperienceProposal,
    LogisticsProposal,
    RegistryEntity,
)
from ..models.schemas import TripRequest
from .structured_llm import invoke_structured
from .tool_gateway import ToolGateway, ToolGatewayError


class LogisticsAgentError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class LogisticsAgent:
    MAX_ATTEMPTS = 2

    def __init__(self, *, llm: Any, gateway: ToolGateway):
        self.llm = llm
        self.gateway = gateway

    def run(self, *, request: TripRequest, experience: ExperienceProposal,
            feedback: Optional[AgentFeedback] = None,
            previous: Optional[LogisticsProposal] = None, attempt: int = 1) -> LogisticsProposal:
        if attempt < 1 or attempt > self.MAX_ATTEMPTS:
            raise LogisticsAgentError("retry_budget_exhausted", "logistics attempt budget exhausted")
        cluster_names = [cluster.name for cluster in experience.clusters]
        max_hotel_queries = min(3, 1 + len(cluster_names))
        max_meal_queries = min(4, max(2, len(cluster_names)))
        hotel_queries = [f"{request.accommodation} hotel"] + [f"hotel near {name}" for name in cluster_names]
        meal_queries = ["restaurants", "local dining"] + [f"food near {name}" for name in cluster_names]
        try:
            for query in hotel_queries[:max_hotel_queries]:
                items = self.gateway.call("logistics", "hotel_search", query_key=query,
                                          query=query, city=request.city, country_code=request.country_code)
                self.gateway.register("logistics", self._entities(items, "hotel"))
            for query in meal_queries[:max_meal_queries]:
                items = self.gateway.call("logistics", "meal_search", query_key=query,
                                          query=query, city=request.city, country_code=request.country_code)
                self.gateway.register("logistics", self._entities(items, "meal"))
        except ToolGatewayError as exc:
            raise LogisticsAgentError(exc.code, str(exc)) from exc

        hotels = self._ids("hotel")
        meals = self._ids("meal")
        feasibility = self._coarse_feasibility(experience, hotels + meals)
        context = {
            "previous_proposal": previous.model_dump() if previous else None,
            "feedback": feedback.model_dump() if feedback else None,
            "remaining_attempts": self.MAX_ATTEMPTS - attempt,
            "registry_summary": self.gateway.registry.summary(),
        }
        prompt = (
            "You are the Logistics specialist. Select only listed hotel and meal source IDs. "
            "Return constraints, infeasible pairs, unknowns, and integer cost assumptions. "
            "Do not create itinerary dates or attractions.\n"
            f"request={request.model_dump()}\nexperience={experience.model_dump()}\n"
            f"hotel_ids={hotels}\nmeal_ids={meals}\ncoarse_feasibility={feasibility}\nrevision={context}"
        )
        try:
            proposal = invoke_structured(self.llm, LogisticsProposal, prompt)
        except Exception as exc:
            raise LogisticsAgentError("structured_output", str(exc)) from exc
        proposal.run_id = self.gateway.registry.run_id
        proposal.version = (previous.version if previous else 0) + 1
        proposal.experience_version = experience.version
        invalid = (set(proposal.hotel_ids) - set(hotels)) | (set(proposal.meal_ids) - set(meals))
        if invalid:
            raise LogisticsAgentError("invalid_source_id", f"invalid logistics IDs: {sorted(invalid)}")
        return proposal

    def _entities(self, items: Any, entity_type: str) -> List[RegistryEntity]:
        entities = []
        for item in list(items or []):
            if not isinstance(item, dict):
                continue
            source_id = str(item.get("id") or item.get("source_id") or "")
            name = str(item.get("name") or "")
            if not source_id or not name:
                continue
            entities.append(RegistryEntity(source_id=source_id, entity_type=entity_type, name=name,
                                            address=str(item.get("address") or ""), location=item.get("location") or None,
                                            rating=item.get("rating"), maps_url=item.get("maps_url"),
                                            website_url=item.get("website_url"), image_url=item.get("image_url"),
                                            photo_names=list(item.get("photo_names") or []),
                                            metadata={"category": item.get("type") or entity_type},
                                            registered_by="logistics"))
        return entities

    def _ids(self, entity_type: str) -> List[str]:
        return [source_id for source_id, entity in self.gateway.registry.entities.items()
                if entity.entity_type == entity_type]

    def _coarse_feasibility(self, experience: ExperienceProposal, logistics_ids: List[str]) -> dict:
        attraction_ids = list(experience.allowed_attraction_ids)
        result = {}
        for source_id in logistics_ids:
            entity = self.gateway.registry.entities[source_id]
            distances = []
            if entity.location:
                for attraction_id in attraction_ids:
                    attraction = self.gateway.registry.entities.get(attraction_id)
                    if attraction and attraction.location:
                        distances.append(self._distance_km(entity.location.latitude, entity.location.longitude,
                                                           attraction.location.latitude, attraction.location.longitude))
            result[source_id] = round(sum(distances) / len(distances), 2) if distances else None
        return result

    @staticmethod
    def _distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        d_lat, d_lon = radians(lat2 - lat1), radians(lon2 - lon1)
        value = sin(d_lat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(d_lon / 2) ** 2
        return 6371.0 * 2 * asin(sqrt(value))

