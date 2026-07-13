"""Deterministic candidate retrieval plus one structured logistics selection."""

from __future__ import annotations

from math import asin, cos, radians, sin, sqrt
from typing import Any, List, Optional

from pydantic import BaseModel, Field

from ..models.multi_agent import (
    AgentFeedback,
    CandidateObservation,
    ExperienceProposal,
    LogisticsProposal,
    RegistryEntity,
    registry_source_id,
)
from ..models.schemas import TripRequest
from .candidate_ranking import alias_map, compact_candidates, normalize_text, resolve_aliases, shortlist
from .evidence_snapshot import AgentEvidenceSnapshot
from .structured_llm import invoke_structured
from .tool_gateway import ToolGateway, ToolGatewayError


class LogisticsAgentError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class LogisticsAliasProposal(BaseModel):
    primary_hotel_alias: str
    hotel_aliases: List[str] = Field(default_factory=list)
    meal_aliases: List[str] = Field(default_factory=list)
    constraints: List[str] = Field(default_factory=list)
    infeasible_pairs: List[List[str]] = Field(default_factory=list)
    unknowns: List[str] = Field(default_factory=list)
    cost_assumptions: dict[str, int] = Field(default_factory=dict)


class LogisticsAgent:
    MAX_ATTEMPTS = 2

    def __init__(self, *, llm: Any, gateway: ToolGateway):
        self.llm = llm
        self.gateway = gateway
        self.trace: dict[str, Any] = {}

    def run(self, *, request: TripRequest, experience: ExperienceProposal,
            feedback: Optional[AgentFeedback] = None,
            previous: Optional[LogisticsProposal] = None, attempt: int = 1,
            evidence_override: Optional[AgentEvidenceSnapshot] = None) -> LogisticsProposal:
        if attempt < 1 or attempt > self.MAX_ATTEMPTS:
            raise LogisticsAgentError("retry_budget_exhausted", "logistics attempt budget exhausted")
        anchor_entities = [self.gateway.registry.entities[source_id] for source_id in experience.core_attraction_ids
                           if source_id in self.gateway.registry.entities]
        anchor_names = sorted({entity.name for entity in anchor_entities})[:2]
        max_hotel_queries = min(3, 1 + len(anchor_names))
        max_meal_queries = min(4, max(2, 2 + len(anchor_names)))
        hotel_queries = [(normalize_text(f"{request.accommodation} hotel"), "base_anchor")] + [
            (normalize_text(f"hotel near {name}"), "supplemental") for name in anchor_names]
        food_preferences = " ".join(sorted(p for p in request.preferences if normalize_text(p)))
        meal_queries = [(normalize_text(f"{request.city} dining {food_preferences}"), "base_anchor"),
                        ("local restaurants", "supplemental")] + [
            (normalize_text(f"food near {name}"), "supplemental") for name in anchor_names]
        try:
            if evidence_override is not None:
                for entity in evidence_override.entities:
                    replayed = entity.model_copy(deep=True)
                    replayed.registered_by = "logistics"
                    self.gateway.registry.add(replayed, actor="logistics")
                if not self._ids("hotel") or not self._ids("meal"):
                    raise LogisticsAgentError("snapshot_mismatch", "replay snapshot lacks hotel or meal evidence")
            for query_index, (query, source_type) in enumerate(
                [] if evidence_override is not None else hotel_queries[:max_hotel_queries]
            ):
                items = self.gateway.call("logistics", "hotel_search", query_key=query,
                                          query=query, city=request.city, country_code=request.country_code)
                self.gateway.register("logistics", self._entities(items, "hotel", source_type=source_type,
                                                                  query=query, query_index=query_index))
                hotels_now = [entity for entity in self.gateway.registry.entities.values()
                              if entity.entity_type == "hotel"]
                if len(hotels_now) >= 8 and sum(entity.location is not None for entity in hotels_now) >= 3:
                    self.gateway.early_stop_reasons["hotels"] = "pool_target_and_coordinate_feasibility"
                    break
            for query_index, (query, source_type) in enumerate(
                [] if evidence_override is not None else meal_queries[:max_meal_queries]
            ):
                items = self.gateway.call("logistics", "meal_search", query_key=query,
                                          query=query, city=request.city, country_code=request.country_code)
                self.gateway.register("logistics", self._entities(items, "meal", source_type=source_type,
                                                                  query=query, query_index=query_index))
                meals_now = [entity for entity in self.gateway.registry.entities.values()
                             if entity.entity_type == "meal"]
                if len(meals_now) >= 12 and self._meal_coverage(request, meals_now):
                    self.gateway.early_stop_reasons["meals"] = "pool_target_and_theme_coverage"
                    break
        except ToolGatewayError as exc:
            raise LogisticsAgentError(exc.code, str(exc)) from exc

        centroid = self._centroid(anchor_entities)
        hotels_ranked = shortlist(self.gateway.registry, "hotel", request, limit=6, centroid=centroid)
        meals_ranked = shortlist(self.gateway.registry, "meal", request, limit=12, centroid=centroid)
        hotel_aliases = alias_map(hotels_ranked, "H")
        meal_aliases = alias_map(meals_ranked, "M")
        feasibility = self._coarse_feasibility(experience, [item.source_id for item in hotels_ranked + meals_ranked])
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
            f"hotels={compact_candidates(hotels_ranked, hotel_aliases)}\n"
            f"meals={compact_candidates(meals_ranked, meal_aliases)}\n"
            f"Select one primary H alias. hotel_aliases contains only optional alternate hotels: do not repeat "
            f"primary_hotel_alias there, and select at most one alternate. Select at most "
            f"{min(6, 2 * request.travel_days)} unique M aliases.\n"
            f"coarse_feasibility={feasibility}\nrevision={context}"
        )
        try:
            alias_proposal = invoke_structured(self.llm, LogisticsAliasProposal, prompt)
        except Exception as exc:
            raise LogisticsAgentError("structured_output", str(exc)) from exc
        try:
            primary = resolve_aliases([alias_proposal.primary_hotel_alias], hotel_aliases, expected_prefix="H")[0]
            alternate_hotels = resolve_aliases(alias_proposal.hotel_aliases, hotel_aliases, expected_prefix="H")
            hotels = list(dict.fromkeys([primary, *alternate_hotels]))
            meals = resolve_aliases(alias_proposal.meal_aliases, meal_aliases, expected_prefix="M")
            if len(hotels) > 2 or len(meals) > min(6, 2 * request.travel_days):
                raise ValueError("invalid logistics cardinality")
            costs = {hotel_aliases[key]: value for key, value in alias_proposal.cost_assumptions.items()
                     if key in hotel_aliases}
        except ValueError as exc:
            raise LogisticsAgentError("invalid_source_id", str(exc)) from exc
        proposal = LogisticsProposal(run_id=self.gateway.registry.run_id,
                                     version=(previous.version if previous else 0) + 1,
                                     experience_version=experience.version, hotel_ids=hotels, meal_ids=meals,
                                     primary_hotel_id=primary, constraints=alias_proposal.constraints,
                                     infeasible_pairs=alias_proposal.infeasible_pairs,
                                     unknowns=alias_proposal.unknowns, cost_assumptions=costs)
        self.trace = {"hotel_pool_ids": sorted(self._ids("hotel")),
                      "meal_pool_ids": sorted(self._ids("meal")),
                      "hotel_shortlist_ids": [item.source_id for item in hotels_ranked],
                      "meal_shortlist_ids": [item.source_id for item in meals_ranked],
                      "hotel_alias_map": hotel_aliases, "meal_alias_map": meal_aliases,
                      "primary_hotel_id": primary, "selected_meal_ids": meals}
        if evidence_override is not None:
            self.trace["evidence_mode"] = "replay"
        return proposal

    def _entities(self, items: Any, entity_type: str, *, source_type: str, query: str,
                  query_index: int) -> List[RegistryEntity]:
        entities = []
        for provider_rank, item in enumerate(list(items or []), 1):
            if not isinstance(item, dict):
                continue
            provider_id = str(item.get("id") or item.get("provider_id") or item.get("source_id") or "")
            name = str(item.get("name") or "")
            if not provider_id or not name:
                continue
            entities.append(RegistryEntity(source_id=registry_source_id(entity_type, provider_id),
                                            provider_id=provider_id, entity_type=entity_type, name=name,
                                            address=str(item.get("address") or ""), location=item.get("location") or None,
                                            rating=item.get("rating"), maps_url=item.get("maps_url"),
                                            user_rating_count=item.get("user_rating_count"),
                                            website_url=item.get("website_url"), image_url=item.get("image_url"),
                                            photo_names=list(item.get("photo_names") or []),
                                            metadata={"category": item.get("type") or entity_type},
                                            registered_by="logistics",
                                            observations=[CandidateObservation(
                                                source_type=source_type, normalized_query=normalize_text(query),
                                                query_index=query_index, provider_rank=provider_rank,
                                                provider_id=provider_id, name=name,
                                                address=str(item.get("address") or ""), location=item.get("location") or None,
                                                rating=item.get("rating"), user_rating_count=item.get("user_rating_count"),
                                                maps_url=item.get("maps_url"), website_url=item.get("website_url"),
                                                image_url=item.get("image_url"), photo_names=list(item.get("photo_names") or []),
                                                metadata={"category": item.get("type") or entity_type})]))
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
    def _centroid(entities: List[RegistryEntity]) -> Optional[tuple[float, float]]:
        locations = [entity.location for entity in entities if entity.location]
        if not locations:
            return None
        return (sum(item.latitude for item in locations) / len(locations),
                sum(item.longitude for item in locations) / len(locations))

    @staticmethod
    def _meal_coverage(request: TripRequest, entities: List[RegistryEntity]) -> bool:
        food_terms = [normalize_text(value) for value in request.preferences
                      if any(term in normalize_text(value) for term in ("food", "dining", "restaurant", "cafe"))]
        if not food_terms:
            return True
        text = normalize_text(" ".join(entity.name + " " + str(entity.metadata) for entity in entities))
        return all(term in text for term in food_terms)

    @staticmethod
    def _distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        d_lat, d_lon = radians(lat2 - lat1), radians(lon2 - lon1)
        value = sin(d_lat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(d_lon / 2) ** 2
        return 6371.0 * 2 * asin(sqrt(value))
