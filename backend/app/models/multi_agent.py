"""Typed contracts shared by the bounded trip-planning agents."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from .schemas import Location

AgentRole = Literal["experience", "logistics", "composer"]
EntityType = Literal["attraction", "hotel", "meal"]


def registry_source_id(entity_type: EntityType, provider_id: str) -> str:
    """Build an entity-type-scoped ID for one request-local registry entry."""
    normalized = str(provider_id or "").strip()
    if not normalized:
        raise ValueError("provider_id must not be blank")
    return f"{entity_type}:{normalized}"


class RegistryEntity(BaseModel):
    """Canonical provider-backed entity stored for one planner run."""

    source_id: str
    provider_id: str = ""
    entity_type: EntityType
    name: str
    address: str = ""
    location: Optional[Location] = None
    rating: Optional[float] = None
    user_rating_count: Optional[int] = None
    provider: str = "google_maps"
    maps_url: Optional[str] = None
    website_url: Optional[str] = None
    image_url: Optional[str] = None
    photo_names: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    registered_by: AgentRole
    observations: List["CandidateObservation"] = Field(default_factory=list)
    query_provenance: List[str] = Field(default_factory=list)
    best_provider_rank: Optional[int] = None
    relevance_score: float = 0.0
    score_components: Dict[str, float] = Field(default_factory=dict)


class CandidateObservation(BaseModel):
    source_type: Literal["place_details", "base_anchor", "supplemental"]
    normalized_query: str = ""
    query_index: int = 0
    provider_rank: int = Field(default=1, ge=1)
    provider_id: str
    name: str = ""
    address: str = ""
    location: Optional[Location] = None
    rating: Optional[float] = None
    user_rating_count: Optional[int] = None
    maps_url: Optional[str] = None
    website_url: Optional[str] = None
    image_url: Optional[str] = None
    photo_names: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CandidateRegistry(BaseModel):
    """Request-local canonical entity registry with explicit ownership."""

    run_id: str
    entities: Dict[str, RegistryEntity] = Field(default_factory=dict)
    revision: int = 0

    def add(self, entity: RegistryEntity, *, actor: AgentRole) -> None:
        if entity.registered_by != actor:
            raise ValueError("registry actor must match entity ownership")
        existing = self.entities.get(entity.source_id)
        if existing is not None and existing.registered_by != actor:
            raise ValueError("an agent cannot overwrite another agent's registry entity")
        if existing is not None:
            entity = self._merge(existing, entity)
        self.entities[entity.source_id] = entity
        self.revision += 1

    @staticmethod
    def _merge(existing: RegistryEntity, incoming: RegistryEntity) -> RegistryEntity:
        observations = list(existing.observations) + list(incoming.observations)
        unique = {item.model_dump_json(): item for item in observations}
        observations = sorted(unique.values(), key=CandidateRegistry._observation_key)
        merged = existing.model_copy(deep=True)
        merged.observations = observations
        for field in ("name", "address", "location", "rating", "user_rating_count", "maps_url",
                      "website_url", "image_url", "photo_names", "metadata"):
            for observation in observations:
                value = getattr(observation, field)
                if value not in (None, "", [], {}):
                    setattr(merged, field, value)
                    break
        merged.query_provenance = sorted({item.normalized_query for item in observations if item.normalized_query})
        ranks = [item.provider_rank for item in observations]
        merged.best_provider_rank = min(ranks) if ranks else None
        return merged

    @staticmethod
    def _observation_key(item: CandidateObservation) -> tuple:
        priority = {"place_details": 0, "base_anchor": 1, "supplemental": 2}[item.source_type]
        return (priority, item.query_index, item.provider_rank, item.provider_id)

    def summary(self) -> Dict[str, Any]:
        by_type = {kind: 0 for kind in ("attraction", "hotel", "meal")}
        for entity in self.entities.values():
            by_type[entity.entity_type] += 1
        return {"run_id": self.run_id, "revision": self.revision, "counts": by_type}


class ProposalBase(BaseModel):
    version: int = Field(default=1, ge=1)
    run_id: str


class ExperienceCluster(BaseModel):
    name: str
    attraction_ids: List[str] = Field(default_factory=list)
    rationale: str = ""


class ExperienceProposal(ProposalBase):
    clusters: List[ExperienceCluster] = Field(default_factory=list, max_length=4)
    rag_chunk_ids: List[str] = Field(default_factory=list)
    uncovered_preferences: List[str] = Field(default_factory=list)
    evidence_sufficient: bool = True
    core_attraction_ids: List[str] = Field(default_factory=list)
    optional_attraction_ids: List[str] = Field(default_factory=list)
    target_attractions: int = 0

    @property
    def allowed_attraction_ids(self) -> set[str]:
        explicit = set(self.core_attraction_ids) | set(self.optional_attraction_ids)
        return explicit or {source_id for cluster in self.clusters for source_id in cluster.attraction_ids}


class LogisticsProposal(ProposalBase):
    experience_version: int = Field(ge=1)
    hotel_ids: List[str] = Field(default_factory=list)
    meal_ids: List[str] = Field(default_factory=list)
    constraints: List[str] = Field(default_factory=list)
    infeasible_pairs: List[List[str]] = Field(default_factory=list)
    unknowns: List[str] = Field(default_factory=list)
    cost_assumptions: Dict[str, int] = Field(default_factory=dict)
    primary_hotel_id: Optional[str] = None


class DraftAttraction(BaseModel):
    source_id: str
    visit_duration: int
    description: str = ""
    ticket_price: int = 0
    cost_status: Literal["known", "estimated", "unknown"] = "unknown"


class DraftMeal(BaseModel):
    meal_type: Literal["breakfast", "lunch", "dinner", "snack"]
    source_id: Optional[str] = None
    generic_name: Optional[str] = None
    description: str = ""
    estimated_cost: int = 0
    cost_status: Literal["known", "estimated", "unknown"] = "unknown"


class DraftDay(BaseModel):
    date: str
    day_index: int = Field(ge=0)
    description: str = ""
    attraction_items: List[DraftAttraction] = Field(default_factory=list)
    meal_items: List[DraftMeal] = Field(default_factory=list)
    hotel_id: Optional[str] = None


class IDBasedItineraryDraft(ProposalBase):
    experience_version: int = Field(ge=1)
    logistics_version: int = Field(ge=1)
    days: List[DraftDay] = Field(default_factory=list)
    overall_suggestions: str = ""
    transportation_estimate: int = 0


class AgentFeedback(BaseModel):
    owner: AgentRole
    codes: List[str] = Field(default_factory=list)
    details: List[Dict[str, Any]] = Field(default_factory=list)


class AgentRetryState(BaseModel):
    experience_attempts: int = 0
    logistics_attempts: int = 0
    composer_attempts: int = 0
    global_revisions: int = 0


class AgentMetric(BaseModel):
    attempts: int = 0
    latency_ms: float = 0.0
    token_usage: int = 0
    tool_calls: Dict[str, int] = Field(default_factory=dict)


class AgentMetrics(BaseModel):
    by_agent: Dict[AgentRole, AgentMetric] = Field(default_factory=dict)
    targeted_retries: List[AgentRole] = Field(default_factory=list)
    invalid_source_ids: List[str] = Field(default_factory=list)
    handoff_trace: List[Dict[str, Any]] = Field(default_factory=list)
    early_stop_reasons: Dict[str, str] = Field(default_factory=dict)
    budget_usage: Dict[str, Any] = Field(default_factory=dict)
    stability_trace: Dict[str, Any] = Field(default_factory=dict)


class CallBudgetLedger(BaseModel):
    role_limits: Dict[str, Dict[str, int]] = Field(default_factory=lambda: {
        "experience": {"llm": 4, "maps": 10, "rag": 2},
        "logistics": {"llm": 2, "maps": 14, "rag": 0},
        "composer": {"llm": 2, "maps": 0, "rag": 0},
    })
    global_limits: Dict[str, int] = Field(default_factory=lambda: {"llm": 8, "maps": 24, "rag": 2})
    role_used: Dict[str, Dict[str, int]] = Field(default_factory=dict)
    global_used: Dict[str, int] = Field(default_factory=lambda: {"llm": 0, "maps": 0, "rag": 0})
    blocked_calls: List[Dict[str, str]] = Field(default_factory=list)

    def consume(self, role: AgentRole, resource: Literal["llm", "maps", "rag"], call_name: str) -> None:
        used = self.role_used.setdefault(role, {"llm": 0, "maps": 0, "rag": 0})
        role_limit = self.role_limits[role][resource]
        global_limit = self.global_limits[resource]
        if used[resource] >= role_limit or self.global_used[resource] >= global_limit:
            self.blocked_calls.append({"role": role, "resource": resource, "call": call_name})
            raise ValueError(f"call budget exhausted: {role}.{resource}")
        used[resource] += 1
        self.global_used[resource] += 1

    def snapshot(self) -> Dict[str, Any]:
        return {"role_limits": self.role_limits, "global_limits": self.global_limits,
                "role_used": self.role_used, "global_used": self.global_used,
                "blocked_calls": self.blocked_calls}
