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
    provider: str = "google_maps"
    maps_url: Optional[str] = None
    website_url: Optional[str] = None
    image_url: Optional[str] = None
    photo_names: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    registered_by: AgentRole


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
        self.entities[entity.source_id] = entity
        self.revision += 1

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

    @property
    def allowed_attraction_ids(self) -> set[str]:
        return {source_id for cluster in self.clusters for source_id in cluster.attraction_ids}


class LogisticsProposal(ProposalBase):
    experience_version: int = Field(ge=1)
    hotel_ids: List[str] = Field(default_factory=list)
    meal_ids: List[str] = Field(default_factory=list)
    constraints: List[str] = Field(default_factory=list)
    infeasible_pairs: List[List[str]] = Field(default_factory=list)
    unknowns: List[str] = Field(default_factory=list)
    cost_assumptions: Dict[str, int] = Field(default_factory=dict)


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
