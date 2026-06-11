"""LangGraph state and supporting models for trip planning."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, TypedDict

from pydantic import BaseModel, Field

from .schemas import Location, TripPlan, TripRequest, WeatherInfo


class RequestContext(BaseModel):
    """Normalized request summary used across retrieval and planning nodes."""

    city: str
    start_date: str
    end_date: str
    travel_days: int
    transportation: str
    accommodation: str
    preferences: List[str] = Field(default_factory=list)
    free_text_input: str = ""
    memory_context: str = ""
    summary: str = ""


class AttractionCandidate(BaseModel):
    """Structured attraction candidate from retrieval."""

    name: str
    address: str = ""
    location: Optional[Location] = None
    source: str = "amap_mcp"
    source_id: str = ""
    raw_text: str = ""


class HotelCandidate(BaseModel):
    """Structured hotel candidate from retrieval."""

    name: str
    address: str = ""
    location: Optional[Location] = None
    source: str = "amap_mcp"
    source_id: str = ""
    raw_text: str = ""


class RAGChunk(BaseModel):
    """Retrieved travel knowledge chunk."""

    chunk_id: str
    source: str
    title: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PlannerInputBundle(BaseModel):
    """Normalized planner payload assembled from graph state."""

    request_context: RequestContext
    attraction_candidates: List[AttractionCandidate] = Field(default_factory=list)
    hotel_candidates: List[HotelCandidate] = Field(default_factory=list)
    weather_info: List[WeatherInfo] = Field(default_factory=list)
    rag_chunks: List[RAGChunk] = Field(default_factory=list)


class EvaluationScores(BaseModel):
    """Structured evaluation scores."""

    schema_score: float = 0.0
    date_coverage_score: float = 0.0
    budget_consistency_score: float = 0.0
    grounding_score: float = 0.0
    pacing_score: float = 0.0
    route_coherence_score: float = 0.0
    preference_match_score: float = 0.0
    attribution_coverage_score: float = 0.0


class UnsupportedEntity(BaseModel):
    """Entity that could not be grounded to retrieval evidence."""

    entity_type: Literal["attraction", "hotel", "claim"]
    name: str
    reason: str = ""


class EvidenceLink(BaseModel):
    """Mapping from a generated itinerary entity to retrieval or RAG evidence."""

    entity_type: Literal["attraction", "hotel", "claim"]
    entity_name: str
    day_index: Optional[int] = None
    evidence_type: Literal["candidate_attraction", "candidate_hotel", "rag_chunk", "none"]
    evidence_id: str = ""
    source_title: str = ""
    source_url: str = ""
    confidence: float = 0.0
    match_reason: str = ""


class EvaluationReport(BaseModel):
    """Graph evaluation result used for routing and debugging."""

    passed: bool = False
    hard_failures: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    quality_warnings: List[str] = Field(default_factory=list)
    scores: EvaluationScores = Field(default_factory=EvaluationScores)
    unsupported_entities: List[UnsupportedEntity] = Field(default_factory=list)
    unsupported_claims: List[str] = Field(default_factory=list)
    evidence_links: List[EvidenceLink] = Field(default_factory=list)
    next_action: Literal[
        "finalize_response",
        "plan_itinerary",
        "retrieve_attractions",
        "retrieve_hotels",
        "fallback_response",
    ] = "plan_itinerary"


class RetryState(BaseModel):
    """Per-node execution counts used for retry routing."""

    prepare_request: int = 0
    retrieve_attractions: int = 0
    retrieve_hotels: int = 0
    retrieve_weather: int = 0
    retrieve_rag_context: int = 0
    plan_itinerary: int = 0
    evaluate_itinerary: int = 0
    finalize_response: int = 0
    fallback_response: int = 0


class RunMetrics(BaseModel):
    """Per-run metrics tracked for observability and later quantification."""

    started_at: float = 0.0
    ended_at: float = 0.0
    end_to_end_ms: float = 0.0
    node_latency_ms: Dict[str, float] = Field(default_factory=dict)
    node_attempts: Dict[str, int] = Field(default_factory=dict)
    evaluation_pass_count: int = 0
    evaluation_attempt_count: int = 0
    first_evaluation_pass: Optional[bool] = None
    final_evaluation_pass: Optional[bool] = None
    recovered_after_retry: bool = False
    fallback_count: int = 0
    schema_failure_count: int = 0
    date_coverage_failure_count: int = 0
    budget_consistency_failure_count: int = 0
    grounding_failure_count: int = 0


class TripGraphState(TypedDict, total=False):
    """Typed LangGraph state."""

    request: TripRequest
    travel_dates: List[str]
    request_context: RequestContext
    candidate_attractions: List[AttractionCandidate]
    candidate_hotels: List[HotelCandidate]
    weather_info: List[WeatherInfo]
    rag_chunks: List[RAGChunk]
    planner_inputs: PlannerInputBundle
    draft_plan_raw: str
    draft_plan: Optional[TripPlan]
    evaluation_report: EvaluationReport
    evaluation_history: List[EvaluationReport]
    retry_counts: RetryState
    decision_trace: List[str]
    metrics: RunMetrics
    final_plan: Optional[TripPlan]
    conversation_id: str
    memory_applied: bool
    memory_summary: str
    memory_profile: Dict[str, Any]
