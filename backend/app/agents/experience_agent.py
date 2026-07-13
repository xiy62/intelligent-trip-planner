"""Bounded attraction and destination-knowledge specialist."""

from __future__ import annotations

import math
from typing import Any, Callable, List, Optional

from pydantic import BaseModel, Field

from ..models.langgraph_state import RAGChunk
from ..models.multi_agent import (
    AgentFeedback,
    CandidateObservation,
    CandidateRegistry,
    ExperienceCluster,
    ExperienceProposal,
    RegistryEntity,
    registry_source_id,
)
from ..models.schemas import TripRequest
from .candidate_ranking import alias_map, compact_candidates, normalize_text, pace_target, resolve_aliases, shortlist
from .evidence_snapshot import ExperienceEvidenceSnapshot
from .structured_llm import invoke_structured
from .tool_gateway import ToolGateway, ToolGatewayError


class ExperienceResearchPlan(BaseModel):
    attraction_queries: List[str] = Field(default_factory=list, max_length=3)
    rag_query: str = ""
    detail_source_ids: List[str] = Field(default_factory=list, max_length=2)


class ExperienceAliasCluster(BaseModel):
    name: str
    attraction_aliases: List[str] = Field(default_factory=list)
    rationale: str = ""


class ExperienceAliasProposal(BaseModel):
    clusters: List[ExperienceAliasCluster] = Field(default_factory=list, max_length=4)
    core_attraction_aliases: List[str] = Field(default_factory=list)
    optional_attraction_aliases: List[str] = Field(default_factory=list)
    rag_chunk_ids: List[str] = Field(default_factory=list)
    uncovered_preferences: List[str] = Field(default_factory=list)
    evidence_sufficient: bool = True


class ExperienceAgentResult(BaseModel):
    proposal: ExperienceProposal
    rag_chunks: List[RAGChunk] = Field(default_factory=list)
    used_deterministic_fallback: bool = False


class ExperienceAgentError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class ExperienceAgent:
    MAX_ATTEMPTS = 2

    def __init__(
        self,
        *,
        llm: Any,
        gateway: ToolGateway,
        deterministic_fallback: Optional[Callable[[TripRequest], tuple[List[dict], List[RAGChunk]]]] = None,
    ):
        self.llm = llm
        self.gateway = gateway
        self.deterministic_fallback = deterministic_fallback
        self.trace: dict[str, Any] = {}

    def run(
        self,
        *,
        request: TripRequest,
        feedback: Optional[AgentFeedback] = None,
        previous: Optional[ExperienceProposal] = None,
        attempt: int = 1,
        evidence_override: Optional[ExperienceEvidenceSnapshot] = None,
    ) -> ExperienceAgentResult:
        if attempt < 1 or attempt > self.MAX_ATTEMPTS:
            raise ExperienceAgentError("retry_budget_exhausted", "experience attempt budget exhausted")
        revision_context = {
            "previous_proposal": previous.model_dump() if previous else None,
            "feedback": feedback.model_dump() if feedback else None,
            "remaining_attempts": self.MAX_ATTEMPTS - attempt,
            "registry_summary": self.gateway.registry.summary(),
        }
        if evidence_override is not None:
            for entity in evidence_override.entities:
                replayed = entity.model_copy(deep=True)
                replayed.registered_by = "experience"
                self.gateway.registry.add(replayed, actor="experience")
            rag_chunks = [item.model_copy(deep=True) for item in evidence_override.rag_chunks]
            if not self._attraction_ids():
                raise ExperienceAgentError("snapshot_mismatch", "replay snapshot contains no attraction evidence")
            proposal = self._build_proposal(request, rag_chunks, revision_context, attempt)
            self.trace["evidence_mode"] = "replay"
            return ExperienceAgentResult(proposal=proposal, rag_chunks=rag_chunks)

        try:
            research = invoke_structured(
                self.llm,
                ExperienceResearchPlan,
                self._research_prompt(request, revision_context),
            )
        except Exception as exc:
            raise ExperienceAgentError("structured_output", str(exc)) from exc

        rag_chunks: List[RAGChunk] = []
        try:
            queries = self._attraction_queries(request, research.attraction_queries)
            for query_index, (query, source_type) in enumerate(queries):
                items = self.gateway.call(
                    "experience", "attraction_search", query_key=query,
                    query=query, city=request.city, country_code=request.country_code, page_size=12,
                )
                self.gateway.register("experience", self._registry_entities(
                    items, source_type=source_type, query=query, query_index=query_index))
                if len(self._attraction_ids()) >= 12 and self._preference_coverage(request):
                    self.gateway.early_stop_reasons["experience"] = "pool_target_and_preference_coverage"
                    break
            if research.rag_query:
                result = self.gateway.call(
                    "experience", "rag_search", query_key=research.rag_query,
                    query=research.rag_query, request=request,
                )
                rag_chunks = [item if isinstance(item, RAGChunk) else RAGChunk.model_validate(item) for item in result]
            for source_id in research.detail_source_ids[:2]:
                if source_id not in self.gateway.registry.entities:
                    continue
                entity = self.gateway.registry.entities[source_id]
                detail = self.gateway.call(
                    "experience", "place_detail", query_key=source_id,
                    source_id=entity.provider_id or entity.source_id,
                )
                entities = self._registry_entities([detail], source_type="place_details",
                                                   query=source_id, query_index=0)
                if entities:
                    self.gateway.register("experience", entities)
        except ToolGatewayError as exc:
            if exc.transient and self.deterministic_fallback is not None:
                return self._fallback(request, attempt)
            if exc.code == "tool_budget_exhausted" and self._attraction_ids():
                pass
            else:
                raise ExperienceAgentError(exc.code, str(exc)) from exc

        if not self._attraction_ids():
            if self.deterministic_fallback is not None:
                return self._fallback(request, attempt)
            raise ExperienceAgentError("no_attraction_evidence", "no valid attraction evidence")
        proposal = self._build_proposal(request, rag_chunks, revision_context, attempt)
        return ExperienceAgentResult(proposal=proposal, rag_chunks=rag_chunks)

    def _build_proposal(self, request: TripRequest, rag_chunks: List[RAGChunk], context: dict,
                        attempt: int) -> ExperienceProposal:
        ranked = shortlist(self.gateway.registry, "attraction", request, limit=12)
        aliases = alias_map(ranked, "A")
        target = min(len(ranked), pace_target(request))
        core_count = min(target, max(2, math.ceil(target * 0.6)))
        optional_count = min(max(0, len(ranked) - core_count), max(0, target - core_count + 2))
        prompt = (
            "You are the Experience specialist. Create at most four thematic clusters. "
            "Use only A aliases below; do not assign dates. Select exactly the requested core and optional counts.\n"
            f"request={request.model_dump()}\ntarget={target} core_count={core_count} optional_count={optional_count}\n"
            f"candidates={compact_candidates(ranked, aliases)}\n"
            f"rag_chunk_ids={[chunk.chunk_id for chunk in rag_chunks]}\nrevision={context}"
        )
        try:
            alias_proposal = invoke_structured(self.llm, ExperienceAliasProposal, prompt)
        except Exception as exc:
            raise ExperienceAgentError("structured_output", str(exc)) from exc
        try:
            core = resolve_aliases(alias_proposal.core_attraction_aliases, aliases, expected_prefix="A")
            optional = resolve_aliases(alias_proposal.optional_attraction_aliases, aliases, expected_prefix="A")
            if len(core) != core_count or len(optional) != optional_count or set(core) & set(optional):
                raise ValueError("invalid attraction cardinality")
            clusters = [ExperienceCluster(name=item.name,
                                           attraction_ids=resolve_aliases(item.attraction_aliases, aliases, expected_prefix="A"),
                                           rationale=item.rationale)
                        for item in alias_proposal.clusters]
        except ValueError as exc:
            raise ExperienceAgentError("invalid_source_id", str(exc)) from exc
        invalid_rag = set(alias_proposal.rag_chunk_ids) - {chunk.chunk_id for chunk in rag_chunks}
        if invalid_rag:
            raise ExperienceAgentError("invalid_source_id", f"invalid RAG IDs: {sorted(invalid_rag)}")
        proposal = ExperienceProposal(run_id=self.gateway.registry.run_id,
                                      version=(context["previous_proposal"] or {}).get("version", 0) + 1,
                                      clusters=clusters, rag_chunk_ids=alias_proposal.rag_chunk_ids,
                                      uncovered_preferences=alias_proposal.uncovered_preferences,
                                      evidence_sufficient=alias_proposal.evidence_sufficient,
                                      core_attraction_ids=core, optional_attraction_ids=optional,
                                      target_attractions=target)
        self.trace = {"candidate_pool_ids": sorted(self._attraction_ids()),
                      "shortlist_ids": [item.source_id for item in ranked],
                      "alias_map": aliases,
                      "core_ids": core, "optional_ids": optional}
        return proposal

    def _fallback(self, request: TripRequest, attempt: int) -> ExperienceAgentResult:
        items, rag_chunks = self.deterministic_fallback(request) if self.deterministic_fallback else ([], [])
        self.gateway.register("experience", self._registry_entities(
            items, source_type="base_anchor", query="deterministic fallback", query_index=0))
        ids = self._attraction_ids()
        if not ids:
            raise ExperienceAgentError("no_attraction_evidence", "agent and fallback found no attractions")
        from ..models.multi_agent import ExperienceCluster
        proposal = ExperienceProposal(
            run_id=self.gateway.registry.run_id,
            version=attempt,
            clusters=[ExperienceCluster(name="Conservative highlights", attraction_ids=ids[:8],
                                        rationale="Deterministic fallback after provider failure")],
            rag_chunk_ids=[chunk.chunk_id for chunk in rag_chunks],
            uncovered_preferences=list(request.preferences),
            evidence_sufficient=False,
            core_attraction_ids=ids[:max(2, min(len(ids), pace_target(request)))],
            optional_attraction_ids=[],
            target_attractions=min(len(ids), pace_target(request)),
        )
        return ExperienceAgentResult(proposal=proposal, rag_chunks=rag_chunks,
                                     used_deterministic_fallback=True)

    def _registry_entities(self, items: Any, *, source_type: str, query: str,
                           query_index: int) -> List[RegistryEntity]:
        entities = []
        for provider_rank, item in enumerate(list(items or []), 1):
            if not isinstance(item, dict):
                continue
            provider_id = str(item.get("id") or item.get("provider_id") or item.get("source_id") or "")
            name = str(item.get("name") or "")
            if not provider_id or not name:
                continue
            entities.append(RegistryEntity(source_id=registry_source_id("attraction", provider_id),
                                            provider_id=provider_id, entity_type="attraction", name=name,
                                            address=str(item.get("address") or ""),
                                            location=item.get("location") or None,
                                            rating=item.get("rating"), maps_url=item.get("maps_url"),
                                            user_rating_count=item.get("user_rating_count"),
                                            website_url=item.get("website_url"), image_url=item.get("image_url"),
                                            photo_names=list(item.get("photo_names") or []),
                                            metadata={"category": item.get("type") or "Attraction"},
                                            registered_by="experience",
                                            observations=[CandidateObservation(
                                                source_type=source_type, normalized_query=normalize_text(query),
                                                query_index=query_index, provider_rank=provider_rank,
                                                provider_id=provider_id, name=name,
                                                address=str(item.get("address") or ""),
                                                location=item.get("location") or None, rating=item.get("rating"),
                                                user_rating_count=item.get("user_rating_count"),
                                                maps_url=item.get("maps_url"), website_url=item.get("website_url"),
                                                image_url=item.get("image_url"),
                                                photo_names=list(item.get("photo_names") or []),
                                                metadata={"category": item.get("type") or "Attraction"})]))
        return entities

    def _attraction_ids(self) -> List[str]:
        return [source_id for source_id, entity in self.gateway.registry.entities.items()
                if entity.entity_type == "attraction" and entity.registered_by == "experience"]

    def _preference_coverage(self, request: TripRequest) -> bool:
        if not request.preferences:
            return True
        text = normalize_text(" ".join(entity.name + " " + str(entity.metadata)
                                       for entity in self.gateway.registry.entities.values()
                                       if entity.entity_type == "attraction"))
        return all(normalize_text(value) in text for value in request.preferences if normalize_text(value))

    @staticmethod
    def _research_prompt(request: TripRequest, context: dict) -> str:
        return (
            "You are a bounded Experience research agent. Return up to two backup attraction queries, "
            "one optional RAG query, and up to two existing source IDs for details. "
            "The runtime executes deterministic city/preference queries first and uses backup queries only "
            "when fewer than two distinct preference queries exist. "
            "Current request overrides memory. Do not produce an itinerary.\n"
            f"request={request.model_dump()}\nrevision={context}"
        )

    @staticmethod
    def _attraction_queries(request: TripRequest, research_queries: List[str]) -> List[tuple[str, str]]:
        """Build an arrival-order-independent anchor plus two stable supplemental queries."""
        city = normalize_text(request.city)
        preferences = sorted({normalize_text(value) for value in request.preferences if normalize_text(value)})
        anchor_terms = " ".join(preferences)
        anchor = normalize_text(f"{city} {anchor_terms} attractions") or "top attractions"
        deterministic = [normalize_text(f"{city} {value} attractions") for value in preferences]
        backups = sorted({normalize_text(value) for value in research_queries if normalize_text(value)})
        supplemental = []
        for query in [*deterministic, *backups]:
            if query != anchor and query not in supplemental:
                supplemental.append(query)
            if len(supplemental) == 2:
                break
        return [(anchor, "base_anchor"), *[(query, "supplemental") for query in supplemental]]
