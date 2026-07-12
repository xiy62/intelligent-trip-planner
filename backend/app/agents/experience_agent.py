"""Bounded attraction and destination-knowledge specialist."""

from __future__ import annotations

from typing import Any, Callable, List, Optional

from pydantic import BaseModel, Field

from ..models.langgraph_state import RAGChunk
from ..models.multi_agent import (
    AgentFeedback,
    CandidateRegistry,
    ExperienceProposal,
    RegistryEntity,
)
from ..models.schemas import TripRequest
from .structured_llm import invoke_structured
from .tool_gateway import ToolGateway, ToolGatewayError


class ExperienceResearchPlan(BaseModel):
    attraction_queries: List[str] = Field(default_factory=list, max_length=3)
    rag_query: str = ""
    detail_source_ids: List[str] = Field(default_factory=list, max_length=2)


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

    def run(
        self,
        *,
        request: TripRequest,
        feedback: Optional[AgentFeedback] = None,
        previous: Optional[ExperienceProposal] = None,
        attempt: int = 1,
    ) -> ExperienceAgentResult:
        if attempt < 1 or attempt > self.MAX_ATTEMPTS:
            raise ExperienceAgentError("retry_budget_exhausted", "experience attempt budget exhausted")
        revision_context = {
            "previous_proposal": previous.model_dump() if previous else None,
            "feedback": feedback.model_dump() if feedback else None,
            "remaining_attempts": self.MAX_ATTEMPTS - attempt,
            "registry_summary": self.gateway.registry.summary(),
        }
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
            for query in research.attraction_queries[:3]:
                items = self.gateway.call(
                    "experience", "attraction_search", query_key=query,
                    query=query, city=request.city, country_code=request.country_code,
                )
                self.gateway.register("experience", self._registry_entities(items))
            if research.rag_query:
                result = self.gateway.call(
                    "experience", "rag_search", query_key=research.rag_query,
                    query=research.rag_query, request=request,
                )
                rag_chunks = [item if isinstance(item, RAGChunk) else RAGChunk.model_validate(item) for item in result]
            for source_id in research.detail_source_ids[:2]:
                if source_id not in self.gateway.registry.entities:
                    continue
                detail = self.gateway.call(
                    "experience", "place_detail", query_key=source_id, source_id=source_id,
                )
                entities = self._registry_entities([detail])
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
        allowed = self._attraction_ids()
        prompt = (
            "You are the Experience specialist. Create at most four thematic clusters. "
            "Use only attraction IDs and RAG chunk IDs listed below; do not assign dates.\n"
            f"request={request.model_dump()}\nregistry_ids={allowed}\n"
            f"rag_chunk_ids={[chunk.chunk_id for chunk in rag_chunks]}\nrevision={context}"
        )
        try:
            proposal = invoke_structured(self.llm, ExperienceProposal, prompt)
        except Exception as exc:
            raise ExperienceAgentError("structured_output", str(exc)) from exc
        proposal.run_id = self.gateway.registry.run_id
        proposal.version = (context["previous_proposal"] or {}).get("version", 0) + 1
        invalid = proposal.allowed_attraction_ids - set(allowed)
        invalid_rag = set(proposal.rag_chunk_ids) - {chunk.chunk_id for chunk in rag_chunks}
        if invalid or invalid_rag:
            raise ExperienceAgentError("invalid_source_id", f"invalid IDs: {sorted(invalid | invalid_rag)}")
        if not proposal.allowed_attraction_ids:
            raise ExperienceAgentError("no_attraction_evidence", "proposal selected no attractions")
        return proposal

    def _fallback(self, request: TripRequest, attempt: int) -> ExperienceAgentResult:
        items, rag_chunks = self.deterministic_fallback(request) if self.deterministic_fallback else ([], [])
        self.gateway.register("experience", self._registry_entities(items))
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
        )
        return ExperienceAgentResult(proposal=proposal, rag_chunks=rag_chunks,
                                     used_deterministic_fallback=True)

    def _registry_entities(self, items: Any) -> List[RegistryEntity]:
        entities = []
        for item in list(items or []):
            if not isinstance(item, dict):
                continue
            source_id = str(item.get("id") or item.get("source_id") or "")
            name = str(item.get("name") or "")
            if not source_id or not name:
                continue
            entities.append(RegistryEntity(source_id=source_id, entity_type="attraction", name=name,
                                            address=str(item.get("address") or ""),
                                            location=item.get("location") or None,
                                            rating=item.get("rating"), maps_url=item.get("maps_url"),
                                            website_url=item.get("website_url"), image_url=item.get("image_url"),
                                            photo_names=list(item.get("photo_names") or []),
                                            metadata={"category": item.get("type") or "Attraction"},
                                            registered_by="experience"))
        return entities

    def _attraction_ids(self) -> List[str]:
        return [source_id for source_id, entity in self.gateway.registry.entities.items()
                if entity.entity_type == "attraction" and entity.registered_by == "experience"]

    @staticmethod
    def _research_prompt(request: TripRequest, context: dict) -> str:
        return (
            "You are a bounded Experience research agent. Return up to three attraction queries, "
            "one optional RAG query, and up to two existing source IDs for details. "
            "Current request overrides memory. Do not produce an itinerary.\n"
            f"request={request.model_dump()}\nrevision={context}"
        )
