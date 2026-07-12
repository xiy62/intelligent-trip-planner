"""Trip-planning API routes."""

from fastapi import APIRouter, Depends, HTTPException

from ..dependencies import require_admin_token
from ...agents.planner_factory import get_trip_planner_agent
from ...models.schemas import MemoryClearRequest, MemoryClearResponse, TripPlanResponse, TripRequest, ValidationSummary
from ...services.memory_service import get_memory_service
from ...services.observability_service import get_observability_service

router = APIRouter(prefix="/trip", tags=["Trip Planning"])


def build_validation_summary(state: dict) -> ValidationSummary | None:
    """Create a sanitized public validation summary from graph state."""
    report = state.get("evaluation_report")
    metrics = state.get("metrics")
    next_action = getattr(report, "next_action", None) if report is not None else None
    fallback_used = bool(
        getattr(metrics, "fallback_count", 0)
        or state.get("final_plan") is None
        or next_action == "fallback_response"
    )
    if report is None and not fallback_used:
        return None

    hard_failures = set(getattr(report, "hard_failures", []) if report is not None else [])
    scores = getattr(report, "scores", None) if report is not None else None
    evidence_links = list(getattr(report, "evidence_links", []) if report is not None else [])
    supported_links = [link for link in evidence_links if getattr(link, "confidence", 0.0) >= 0.65]

    evidence_summary = None
    if report is not None:
        if evidence_links:
            summary_parts = ["Attractions and hotels were checked against retrieved map/RAG evidence."]
            if any(getattr(link, "entity_type", "") == "meal" for link in evidence_links):
                summary_parts.append("Concrete named restaurants were checked against retrieved restaurant candidates.")
            if any(getattr(link, "evidence_type", "") == "rag_chunk" for link in evidence_links):
                summary_parts.append("Destination context was cross-checked with retrieved travel knowledge.")
            evidence_summary = " ".join(summary_parts)
        else:
            evidence_summary = "The plan was checked for date coverage, budget consistency, pacing, and route coherence."

    return ValidationSummary(
        validated=bool(report and report.passed and not fallback_used),
        fallback_used=fallback_used,
        date_coverage_passed=bool(report and "date_coverage" not in hard_failures),
        budget_consistency_passed=bool(report and "budget_consistency" not in hard_failures),
        grounding_score=getattr(scores, "grounding_score", None),
        attribution_coverage_score=getattr(scores, "attribution_coverage_score", None),
        pacing_score=getattr(scores, "pacing_score", None),
        route_coherence_score=getattr(scores, "route_coherence_score", None),
        quality_warnings=list(getattr(report, "quality_warnings", []) if report is not None else []),
        grounded_entity_count=len(supported_links) if report is not None else None,
        checked_entity_count=len(evidence_links) if report is not None else None,
        evidence_summary=evidence_summary,
    )


@router.post(
    "/plan",
    response_model=TripPlanResponse,
    summary="Generate a trip plan",
    description="Generate and validate a detailed itinerary from a structured travel request.",
)
async def plan_trip(request: TripRequest):
    """Run the LangGraph planning workflow."""
    try:
        print(
            f"Planning request: city={request.city} "
            f"dates={request.start_date}..{request.end_date} days={request.travel_days}"
        )
        agent = get_trip_planner_agent()
        state = agent.plan_trip_with_state(request)
        trip_plan = state.get("final_plan")
        try:
            get_observability_service().persist_state(
                state,
                source="runtime",
                rag_mode=getattr(agent, "rag_mode", ""),
            )
        except Exception as observability_error:
            print(f"Observability persistence failed; returning plan: {observability_error}")

        return TripPlanResponse(
            success=True,
            message="Trip plan generated successfully",
            data=trip_plan,
            conversation_id=state.get("conversation_id"),
            memory_applied=bool(state.get("memory_applied")),
            memory_summary=state.get("memory_summary") or None,
            memory_profile=state.get("memory_profile") or None,
            memory_conflicts=list(state.get("memory_conflicts") or []),
            validation_summary=build_validation_summary(state),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Trip planning failed: {exc}") from exc


@router.get(
    "/health",
    summary="Check planner health",
    description="Return active workflow metadata and service availability.",
)
async def health_check():
    """Return planner health metadata."""
    try:
        summary = get_trip_planner_agent().health_summary()
        return {
            "status": "healthy",
            "service": "trip-planner",
            "planner_name": summary["planner_name"],
            "workflow": summary["workflow"],
            "nodes": summary["nodes"],
        }
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Planner unavailable: {exc}") from exc


@router.post(
    "/memory/clear",
    response_model=MemoryClearResponse,
    summary="Clear anonymous preference memory",
    description="Delete persisted preference memory for an anonymous profile ID.",
)
async def clear_memory(request: MemoryClearRequest):
    """Clear anonymous profile memory."""
    try:
        get_memory_service().clear_profile(request.profile_id)
        return MemoryClearResponse(
            success=True,
            message="Anonymous preference memory cleared",
            profile_id=request.profile_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Memory cleanup failed: {exc}") from exc


@router.get(
    "/memory/{profile_id}",
    summary="Inspect anonymous preference memory",
    description="Return stored preference metadata for local debugging.",
)
async def get_memory_profile(profile_id: str, _admin: None = Depends(require_admin_token)):
    """Return anonymous profile memory for local debugging."""
    try:
        memory_service = get_memory_service()
        profile = memory_service.get_profile(profile_id)
        return {
            "success": True,
            "profile_id": profile_id,
            "data": profile,
            "memory_summary": memory_service.build_memory_context(profile_id),
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Memory lookup failed: {exc}") from exc
