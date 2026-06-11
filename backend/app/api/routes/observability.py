"""Local observability API for planner run traces."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from ...services.observability_service import get_observability_service

router = APIRouter(prefix="/observability", tags=["Observability"])


@router.get(
    "/summary",
    summary="Planner observability summary",
    description="Return aggregate evaluation, retry, latency, and failure metrics for persisted runs.",
)
async def get_observability_summary():
    """Return aggregate observability metrics."""
    try:
        return {
            "success": True,
            "data": get_observability_service().summary(),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read observability summary: {exc}")


@router.get(
    "/runs",
    summary="List planner runs",
    description="Return recent planner runs with optional filters.",
)
async def list_observability_runs(
    limit: int = Query(default=50, ge=1, le=500),
    source: Optional[str] = Query(default=None),
    city: Optional[str] = Query(default=None),
    passed: Optional[bool] = Query(default=None),
    failure_type: Optional[str] = Query(default=None),
):
    """Return recent runs."""
    try:
        return {
            "success": True,
            "data": get_observability_service().list_runs(
                limit=limit,
                source=source,
                city=city,
                passed=passed,
                failure_type=failure_type,
            ),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read observability runs: {exc}")


@router.get(
    "/runs/{run_id}",
    summary="Get planner run detail",
    description="Return one planner run plus ordered node/evaluation/routing events.",
)
async def get_observability_run_detail(run_id: str):
    """Return run detail."""
    try:
        detail = get_observability_service().get_run_detail(run_id)
        if detail is None:
            raise HTTPException(status_code=404, detail="run_id does not exist")
        return {
            "success": True,
            "data": detail,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read observability run: {exc}")


@router.delete(
    "/runs",
    summary="Delete persisted planner runs",
    description="Local development cleanup endpoint. Optionally filter by source.",
)
async def delete_observability_runs(source: Optional[str] = Query(default=None)):
    """Delete persisted observability runs."""
    try:
        deleted = get_observability_service().delete_runs(source=source)
        return {
            "success": True,
            "deleted": deleted,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to delete observability runs: {exc}")
