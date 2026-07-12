"""Active planner factory. Runtime switching is intentionally unsupported."""

from __future__ import annotations

from typing import Optional

from ..config import get_settings
from .multi_agent_trip_planner import MultiAgentTripPlanner

_planner: Optional[MultiAgentTripPlanner] = None


def get_trip_planner_agent() -> MultiAgentTripPlanner:
    """Return the active planner without reading a runtime mode flag."""
    global _planner
    if _planner is None:
        _planner = MultiAgentTripPlanner(rag_mode=get_settings().rag_mode)
    return _planner


def reset_trip_planner_agent() -> None:
    global _planner
    _planner = None
