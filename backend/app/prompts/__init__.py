"""Prompt package for trip planning runtime and legacy baseline."""

from .attraction import ATTRACTION_AGENT_PROMPT, build_attraction_query, build_attraction_search_terms
from .hotel import HOTEL_AGENT_PROMPT, build_hotel_query, build_hotel_search_terms
from .planner import PLANNER_AGENT_PROMPT, build_planner_prompt, build_retry_feedback

__all__ = [
    "ATTRACTION_AGENT_PROMPT",
    "HOTEL_AGENT_PROMPT",
    "PLANNER_AGENT_PROMPT",
    "build_attraction_query",
    "build_attraction_search_terms",
    "build_hotel_query",
    "build_hotel_search_terms",
    "build_planner_prompt",
    "build_retry_feedback",
]
