"""Prompts and query builders for attraction retrieval."""

from __future__ import annotations

from ..models.schemas import TripRequest


ATTRACTION_AGENT_PROMPT = """You are an attraction retrieval specialist. Search for places that match the city and user preferences.

Important:
Use the map search tool. Do not invent attractions.

Tool name: map_search_poi
"""


def build_attraction_search_terms(request: TripRequest, retry_count: int = 0) -> list[str]:
    terms: list[str] = []
    if request.preferences:
        terms.extend(request.preferences[:2])
    else:
        terms.append("attractions")
    if retry_count > 0 or request.preferences:
        terms.append("top attractions")
    deduped: list[str] = []
    for term in terms:
        if term and term not in deduped:
            deduped.append(term)
    return deduped or ["attractions"]


def build_attraction_query(request: TripRequest) -> str:
    keyword = build_attraction_search_terms(request)[0]
    return (
        f"Search for {keyword} in {request.city} using map_search_poi."
    )
