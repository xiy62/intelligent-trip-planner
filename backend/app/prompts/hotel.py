"""Prompts and query builders for hotel retrieval."""

from __future__ import annotations

from ..models.schemas import TripRequest


HOTEL_AGENT_PROMPT = """You are a hotel retrieval specialist. Search for hotels that match the city and accommodation preference.

Important:
Use the map search tool. Do not invent hotels.

Tool name: map_search_poi
"""


def build_hotel_search_terms(request: TripRequest) -> list[str]:
    terms = [request.accommodation]
    lowered = request.accommodation.lower()
    if "hotel" not in lowered and "inn" not in lowered:
        terms.append("hotels")
    deduped: list[str] = []
    for term in terms:
        if term and term not in deduped:
            deduped.append(term)
    return deduped or ["hotels"]


def build_hotel_query(request: TripRequest) -> str:
    keyword = build_hotel_search_terms(request)[0]
    return f"Search for {keyword} in {request.city}."
