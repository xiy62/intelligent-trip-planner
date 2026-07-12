"""Deterministic evidence scoring, shortlists, and prompt-local aliases."""

from __future__ import annotations

import math
import re
from typing import Dict, Iterable, List, Optional

from ..models.multi_agent import CandidateRegistry, EntityType, RegistryEntity
from ..models.schemas import TripRequest


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def normalize_text(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", (value or "").lower()))


def provider_rank_score(rank: Optional[int]) -> float:
    if rank is None:
        return 0.0
    return clamp((8 - rank) / 7)


def bayesian_adjusted_rating(rating: Optional[float], review_count: Optional[int], *, prior: float = 4.2,
                             prior_weight: int = 50) -> Optional[float]:
    if rating is None or review_count is None or review_count < 0:
        return None
    return (review_count / (review_count + prior_weight)) * rating + (prior_weight / (review_count + prior_weight)) * prior


def rating_confidence(rating: Optional[float], review_count: Optional[int]) -> float:
    adjusted = bayesian_adjusted_rating(rating, review_count)
    return clamp(adjusted / 5) if adjusted is not None else 0.0


def pace_target(request: TripRequest) -> int:
    text = normalize_text(" ".join(request.preferences) + " " + (request.free_text_input or ""))
    if any(term in text for term in ("relaxed", "slow", "easy", "not rushed", "leisurely")):
        multiplier = 1.5
    elif any(term in text for term in ("packed", "maximize", "fast paced", "as much as possible")):
        multiplier = 2.5
    else:
        multiplier = 2.0
    return max(3, min(10, round(request.travel_days * multiplier)))


def _term_match(entity: RegistryEntity, terms: Iterable[str]) -> float:
    haystack = normalize_text(" ".join([entity.name, entity.address, str(entity.metadata), *entity.query_provenance]))
    normalized = [normalize_text(term) for term in terms if normalize_text(term)]
    return sum(1 for term in normalized if term in haystack) / len(normalized) if normalized else 0.0


def _proximity(entity: RegistryEntity, centroid: Optional[tuple[float, float]]) -> float:
    if entity.location is None or centroid is None:
        return 0.0
    lat, lon = entity.location.latitude, entity.location.longitude
    c_lat, c_lon = centroid
    km = math.sqrt(((lat - c_lat) * 111) ** 2 + ((lon - c_lon) * 85) ** 2)
    return clamp(1 - km / 15)


def rank_entities(registry: CandidateRegistry, entity_type: EntityType, request: TripRequest, *,
                  centroid: Optional[tuple[float, float]] = None) -> List[RegistryEntity]:
    result = []
    for entity in registry.entities.values():
        if entity.entity_type != entity_type or not entity.name:
            continue
        query_hits = min(len(set(entity.query_provenance)), 3) / 3
        rating = rating_confidence(entity.rating, entity.user_rating_count)
        rank = provider_rank_score(entity.best_provider_rank)
        proximity = _proximity(entity, centroid)
        if entity_type == "attraction":
            match = _term_match(entity, request.preferences)
            components = {"preference_match": match, "query_hit_score": query_hits,
                          "rating_confidence": rating, "provider_rank_score": rank}
            score = 3 * match + 1.5 * query_hits + 0.5 * rating + 0.5 * rank
        elif entity_type == "hotel":
            match = _term_match(entity, [request.accommodation])
            components = {"accommodation_match": match, "proximity_score": proximity,
                          "query_hit_score": query_hits, "rating_confidence": rating,
                          "provider_rank_score": rank}
            score = 3 * match + 2 * proximity + 1.5 * query_hits + rating + 0.5 * rank
        else:
            terms = list(request.preferences) + ["food", "dining", "restaurant"]
            match = _term_match(entity, terms)
            components = {"preference_theme_match": match, "proximity_score": proximity,
                          "query_hit_score": query_hits, "rating_confidence": rating,
                          "provider_rank_score": rank}
            score = 3 * match + 1.5 * proximity + 1.5 * query_hits + rating + 0.5 * rank
        ranked = entity.model_copy(deep=True)
        ranked.relevance_score = round(score, 6)
        ranked.score_components = {key: round(value, 6) for key, value in components.items()}
        result.append(ranked)
    return sorted(result, key=lambda item: (-item.relevance_score, item.provider_id))


def shortlist(registry: CandidateRegistry, entity_type: EntityType, request: TripRequest, *, limit: int,
              centroid: Optional[tuple[float, float]] = None) -> List[RegistryEntity]:
    return rank_entities(registry, entity_type, request, centroid=centroid)[:limit]


def alias_map(items: List[RegistryEntity], prefix: str) -> Dict[str, str]:
    return {f"{prefix}{index}": item.source_id for index, item in enumerate(items, 1)}


def resolve_aliases(values: Iterable[str], aliases: Dict[str, str], *, expected_prefix: str) -> List[str]:
    resolved: List[str] = []
    for value in values:
        if not isinstance(value, str) or not value.startswith(expected_prefix) or value not in aliases:
            raise ValueError(f"invalid {expected_prefix} alias: {value}")
        source_id = aliases[value]
        if source_id in resolved:
            raise ValueError(f"duplicate {expected_prefix} alias: {value}")
        resolved.append(source_id)
    return resolved


def compact_candidates(items: List[RegistryEntity], aliases: Dict[str, str]) -> List[dict]:
    reverse = {source_id: alias for alias, source_id in aliases.items()}
    return [{"alias": reverse[item.source_id], "name": item.name, "address": item.address,
             "rating": item.rating, "review_count": item.user_rating_count,
             "score": item.relevance_score, "score_components": item.score_components}
            for item in items]
