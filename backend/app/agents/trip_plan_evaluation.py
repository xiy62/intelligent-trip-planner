"""Shared evaluation utilities for legacy and LangGraph trip planners."""

from __future__ import annotations

import re
from math import asin, cos, radians, sin, sqrt
from typing import List, Optional

from ..models.langgraph_state import (
    AttractionCandidate,
    EvidenceLink,
    EvaluationReport,
    EvaluationScores,
    HotelCandidate,
    RAGChunk,
    RetryState,
    UnsupportedEntity,
)
from ..models.schemas import Location, TripPlan, TripRequest


def normalize_entity_name(value: str) -> str:
    """Normalize entity names for simple grounding checks."""
    normalized = (value or "").strip().lower()
    normalized = re.sub(r"\s+", "", normalized)
    normalized = re.sub(r"[()（）·\-—,:：'\"“”‘’]", "", normalized)
    for suffix in ("景区", "景点", "公园", "博物院", "博物馆", "酒店", "青年酒店", "商旅酒店", "连锁酒店", "大酒店"):
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
    return normalized


def entity_matches(name: str, candidate_names: List[str], rag_text: str) -> bool:
    """Allow exact and containment-based grounding matches."""
    normalized = normalize_entity_name(name)
    if not normalized:
        return False
    if normalized in candidate_names:
        return True
    for candidate in candidate_names:
        if not candidate:
            continue
        if normalized in candidate or candidate in normalized:
            return True
    return normalized in normalize_entity_name(rag_text)


def match_entity_to_evidence(
    *,
    entity_type: str,
    entity_name: str,
    day_index: Optional[int],
    candidates: List[AttractionCandidate] | List[HotelCandidate],
    rag_chunks: List[RAGChunk],
) -> EvidenceLink:
    """Return the strongest deterministic evidence link for an itinerary entity."""
    normalized = normalize_entity_name(entity_name)
    if not normalized:
        return EvidenceLink(
            entity_type=entity_type,
            entity_name=entity_name,
            day_index=day_index,
            evidence_type="none",
            match_reason="empty entity name",
        )

    candidate_evidence_type = (
        "candidate_attraction" if entity_type == "attraction" else "candidate_hotel"
    )
    for candidate in candidates:
        candidate_normalized = normalize_entity_name(candidate.name)
        if not candidate_normalized:
            continue
        if normalized == candidate_normalized:
            return EvidenceLink(
                entity_type=entity_type,
                entity_name=entity_name,
                day_index=day_index,
                evidence_type=candidate_evidence_type,
                evidence_id=candidate.source_id or candidate.name,
                source_title=candidate.name,
                confidence=1.0,
                match_reason="exact normalized candidate name match",
            )
        if normalized in candidate_normalized or candidate_normalized in normalized:
            return EvidenceLink(
                entity_type=entity_type,
                entity_name=entity_name,
                day_index=day_index,
                evidence_type=candidate_evidence_type,
                evidence_id=candidate.source_id or candidate.name,
                source_title=candidate.name,
                confidence=0.85,
                match_reason="partial normalized candidate name match",
            )

    for chunk in rag_chunks:
        searchable = normalize_entity_name(f"{chunk.title} {chunk.content}")
        if normalized and normalized in searchable:
            metadata = dict(chunk.metadata)
            return EvidenceLink(
                entity_type=entity_type,
                entity_name=entity_name,
                day_index=day_index,
                evidence_type="rag_chunk",
                evidence_id=metadata.get("doc_id") or chunk.chunk_id,
                source_title=chunk.title,
                source_url=str(metadata.get("source_url", "")),
                confidence=0.65,
                match_reason="entity name appears in retrieved RAG chunk",
            )

    return EvidenceLink(
        entity_type=entity_type,
        entity_name=entity_name,
        day_index=day_index,
        evidence_type="none",
        confidence=0.0,
        match_reason="no matching candidate or RAG evidence found",
    )


def haversine_km(origin: Location, destination: Location) -> float:
    """Calculate approximate geographic distance between two coordinates."""
    radius_km = 6371.0
    lat1 = radians(float(origin.latitude))
    lon1 = radians(float(origin.longitude))
    lat2 = radians(float(destination.latitude))
    lon2 = radians(float(destination.longitude))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    value = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * radius_km * asin(sqrt(value))


def route_distance_threshold_km(transportation: str) -> float:
    """Return a soft same-day POI jump threshold by requested transportation mode."""
    normalized = normalized_text(transportation)
    if any(term in normalized for term in ("步行", "徒步", "citywalk", "walk")):
        return 5.0
    if any(term in normalized for term in ("打车", "出租", "自驾", "驾车", "taxi", "car", "drive")):
        return 30.0
    return 15.0


def evaluate_pacing_quality(draft_plan: TripPlan) -> tuple[float, List[str]]:
    """Score whether each day is reasonably paced for a real itinerary."""
    day_scores: List[float] = []
    warnings: List[str] = []
    for day in draft_plan.days:
        attraction_count = len(day.attractions)
        visit_minutes = sum(max(0, attraction.visit_duration or 0) for attraction in day.attractions)
        meal_buffer = len(day.meals) * 45
        transit_buffer = max(0, attraction_count - 1) * 40
        active_minutes = visit_minutes + meal_buffer + transit_buffer
        score = 1.0
        if attraction_count == 0:
            score -= 0.6
            warnings.append(f"pacing_day_{day.day_index}_no_attractions")
        if attraction_count > 4:
            score -= min(0.35, (attraction_count - 4) * 0.12)
            warnings.append(f"pacing_day_{day.day_index}_too_many_attractions")
        if active_minutes > 600:
            score -= 0.35
            warnings.append(f"pacing_day_{day.day_index}_overloaded")
        elif active_minutes > 540:
            score -= 0.2
            warnings.append(f"pacing_day_{day.day_index}_tight")
        if attraction_count > 0 and active_minutes < 180:
            score -= 0.2
            warnings.append(f"pacing_day_{day.day_index}_underfilled")
        day_scores.append(max(0.0, round(score, 4)))
    if not day_scores:
        return 0.0, ["pacing_no_days"]
    return round(sum(day_scores) / len(day_scores), 4), warnings


def evaluate_route_coherence(draft_plan: TripPlan, request: TripRequest) -> tuple[float, List[str]]:
    """Score whether same-day attraction ordering is geographically coherent."""
    warnings: List[str] = []
    segment_scores: List[float] = []
    threshold_km = route_distance_threshold_km(request.transportation)
    for day in draft_plan.days:
        attractions = day.attractions
        if len(attractions) <= 1:
            continue
        comparable_segments = 0
        for index in range(len(attractions) - 1):
            origin = attractions[index].location
            destination = attractions[index + 1].location
            if origin is None or destination is None:
                continue
            comparable_segments += 1
            distance_km = haversine_km(origin, destination)
            if distance_km <= threshold_km:
                segment_scores.append(1.0)
            else:
                excess_ratio = min(1.0, (distance_km - threshold_km) / threshold_km)
                segment_scores.append(max(0.0, 1.0 - 0.7 - (0.3 * excess_ratio)))
                warnings.append(
                    f"route_day_{day.day_index}_long_jump_{round(distance_km, 1)}km"
                )
        if comparable_segments == 0 and len(attractions) > 1:
            warnings.append(f"route_day_{day.day_index}_missing_coordinates")
            segment_scores.append(0.5)
    if not segment_scores:
        return 1.0, warnings
    return round(sum(segment_scores) / len(segment_scores), 4), warnings


def preference_terms(request: TripRequest) -> List[str]:
    """Build deterministic preference terms from explicit labels and free text."""
    values = list(request.preferences)
    if request.free_text_input:
        free_text = request.free_text_input
        known_terms = [
            "历史文化",
            "城市经典",
            "经典路线",
            "美食",
            "夜景",
            "自然风光",
            "轻松",
            "拍照",
            "博物馆",
            "胡同",
            "中轴线",
            "故宫",
            "长城",
            "外滩",
            "西湖",
            "珠江",
            "不要太赶",
        ]
        values.extend(term for term in known_terms if term in free_text)
        values.extend(re.split(r"[\s,，。；;、/|和与]+", free_text))
    terms: List[str] = []
    for value in values:
        term = normalized_text(value)
        for prefix in ("希望以", "希望", "想要", "想", "以"):
            if term.startswith(prefix):
                term = term[len(prefix):]
        for suffix in ("为主", "为重点", "优先"):
            if term.endswith(suffix):
                term = term[: -len(suffix)]
        if len(term) >= 2 and term not in terms:
            terms.append(term)
    return terms


def evaluate_preference_match(draft_plan: TripPlan, request: TripRequest, rag_chunks: List[RAGChunk]) -> tuple[float, List[str]]:
    """Score whether generated itinerary text reflects explicit user preferences."""
    terms = preference_terms(request)
    if not terms:
        return 1.0, []
    plan_text_parts: List[str] = [draft_plan.overall_suggestions]
    for day in draft_plan.days:
        plan_text_parts.extend(
            [
                day.description,
                day.transportation,
                day.accommodation,
                " ".join(attraction.name for attraction in day.attractions),
                " ".join(attraction.description for attraction in day.attractions),
                " ".join(meal.name for meal in day.meals),
                " ".join(str(meal.description or "") for meal in day.meals),
            ]
        )
    rag_text = " ".join(f"{chunk.title} {chunk.content}" for chunk in rag_chunks)
    searchable = normalized_text(" ".join(plan_text_parts) + " " + rag_text)
    matched = [term for term in terms if term in searchable]
    score = len(matched) / len(terms)
    warnings = []
    missing = [term for term in terms if term not in matched]
    if missing:
        warnings.append(f"preference_terms_missing:{','.join(missing[:5])}")
    return round(score, 4), warnings


def next_action_with_retry_budget(retry_counts: RetryState, action: str, max_retries: int) -> str:
    """Route to fallback when a node already exhausted retry budget."""
    attempts = getattr(retry_counts, action, 0)
    if attempts >= max_retries + 1:
        return "fallback_response"
    return action


def normalized_text(value: str) -> str:
    """Normalize preference text for request-alignment checks."""
    normalized = (value or "").strip().lower()
    normalized = re.sub(r"\s+", "", normalized)
    return re.sub(r"[()（）·\-—,:：'\"“”‘’/、+，。|]", "", normalized)


def field_matches_request(actual: str, expected: str) -> bool:
    """Return whether generated text preserves an explicit request field."""
    actual_norm = normalized_text(actual)
    expected_norm = normalized_text(expected)
    if not expected_norm:
        return True
    if not actual_norm:
        return False
    return expected_norm in actual_norm or actual_norm in expected_norm


def evaluate_trip_plan(
    *,
    request: TripRequest,
    travel_dates: List[str],
    draft_plan: Optional[TripPlan],
    candidate_attractions: List[AttractionCandidate],
    candidate_hotels: List[HotelCandidate],
    rag_chunks: List[RAGChunk],
    retry_counts: RetryState,
    max_retries: int,
) -> EvaluationReport:
    """Evaluate a trip plan against schema, date, budget, and grounding checks."""
    scores = EvaluationScores()
    hard_failures: List[str] = []
    warnings: List[str] = []
    quality_warnings: List[str] = []
    unsupported_entities: List[UnsupportedEntity] = []
    unsupported_claims: List[str] = []
    evidence_links: List[EvidenceLink] = []

    if draft_plan is None:
        hard_failures.append("schema_correctness")
        scores.schema_score = 0.0
        return EvaluationReport(
            passed=False,
            hard_failures=hard_failures,
            warnings=warnings,
            quality_warnings=quality_warnings,
            scores=scores,
            unsupported_entities=unsupported_entities,
            unsupported_claims=unsupported_claims,
            evidence_links=evidence_links,
            next_action=next_action_with_retry_budget(retry_counts, "plan_itinerary", max_retries),
        )

    scores.schema_score = 1.0

    day_dates = [day.date for day in draft_plan.days]
    weather_dates = [item.date for item in draft_plan.weather_info]
    matched_day_dates = len([d for d in travel_dates if d in day_dates])
    matched_weather_dates = len([d for d in travel_dates if d in weather_dates])
    if travel_dates:
        scores.date_coverage_score = (
            (matched_day_dates / len(travel_dates)) + (matched_weather_dates / len(travel_dates))
        ) / 2.0
    if scores.date_coverage_score < 1.0:
        hard_failures.append("date_coverage")

    budget = draft_plan.budget
    total_attractions = sum(
        (attraction.ticket_price or 0)
        for day in draft_plan.days
        for attraction in day.attractions
    )
    total_hotels = sum(
        ((day.hotel.estimated_cost or 0) if day.hotel is not None else 0)
        for day in draft_plan.days
    )
    total_meals = sum(
        (meal.estimated_cost or 0)
        for day in draft_plan.days
        for meal in day.meals
    )
    if budget is None:
        scores.budget_consistency_score = 0.0
        hard_failures.append("budget_consistency")
    else:
        matches = 0
        total_checks = 4
        if budget.total_attractions == total_attractions:
            matches += 1
        if budget.total_hotels == total_hotels:
            matches += 1
        if budget.total_meals == total_meals:
            matches += 1
        if budget.total == (
            budget.total_attractions
            + budget.total_hotels
            + budget.total_meals
            + budget.total_transportation
        ):
            matches += 1
        scores.budget_consistency_score = matches / total_checks
        if scores.budget_consistency_score < 0.75:
            hard_failures.append("budget_consistency")
        elif scores.budget_consistency_score < 1.0:
            warnings.append("budget_total_not_fully_aligned")

    alignment_issues: List[str] = []
    expected_transportation = request.transportation.strip()
    expected_accommodation = request.accommodation.strip()
    for day in draft_plan.days:
        day_label = f"day_index={day.day_index}"
        if expected_transportation and not field_matches_request(day.transportation, expected_transportation):
            alignment_issues.append(
                f"{day_label} transportation='{day.transportation}' does not match current request '{expected_transportation}'"
            )
        if expected_accommodation and not field_matches_request(day.accommodation, expected_accommodation):
            alignment_issues.append(
                f"{day_label} accommodation='{day.accommodation}' does not match current request '{expected_accommodation}'"
            )
    if alignment_issues:
        hard_failures.append("current_request_alignment")
        unsupported_claims.extend(alignment_issues)

    attraction_names = [normalize_entity_name(candidate.name) for candidate in candidate_attractions]
    hotel_names = [normalize_entity_name(candidate.name) for candidate in candidate_hotels]
    rag_text = " ".join(chunk.content for chunk in rag_chunks)

    total_entities = 0
    grounded_entities = 0
    for day in draft_plan.days:
        for attraction in day.attractions:
            total_entities += 1
            evidence_link = match_entity_to_evidence(
                entity_type="attraction",
                entity_name=attraction.name,
                day_index=day.day_index,
                candidates=candidate_attractions,
                rag_chunks=rag_chunks,
            )
            evidence_links.append(evidence_link)
            if entity_matches(attraction.name, attraction_names, rag_text):
                grounded_entities += 1
            else:
                unsupported_entities.append(
                    UnsupportedEntity(
                        entity_type="attraction",
                        name=attraction.name,
                        reason="not found in retrieved attraction candidates or rag context",
                    )
                )
        if day.hotel is not None and day.hotel.name:
            total_entities += 1
            evidence_link = match_entity_to_evidence(
                entity_type="hotel",
                entity_name=day.hotel.name,
                day_index=day.day_index,
                candidates=candidate_hotels,
                rag_chunks=rag_chunks,
            )
            evidence_links.append(evidence_link)
            if entity_matches(day.hotel.name, hotel_names, rag_text):
                grounded_entities += 1
            else:
                unsupported_entities.append(
                    UnsupportedEntity(
                        entity_type="hotel",
                        name=day.hotel.name,
                        reason="not found in retrieved hotel candidates or rag context",
                    )
                )
    if total_entities:
        scores.grounding_score = grounded_entities / total_entities
    else:
        warnings.append("no_entities_found_for_grounding")
        scores.grounding_score = 0.0
    supported_links = [link for link in evidence_links if link.confidence >= 0.65]
    scores.attribution_coverage_score = (
        len(supported_links) / len(evidence_links) if evidence_links else 0.0
    )
    if evidence_links and scores.attribution_coverage_score < 0.8:
        quality_warnings.append("low_attribution_coverage")

    if any(item.entity_type == "attraction" for item in unsupported_entities):
        hard_failures.append("retrieval_grounding_attractions")
    if any(item.entity_type == "hotel" for item in unsupported_entities):
        hard_failures.append("retrieval_grounding_hotels")

    scores.pacing_score, pacing_warnings = evaluate_pacing_quality(draft_plan)
    scores.route_coherence_score, route_warnings = evaluate_route_coherence(draft_plan, request)
    scores.preference_match_score, preference_warnings = evaluate_preference_match(
        draft_plan, request, rag_chunks
    )
    quality_warnings.extend(pacing_warnings)
    quality_warnings.extend(route_warnings)
    quality_warnings.extend(preference_warnings)
    if scores.pacing_score < 0.75 and "low_pacing_score" not in quality_warnings:
        quality_warnings.append("low_pacing_score")
    if scores.route_coherence_score < 0.75 and "low_route_coherence_score" not in quality_warnings:
        quality_warnings.append("low_route_coherence_score")
    if scores.preference_match_score < 0.6 and "low_preference_match_score" not in quality_warnings:
        quality_warnings.append("low_preference_match_score")

    next_action = "finalize_response"
    passed = len(hard_failures) == 0
    if not passed:
        if "schema_correctness" in hard_failures:
            next_action = next_action_with_retry_budget(retry_counts, "plan_itinerary", max_retries)
        elif "retrieval_grounding_attractions" in hard_failures:
            next_action = next_action_with_retry_budget(retry_counts, "retrieve_attractions", max_retries)
        elif "retrieval_grounding_hotels" in hard_failures:
            next_action = next_action_with_retry_budget(retry_counts, "retrieve_hotels", max_retries)
        else:
            next_action = next_action_with_retry_budget(retry_counts, "plan_itinerary", max_retries)

    return EvaluationReport(
        passed=passed,
        hard_failures=hard_failures,
        warnings=warnings,
        quality_warnings=quality_warnings,
        scores=scores,
        unsupported_entities=unsupported_entities,
        unsupported_claims=unsupported_claims,
        evidence_links=evidence_links,
        next_action=next_action,
    )
