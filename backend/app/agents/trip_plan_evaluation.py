"""Shared evaluation utilities for LangGraph trip planning."""

from __future__ import annotations

import re
from math import asin, cos, radians, sin, sqrt
from typing import Dict, List, Optional

from ..models.langgraph_state import (
    AttractionCandidate,
    EvidenceLink,
    EvaluationReport,
    EvaluationScores,
    HotelCandidate,
    MealCandidate,
    RAGChunk,
    RouteTimeEstimate,
    RetryState,
    UnsupportedEntity,
)
from ..models.schemas import Location, Meal, TripPlan, TripRequest


GENERIC_MEAL_NAME_TERMS = (
    "local cafe",
    "local restaurant",
    "near the",
    "nearby",
    "neighborhood",
    "street food",
    "food hall",
    "food market",
    "market lunch",
    "market dinner",
    "museum cafe",
    "hotel breakfast",
)

DEFAULT_MAX_SEGMENT_MINUTES_BY_MODE = {
    "walking": 30,
    "transit": 45,
    "driving": 35,
    "bicycling": 30,
}

DEFAULT_MAX_DAILY_TRANSIT_MINUTES_BY_MODE = {
    "walking": 90,
    "transit": 150,
    "driving": 120,
    "bicycling": 120,
}


def normalize_entity_name(value: str) -> str:
    """Normalize entity names for simple grounding checks."""
    normalized = (value or "").strip().lower()
    normalized = re.sub(r"\s+", "", normalized)
    normalized = re.sub(r"[()（）·\-—,:：'\"“”‘’]", "", normalized)
    for suffix in (
        "scenicarea",
        "attraction",
        "park",
        "museum",
        "hotel",
        "inn",
        "hostel",
        "resort",
        "景区",
        "景点",
        "公园",
        "博物院",
        "博物馆",
        "酒店",
        "青年酒店",
        "商旅酒店",
        "连锁酒店",
        "大酒店",
    ):
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


def is_concrete_meal_recommendation(meal: Meal) -> bool:
    """Return whether a meal is a concrete restaurant claim that requires evidence."""
    name = (meal.name or "").strip()
    address = (meal.address or "").strip()
    if not name or not address:
        return False

    normalized_name = name.lower()
    generic_labels = {"breakfast", "lunch", "dinner", "snack", meal.type.strip().lower()}
    if normalized_name in generic_labels:
        return False
    if any(term in normalized_name for term in GENERIC_MEAL_NAME_TERMS):
        return False
    if normalized_name in {"cafe", "restaurant", "bar", "bakery"}:
        return False
    return True


def match_entity_to_evidence(
    *,
    entity_type: str,
    entity_name: str,
    day_index: Optional[int],
    candidates: List[AttractionCandidate] | List[HotelCandidate] | List[MealCandidate],
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

    candidate_evidence_type = {
        "attraction": "candidate_attraction",
        "hotel": "candidate_hotel",
        "meal": "candidate_meal",
    }.get(entity_type, "none")
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
    if any(term in normalized for term in ("步行", "徒步", "citywalk", "walk", "walking")):
        return 5.0
    if any(term in normalized for term in ("打车", "出租", "自驾", "驾车", "taxi", "car", "drive", "driving", "rideshare")):
        return 30.0
    return 15.0


def route_type_for_transportation(transportation: str) -> str:
    """Map request/day transportation text to provider route modes."""
    normalized = normalized_text(transportation)
    if any(term in normalized for term in ("transit", "publictransit", "subway", "bus", "metro", "地铁", "公交")):
        return "transit"
    if any(term in normalized for term in ("car", "drive", "driving", "taxi", "rideshare", "uber", "lyft", "自驾", "打车")):
        return "driving"
    if any(term in normalized for term in ("bike", "cycling", "bicycling", "bicycle", "骑行")):
        return "bicycling"
    return "walking"


def threshold_for_route_type(thresholds: Optional[Dict[str, int]], route_type: str, default_value: int) -> int:
    """Return a route-time threshold using a canonical route type with aliases."""
    values = thresholds or {}
    aliases = {
        "transit": ("transit", "public_transit", "public transit"),
        "driving": ("driving", "drive", "car", "taxi"),
        "bicycling": ("bicycling", "cycling", "bike"),
        "walking": ("walking", "walk"),
    }
    for key in aliases.get(route_type, (route_type,)):
        if key in values:
            try:
                return int(values[key])
            except (TypeError, ValueError):
                break
    return default_value


def route_time_score(duration_minutes: float, threshold_minutes: int) -> float:
    """Score one route-time segment against a mode-specific maximum."""
    if threshold_minutes <= 0 or duration_minutes <= threshold_minutes:
        return 1.0
    excess_ratio = min(1.0, (duration_minutes - threshold_minutes) / threshold_minutes)
    return max(0.0, 1.0 - 0.7 - (0.3 * excess_ratio))


def route_estimate_key(day_index: int, segment_index: int) -> str:
    """Stable key for a same-day attraction transfer estimate."""
    return f"{day_index}:{segment_index}"


def estimate_map_by_segment(route_time_estimates: Optional[List[RouteTimeEstimate]]) -> Dict[str, RouteTimeEstimate]:
    """Index route-time estimates by day and segment."""
    estimates: Dict[str, RouteTimeEstimate] = {}
    for estimate in route_time_estimates or []:
        estimates[route_estimate_key(estimate.day_index, estimate.segment_index)] = estimate
    return estimates


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


def evaluate_haversine_route_segment(day_index: int, transportation: str, origin: Location | None, destination: Location | None) -> tuple[float, List[str]]:
    """Score one same-day attraction transfer using coordinate distance."""
    warnings: List[str] = []
    threshold_km = route_distance_threshold_km(transportation)
    if origin is None or destination is None:
        warnings.append(f"route_day_{day_index}_missing_coordinates")
        return 0.5, warnings
    distance_km = haversine_km(origin, destination)
    if distance_km <= threshold_km:
        return 1.0, warnings
    excess_ratio = min(1.0, (distance_km - threshold_km) / threshold_km)
    warnings.append(f"route_day_{day_index}_long_jump_{round(distance_km, 1)}km")
    return max(0.0, 1.0 - 0.7 - (0.3 * excess_ratio)), warnings


def evaluate_route_coherence(
    draft_plan: TripPlan,
    request: TripRequest,
    route_time_estimates: Optional[List[RouteTimeEstimate]] = None,
    route_time_evaluation_enabled: bool = False,
    max_segment_minutes_by_mode: Optional[Dict[str, int]] = None,
    max_daily_transit_minutes_by_mode: Optional[Dict[str, int]] = None,
) -> tuple[float, List[str]]:
    """Score whether same-day attraction ordering is geographically coherent."""
    warnings: List[str] = []
    segment_scores: List[float] = []
    estimates_by_segment = estimate_map_by_segment(route_time_estimates)
    for day in draft_plan.days:
        attractions = day.attractions
        if len(attractions) <= 1:
            continue
        comparable_segments = len(attractions) - 1
        daily_route_minutes = 0.0
        daily_route_estimate_count = 0
        route_type = route_type_for_transportation(day.transportation or request.transportation)
        for index in range(len(attractions) - 1):
            origin = attractions[index].location
            destination = attractions[index + 1].location
            estimate = estimates_by_segment.get(route_estimate_key(day.day_index, index))
            if (
                route_time_evaluation_enabled
                and estimate is not None
                and not estimate.error
                and estimate.duration_minutes is not None
                and estimate.duration_minutes > 0
            ):
                duration = float(estimate.duration_minutes)
                segment_threshold = threshold_for_route_type(
                    max_segment_minutes_by_mode,
                    route_type,
                    DEFAULT_MAX_SEGMENT_MINUTES_BY_MODE[route_type],
                )
                segment_scores.append(route_time_score(duration, segment_threshold))
                daily_route_minutes += duration
                daily_route_estimate_count += 1
                if duration > segment_threshold:
                    warnings.append(
                        f"route_day_{day.day_index}_long_transfer_{round(duration)}min"
                    )
            else:
                if route_time_evaluation_enabled:
                    warnings.append(f"route_time_fallback_day_{day.day_index}_segment_{index}")
                score, segment_warnings = evaluate_haversine_route_segment(
                    day.day_index,
                    day.transportation or request.transportation,
                    origin,
                    destination,
                )
                segment_scores.append(score)
                warnings.extend(segment_warnings)
        if route_time_evaluation_enabled and daily_route_estimate_count:
            daily_threshold = threshold_for_route_type(
                max_daily_transit_minutes_by_mode,
                route_type,
                DEFAULT_MAX_DAILY_TRANSIT_MINUTES_BY_MODE[route_type],
            )
            if daily_route_minutes > daily_threshold:
                warnings.append(
                    f"route_day_{day.day_index}_total_transit_{round(daily_route_minutes)}min"
                )
                segment_scores.append(route_time_score(daily_route_minutes, daily_threshold))
    if not segment_scores:
        return 1.0, warnings
    return round(sum(segment_scores) / len(segment_scores), 4), warnings


def preference_terms(request: TripRequest) -> List[str]:
    """Build deterministic preference terms from explicit labels and free text."""
    values = list(request.preferences)
    if request.free_text_input:
        free_text = request.free_text_input
        known_terms = [
            "history",
            "culture",
            "museum",
            "museums",
            "food",
            "restaurants",
            "nightlife",
            "skyline",
            "nature",
            "outdoors",
            "parks",
            "relaxed",
            "relaxing",
            "photography",
            "citywalk",
            "classic route",
            "not too rushed",
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
        for prefix in ("i want", "please", "prefer", "希望以", "希望", "想要", "想", "以"):
            if term.startswith(prefix):
                term = term[len(prefix):]
        for suffix in ("focused", "first", "为主", "为重点", "优先"):
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
    candidate_meals: Optional[List[MealCandidate]] = None,
    route_time_estimates: Optional[List[RouteTimeEstimate]] = None,
    route_time_evaluation_enabled: bool = False,
    max_segment_minutes_by_mode: Optional[Dict[str, int]] = None,
    max_daily_transit_minutes_by_mode: Optional[Dict[str, int]] = None,
    quality_retry_enabled: bool = False,
    min_pacing_score: float = 0.75,
    min_route_coherence_score: float = 0.75,
    min_preference_match_score: float = 0.60,
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

    empty_attraction_days = [
        day.day_index for day in draft_plan.days if len(day.attractions) == 0
    ]
    if empty_attraction_days:
        hard_failures.append("content_completeness_attractions")
        unsupported_claims.append(
            "days missing attraction recommendations: "
            + ", ".join(str(day_index) for day_index in empty_attraction_days)
        )

    attraction_names = [normalize_entity_name(candidate.name) for candidate in candidate_attractions]
    hotel_names = [normalize_entity_name(candidate.name) for candidate in candidate_hotels]
    meal_candidates = candidate_meals or []
    meal_names = [normalize_entity_name(candidate.name) for candidate in meal_candidates]
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
        for meal in day.meals:
            if not is_concrete_meal_recommendation(meal):
                continue
            total_entities += 1
            evidence_link = match_entity_to_evidence(
                entity_type="meal",
                entity_name=meal.name,
                day_index=day.day_index,
                candidates=meal_candidates,
                rag_chunks=rag_chunks,
            )
            evidence_links.append(evidence_link)
            if entity_matches(meal.name, meal_names, rag_text):
                grounded_entities += 1
            else:
                unsupported_entities.append(
                    UnsupportedEntity(
                        entity_type="meal",
                        name=meal.name,
                        reason="concrete restaurant recommendation with address was not found in retrieved meal candidates or rag context",
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
    if any(item.entity_type == "meal" for item in unsupported_entities):
        hard_failures.append("retrieval_grounding_meals")

    scores.pacing_score, pacing_warnings = evaluate_pacing_quality(draft_plan)
    scores.route_coherence_score, route_warnings = evaluate_route_coherence(
        draft_plan,
        request,
        route_time_estimates=route_time_estimates,
        route_time_evaluation_enabled=route_time_evaluation_enabled,
        max_segment_minutes_by_mode=max_segment_minutes_by_mode,
        max_daily_transit_minutes_by_mode=max_daily_transit_minutes_by_mode,
    )
    scores.preference_match_score, preference_warnings = evaluate_preference_match(
        draft_plan, request, rag_chunks
    )
    quality_warnings.extend(pacing_warnings)
    quality_warnings.extend(route_warnings)
    quality_warnings.extend(preference_warnings)
    if scores.pacing_score < min_pacing_score and "low_pacing_score" not in quality_warnings:
        quality_warnings.append("low_pacing_score")
    if scores.route_coherence_score < min_route_coherence_score and "low_route_coherence_score" not in quality_warnings:
        quality_warnings.append("low_route_coherence_score")
    if scores.preference_match_score < min_preference_match_score and "low_preference_match_score" not in quality_warnings:
        quality_warnings.append("low_preference_match_score")

    strict_quality_reasons: List[str] = []
    if scores.pacing_score < min_pacing_score:
        strict_quality_reasons.append(
            f"pacing_score={scores.pacing_score:.4f} below min_pacing_score={min_pacing_score:.4f}"
        )
    if scores.route_coherence_score < min_route_coherence_score:
        strict_quality_reasons.append(
            "route_coherence_score="
            f"{scores.route_coherence_score:.4f} below min_route_coherence_score={min_route_coherence_score:.4f}"
        )
    if scores.preference_match_score < min_preference_match_score:
        strict_quality_reasons.append(
            "preference_match_score="
            f"{scores.preference_match_score:.4f} below min_preference_match_score={min_preference_match_score:.4f}"
        )

    next_action = "finalize_response"
    passed = len(hard_failures) == 0
    if not passed:
        if "schema_correctness" in hard_failures:
            next_action = next_action_with_retry_budget(retry_counts, "plan_itinerary", max_retries)
        elif "content_completeness_attractions" in hard_failures:
            action = "plan_itinerary" if candidate_attractions else "retrieve_attractions"
            next_action = next_action_with_retry_budget(retry_counts, action, max_retries)
        elif "retrieval_grounding_attractions" in hard_failures:
            next_action = next_action_with_retry_budget(retry_counts, "retrieve_attractions", max_retries)
        elif "retrieval_grounding_hotels" in hard_failures:
            next_action = next_action_with_retry_budget(retry_counts, "retrieve_hotels", max_retries)
        elif "retrieval_grounding_meals" in hard_failures:
            next_action = next_action_with_retry_budget(retry_counts, "retrieve_meals", max_retries)
        else:
            next_action = next_action_with_retry_budget(retry_counts, "plan_itinerary", max_retries)
    elif quality_retry_enabled and strict_quality_reasons:
        passed = False
        warnings.append("strict_quality_retry_triggered")
        unsupported_claims.extend(
            f"strict_quality_retry: {reason}" for reason in strict_quality_reasons
        )
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
