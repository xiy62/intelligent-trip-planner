"""Benchmark deterministic validators against adversarial itinerary failures."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agents.trip_plan_evaluation import evaluate_trip_plan
from app.models.langgraph_state import (
    AttractionCandidate,
    HotelCandidate,
    RAGChunk,
    RetryState,
)
from app.models.schemas import (
    Attraction,
    Budget,
    DayPlan,
    Hotel,
    Location,
    Meal,
    TripPlan,
    TripRequest,
    WeatherInfo,
)


VALIDATOR_CATEGORIES = [
    "request_validation",
    "schema_correctness",
    "date_coverage",
    "budget_consistency",
    "current_request_alignment",
    "content_completeness_attractions",
    "retrieval_grounding_attractions",
    "retrieval_grounding_hotels",
]

DEFAULT_REQUEST = {
    "city": "New York",
    "start_date": "2026-07-01",
    "end_date": "2026-07-02",
    "travel_days": 2,
    "transportation": "Public transit",
    "accommodation": "Mid-range hotel",
    "preferences": ["Museums", "Food"],
    "free_text_input": "Keep the itinerary relaxed and avoid unsupported recommendations.",
}


def travel_dates_for_request(request: TripRequest) -> List[str]:
    start = datetime.strptime(request.start_date, "%Y-%m-%d").date()
    return [
        (start + timedelta(days=offset)).isoformat()
        for offset in range(request.travel_days)
    ]


def base_attraction_candidates() -> List[AttractionCandidate]:
    return [
        AttractionCandidate(
            name="The Metropolitan Museum of Art",
            source_id="google-met",
            latitude=40.7794,
            longitude=-73.9632,
        ),
        AttractionCandidate(
            name="Central Park",
            source_id="google-central-park",
            latitude=40.7829,
            longitude=-73.9654,
        ),
        AttractionCandidate(
            name="Museum of Modern Art",
            source_id="google-moma",
            latitude=40.7614,
            longitude=-73.9776,
        ),
    ]


def base_hotel_candidates() -> List[HotelCandidate]:
    return [
        HotelCandidate(
            name="Arlo Midtown",
            source_id="google-arlo-midtown",
            latitude=40.7547,
            longitude=-73.9935,
        )
    ]


def base_rag_chunks() -> List[RAGChunk]:
    return [
        RAGChunk(
            chunk_id="new-york-museum-mile-central-park-001-overview",
            source="knowledge",
            title="New York Museum Mile and Central Park",
            content=(
                "The Metropolitan Museum of Art, Central Park, and the Museum of Modern Art "
                "can be planned as a relaxed public-transit museum and park itinerary."
            ),
            metadata={
                "doc_id": "new-york-museum-mile-central-park-001",
                "city": "New York",
                "theme": "museums,parks",
                "source_url": "https://www.nyctourism.com/",
            },
        )
    ]


def build_base_plan(request: TripRequest) -> TripPlan:
    dates = travel_dates_for_request(request)
    hotel = Hotel(
        name="Arlo Midtown",
        address="351 W 38th St, New York, NY",
        location=Location(longitude=-73.9935, latitude=40.7547),
        estimated_cost=260,
        price_range="Mid-range",
    )
    day_one_attractions = [
        Attraction(
            name="The Metropolitan Museum of Art",
            address="1000 5th Ave, New York, NY",
            location=Location(longitude=-73.9632, latitude=40.7794),
            visit_duration=180,
            description="Museum-focused stop matching the user's culture preference.",
            ticket_price=30,
        ),
        Attraction(
            name="Central Park",
            address="New York, NY",
            location=Location(longitude=-73.9654, latitude=40.7829),
            visit_duration=120,
            description="Relaxed park walk near Museum Mile.",
            ticket_price=0,
        ),
    ]
    day_two_attractions = [
        Attraction(
            name="Museum of Modern Art",
            address="11 W 53rd St, New York, NY",
            location=Location(longitude=-73.9776, latitude=40.7614),
            visit_duration=150,
            description="Museum and design stop with nearby food options.",
            ticket_price=30,
        )
    ]
    days = [
        DayPlan(
            date=dates[0],
            day_index=0,
            description="Museum Mile and Central Park at a relaxed pace.",
            transportation=request.transportation,
            accommodation=request.accommodation,
            hotel=hotel,
            attractions=day_one_attractions,
            meals=[
                Meal(type="lunch", name="Museum cafe lunch", description="Simple lunch near the Met", estimated_cost=35),
                Meal(type="dinner", name="Midtown casual dinner", description="Food-focused dinner", estimated_cost=45),
            ],
        ),
        DayPlan(
            date=dates[1],
            day_index=1,
            description="MoMA visit and nearby food route.",
            transportation=request.transportation,
            accommodation=request.accommodation,
            hotel=hotel,
            attractions=day_two_attractions,
            meals=[
                Meal(type="lunch", name="MoMA neighborhood lunch", description="Food near the museum", estimated_cost=35),
                Meal(type="dinner", name="Relaxed Midtown dinner", description="Not rushed", estimated_cost=45),
            ],
        ),
    ]
    return normalize_budget(
        TripPlan(
            city=request.city,
            start_date=request.start_date,
            end_date=request.end_date,
            days=days,
            weather_info=[
                WeatherInfo(date=date, day_weather="Partly cloudy", night_weather="Clear")
                for date in dates
            ],
            overall_suggestions="Use public transit and keep the museum route relaxed.",
            budget=Budget(),
        )
    )


def normalize_budget(plan: TripPlan) -> TripPlan:
    total_attractions = sum(
        attraction.ticket_price for day in plan.days for attraction in day.attractions
    )
    total_hotels = sum(day.hotel.estimated_cost for day in plan.days if day.hotel)
    total_meals = sum(meal.estimated_cost for day in plan.days for meal in day.meals)
    total_transportation = 60
    plan.budget = Budget(
        total_attractions=total_attractions,
        total_hotels=total_hotels,
        total_meals=total_meals,
        total_transportation=total_transportation,
        total=total_attractions + total_hotels + total_meals + total_transportation,
    )
    return plan


def build_retry_state(mutation: str) -> RetryState:
    retry_counts = RetryState()
    if mutation == "retry_exhausted_schema":
        retry_counts.plan_itinerary = 3
    return retry_counts


def apply_mutation(
    mutation: str,
    plan: TripPlan,
    candidate_attractions: List[AttractionCandidate],
    candidate_hotels: List[HotelCandidate],
) -> Optional[TripPlan]:
    if mutation == "none":
        return plan
    if mutation in {"schema_missing", "retry_exhausted_schema"}:
        return None
    if mutation == "wrong_date_coverage":
        plan.days[-1].date = "2026-07-05"
        plan.weather_info = plan.weather_info[:1]
        return plan
    if mutation == "bad_budget":
        plan.budget = Budget(
            total_attractions=999,
            total_hotels=1,
            total_meals=2,
            total_transportation=3,
            total=4,
        )
        return plan
    if mutation == "hallucinated_attraction":
        plan.days[0].attractions[0].name = "Moon Base Observation Deck"
        plan.days[0].attractions[0].description = "Unsupported attraction that was not retrieved."
        return plan
    if mutation == "hallucinated_hotel":
        if plan.days[0].hotel is not None:
            plan.days[0].hotel.name = "Fictional Palace Hotel"
        return plan
    if mutation in {"missing_attractions", "missing_attractions_no_candidates"}:
        plan.days[0].attractions = []
        normalize_budget(plan)
        if mutation == "missing_attractions_no_candidates":
            candidate_attractions.clear()
        return plan
    if mutation == "current_request_mismatch":
        for day in plan.days:
            day.transportation = "Private helicopter"
            day.accommodation = "Luxury resort"
        return plan
    raise ValueError(f"Unknown adversarial mutation: {mutation}")


def build_request_payload(case: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(DEFAULT_REQUEST)
    payload.update(case.get("request_overrides", {}))
    return payload


def run_case(case: Dict[str, Any]) -> Dict[str, Any]:
    expected_failures = list(case.get("expected_failures", []))
    expected_next_action = str(case.get("expected_next_action", ""))
    try:
        request = TripRequest(**build_request_payload(case))
    except ValidationError as exc:
        actual_failures = ["request_validation"]
        return {
            "case_id": case["case_id"],
            "description": case.get("description", ""),
            "mutation": case.get("mutation", ""),
            "expected_failures": expected_failures,
            "actual_failures": actual_failures,
            "expected_next_action": expected_next_action,
            "actual_next_action": "reject_request",
            "passed": False,
            "detected_expected_failure": bool(set(expected_failures) & set(actual_failures)),
            "validation_error": str(exc),
            "scores": {},
            "quality_warnings": [],
        }

    candidate_attractions = copy.deepcopy(base_attraction_candidates())
    candidate_hotels = copy.deepcopy(base_hotel_candidates())
    rag_chunks = copy.deepcopy(base_rag_chunks())
    mutation = str(case.get("mutation", "none"))
    plan = apply_mutation(
        mutation,
        build_base_plan(request),
        candidate_attractions,
        candidate_hotels,
    )
    report = evaluate_trip_plan(
        request=request,
        travel_dates=travel_dates_for_request(request),
        draft_plan=plan,
        candidate_attractions=candidate_attractions,
        candidate_hotels=candidate_hotels,
        rag_chunks=rag_chunks,
        retry_counts=build_retry_state(mutation),
        max_retries=2,
    )
    actual_failures = list(report.hard_failures)
    return {
        "case_id": case["case_id"],
        "description": case.get("description", ""),
        "mutation": mutation,
        "expected_failures": expected_failures,
        "actual_failures": actual_failures,
        "expected_next_action": expected_next_action,
        "actual_next_action": report.next_action,
        "passed": report.passed,
        "detected_expected_failure": bool(set(expected_failures) & set(actual_failures))
        if expected_failures
        else not actual_failures,
        "scores": report.scores.model_dump(),
        "quality_warnings": list(report.quality_warnings),
        "unsupported_entities": [entity.model_dump() for entity in report.unsupported_entities],
        "unsupported_claims": list(report.unsupported_claims),
    }


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def precision_recall_f1(results: Iterable[Dict[str, Any]], category: str) -> Dict[str, float]:
    true_positive = false_positive = false_negative = true_negative = 0
    for result in results:
        expected = category in set(result.get("expected_failures", []))
        actual = category in set(result.get("actual_failures", []))
        if expected and actual:
            true_positive += 1
        elif not expected and actual:
            false_positive += 1
        elif expected and not actual:
            false_negative += 1
        else:
            true_negative += 1
    precision = safe_divide(true_positive, true_positive + false_positive)
    recall = safe_divide(true_positive, true_positive + false_negative)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    false_positive_rate = safe_divide(false_positive, false_positive + true_negative)
    false_negative_rate = safe_divide(false_negative, false_negative + true_positive)
    return {
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "tn": true_negative,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "false_positive_rate": round(false_positive_rate, 4),
        "false_negative_rate": round(false_negative_rate, 4),
    }


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    expected_failure_cases = [
        result for result in results if result.get("expected_failures")
    ]
    detected_failure_cases = [
        result for result in expected_failure_cases if result.get("detected_expected_failure")
    ]
    unsafe_escapes = [
        result for result in expected_failure_cases if not result.get("detected_expected_failure")
    ]
    routing_matches = [
        result
        for result in results
        if result.get("expected_next_action") == result.get("actual_next_action")
    ]
    fallback_cases = [
        result for result in results if result.get("expected_next_action") == "fallback_response"
    ]
    fallback_matches = [
        result for result in fallback_cases if result.get("actual_next_action") == "fallback_response"
    ]
    per_category = {
        category: precision_recall_f1(results, category)
        for category in VALIDATOR_CATEGORIES
    }
    return {
        "case_count": len(results),
        "expected_failure_case_count": len(expected_failure_cases),
        "failure_detection_rate": round(
            safe_divide(len(detected_failure_cases), len(expected_failure_cases)),
            4,
        ),
        "unsafe_plan_escape_rate": round(
            safe_divide(len(unsafe_escapes), len(expected_failure_cases)),
            4,
        ),
        "routing_action_accuracy": round(
            safe_divide(len(routing_matches), len(results)),
            4,
        ),
        "fallback_correctness_rate": round(
            safe_divide(len(fallback_matches), len(fallback_cases)),
            4,
        )
        if fallback_cases
        else None,
        "per_validator": per_category,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run adversarial evaluator calibration benchmark.")
    parser.add_argument(
        "--dataset",
        default="benchmarks/adversarial_evaluation_cases.json",
        help="Path to adversarial evaluation case JSON relative to backend/ or absolute path.",
    )
    parser.add_argument(
        "--output",
        default="benchmarks/results/adversarial_evaluation_benchmark.json",
        help="Path to output JSON relative to backend/ or absolute path.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    output_path = Path(args.output)
    if not dataset_path.is_absolute():
        dataset_path = ROOT / dataset_path
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cases = json.loads(dataset_path.read_text(encoding="utf-8"))
    results = [run_case(case) for case in cases]
    output = {
        "dataset": str(dataset_path),
        "summary": summarize(results),
        "results": results,
    }
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2))
    print(f"\nSaved adversarial evaluation benchmark results to {output_path}")


if __name__ == "__main__":
    main()
