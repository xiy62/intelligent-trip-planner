"""Planner prompts for LangChain-native execution."""

from __future__ import annotations

from typing import Iterable, Sequence

from ..models.langgraph_state import EvaluationReport, PlannerInputBundle
from ..models.schemas import TripRequest
from .shared import TRIP_JSON_RULES


PLANNER_AGENT_PROMPT = """You are an itinerary planning specialist. Generate a detailed trip plan from retrieved attractions, hotels, weather, and travel knowledge.

Return JSON with these required fields:
- city
- start_date
- end_date
- days
- weather_info
- overall_suggestions
- budget

The budget field must include:
- total_attractions
- total_hotels
- total_meals
- total_transportation
- total
"""


def build_retry_feedback(
    report: EvaluationReport | None,
    planner_inputs: PlannerInputBundle,
    travel_dates: Sequence[str],
) -> str:
    if report is None or report.passed:
        return ""

    attraction_names = [candidate.name for candidate in planner_inputs.attraction_candidates]
    hotel_names = [candidate.name for candidate in planner_inputs.hotel_candidates]
    unsupported_attractions = [
        item.name for item in report.unsupported_entities if item.entity_type == "attraction"
    ]
    unsupported_hotels = [
        item.name for item in report.unsupported_entities if item.entity_type == "hotel"
    ]

    feedback_lines = [
        "",
        "**The previous draft failed evaluation. Regenerate the plan and fix these issues:**",
        f"- Hard failures: {', '.join(report.hard_failures) if report.hard_failures else 'none'}",
    ]
    if unsupported_attractions:
        feedback_lines.append(
            f"- These attractions were unsupported in the previous draft and must not be reused: {', '.join(unsupported_attractions)}"
        )
    if unsupported_hotels:
        feedback_lines.append(
            f"- These hotels were unsupported in the previous draft and must not be reused: {', '.join(unsupported_hotels)}"
        )
    if attraction_names:
        feedback_lines.append(
            f"- Attractions must be selected only from these candidates, using exact names: {', '.join(attraction_names)}"
        )
    if hotel_names:
        feedback_lines.append(
            f"- Hotels must be selected only from these candidates, using exact names: {', '.join(hotel_names)}"
        )
    if "budget_consistency" in report.hard_failures or "budget_total_not_fully_aligned" in report.warnings:
        feedback_lines.append(
            "- budget.total_attractions, budget.total_hotels, and budget.total_meals must equal itemized day-level totals; budget.total must equal the sum of all budget categories."
        )
    if "current_request_alignment" in report.hard_failures:
        feedback_lines.append(
            "- Current request fields have highest priority: days[].transportation and days[].accommodation must exactly match the current request. Historical memory must not override the current request."
        )
        if report.unsupported_claims:
            feedback_lines.append(f"- Current-request alignment issues: {'; '.join(report.unsupported_claims)}")
    if "date_coverage" in report.hard_failures:
        feedback_lines.append(
            f"- days[].date and weather_info[].date must fully cover these dates: {', '.join(travel_dates)}"
        )
    feedback_lines.append("- Output must still be complete JSON only. Do not include explanatory prose.")
    return "\n".join(feedback_lines)


def build_planner_prompt(
    request: TripRequest,
    attractions_text: str,
    weather_text: str,
    hotels_text: str,
    rag_text: str,
    retry_feedback: str = "",
    format_instructions: str = "",
    memory_context: str = "",
) -> str:
    prompt = f"""Generate a {request.travel_days}-day itinerary for {request.city} using the information below.

**Current request (highest priority, authoritative):**
- City: {request.city}
- Dates: {request.start_date} to {request.end_date}
- Travel days: {request.travel_days}
- Transportation: {request.transportation}
- Accommodation preference: {request.accommodation}
- Preferences: {', '.join(request.preferences) if request.preferences else 'none'}

**Attraction candidates:**
{attractions_text}

**Weather:**
{weather_text}

**Hotel candidates:**
{hotels_text}

**Travel knowledge reference:**
{rag_text}

**Anonymous preference memory (soft context, lower priority than the current request):**
{memory_context or 'none'}

Memory rule: use historical preferences only to personalize unspecified or ambiguous choices. Do not copy old preferences just because they appear in memory. If memory conflicts with this request's city, dates, transportation, accommodation, preferences, or free-text requirements, follow the current request.

**Output rules:**
{TRIP_JSON_RULES}
"""
    if request.free_text_input:
        prompt += f"\n**Additional requirements:** {request.free_text_input}\n"
    if retry_feedback:
        prompt += f"\n{retry_feedback}\n"
    if format_instructions:
        prompt += f"\n**Format instructions:**\n{format_instructions}\n"
    return prompt
