"""Tool-free itinerary composer that emits source IDs only."""

from __future__ import annotations

from typing import Any, List, Optional

from ..models.multi_agent import AgentFeedback, ExperienceProposal, IDBasedItineraryDraft, LogisticsProposal
from ..models.schemas import TripRequest, WeatherInfo
from .candidate_ranking import resolve_aliases
from .structured_llm import invoke_structured


class ComposerAgentError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class ComposerAgent:
    MAX_ATTEMPTS = 2

    def __init__(self, *, llm: Any):
        self.llm = llm

    def run(self, *, request: TripRequest, experience: ExperienceProposal,
            logistics: LogisticsProposal, weather_info: List[WeatherInfo],
            feedback: Optional[AgentFeedback] = None,
            previous: Optional[IDBasedItineraryDraft] = None,
            attempt: int = 1) -> IDBasedItineraryDraft:
        if attempt < 1 or attempt > self.MAX_ATTEMPTS:
            raise ComposerAgentError("retry_budget_exhausted", "composer attempt budget exhausted")
        context = {"previous_draft": previous.model_dump() if previous else None,
                   "feedback": feedback.model_dump() if feedback else None,
                   "remaining_attempts": self.MAX_ATTEMPTS - attempt}
        attraction_ids = list(dict.fromkeys(experience.core_attraction_ids + experience.optional_attraction_ids))
        attraction_aliases = {f"A{index}": source_id for index, source_id in enumerate(attraction_ids, 1)}
        hotel_aliases = {f"H{index}": source_id for index, source_id in enumerate(logistics.hotel_ids, 1)}
        meal_aliases = {f"M{index}": source_id for index, source_id in enumerate(logistics.meal_ids, 1)}
        reverse_a = {value: key for key, value in attraction_aliases.items()}
        reverse_h = {value: key for key, value in hotel_aliases.items()}
        reverse_m = {value: key for key, value in meal_aliases.items()}
        compact_experience = {
            "version": experience.version, "target": experience.target_attractions,
            "core": [reverse_a[item] for item in experience.core_attraction_ids],
            "optional": [reverse_a[item] for item in experience.optional_attraction_ids],
        }
        compact_logistics = {
            "version": logistics.version, "primary_hotel": reverse_h.get(logistics.primary_hotel_id or ""),
            "hotels": list(hotel_aliases), "meals": list(meal_aliases),
        }
        prompt = (
            "You are the tool-free Composer. Build the exact ordered request date set with continuous zero-based "
            "day indices. Use only IDs in the two proposals. Named meals require a meal source ID; otherwise use "
            "a generic meal name with no address or POI ID. You may author ordering, duration, descriptions and "
            "cost estimates, but never provider names, addresses, coordinates, ratings, or links.\n"
            "Return A/H/M aliases in source_id and hotel_id fields. Use every core A alias, use the primary H alias every day, "
            "and never repeat an M alias.\n"
            f"request={request.model_dump()}\nexperience={compact_experience}\n"
            f"logistics={compact_logistics}\nweather={[item.model_dump() for item in weather_info]}\n"
            f"revision={context}"
        )
        try:
            draft = invoke_structured(self.llm, IDBasedItineraryDraft, prompt)
        except Exception as exc:
            raise ComposerAgentError("structured_output", str(exc)) from exc
        try:
            used_attractions = []
            used_meals = []
            for day in draft.days:
                for item in day.attraction_items:
                    item.source_id = resolve_aliases([item.source_id], attraction_aliases, expected_prefix="A")[0]
                    used_attractions.append(item.source_id)
                for item in day.meal_items:
                    if item.source_id:
                        item.source_id = resolve_aliases([item.source_id], meal_aliases, expected_prefix="M")[0]
                        used_meals.append(item.source_id)
                if day.hotel_id:
                    day.hotel_id = resolve_aliases([day.hotel_id], hotel_aliases, expected_prefix="H")[0]
                if day.hotel_id != logistics.primary_hotel_id:
                    raise ValueError("composer must use primary hotel every day")
            if not set(experience.core_attraction_ids) <= set(used_attractions):
                raise ValueError("composer omitted core attractions")
            if len(used_attractions) != len(set(used_attractions)) or len(used_meals) != len(set(used_meals)):
                raise ValueError("composer duplicated an alias")
            if feedback is None and len(used_attractions) != experience.target_attractions:
                raise ValueError("composer did not meet attraction target")
        except ValueError as exc:
            raise ComposerAgentError("invalid_source_id", str(exc)) from exc
        draft.run_id = experience.run_id
        draft.version = (previous.version if previous else 0) + 1
        draft.experience_version = experience.version
        draft.logistics_version = logistics.version
        return draft
