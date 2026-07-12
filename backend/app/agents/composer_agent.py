"""Tool-free itinerary composer that emits source IDs only."""

from __future__ import annotations

from typing import Any, List, Optional

from ..models.multi_agent import AgentFeedback, ExperienceProposal, IDBasedItineraryDraft, LogisticsProposal
from ..models.schemas import TripRequest, WeatherInfo
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
        prompt = (
            "You are the tool-free Composer. Build the exact ordered request date set with continuous zero-based "
            "day indices. Use only IDs in the two proposals. Named meals require a meal source ID; otherwise use "
            "a generic meal name with no address or POI ID. You may author ordering, duration, descriptions and "
            "cost estimates, but never provider names, addresses, coordinates, ratings, or links.\n"
            f"request={request.model_dump()}\nexperience={experience.model_dump()}\n"
            f"logistics={logistics.model_dump()}\nweather={[item.model_dump() for item in weather_info]}\n"
            f"revision={context}"
        )
        try:
            draft = invoke_structured(self.llm, IDBasedItineraryDraft, prompt)
        except Exception as exc:
            raise ComposerAgentError("structured_output", str(exc)) from exc
        draft.run_id = experience.run_id
        draft.version = (previous.version if previous else 0) + 1
        draft.experience_version = experience.version
        draft.logistics_version = logistics.version
        return draft

