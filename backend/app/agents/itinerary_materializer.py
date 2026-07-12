"""Deterministically materialize ID-only drafts from canonical registry entities."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from ..models.multi_agent import (
    CandidateRegistry,
    ExperienceProposal,
    IDBasedItineraryDraft,
    LogisticsProposal,
    RegistryEntity,
)
from ..models.schemas import Attraction, Budget, DayPlan, Hotel, Meal, TripPlan, TripRequest, WeatherInfo


class MaterializationFailure(BaseModel):
    code: Literal[
        "unknown_id",
        "wrong_entity_type",
        "stale_proposal_version",
        "date_mismatch",
        "duplicate_date",
        "duplicate_attraction",
        "invalid_duration",
        "invalid_cost",
        "missing_canonical_location",
        "unapproved_id",
        "invalid_generic_meal",
    ]
    path: str
    message: str
    source_id: str = ""


class MaterializationResult(BaseModel):
    plan: Optional[TripPlan] = None
    failures: List[MaterializationFailure] = Field(default_factory=list)
    canonical_field_hallucination_count: int = 0

    @property
    def succeeded(self) -> bool:
        return self.plan is not None and not self.failures


class ItineraryMaterializer:
    """Turn agent-selected source IDs into a public plan without trusting provider fields from an LLM."""

    def materialize(
        self,
        *,
        request: TripRequest,
        registry: CandidateRegistry,
        experience: ExperienceProposal,
        logistics: LogisticsProposal,
        draft: IDBasedItineraryDraft,
        weather_info: List[WeatherInfo],
    ) -> MaterializationResult:
        failures: List[MaterializationFailure] = []
        if experience.run_id != registry.run_id or logistics.run_id != registry.run_id or draft.run_id != registry.run_id:
            failures.append(self._failure("stale_proposal_version", "run_id", "proposal belongs to another run"))
        if logistics.experience_version != experience.version:
            failures.append(self._failure("stale_proposal_version", "logistics.experience_version", "stale experience proposal"))
        if draft.experience_version != experience.version:
            failures.append(self._failure("stale_proposal_version", "draft.experience_version", "stale experience proposal"))
        if draft.logistics_version != logistics.version:
            failures.append(self._failure("stale_proposal_version", "draft.logistics_version", "stale logistics proposal"))

        expected_dates = [
            (datetime.strptime(request.start_date, "%Y-%m-%d") + timedelta(days=index)).strftime("%Y-%m-%d")
            for index in range(request.travel_days)
        ]
        actual_dates = [day.date for day in draft.days]
        if len(actual_dates) != len(set(actual_dates)):
            failures.append(self._failure("duplicate_date", "days", "draft contains duplicate dates"))
        if actual_dates != expected_dates or [day.day_index for day in draft.days] != list(range(request.travel_days)):
            failures.append(self._failure("date_mismatch", "days", "draft must contain the exact ordered request dates"))

        seen_attractions: set[str] = set()
        days: List[DayPlan] = []
        incomplete = False
        attraction_total = hotel_total = meal_total = 0
        allowed_attractions = experience.allowed_attraction_ids
        allowed_hotels = set(logistics.hotel_ids)
        allowed_meals = set(logistics.meal_ids)

        for day_pos, day in enumerate(draft.days):
            attractions: List[Attraction] = []
            meals: List[Meal] = []
            for item_pos, item in enumerate(day.attraction_items):
                path = f"days.{day_pos}.attractions.{item_pos}"
                if item.visit_duration < 15 or item.visit_duration > 720:
                    failures.append(self._failure("invalid_duration", path, "visit duration is outside allowed bounds", item.source_id))
                    continue
                if item.ticket_price < 0:
                    failures.append(self._failure("invalid_cost", path, "ticket price cannot be negative", item.source_id))
                    continue
                entity = self._resolve(registry, item.source_id, "attraction", path, failures)
                if entity is None:
                    continue
                if item.source_id not in allowed_attractions:
                    failures.append(self._failure("unapproved_id", path, "attraction is not allowed by current proposal", item.source_id))
                    continue
                if item.source_id in seen_attractions:
                    failures.append(self._failure("duplicate_attraction", path, "attraction assigned more than once", item.source_id))
                    continue
                if entity.location is None:
                    failures.append(self._failure("missing_canonical_location", path, "attraction has no provider coordinates", item.source_id))
                    continue
                seen_attractions.add(item.source_id)
                incomplete = incomplete or item.cost_status == "unknown"
                attraction_total += item.ticket_price
                attractions.append(
                    Attraction(
                        name=entity.name,
                        address=entity.address,
                        location=entity.location,
                        visit_duration=item.visit_duration,
                        description=item.description,
                        category=str(entity.metadata.get("category") or "Attraction"),
                        rating=entity.rating,
                        photos=list(entity.photo_names),
                        poi_id=entity.source_id,
                        image_url=entity.image_url,
                        maps_url=entity.maps_url,
                        website_url=entity.website_url,
                        ticket_price=item.ticket_price,
                        cost_status=item.cost_status,
                    )
                )

            for item_pos, item in enumerate(day.meal_items):
                path = f"days.{day_pos}.meals.{item_pos}"
                if item.source_id:
                    entity = self._resolve(registry, item.source_id, "meal", path, failures)
                    if entity is None:
                        continue
                    if item.source_id not in allowed_meals:
                        failures.append(self._failure("unapproved_id", path, "meal is not allowed by current proposal", item.source_id))
                        continue
                    name, address, location = entity.name, entity.address or None, entity.location
                    image_url, maps_url, website_url = entity.image_url, entity.maps_url, entity.website_url
                    poi_id = entity.source_id
                elif item.generic_name:
                    name, address, location = item.generic_name, None, None
                    image_url = maps_url = website_url = None
                    poi_id = ""
                else:
                    failures.append(self._failure("invalid_generic_meal", path, "generic meal requires a name"))
                    continue
                incomplete = incomplete or item.cost_status == "unknown"
                meal_total += item.estimated_cost
                meals.append(Meal(type=item.meal_type, name=name, address=address, location=location,
                                  description=item.description, estimated_cost=item.estimated_cost,
                                  image_url=image_url, maps_url=maps_url, website_url=website_url,
                                  poi_id=poi_id, cost_status=item.cost_status))

            hotel = None
            if day.hotel_id:
                path = f"days.{day_pos}.hotel"
                entity = self._resolve(registry, day.hotel_id, "hotel", path, failures)
                if entity is not None:
                    if day.hotel_id not in allowed_hotels:
                        failures.append(self._failure("unapproved_id", path, "hotel is not allowed by current proposal", day.hotel_id))
                    else:
                        estimate = max(0, int(logistics.cost_assumptions.get(day.hotel_id, 0)))
                        status = "estimated" if day.hotel_id in logistics.cost_assumptions else "unknown"
                        incomplete = incomplete or status == "unknown"
                        hotel_total += estimate
                        hotel = Hotel(name=entity.name, address=entity.address, location=entity.location,
                                      price_range="", rating=str(entity.rating or ""), distance="",
                                      type="Hotel", estimated_cost=estimate, image_url=entity.image_url,
                                      maps_url=entity.maps_url, website_url=entity.website_url,
                                      poi_id=entity.source_id, cost_status=status)

            days.append(DayPlan(date=day.date, day_index=day.day_index, description=day.description,
                                transportation=request.transportation, accommodation=request.accommodation,
                                hotel=hotel, attractions=attractions, meals=meals))

        if failures:
            return MaterializationResult(failures=failures)
        transportation_total = draft.transportation_estimate
        budget = Budget(total_attractions=attraction_total, total_hotels=hotel_total,
                        total_meals=meal_total, total_transportation=transportation_total,
                        total=attraction_total + hotel_total + meal_total + transportation_total,
                        estimate_incomplete=incomplete, currency="USD")
        weather_by_date = {item.date: item for item in weather_info}
        authoritative_weather = [weather_by_date[date] for date in expected_dates if date in weather_by_date]
        return MaterializationResult(plan=TripPlan(city=request.city, start_date=request.start_date,
                                                   end_date=request.end_date, days=days,
                                                   weather_info=authoritative_weather,
                                                   overall_suggestions=draft.overall_suggestions,
                                                   budget=budget))

    def _resolve(self, registry: CandidateRegistry, source_id: str, expected_type: str, path: str,
                 failures: List[MaterializationFailure]) -> Optional[RegistryEntity]:
        entity = registry.entities.get(source_id)
        if entity is None:
            failures.append(self._failure("unknown_id", path, "source ID is not in the request registry", source_id))
            return None
        if entity.entity_type != expected_type:
            failures.append(self._failure("wrong_entity_type", path, f"expected {expected_type}, got {entity.entity_type}", source_id))
            return None
        return entity

    @staticmethod
    def _failure(code: str, path: str, message: str, source_id: str = "") -> MaterializationFailure:
        return MaterializationFailure(code=code, path=path, message=message, source_id=source_id)
