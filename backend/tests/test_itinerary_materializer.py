import unittest

from app.agents.itinerary_materializer import ItineraryMaterializer
from app.models.multi_agent import (
    CandidateRegistry,
    DraftAttraction,
    DraftDay,
    DraftMeal,
    ExperienceCluster,
    ExperienceProposal,
    IDBasedItineraryDraft,
    LogisticsProposal,
    RegistryEntity,
)
from app.models.schemas import Location, TripRequest, WeatherInfo


class ItineraryMaterializerTests(unittest.TestCase):
    def setUp(self):
        self.request = TripRequest(city="New York", start_date="2026-06-01", end_date="2026-06-01",
                                   travel_days=1, transportation="Public transit",
                                   accommodation="Mid-range hotel", preferences=["Museums"])
        self.registry = CandidateRegistry(run_id="run-1")
        for entity in (
            RegistryEntity(source_id="a1", entity_type="attraction", name="Canonical Museum",
                           address="1 Museum Way", location=Location(longitude=-73.9, latitude=40.7),
                           rating=4.8, maps_url="https://maps/a1", registered_by="experience"),
            RegistryEntity(source_id="h1", entity_type="hotel", name="Canonical Hotel",
                           address="2 Hotel Way", location=Location(longitude=-73.91, latitude=40.71),
                           registered_by="logistics"),
            RegistryEntity(source_id="m1", entity_type="meal", name="Canonical Cafe",
                           address="3 Cafe Way", location=Location(longitude=-73.92, latitude=40.72),
                           registered_by="logistics"),
        ):
            self.registry.add(entity, actor=entity.registered_by)
        self.experience = ExperienceProposal(run_id="run-1", version=2,
                                             clusters=[ExperienceCluster(name="Museums", attraction_ids=["a1"])])
        self.logistics = LogisticsProposal(run_id="run-1", version=3, experience_version=2,
                                           hotel_ids=["h1"], meal_ids=["m1"], cost_assumptions={"h1": 200})
        self.weather = [WeatherInfo(date="2026-06-01", day_weather="Clear", night_weather="Clear",
                                    day_temp=25, night_temp=18)]

    def draft(self, **updates):
        value = IDBasedItineraryDraft(
            run_id="run-1", version=1, experience_version=2, logistics_version=3,
            days=[DraftDay(date="2026-06-01", day_index=0, description="A good day",
                           attraction_items=[DraftAttraction(source_id="a1", visit_duration=120,
                                                            description="LLM-authored narrative",
                                                            ticket_price=25, cost_status="known")],
                           meal_items=[DraftMeal(meal_type="lunch", source_id="m1",
                                                 estimated_cost=30, cost_status="unknown")],
                           hotel_id="h1")],
            transportation_estimate=15,
        )
        return value.model_copy(update=updates)

    def test_materializes_only_canonical_provider_fields_and_deterministic_budget(self):
        result = ItineraryMaterializer().materialize(request=self.request, registry=self.registry,
                                                     experience=self.experience, logistics=self.logistics,
                                                     draft=self.draft(), weather_info=self.weather)
        self.assertTrue(result.succeeded)
        attraction = result.plan.days[0].attractions[0]
        self.assertEqual(attraction.name, "Canonical Museum")
        self.assertEqual(attraction.poi_id, "a1")
        self.assertEqual(result.plan.budget.total, 270)
        self.assertTrue(result.plan.budget.estimate_incomplete)
        self.assertEqual(result.canonical_field_hallucination_count, 0)

    def test_rejects_unknown_id_stale_version_extra_date_and_duplicate_poi(self):
        invalid = self.draft(logistics_version=2)
        invalid.days[0].attraction_items.append(DraftAttraction(source_id="a1", visit_duration=60))
        invalid.days[0].meal_items.append(DraftMeal(meal_type="dinner", source_id="missing"))
        invalid.days.append(DraftDay(date="2026-06-02", day_index=1))
        result = ItineraryMaterializer().materialize(request=self.request, registry=self.registry,
                                                     experience=self.experience, logistics=self.logistics,
                                                     draft=invalid, weather_info=self.weather)
        codes = {failure.code for failure in result.failures}
        self.assertFalse(result.succeeded)
        self.assertTrue({"stale_proposal_version", "date_mismatch", "duplicate_attraction", "unknown_id"} <= codes)

    def test_rejects_wrong_entity_type_and_unapproved_id(self):
        invalid = self.draft()
        invalid.days[0].attraction_items[0].source_id = "h1"
        result = ItineraryMaterializer().materialize(request=self.request, registry=self.registry,
                                                     experience=self.experience, logistics=self.logistics,
                                                     draft=invalid, weather_info=self.weather)
        self.assertIn("wrong_entity_type", {failure.code for failure in result.failures})


if __name__ == "__main__":
    unittest.main()
