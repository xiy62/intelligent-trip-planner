import unittest

from app.agents.canonical_audit import audit_canonical_fields
from app.agents.itinerary_materializer import ItineraryMaterializer
from app.models.multi_agent import (
    CandidateRegistry,
    DraftAttraction,
    DraftDay,
    DraftMeal,
    ExperienceProposal,
    IDBasedItineraryDraft,
    LogisticsProposal,
    RegistryEntity,
)
from app.models.schemas import Location, TripRequest


class CanonicalAuditTests(unittest.TestCase):
    def setUp(self):
        self.request = TripRequest(city="New York", start_date="2026-06-01", end_date="2026-06-01",
                                   travel_days=1, transportation="Public transit",
                                   accommodation="Mid-range hotel", preferences=["Museums"])
        self.registry = CandidateRegistry(run_id="run-1")
        entities = [
            RegistryEntity(source_id="attraction:a1", provider_id="a1", entity_type="attraction",
                           name="Museum", address="1 Main", location=Location(longitude=-73.9, latitude=40.7),
                           rating=4.8, photo_names=["photo-a"], maps_url="https://maps/a",
                           website_url="https://a.example", image_url="https://img/a",
                           metadata={"category": "Museum"}, registered_by="experience"),
            RegistryEntity(source_id="hotel:h1", provider_id="h1", entity_type="hotel",
                           name="Hotel", address="2 Main", location=Location(longitude=-73.91, latitude=40.71),
                           rating=4.5, maps_url="https://maps/h", website_url="https://h.example",
                           image_url="https://img/h", registered_by="logistics"),
            RegistryEntity(source_id="meal:m1", provider_id="m1", entity_type="meal",
                           name="Cafe", address="3 Main", location=Location(longitude=-73.92, latitude=40.72),
                           maps_url="https://maps/m", website_url="https://m.example",
                           image_url="https://img/m", registered_by="logistics"),
        ]
        for entity in entities:
            self.registry.add(entity, actor=entity.registered_by)
        experience = ExperienceProposal(run_id="run-1", core_attraction_ids=["attraction:a1"])
        logistics = LogisticsProposal(run_id="run-1", experience_version=1,
                                       hotel_ids=["hotel:h1"], primary_hotel_id="hotel:h1",
                                       meal_ids=["meal:m1"])
        draft = IDBasedItineraryDraft(
            run_id="run-1", experience_version=1, logistics_version=1,
            days=[DraftDay(date="2026-06-01", day_index=0, hotel_id="hotel:h1",
                           attraction_items=[DraftAttraction(source_id="attraction:a1", visit_duration=90)],
                           meal_items=[DraftMeal(meal_type="lunch", source_id="meal:m1"),
                                       DraftMeal(meal_type="dinner", generic_name="Flexible dinner")])],
        )
        result = ItineraryMaterializer().materialize(request=self.request, registry=self.registry,
                                                     experience=experience, logistics=logistics,
                                                     draft=draft, weather_info=[])
        self.assertTrue(result.succeeded)
        self.plan = result.plan

    def test_clean_materialized_plan_has_no_mismatches_and_skips_generic_meal(self):
        result = audit_canonical_fields(self.plan, self.registry)
        self.assertEqual(result.mismatch_count, 0)
        self.assertEqual(result.audited_entity_count, 3)
        self.assertEqual(result.skipped_generic_meal_count, 1)

    def test_detects_each_provider_owned_field_tamper(self):
        attraction = self.plan.days[0].attractions[0]
        attraction.name = "Invented"
        attraction.address = "Wrong"
        attraction.location = Location(longitude=0, latitude=0)
        attraction.rating = 1.0
        attraction.photos = ["wrong"]
        attraction.category = "Park"
        attraction.maps_url = "https://wrong"
        attraction.website_url = "https://wrong-site"
        attraction.image_url = "https://wrong-image"
        fields = {item.field for item in audit_canonical_fields(self.plan, self.registry).mismatches}
        self.assertEqual(fields, {"name", "address", "location", "rating", "photos", "category",
                                  "maps_url", "website_url", "image_url"})

    def test_detects_wrong_type_and_unknown_identity(self):
        self.plan.days[0].attractions[0].poi_id = "h1"
        self.plan.days[0].hotel.poi_id = "missing"
        mismatches = audit_canonical_fields(self.plan, self.registry).mismatches
        self.assertEqual([(item.entity_type, item.field) for item in mismatches],
                         [("attraction", "identity_type"), ("hotel", "identity")])


if __name__ == "__main__":
    unittest.main()
