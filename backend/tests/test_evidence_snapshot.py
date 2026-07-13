import unittest

from app.agents.evidence_snapshot import (
    AgentEvidenceSnapshot,
    EvidenceSnapshotFile,
    EvidenceSnapshotMismatch,
    ExperienceEvidenceSnapshot,
    WorkflowEvidenceSnapshot,
    request_fingerprint,
    request_snapshot,
)
from app.agents.experience_agent import ExperienceAgent
from app.agents.logistics_agent import LogisticsAgent
from app.agents.tool_gateway import ToolGateway
from app.models.multi_agent import CandidateRegistry, ExperienceProposal, RegistryEntity
from app.models.schemas import Location, TripRequest, WeatherInfo
from tests.test_experience_agent import FakeLLM


def request():
    return TripRequest(city="New York", start_date="2026-06-01", end_date="2026-06-01",
                       travel_days=1, transportation="Public transit",
                       accommodation="Mid-range hotel", preferences=["Museums"])


class EvidenceSnapshotTests(unittest.TestCase):
    def snapshot(self):
        value = WorkflowEvidenceSnapshot(
            request_fingerprint=request_fingerprint(request()),
            request=request_snapshot(request()),
            experience=ExperienceEvidenceSnapshot(entities=[
                RegistryEntity(source_id="attraction:a1", provider_id="a1", entity_type="attraction",
                               name="Museum", location=Location(longitude=-73.9, latitude=40.7),
                               registered_by="experience")
            ]),
            logistics=AgentEvidenceSnapshot(entities=[
                RegistryEntity(source_id="hotel:h1", provider_id="h1", entity_type="hotel",
                               name="Hotel", location=Location(longitude=-73.91, latitude=40.71),
                               registered_by="logistics"),
                RegistryEntity(source_id="meal:m1", provider_id="m1", entity_type="meal",
                               name="Cafe", location=Location(longitude=-73.92, latitude=40.72),
                               registered_by="logistics"),
            ]),
            weather_info=[WeatherInfo(date="2026-06-01", day_weather="Clear")],
        )
        return value

    def test_snapshot_round_trip_and_integrity_validation(self):
        workflow = self.snapshot()
        snapshot = EvidenceSnapshotFile(metadata={"dataset_sha256": "dataset"},
                                        cases={workflow.request_fingerprint: workflow}).with_hash()
        loaded = EvidenceSnapshotFile.model_validate_json(snapshot.model_dump_json())
        loaded.validate_integrity()
        self.assertEqual(loaded.cases[workflow.request_fingerprint].request, request_snapshot(request()))
        loaded.metadata["dataset_sha256"] = "tampered"
        with self.assertRaises(EvidenceSnapshotMismatch):
            loaded.validate_integrity()

    def test_request_fingerprint_changes_with_request_payload(self):
        changed = request().model_copy(update={"city": "Chicago"})
        self.assertNotEqual(request_fingerprint(request()), request_fingerprint(changed))

    def test_replay_evidence_skips_experience_and_logistics_tools(self):
        workflow = self.snapshot()

        def provider_escape(**kwargs):
            raise AssertionError("provider should not be called in replay")

        registry = CandidateRegistry(run_id="run-1")
        experience_gateway = ToolGateway(
            registry=registry,
            tools={"attraction_search": provider_escape, "rag_search": provider_escape,
                   "place_detail": provider_escape},
        )
        experience_llm = FakeLLM([{
            "clusters": [{"name": "Museums", "attraction_aliases": ["A1"]}],
            "core_attraction_aliases": ["A1"], "optional_attraction_aliases": [],
            "rag_chunk_ids": [],
        }])
        experience = ExperienceAgent(llm=experience_llm, gateway=experience_gateway).run(
            request=request(), evidence_override=workflow.experience,
        ).proposal
        self.assertEqual(len(experience_llm.calls), 1)
        self.assertEqual(sum(experience_gateway.call_counts.values()), 0)

        logistics_gateway = ToolGateway(
            registry=registry,
            tools={"hotel_search": provider_escape, "meal_search": provider_escape},
        )
        logistics_llm = FakeLLM([{
            "primary_hotel_alias": "H1", "hotel_aliases": [], "meal_aliases": ["M1"],
            "constraints": [], "unknowns": [], "cost_assumptions": {"H1": 200},
        }])
        proposal = LogisticsAgent(llm=logistics_llm, gateway=logistics_gateway).run(
            request=request(), experience=experience, evidence_override=workflow.logistics,
        )
        self.assertEqual(proposal.primary_hotel_id, "hotel:h1")
        self.assertEqual(sum(logistics_gateway.call_counts.values()), 0)


if __name__ == "__main__":
    unittest.main()
