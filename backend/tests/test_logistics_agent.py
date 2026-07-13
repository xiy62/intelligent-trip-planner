import json
import unittest

from app.agents.evidence_snapshot import AgentEvidenceSnapshot
from app.agents.logistics_agent import LogisticsAgent, LogisticsAgentError
from app.agents.tool_gateway import ToolGateway
from app.models.multi_agent import CandidateRegistry, ExperienceProposal, RegistryEntity
from app.models.schemas import Location, TripRequest


class FakeLLM:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def invoke(self, prompt):
        self.calls.append(prompt)
        return type("Message", (), {"content": json.dumps(self.response)})()


def request():
    return TripRequest(city="New York", start_date="2026-06-01", end_date="2026-06-01",
                       travel_days=1, transportation="Public transit",
                       accommodation="Mid-range hotel", preferences=["Museums"])


class LogisticsAgentTests(unittest.TestCase):
    def setUp(self):
        self.registry = CandidateRegistry(run_id="run-1")
        for entity in (
            RegistryEntity(source_id="attraction:a1", provider_id="a1", entity_type="attraction",
                           name="Museum One", location=Location(longitude=-73.98, latitude=40.76),
                           metadata={"category": "Museums"}, registered_by="experience"),
            RegistryEntity(source_id="attraction:a2", provider_id="a2", entity_type="attraction",
                           name="Museum Two", location=Location(longitude=-73.97, latitude=40.75),
                           metadata={"category": "Museums"}, registered_by="experience"),
        ):
            self.registry.add(entity, actor="experience")
        self.experience = ExperienceProposal(run_id="run-1", core_attraction_ids=["attraction:a2"])

    def agent(self, response):
        return LogisticsAgent(
            llm=FakeLLM(response),
            gateway=ToolGateway(registry=self.registry, tools={}),
        )

    @staticmethod
    def evidence(hotels):
        return AgentEvidenceSnapshot(entities=[*hotels,
            RegistryEntity(source_id="meal:m1", provider_id="m1", entity_type="meal", name="Cafe",
                           location=Location(longitude=-73.98, latitude=40.76), registered_by="logistics")])

    @staticmethod
    def hotel(provider_id, name, longitude, latitude):
        return RegistryEntity(source_id=f"hotel:{provider_id}", provider_id=provider_id, entity_type="hotel",
                              name=name, location=Location(longitude=longitude, latitude=latitude),
                              registered_by="logistics")

    @staticmethod
    def response(primary):
        return {"primary_hotel_alias": primary, "hotel_aliases": [], "meal_aliases": ["M1"],
                "constraints": [], "infeasible_pairs": [], "unknowns": [],
                "cost_assumptions": {"H1": 200}}

    def test_stable_attraction_anchors_do_not_depend_on_experience_core_order(self):
        agent = self.agent(self.response("H1"))
        ranked, names, centroid = agent._stable_attraction_context(request())
        self.assertEqual([item.source_id for item in ranked], ["attraction:a1", "attraction:a2"])
        self.assertEqual(names, ["Museum One", "Museum Two"])
        self.assertIsNotNone(centroid)

    def test_dominant_hotel_uses_deterministic_top_candidate(self):
        hotels = [self.hotel("h1", "Mid-range Hotel Central", -73.98, 40.76),
                  self.hotel("h2", "Remote Motel", -74.30, 41.10)]
        agent = self.agent(self.response("H2"))
        proposal = agent.run(request=request(), experience=self.experience,
                             evidence_override=self.evidence(hotels))
        self.assertEqual(proposal.primary_hotel_id, "hotel:h1")
        self.assertEqual(agent.trace["agent_selected_primary_hotel_id"], "hotel:h2")
        self.assertEqual(agent.trace["primary_hotel_selection_mode"], "deterministic_dominant")

    def test_competitive_hotel_remains_agent_selected(self):
        hotels = [self.hotel("h1", "Mid-range Hotel One", -73.98, 40.76),
                  self.hotel("h2", "Mid-range Hotel Two", -73.98, 40.76)]
        agent = self.agent(self.response("H2"))
        proposal = agent.run(request=request(), experience=self.experience,
                             evidence_override=self.evidence(hotels))
        self.assertEqual(proposal.primary_hotel_id, "hotel:h2")
        self.assertEqual(agent.trace["primary_hotel_selection_mode"], "agent_competitive")
        self.assertEqual(set(agent.trace["competitive_hotel_ids"]), {"hotel:h1", "hotel:h2"})
        self.assertIn("default to the first competitive alias", agent.llm.calls[0])

    def test_noncompetitive_agent_alias_is_rejected(self):
        hotels = [self.hotel("h1", "Mid-range Hotel One", -73.98, 40.76),
                  self.hotel("h2", "Mid-range Hotel Two", -73.98, 40.76),
                  self.hotel("h3", "Remote Motel", -74.30, 41.10)]
        agent = self.agent(self.response("H3"))
        with self.assertRaises(LogisticsAgentError) as context:
            agent.run(request=request(), experience=self.experience,
                      evidence_override=self.evidence(hotels))
        self.assertEqual(context.exception.code, "invalid_source_id")


if __name__ == "__main__":
    unittest.main()
