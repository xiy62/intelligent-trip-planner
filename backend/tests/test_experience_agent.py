import json
import unittest

from app.agents.experience_agent import ExperienceAgent, ExperienceAgentError
from app.agents.tool_gateway import ToolGateway
from app.models.multi_agent import CandidateRegistry
from app.models.schemas import TripRequest


class FakeLLM:
    def __init__(self, values):
        self.values = list(values)
        self.calls = []

    def invoke(self, prompt):
        self.calls.append(prompt)
        value = self.values.pop(0)
        return type("Message", (), {"content": json.dumps(value)})()


def request():
    return TripRequest(city="New York", start_date="2026-06-01", end_date="2026-06-01",
                       travel_days=1, transportation="Public transit",
                       accommodation="Mid-range hotel", preferences=["Museums"])


class ExperienceAgentTests(unittest.TestCase):
    def test_bounded_loop_registers_evidence_and_returns_typed_proposal(self):
        calls = []

        def attraction_search(**kwargs):
            calls.append(kwargs["query"])
            return [{"id": "a1", "name": "Museum", "address": "1 Way",
                     "location": {"longitude": -73.9, "latitude": 40.7}}]

        gateway = ToolGateway(registry=CandidateRegistry(run_id="run-1"),
                              tools={"attraction_search": attraction_search,
                                     "rag_search": lambda **kwargs: [{"chunk_id": "r1", "source": "guide",
                                                                       "title": "Museum guide", "content": "Visit early"}],
                                     "place_detail": lambda **kwargs: {}},
                              budgets={"attraction_search": 3, "rag_search": 1, "place_detail": 2})
        llm = FakeLLM([
            {"attraction_queries": ["museum", "museum"], "rag_query": "museum planning"},
            {"clusters": [{"name": "Art", "attraction_aliases": ["A1"]}],
             "core_attraction_aliases": ["A1"], "optional_attraction_aliases": [],
             "rag_chunk_ids": ["r1"], "uncovered_preferences": [], "evidence_sufficient": True},
        ])
        result = ExperienceAgent(llm=llm, gateway=gateway).run(request=request())
        self.assertEqual(result.proposal.run_id, "run-1")
        self.assertEqual(result.proposal.version, 1)
        self.assertEqual(result.proposal.allowed_attraction_ids, {"attraction:a1"})
        self.assertEqual(calls, ["museums attractions", "museum"])
        self.assertEqual(gateway.call_counts["attraction_search"], 2)
        self.assertEqual(len(llm.calls), 2)

    def test_transient_provider_failure_uses_deterministic_fallback(self):
        def fail(**kwargs):
            raise TimeoutError("provider timeout")

        gateway = ToolGateway(registry=CandidateRegistry(run_id="run-1"),
                              tools={"attraction_search": fail}, budgets={"attraction_search": 3})
        llm = FakeLLM([{"attraction_queries": ["museum"]}])
        result = ExperienceAgent(
            llm=llm,
            gateway=gateway,
            deterministic_fallback=lambda req: ([{"id": "fallback-a", "name": "Fallback Museum",
                                                   "location": {"longitude": -73.9, "latitude": 40.7}}], []),
        ).run(request=request())
        self.assertTrue(result.used_deterministic_fallback)
        self.assertFalse(result.proposal.evidence_sufficient)

    def test_invalid_source_id_is_rejected_without_message_history(self):
        gateway = ToolGateway(registry=CandidateRegistry(run_id="run-1"),
                              tools={"attraction_search": lambda **kwargs: [{"id": "a1", "name": "Museum",
                                                                             "location": {"longitude": 1, "latitude": 1}}]},
                              budgets={"attraction_search": 3})
        llm = FakeLLM([
            {"attraction_queries": ["museum"]},
            {"clusters": [{"name": "Bad", "attraction_aliases": ["A99"]}],
             "core_attraction_aliases": ["A99"], "optional_attraction_aliases": []},
        ])
        with self.assertRaises(ExperienceAgentError) as context:
            ExperienceAgent(llm=llm, gateway=gateway).run(request=request())
        self.assertEqual(context.exception.code, "invalid_source_id")
        self.assertNotIn("messages", llm.calls[-1])


if __name__ == "__main__":
    unittest.main()
