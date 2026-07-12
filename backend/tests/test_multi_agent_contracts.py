import unittest

from app.agents.tool_gateway import ToolGateway, ToolGatewayError
from app.models.multi_agent import CandidateRegistry, RegistryEntity


class MultiAgentContractTests(unittest.TestCase):
    def test_registry_enforces_agent_ownership(self):
        registry = CandidateRegistry(run_id="run-1")
        registry.add(
            RegistryEntity(
                source_id="poi-1",
                entity_type="attraction",
                name="Museum",
                registered_by="experience",
            ),
            actor="experience",
        )
        with self.assertRaises(ValueError):
            registry.add(
                RegistryEntity(
                    source_id="poi-1",
                    entity_type="hotel",
                    name="Hotel",
                    registered_by="logistics",
                ),
                actor="logistics",
            )

    def test_gateway_enforces_budget_and_deduplicates_queries(self):
        calls = []

        def search(**kwargs):
            calls.append(kwargs)
            return ["result"]

        gateway = ToolGateway(
            registry=CandidateRegistry(run_id="run-1"),
            tools={"attraction_search": search},
            budgets={"attraction_search": 1},
        )
        first = gateway.call(
            "experience", "attraction_search", query_key="museum", query="museum"
        )
        second = gateway.call(
            "experience", "attraction_search", query_key="Museum", query="museum"
        )
        self.assertEqual(first, second)
        self.assertEqual(len(calls), 1)
        with self.assertRaises(ToolGatewayError) as context:
            gateway.call("experience", "attraction_search", query_key="park", query="park")
        self.assertEqual(context.exception.code, "tool_budget_exhausted")

    def test_composer_has_no_tools(self):
        gateway = ToolGateway(registry=CandidateRegistry(run_id="run-1"))
        with self.assertRaises(ToolGatewayError) as context:
            gateway.call("composer", "attraction_search", query_key="museum")
        self.assertEqual(context.exception.code, "tool_not_allowed")


if __name__ == "__main__":
    unittest.main()
