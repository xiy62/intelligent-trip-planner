import unittest

from app.agents.tool_gateway import ToolGateway, ToolGatewayError
from app.models.multi_agent import CallBudgetLedger, CandidateRegistry, RegistryEntity, registry_source_id


class MultiAgentContractTests(unittest.TestCase):
    def test_registry_source_ids_are_scoped_by_entity_type(self):
        self.assertEqual(registry_source_id("attraction", "shared"), "attraction:shared")
        self.assertEqual(registry_source_id("hotel", "shared"), "hotel:shared")
        self.assertEqual(registry_source_id("meal", "shared"), "meal:shared")
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

    def test_cumulative_role_and_global_budgets_cannot_reset_between_attempts(self):
        ledger = CallBudgetLedger(
            role_limits={"experience": {"llm": 1, "maps": 1, "rag": 1},
                         "logistics": {"llm": 1, "maps": 1, "rag": 0},
                         "composer": {"llm": 1, "maps": 0, "rag": 0}},
            global_limits={"llm": 2, "maps": 1, "rag": 1},
        )
        ledger.consume("experience", "maps", "search")
        with self.assertRaises(ValueError):
            ledger.consume("experience", "maps", "retry_search")
        with self.assertRaises(ValueError):
            ledger.consume("logistics", "maps", "global_search")
        self.assertEqual(ledger.global_used["maps"], 1)
        self.assertEqual(len(ledger.blocked_calls), 2)


if __name__ == "__main__":
    unittest.main()
