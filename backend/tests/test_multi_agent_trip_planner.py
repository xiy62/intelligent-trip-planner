import json
import unittest

from app.agents.multi_agent_trip_planner import MultiAgentTripPlanner
from app.models.langgraph_state import EvaluationReport, RAGChunk, RouteFailureDetail
from app.models.multi_agent import AgentRetryState, CandidateRegistry
from app.models.schemas import TripRequest, WeatherInfo


class FakeLLM:
    def __init__(self, responses=None):
        self.responses = list(responses or [
            {"attraction_queries": ["museum"], "rag_query": "museum planning"},
            {"clusters": [{"name": "Museums", "attraction_aliases": ["A1"]}],
             "core_attraction_aliases": ["A1"], "optional_attraction_aliases": [],
             "rag_chunk_ids": ["r1"], "evidence_sufficient": True},
            {"primary_hotel_alias": "H1", "hotel_aliases": [], "meal_aliases": ["M1"],
             "constraints": [], "infeasible_pairs": [], "unknowns": [], "cost_assumptions": {"H1": 200}},
            {"version": 1, "run_id": "ignored", "experience_version": 1, "logistics_version": 1,
             "days": [{"date": "2026-06-01", "day_index": 0, "description": "Museum day",
                       "attraction_items": [{"source_id": "A1", "visit_duration": 120,
                                             "description": "Explore the collection", "ticket_price": 25,
                                             "cost_status": "known"}],
                       "meal_items": [{"meal_type": "lunch", "source_id": "M1",
                                       "estimated_cost": 25, "cost_status": "estimated"}],
                       "hotel_id": "H1"}],
             "overall_suggestions": "Use public transit.", "transportation_estimate": 15},
        ])
        self.calls = []

    def invoke(self, prompt):
        self.calls.append(prompt)
        return type("Message", (), {"content": json.dumps(self.responses.pop(0))})()


class SearchTool:
    name = "map_search_poi"

    def __init__(self):
        self.calls = []

    def invoke(self, payload):
        self.calls.append(payload)
        query = payload["keywords"].lower()
        common_location = {"longitude": -73.98, "latitude": 40.76}
        if "hotel" in query:
            return [{"id": "h1", "name": "Registry Hotel", "address": "2 Hotel Way", "location": common_location}]
        if any(word in query for word in ("restaurant", "dining", "food")):
            return [{"id": "m1", "name": "Registry Cafe", "address": "3 Cafe Way", "location": common_location}]
        return [{"id": "a1", "name": "Registry Museum", "address": "1 Museum Way", "location": common_location}]


class FakeMapService:
    def __init__(self, tool=None):
        self.tool = tool or SearchTool()

    def get_langchain_tools(self):
        return [self.tool]

    def plan_route(self, **kwargs):
        return {"duration": 600, "distance": 800}


class FakeRAGService:
    def retrieve_chunks(self, request, **kwargs):
        return [RAGChunk(chunk_id="r1", source="guide", title="Museum planning",
                         content="Visit the museum early and use transit.")]


class FakeWeatherService:
    def get_weather_for_trip(self, **kwargs):
        return [WeatherInfo(date="2026-06-01", day_weather="Clear", night_weather="Clear",
                            day_temp=25, night_temp=18)]


class FakeMemoryService:
    def get_profile_snapshot(self, profile_id):
        return None

    def build_memory_context_for_request(self, profile, request):
        return "", []

    def record_successful_plan(self, **kwargs):
        return None


class MultiAgentTripPlannerTests(unittest.TestCase):
    def test_full_happy_path_has_three_independent_model_roles_and_canonical_fields(self):
        llm = FakeLLM()
        planner = MultiAgentTripPlanner(llm=llm, map_service=FakeMapService(),
                                        rag_service=FakeRAGService(), weather_service=FakeWeatherService(),
                                        memory_service=FakeMemoryService(), rag_mode="local_lightweight")
        request = TripRequest(city="New York", start_date="2026-06-01", end_date="2026-06-01",
                              travel_days=1, transportation="Public transit",
                              accommodation="Mid-range hotel", preferences=["Museums"])
        state = planner.invoke_graph(request)
        self.assertEqual(len(llm.calls), 4)
        self.assertEqual(set(state["agent_metrics"].by_agent), {"experience", "logistics", "composer"})
        self.assertEqual(state["final_plan"].days[0].attractions[0].name, "Registry Museum")
        self.assertEqual(state["final_plan"].days[0].hotel.name, "Registry Hotel")
        self.assertEqual(state["final_plan"].days[0].meals[0].name, "Registry Cafe")
        self.assertEqual(state["final_plan"].days[0].attractions[0].poi_id, "a1")
        self.assertEqual(state["final_plan"].days[0].hotel.poi_id, "h1")
        self.assertEqual(state["final_plan"].days[0].meals[0].poi_id, "m1")
        self.assertFalse(state["materialization_failures"])
        self.assertEqual(planner.health_summary()["workflow"], "langgraph_multi_agent")
        self.assertEqual(planner.search_poi_tool.calls[0]["page_size"], 12)
        self.assertIn("initial_attractions_per_day=[1]", llm.calls[3])
        self.assertIn("no alias occurs twice", llm.calls[3])

    def test_same_provider_id_can_exist_as_attraction_hotel_and_meal(self):
        class CollidingSearchTool(SearchTool):
            def invoke(self, payload):
                query = payload["keywords"].lower()
                location = {"longitude": -73.98, "latitude": 40.76}
                if "hotel" in query:
                    return [{"id": "shared", "name": "Shared Hotel", "location": location}]
                if any(word in query for word in ("restaurant", "dining", "food")):
                    return [{"id": "shared", "name": "Shared Cafe", "location": location}]
                return [{"id": "shared", "name": "Shared Museum", "location": location}]

        responses = [
            {"attraction_queries": ["museum"], "rag_query": "museum planning"},
            {"clusters": [{"name": "Museums", "attraction_aliases": ["A1"]}],
             "core_attraction_aliases": ["A1"], "optional_attraction_aliases": [],
             "rag_chunk_ids": ["r1"]},
            {"primary_hotel_alias": "H1", "hotel_aliases": [], "meal_aliases": ["M1"],
             "cost_assumptions": {"H1": 200}},
            {"version": 1, "run_id": "ignored", "experience_version": 1, "logistics_version": 1,
             "days": [{"date": "2026-06-01", "day_index": 0, "description": "Day",
                       "attraction_items": [{"source_id": "A1", "visit_duration": 120}],
                       "meal_items": [{"meal_type": "lunch", "source_id": "M1"}],
                       "hotel_id": "H1"}]},
        ]
        planner = MultiAgentTripPlanner(llm=FakeLLM(responses),
                                        map_service=FakeMapService(CollidingSearchTool()),
                                        rag_service=FakeRAGService(), weather_service=FakeWeatherService(),
                                        memory_service=FakeMemoryService(), rag_mode="local_lightweight")
        state = planner.invoke_graph(TripRequest(city="New York", start_date="2026-06-01",
                                                 end_date="2026-06-01", travel_days=1,
                                                 transportation="Public transit",
                                                 accommodation="Mid-range hotel", preferences=["Museums"]))
        self.assertTrue(state["evaluation_report"].passed)
        self.assertEqual(set(state["candidate_registry"].entities),
                         {"attraction:shared", "hotel:shared", "meal:shared"})
        self.assertEqual(state["final_plan"].days[0].attractions[0].poi_id, "shared")

    def test_invalid_experience_id_gets_one_bounded_revision_before_fallback(self):
        responses = [
            {"attraction_queries": ["museum"], "rag_query": "museum planning"},
            {"clusters": [{"name": "Bad", "attraction_aliases": ["A99"]}],
             "core_attraction_aliases": ["A99"], "optional_attraction_aliases": []},
            {"attraction_queries": ["museum"], "rag_query": "museum planning"},
            {"clusters": [{"name": "Museums", "attraction_aliases": ["A1"]}],
             "core_attraction_aliases": ["A1"], "optional_attraction_aliases": [],
             "rag_chunk_ids": ["r1"]},
            {"primary_hotel_alias": "H1", "hotel_aliases": [], "meal_aliases": ["M1"],
             "cost_assumptions": {"H1": 200}},
            {"version": 1, "run_id": "ignored", "experience_version": 1, "logistics_version": 1,
             "days": [{"date": "2026-06-01", "day_index": 0, "description": "Day",
                       "attraction_items": [{"source_id": "A1", "visit_duration": 120}],
                       "meal_items": [{"meal_type": "lunch", "source_id": "M1"}],
                       "hotel_id": "H1"}]},
        ]
        llm = FakeLLM(responses)
        planner = MultiAgentTripPlanner(llm=llm, map_service=FakeMapService(),
                                        rag_service=FakeRAGService(), weather_service=FakeWeatherService(),
                                        memory_service=FakeMemoryService(), rag_mode="local_lightweight")
        state = planner.invoke_graph(TripRequest(city="New York", start_date="2026-06-01",
                                                 end_date="2026-06-01", travel_days=1,
                                                 transportation="Public transit",
                                                 accommodation="Mid-range hotel", preferences=["Museums"]))
        self.assertTrue(state["evaluation_report"].passed)
        self.assertEqual(state["agent_retry_state"].experience_attempts, 2)
        self.assertEqual(state["agent_metrics"].targeted_retries, ["experience"])
        self.assertEqual(len(llm.calls), 6)

    def test_failure_owner_routing_and_retry_budgets_are_deterministic(self):
        planner = object.__new__(MultiAgentTripPlanner)
        attraction_report = EvaluationReport(
            passed=False,
            hard_failures=["schema_correctness"],
            materialization_failures=[{"code": "unknown_id", "path": "days.0.attractions.0", "source_id": "bad"}],
        )
        state = {"candidate_registry": CandidateRegistry(run_id="run-1"),
                 "materialization_failures": attraction_report.materialization_failures,
                 "agent_retry_state": AgentRetryState(experience_attempts=1, logistics_attempts=1,
                                                       composer_attempts=1)}
        self.assertEqual(planner._failure_owner(state, attraction_report), "experience")
        attraction_report.failure_owner = "experience"
        state["evaluation_report"] = attraction_report
        self.assertEqual(planner._route_multi_result(state), "experience_retry")
        state["agent_retry_state"].experience_attempts = 2
        self.assertEqual(planner._route_multi_result(state), "fallback_response")

    def test_missing_route_data_does_not_trigger_llm_retry(self):
        planner = object.__new__(MultiAgentTripPlanner)
        report = EvaluationReport(passed=False, route_failure_details=[
            RouteFailureDetail(day_index=0, segment_indices=[0], kind="missing_route_data")
        ])
        state = {"materialization_failures": [], "agent_retry_state": AgentRetryState()}
        self.assertIsNone(planner._failure_owner(state, report))

    def test_route_problem_escalates_from_composer_to_logistics_candidate_set(self):
        planner = object.__new__(MultiAgentTripPlanner)
        report = EvaluationReport(passed=False, quality_warnings=[
            "route_day_0_long_transfer_90min", "low_route_coherence_score"
        ])
        first_state = {"materialization_failures": [],
                       "agent_retry_state": AgentRetryState(composer_attempts=1)}
        report.route_failure_details = planner._route_failure_details(first_state, report)
        self.assertEqual(report.route_failure_details[0].kind, "ordering_problem")
        self.assertEqual(planner._failure_owner(first_state, report), "composer")
        second_state = {"materialization_failures": [],
                        "agent_retry_state": AgentRetryState(composer_attempts=2)}
        report.route_failure_details = planner._route_failure_details(second_state, report)
        self.assertEqual(report.route_failure_details[0].kind, "candidate_set_problem")
        self.assertEqual(planner._failure_owner(second_state, report), "logistics")

    def test_factory_has_no_single_agent_mode_branch(self):
        from app.agents import planner_factory

        original = planner_factory.MultiAgentTripPlanner
        sentinel = object()
        try:
            planner_factory.reset_trip_planner_agent()
            planner_factory.MultiAgentTripPlanner = lambda **kwargs: sentinel
            self.assertIs(planner_factory.get_trip_planner_agent(), sentinel)
        finally:
            planner_factory.MultiAgentTripPlanner = original
            planner_factory.reset_trip_planner_agent()


if __name__ == "__main__":
    unittest.main()
