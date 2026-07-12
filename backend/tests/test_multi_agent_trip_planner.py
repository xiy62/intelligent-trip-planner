import json
import unittest

from app.agents.multi_agent_trip_planner import MultiAgentTripPlanner
from app.models.langgraph_state import RAGChunk
from app.models.schemas import TripRequest, WeatherInfo


class FakeLLM:
    def __init__(self):
        self.responses = [
            {"attraction_queries": ["museum"], "rag_query": "museum planning"},
            {"version": 1, "run_id": "ignored", "clusters": [{"name": "Museums", "attraction_ids": ["a1"]}],
             "rag_chunk_ids": ["r1"], "evidence_sufficient": True},
            {"version": 1, "run_id": "ignored", "experience_version": 1,
             "hotel_ids": ["h1"], "meal_ids": ["m1"], "constraints": [], "infeasible_pairs": [],
             "unknowns": [], "cost_assumptions": {"h1": 200}},
            {"version": 1, "run_id": "ignored", "experience_version": 1, "logistics_version": 1,
             "days": [{"date": "2026-06-01", "day_index": 0, "description": "Museum day",
                       "attraction_items": [{"source_id": "a1", "visit_duration": 120,
                                             "description": "Explore the collection", "ticket_price": 25,
                                             "cost_status": "known"}],
                       "meal_items": [{"meal_type": "lunch", "source_id": "m1",
                                       "estimated_cost": 25, "cost_status": "estimated"}],
                       "hotel_id": "h1"}],
             "overall_suggestions": "Use public transit.", "transportation_estimate": 15},
        ]
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
    def __init__(self):
        self.tool = SearchTool()

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
        self.assertFalse(state["materialization_failures"])
        self.assertEqual(planner.health_summary()["workflow"], "langgraph_multi_agent")


if __name__ == "__main__":
    unittest.main()
