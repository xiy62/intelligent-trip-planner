"""Tests for the LangGraph-native trip planner."""

from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path

from app.agents.langgraph_trip_planner import LangGraphTripPlanner
from app.models.schemas import TripPlan, TripRequest, WeatherInfo
from app.services.memory_service import MemoryService


def build_request() -> TripRequest:
    return TripRequest(
        city="北京",
        start_date="2026-06-01",
        end_date="2026-06-02",
        travel_days=2,
        transportation="公共交通",
        accommodation="经济型酒店",
        preferences=["历史文化", "美食"],
        free_text_input="希望行程不要太赶",
    )


def build_valid_plan_json(
    attraction_name: str = "故宫",
    hotel_name: str = "如家酒店",
    transportation: str = "公共交通",
    accommodation: str = "经济型酒店",
) -> str:
    return f"""```json
{{
  "city": "北京",
  "start_date": "2026-06-01",
  "end_date": "2026-06-02",
  "days": [
    {{
      "date": "2026-06-01",
      "day_index": 0,
      "description": "第1天行程",
      "transportation": "{transportation}",
      "accommodation": "{accommodation}",
      "hotel": {{
        "name": "{hotel_name}",
        "address": "东城区酒店1号",
        "estimated_cost": 300
      }},
      "attractions": [
        {{
          "name": "{attraction_name}",
          "address": "东城区景山前街4号",
          "location": {{"longitude": 116.397, "latitude": 39.917}},
          "visit_duration": 180,
          "description": "历史文化景点",
          "category": "景点",
          "ticket_price": 60
        }},
        {{
          "name": "天坛",
          "address": "东城区天坛路",
          "location": {{"longitude": 116.41, "latitude": 39.88}},
          "visit_duration": 120,
          "description": "世界文化遗产",
          "category": "景点",
          "ticket_price": 15
        }}
      ],
      "meals": [
        {{"type": "breakfast", "name": "早餐", "estimated_cost": 30}},
        {{"type": "lunch", "name": "午餐", "estimated_cost": 50}},
        {{"type": "dinner", "name": "晚餐", "estimated_cost": 80}}
      ]
    }},
    {{
      "date": "2026-06-02",
      "day_index": 1,
      "description": "第2天行程",
      "transportation": "{transportation}",
      "accommodation": "{accommodation}",
      "hotel": {{
        "name": "{hotel_name}",
        "address": "东城区酒店1号",
        "estimated_cost": 300
      }},
      "attractions": [
        {{
          "name": "景山公园",
          "address": "景山西街",
          "location": {{"longitude": 116.397, "latitude": 39.924}},
          "visit_duration": 90,
          "description": "适合步行",
          "category": "景点",
          "ticket_price": 10
        }},
        {{
          "name": "南锣鼓巷",
          "address": "东城区南锣鼓巷",
          "location": {{"longitude": 116.403, "latitude": 39.94}},
          "visit_duration": 120,
          "description": "适合美食探索",
          "category": "景点",
          "ticket_price": 0
        }}
      ],
      "meals": [
        {{"type": "breakfast", "name": "早餐", "estimated_cost": 30}},
        {{"type": "lunch", "name": "午餐", "estimated_cost": 50}},
        {{"type": "dinner", "name": "晚餐", "estimated_cost": 80}}
      ]
    }}
  ],
  "weather_info": [
    {{
      "date": "2026-09-01",
      "day_weather": "暴雨",
      "night_weather": "暴雨",
      "day_temp": 99,
      "night_temp": 88,
      "wind_direction": "北风",
      "wind_power": "10级"
    }}
  ],
  "overall_suggestions": "请舒适出行",
  "budget": {{
    "total_attractions": 85,
    "total_hotels": 600,
    "total_meals": 320,
    "total_transportation": 120,
    "total": 1125
  }}
}}
```"""


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def invoke(self, prompt: str):
        self.calls.append(prompt)
        if len(self.responses) == 1:
            content = self.responses[0]
        else:
            content = self.responses.pop(0)
        return type("FakeMessage", (), {"content": content})()


def parse_candidate_response(response: str):
    items = []
    for index, line in enumerate(response.splitlines()):
        if "地址:" not in line:
            continue
        name, address = line.split("地址:", 1)
        name = name.split(".", 1)[-1].strip(" -")
        items.append({"id": f"fake-{index}-{name}", "name": name, "address": address.strip()})
    return items


class FakeSearchTool:
    name = "amap_search_poi"

    def __init__(self, attraction_responses, hotel_responses):
        self.attraction_rounds = [parse_candidate_response(item) for item in attraction_responses]
        self.hotel_items = parse_candidate_response(hotel_responses[0])
        self.attraction_calls = 0

    def invoke(self, payload):
        keyword = payload["keywords"]
        if "酒店" in keyword or "宾馆" in keyword:
            return list(self.hotel_items)
        round_index = min(self.attraction_calls // 3, len(self.attraction_rounds) - 1)
        self.attraction_calls += 1
        return list(self.attraction_rounds[round_index])


class FakeAmapService:
    def __init__(self, attraction_responses, hotel_responses):
        self.tool = FakeSearchTool(attraction_responses, hotel_responses)

    def get_langchain_tools(self):
        return [self.tool]


class FakeWeatherService:
    def __init__(self):
        self.results = [
            WeatherInfo(
                date="2026-06-01",
                day_weather="晴",
                night_weather="多云",
                day_temp=30,
                night_temp=20,
                wind_direction="南风",
                wind_power="3级",
            ),
            WeatherInfo(
                date="2026-06-02",
                day_weather="阴",
                night_weather="小雨",
                day_temp=28,
                night_temp=18,
                wind_direction="东风",
                wind_power="2级",
            ),
        ]

    def get_weather_for_trip(self, city: str, start_date: str, travel_days: int):
        return list(self.results)

    def format_weather_for_planner(self, city: str, weather_info):
        lines = [f"{city}天气如下（已按行程日期对齐）："]
        for item in weather_info:
            lines.append(f"- {item.date}：白天{item.day_weather}，夜间{item.night_weather}")
        return "\n".join(lines)


class FakeNativeRuntime:
    def __init__(self, attraction_responses, hotel_responses, planner_responses):
        self.amap_service = FakeAmapService(attraction_responses, hotel_responses)
        self.llm = FakeLLM(planner_responses)
        self.weather_service = FakeWeatherService()

    def build_planner(self, **kwargs):
        return LangGraphTripPlanner(
            llm=self.llm,
            amap_service=self.amap_service,
            weather_service=self.weather_service,
            **kwargs,
        )


class LangGraphTripPlannerTests(unittest.TestCase):
    def test_weather_authority_and_checkpoint_snapshot(self):
        runtime = FakeNativeRuntime(
            attraction_responses=[
                "1. 故宫 - 地址: 东城区景山前街4号\n2. 天坛 - 地址: 东城区天坛路\n3. 景山公园 - 地址: 景山西街\n4. 南锣鼓巷 - 地址: 东城区南锣鼓巷"
            ],
            hotel_responses=["1. 如家酒店 - 地址: 东城区酒店1号"],
            planner_responses=[build_valid_plan_json()],
        )
        planner = runtime.build_planner()
        thread_id = "weather-authority-thread"
        state = planner.invoke_graph(build_request(), thread_id=thread_id)

        final_plan = state["final_plan"]
        self.assertEqual(final_plan.weather_info[0].date, "2026-06-01")
        self.assertEqual(final_plan.weather_info[0].day_weather, "晴")
        self.assertEqual(final_plan.weather_info[1].night_weather, "小雨")

        snapshot = planner.get_state_snapshot(thread_id)
        self.assertEqual(snapshot.values["final_plan"].city, "北京")
        self.assertEqual(snapshot.values["metrics"].evaluation_pass_count, 1)

    def test_planner_malformed_json_retries_then_succeeds(self):
        runtime = FakeNativeRuntime(
            attraction_responses=[
                "1. 故宫 - 地址: 东城区景山前街4号\n2. 天坛 - 地址: 东城区天坛路\n3. 景山公园 - 地址: 景山西街\n4. 南锣鼓巷 - 地址: 东城区南锣鼓巷"
            ],
            hotel_responses=["1. 如家酒店 - 地址: 东城区酒店1号"],
            planner_responses=["not-json", build_valid_plan_json()],
        )
        planner = runtime.build_planner(max_retries=2)
        state = planner.invoke_graph(build_request())

        self.assertEqual(state["final_plan"].city, "北京")
        self.assertEqual(state["retry_counts"].plan_itinerary, 2)
        self.assertEqual(state["metrics"].schema_failure_count, 1)
        self.assertTrue(state["evaluation_report"].passed)

    def test_grounding_failure_retries_attraction_retrieval(self):
        runtime = FakeNativeRuntime(
            attraction_responses=[
                "1. 颐和园 - 地址: 海淀区",
                "1. 故宫 - 地址: 东城区景山前街4号\n2. 天坛 - 地址: 东城区天坛路\n3. 景山公园 - 地址: 景山西街\n4. 南锣鼓巷 - 地址: 东城区南锣鼓巷",
            ],
            hotel_responses=["1. 如家酒店 - 地址: 东城区酒店1号"],
            planner_responses=[build_valid_plan_json(), build_valid_plan_json()],
        )
        planner = runtime.build_planner(max_retries=2)
        state = planner.invoke_graph(build_request())

        self.assertEqual(state["final_plan"].city, "北京")
        self.assertTrue(state["evaluation_report"].passed)
        self.assertGreaterEqual(state["retry_counts"].retrieve_attractions, 1)
        self.assertEqual(state["metrics"].grounding_failure_count, 1)

    def test_retry_exhaustion_falls_back(self):
        runtime = FakeNativeRuntime(
            attraction_responses=["1. 故宫 - 地址: 东城区景山前街4号"],
            hotel_responses=["1. 如家酒店 - 地址: 东城区酒店1号"],
            planner_responses=["bad-json", "still-bad-json"],
        )
        planner = runtime.build_planner(max_retries=1)
        state = planner.invoke_graph(build_request())

        self.assertTrue(state["final_plan"].overall_suggestions.startswith("这是为您规划的北京2日游行程"))
        self.assertEqual(state["metrics"].fallback_count, 1)
        self.assertEqual(state["retry_counts"].fallback_response, 1)

    def test_anonymous_profile_memory_is_injected_as_soft_context(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_service = MemoryService(Path(tmpdir) / "memory.sqlite3")
            profile_id = "profile_test_123"
            first_request = build_request().model_copy(update={"profile_id": profile_id})
            first_plan = TripPlan(
                **json.loads(build_valid_plan_json().split("```json\n", 1)[1].rsplit("\n```", 1)[0])
            )
            memory_service.update_after_success(
                profile_id=profile_id,
                conversation_id="first-conversation",
                request=first_request,
                plan=first_plan,
                memory_applied=False,
                memory_summary="",
            )

            runtime = FakeNativeRuntime(
                attraction_responses=[
                    "1. 故宫 - 地址: 东城区景山前街4号\n2. 天坛 - 地址: 东城区天坛路\n3. 景山公园 - 地址: 景山西街\n4. 南锣鼓巷 - 地址: 东城区南锣鼓巷"
                ],
                hotel_responses=["1. 如家酒店 - 地址: 东城区酒店1号"],
                planner_responses=[build_valid_plan_json()],
            )
            planner = runtime.build_planner(memory_service=memory_service)
            current_request = build_request().model_copy(
                update={
                    "profile_id": profile_id,
                    "conversation_id": "second-conversation",
                    "accommodation": "豪华酒店",
                }
            )
            state = planner.invoke_graph(current_request)

            self.assertTrue(state["memory_applied"])
            self.assertIn("历史文化", state["memory_summary"])
            self.assertEqual(state["memory_profile"]["accommodation"], "经济型酒店")
            self.assertIn("北京", state["memory_profile"]["recent_cities"])
            self.assertEqual(state["conversation_id"], "second-conversation")
            planner_prompt = runtime.llm.calls[0]
            self.assertIn("匿名偏好记忆", planner_prompt)
            self.assertIn("当前请求优先", planner_prompt)
            self.assertIn("住宿: 豪华酒店", planner_prompt)

    def test_current_request_alignment_guardrail_overrides_memory_conflict(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_service = MemoryService(Path(tmpdir) / "memory.sqlite3")
            profile_id = "profile_conflict_123"
            first_request = build_request().model_copy(update={"profile_id": profile_id})
            first_plan = TripPlan(
                **json.loads(build_valid_plan_json().split("```json\n", 1)[1].rsplit("\n```", 1)[0])
            )
            memory_service.update_after_success(
                profile_id=profile_id,
                conversation_id="first-conversation",
                request=first_request,
                plan=first_plan,
                memory_applied=False,
                memory_summary="",
            )

            runtime = FakeNativeRuntime(
                attraction_responses=[
                    "1. 故宫 - 地址: 东城区景山前街4号\n2. 天坛 - 地址: 东城区天坛路\n3. 景山公园 - 地址: 景山西街\n4. 南锣鼓巷 - 地址: 东城区南锣鼓巷"
                ],
                hotel_responses=["1. 如家酒店 - 地址: 东城区酒店1号"],
                planner_responses=[
                    build_valid_plan_json(accommodation="经济型酒店"),
                ],
            )
            planner = runtime.build_planner(max_retries=2, memory_service=memory_service)
            request = build_request().model_copy(
                update={
                    "profile_id": profile_id,
                    "accommodation": "豪华酒店",
                }
            )
            state = planner.invoke_graph(request)

            self.assertTrue(state["evaluation_report"].passed)
            self.assertEqual(state["retry_counts"].plan_itinerary, 1)
            self.assertEqual(state["final_plan"].days[0].accommodation, "豪华酒店")
            self.assertEqual(state["final_plan"].days[1].accommodation, "豪华酒店")
            self.assertEqual(memory_service.get_profile(profile_id)["accommodation"], "豪华酒店")

    def test_fallback_does_not_write_anonymous_profile_memory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_service = MemoryService(Path(tmpdir) / "memory.sqlite3")
            profile_id = "profile_fallback_123"
            runtime = FakeNativeRuntime(
                attraction_responses=["1. 故宫 - 地址: 东城区景山前街4号"],
                hotel_responses=["1. 如家酒店 - 地址: 东城区酒店1号"],
                planner_responses=["bad-json", "still-bad-json"],
            )
            planner = runtime.build_planner(max_retries=1, memory_service=memory_service)
            request = build_request().model_copy(update={"profile_id": profile_id})
            state = planner.invoke_graph(request)

            self.assertTrue(state["final_plan"].overall_suggestions.startswith("这是为您规划的北京2日游行程"))
            self.assertIsNone(memory_service.get_profile(profile_id))


if __name__ == "__main__":
    unittest.main()
