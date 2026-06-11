"""Planner prompts for legacy and LangChain-native execution."""

from __future__ import annotations

from typing import Iterable, Sequence

from ..models.langgraph_state import EvaluationReport, PlannerInputBundle
from ..models.schemas import TripRequest
from .shared import TRIP_JSON_RULES


PLANNER_AGENT_PROMPT = """你是行程规划专家。你的任务是根据景点信息和天气信息,生成详细的旅行计划。

请严格按照JSON返回旅行计划，字段必须包括:
- city
- start_date
- end_date
- days
- weather_info
- overall_suggestions
- budget

预算字段必须包含:
- total_attractions
- total_hotels
- total_meals
- total_transportation
- total
"""


def build_retry_feedback(
    report: EvaluationReport | None,
    planner_inputs: PlannerInputBundle,
    travel_dates: Sequence[str],
) -> str:
    if report is None or report.passed:
        return ""

    attraction_names = [candidate.name for candidate in planner_inputs.attraction_candidates]
    hotel_names = [candidate.name for candidate in planner_inputs.hotel_candidates]
    unsupported_attractions = [
        item.name for item in report.unsupported_entities if item.entity_type == "attraction"
    ]
    unsupported_hotels = [
        item.name for item in report.unsupported_entities if item.entity_type == "hotel"
    ]

    feedback_lines = [
        "",
        "**上一轮生成失败，请严格修正以下问题后重新生成:**",
        f"- Hard failures: {', '.join(report.hard_failures) if report.hard_failures else '无'}",
    ]
    if unsupported_attractions:
        feedback_lines.append(
            f"- 以下景点上一轮未通过grounding校验，禁止再次使用: {', '.join(unsupported_attractions)}"
        )
    if unsupported_hotels:
        feedback_lines.append(
            f"- 以下酒店上一轮未通过grounding校验，禁止再次使用: {', '.join(unsupported_hotels)}"
        )
    if attraction_names:
        feedback_lines.append(
            f"- 景点必须只从以下候选中选择，名称必须完全一致: {', '.join(attraction_names)}"
        )
    if hotel_names:
        feedback_lines.append(
            f"- 酒店必须只从以下候选中选择，名称必须完全一致: {', '.join(hotel_names)}"
        )
    if "budget_consistency" in report.hard_failures or "budget_total_not_fully_aligned" in report.warnings:
        feedback_lines.append(
            "- budget.total_attractions、budget.total_hotels、budget.total_meals 必须分别等于各天明细汇总；budget.total 必须等于四项子预算之和。"
        )
    if "current_request_alignment" in report.hard_failures:
        feedback_lines.append(
            "- 当前请求字段优先级最高：days[].transportation 必须匹配本次请求的交通方式，days[].accommodation 必须匹配本次请求的住宿偏好；历史记忆不能覆盖本次请求。"
        )
        if report.unsupported_claims:
            feedback_lines.append(f"- 当前请求对齐问题: {'; '.join(report.unsupported_claims)}")
    if "date_coverage" in report.hard_failures:
        feedback_lines.append(
            f"- days[].date 和 weather_info[].date 必须完整覆盖以下日期: {', '.join(travel_dates)}"
        )
    feedback_lines.append("- 输出必须仍然是完整JSON，不要输出解释性文字。")
    return "\n".join(feedback_lines)


def build_planner_prompt(
    request: TripRequest,
    attractions_text: str,
    weather_text: str,
    hotels_text: str,
    rag_text: str,
    retry_feedback: str = "",
    format_instructions: str = "",
    memory_context: str = "",
) -> str:
    prompt = f"""请根据以下信息生成{request.city}的{request.travel_days}天旅行计划:

**基本信息:**
- 城市: {request.city}
- 日期: {request.start_date} 至 {request.end_date}
- 天数: {request.travel_days}天
- 交通方式: {request.transportation}
- 住宿: {request.accommodation}
- 偏好: {', '.join(request.preferences) if request.preferences else '无'}

**景点候选:**
{attractions_text}

**天气信息:**
{weather_text}

**酒店候选:**
{hotels_text}

**旅行知识参考:**
{rag_text}

**匿名偏好记忆:**
{memory_context or '无'}

记忆使用原则: 历史偏好只能作为软约束；如果它与本次请求中的城市、日期、交通、住宿、偏好或额外要求冲突，必须以本次请求为准。

**输出规则:**
{TRIP_JSON_RULES}
"""
    if request.free_text_input:
        prompt += f"\n**额外要求:** {request.free_text_input}\n"
    if retry_feedback:
        prompt += f"\n{retry_feedback}\n"
    if format_instructions:
        prompt += f"\n**格式要求:**\n{format_instructions}\n"
    return prompt
