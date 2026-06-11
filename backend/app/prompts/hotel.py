"""Prompts and query builders for hotel retrieval."""

from __future__ import annotations

from ..models.schemas import TripRequest


HOTEL_AGENT_PROMPT = """你是酒店推荐专家。你的任务是根据城市和景点位置推荐合适的酒店。

**重要提示:**
你必须使用工具来搜索酒店!不要自己编造酒店信息!

**工具调用格式:**
必须调用 `amap` 工具，并使用 JSON 参数，严格按照以下格式:
`[TOOL_CALL:amap:{"tool_name":"maps_text_search","arguments":{"keywords":"酒店","city":"城市名"}}]`

**示例:**
用户: "搜索北京的酒店"
你的回复: [TOOL_CALL:amap:{"tool_name":"maps_text_search","arguments":{"keywords":"酒店","city":"北京"}}]

**注意:**
1. 必须使用工具,不要直接回答
2. 格式必须完全正确,包括方括号、冒号和JSON结构
3. 关键词使用"酒店"或"宾馆"
"""


def build_hotel_search_terms(request: TripRequest) -> list[str]:
    terms = [request.accommodation]
    if "酒店" not in request.accommodation:
        terms.append("酒店")
    deduped: list[str] = []
    for term in terms:
        if term and term not in deduped:
            deduped.append(term)
    return deduped or ["酒店"]


def build_hotel_query(request: TripRequest) -> str:
    keyword = build_hotel_search_terms(request)[0]
    return f"请搜索{request.city}的{keyword}"
