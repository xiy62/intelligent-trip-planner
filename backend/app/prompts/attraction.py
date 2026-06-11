"""Prompts and query builders for attraction retrieval."""

from __future__ import annotations

from ..models.schemas import TripRequest


ATTRACTION_AGENT_PROMPT = """你是景点搜索专家。你的任务是根据城市和用户偏好搜索合适的景点。

**重要提示:**
你必须使用工具来搜索景点!不要自己编造景点信息!

**工具调用格式:**
必须调用 `amap` 工具，并使用 JSON 参数，严格按照以下格式:
`[TOOL_CALL:amap:{"tool_name":"maps_text_search","arguments":{"keywords":"景点关键词","city":"城市名"}}]`

**示例:**
用户: "搜索北京的历史文化景点"
你的回复: [TOOL_CALL:amap:{"tool_name":"maps_text_search","arguments":{"keywords":"历史文化","city":"北京"}}]

用户: "搜索上海的公园"
你的回复: [TOOL_CALL:amap:{"tool_name":"maps_text_search","arguments":{"keywords":"公园","city":"上海"}}]

**注意:**
1. 必须使用工具,不要直接回答
2. 格式必须完全正确,包括方括号、冒号和JSON结构
"""


def build_attraction_search_terms(request: TripRequest, retry_count: int = 0) -> list[str]:
    terms: list[str] = []
    if request.preferences:
        terms.extend(request.preferences[:2])
    else:
        terms.append("景点")
    if retry_count > 0 or request.preferences:
        terms.append("著名景点")
    deduped: list[str] = []
    for term in terms:
        if term and term not in deduped:
            deduped.append(term)
    return deduped or ["景点"]


def build_attraction_query(request: TripRequest) -> str:
    keyword = build_attraction_search_terms(request)[0]
    return (
        f"请使用amap工具搜索{request.city}的{keyword}相关景点。\n"
        f"[TOOL_CALL:amap:{{\"tool_name\":\"maps_text_search\",\"arguments\":{{\"keywords\":\"{keyword}\",\"city\":\"{request.city}\"}}}}]"
    )
