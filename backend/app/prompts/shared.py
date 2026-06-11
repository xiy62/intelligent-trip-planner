"""Shared prompt fragments for trip planning."""

from __future__ import annotations


TRIP_JSON_RULES = """请严格遵守以下要求:
1. 输出必须是完整JSON，不要输出解释性文字
2. 每天安排2-3个景点
3. 每天必须包含早中晚三餐
4. 每天推荐一个具体的酒店(从酒店信息中选择)
5. 考虑景点之间的距离和交通方式
6. weather_info数组必须覆盖所有出行日期
7. 温度必须是纯数字(不要带°C等单位)
8. 提供预算信息，并保证 budget 字段与各天明细自洽
9. 景点和酒店名称优先与候选列表完全一致
10. days[].transportation 必须填写本次请求的交通方式，不要改写成其他交通方式
11. days[].accommodation 必须填写本次请求的住宿偏好；具体酒店名称放在 days[].hotel.name，不要把酒店名写到 accommodation 字段
"""
