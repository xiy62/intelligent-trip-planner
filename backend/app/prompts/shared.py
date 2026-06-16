"""Shared prompt fragments for trip planning."""

from __future__ import annotations


TRIP_JSON_RULES = """Follow these requirements strictly:
1. Return complete JSON only; do not include explanatory prose.
2. Plan 2-3 attractions per day when enough candidates exist.
3. Include breakfast, lunch, and dinner for every day.
4. Recommend one concrete hotel per day from the hotel candidates.
5. Consider distance, route coherence, and the requested transportation mode.
6. The weather_info array must cover every travel date.
7. Temperatures must be plain numbers without °C/°F suffixes.
8. Include budget data and keep budget totals consistent with day-level items.
9. Prefer exact attraction and hotel names from the candidate lists.
10. days[].transportation must match the current request transportation exactly.
11. days[].accommodation must match the current request accommodation preference; put the concrete hotel name in days[].hotel.name, not in accommodation.
"""
