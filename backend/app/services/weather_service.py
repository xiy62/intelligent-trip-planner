"""Weather service with Open-Meteo as the default provider."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import httpx

from ..config import get_settings
from ..models.schemas import WeatherInfo


WEATHER_CODE_MAP: Dict[int, Tuple[str, str]] = {
    0: ("Clear", "Clear"),
    1: ("Mostly clear", "Mostly clear"),
    2: ("Partly cloudy", "Partly cloudy"),
    3: ("Overcast", "Overcast"),
    45: ("Fog", "Fog"),
    48: ("Rime fog", "Rime fog"),
    51: ("Light drizzle", "Light drizzle"),
    53: ("Drizzle", "Drizzle"),
    55: ("Dense drizzle", "Dense drizzle"),
    56: ("Freezing drizzle", "Freezing drizzle"),
    57: ("Dense freezing drizzle", "Dense freezing drizzle"),
    61: ("Light rain", "Light rain"),
    63: ("Rain", "Rain"),
    65: ("Heavy rain", "Heavy rain"),
    66: ("Freezing rain", "Freezing rain"),
    67: ("Heavy freezing rain", "Heavy freezing rain"),
    71: ("Light snow", "Light snow"),
    73: ("Snow", "Snow"),
    75: ("Heavy snow", "Heavy snow"),
    77: ("Snow grains", "Snow grains"),
    80: ("Rain showers", "Rain showers"),
    81: ("Rain showers", "Rain showers"),
    82: ("Violent rain showers", "Violent rain showers"),
    85: ("Snow showers", "Snow showers"),
    86: ("Snow showers", "Snow showers"),
    95: ("Thunderstorm", "Thunderstorm"),
    96: ("Thunderstorm with hail", "Thunderstorm with hail"),
    99: ("Thunderstorm with heavy hail", "Thunderstorm with heavy hail"),
}

OPENMETEO_CITY_ALIASES: Dict[str, str] = {
    "北京": "Beijing",
    "北京市": "Beijing",
    "上海": "Shanghai",
    "上海市": "Shanghai",
    "广州": "Guangzhou",
    "广州市": "Guangzhou",
    "成都": "Chengdu",
    "成都市": "Chengdu",
    "杭州": "Hangzhou",
    "杭州市": "Hangzhou",
    "深圳": "Shenzhen",
    "深圳市": "Shenzhen",
    "重庆": "Chongqing",
    "重庆市": "Chongqing",
    "天津": "Tianjin",
    "天津市": "Tianjin",
    "南京": "Nanjing",
    "南京市": "Nanjing",
    "西安": "Xi'an",
    "西安市": "Xi'an",
    "武汉": "Wuhan",
    "武汉市": "Wuhan",
    "苏州": "Suzhou",
    "苏州市": "Suzhou",
}


def _wind_direction_from_degree(deg: Optional[float]) -> str:
    if deg is None:
        return ""
    directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    idx = int((deg % 360) / 45.0 + 0.5) % 8
    return directions[idx]


class BaseWeatherProvider(ABC):
    @abstractmethod
    def get_forecast(self, city: str, start_date: str, travel_days: int) -> List[WeatherInfo]:
        """Return weather aligned to trip dates."""


class OpenMeteoWeatherProvider(BaseWeatherProvider):
    """Globally available Open-Meteo provider that requires no API key."""

    GEO_URL = "https://geocoding-api.open-meteo.com/v1/search"
    FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
    MAX_FORECAST_DAYS = 16

    def __init__(self):
        self._timeout = 15.0

    def get_forecast(self, city: str, start_date: str, travel_days: int) -> List[WeatherInfo]:
        dates = self._travel_dates(start_date, travel_days)
        if not self._within_forecast_horizon(start_date):
            return self._unknown_weather(dates)

        lat_lon = self._geocode_city(city)
        if not lat_lon:
            return self._unknown_weather(dates)

        lat, lon = lat_lon
        data = self._fetch_forecast(lat, lon, start_date, travel_days)
        if not data:
            return self._unknown_weather(dates)

        daily = data.get("daily", {})
        times = daily.get("time", [])
        codes = daily.get("weather_code", [])
        max_t = daily.get("temperature_2m_max", [])
        min_t = daily.get("temperature_2m_min", [])
        wind_speed = daily.get("wind_speed_10m_max", [])
        wind_deg = daily.get("wind_direction_10m_dominant", [])

        by_date: Dict[str, WeatherInfo] = {}
        for i, d in enumerate(times):
            if i >= len(codes):
                continue
            code = codes[i]
            if code is None:
                day_weather, night_weather = ("Unknown", "Unknown")
            else:
                day_weather, night_weather = WEATHER_CODE_MAP.get(int(code), ("Unknown", "Unknown"))
            ws = wind_speed[i] if i < len(wind_speed) else None
            wd = wind_deg[i] if i < len(wind_deg) else None
            by_date[d] = WeatherInfo(
                date=d,
                day_weather=day_weather,
                night_weather=night_weather,
                day_temp=int(max_t[i]) if i < len(max_t) and max_t[i] is not None else 0,
                night_temp=int(min_t[i]) if i < len(min_t) and min_t[i] is not None else 0,
                wind_direction=_wind_direction_from_degree(wd),
                wind_power=f"{int(round(ws))} km/h" if ws is not None else ""
            )

        results: List[WeatherInfo] = []
        for d in dates:
            if d in by_date:
                results.append(by_date[d])
            else:
                results.append(
                    WeatherInfo(
                        date=d,
                        day_weather="Unknown",
                        night_weather="Unknown",
                        day_temp=0,
                        night_temp=0,
                        wind_direction="",
                        wind_power=""
                    )
                )
        return results

    def _geocode_city(self, city: str) -> Optional[Tuple[float, float]]:
        queries = self._build_geocoding_queries(city)
        candidates = []
        try:
            with httpx.Client(timeout=self._timeout) as client:
                for query in queries:
                    params = {"name": query, "count": 10, "language": "zh", "format": "json"}
                    resp = client.get(self.GEO_URL, params=params)
                    resp.raise_for_status()
                    data = resp.json()
                    results = data.get("results") or []
                    candidates.extend(results)
            best = self._select_best_result(city, candidates)
            if not best:
                return None
            return float(best["latitude"]), float(best["longitude"])
        except Exception as e:
            print(f"Open-Meteo geocoding failed: {str(e)}")
            return None

    def _build_geocoding_queries(self, city: str) -> List[str]:
        normalized = city.strip()
        queries = []
        alias = OPENMETEO_CITY_ALIASES.get(normalized)
        if alias:
            queries.append(alias)
        queries.append(normalized)
        deduped: List[str] = []
        seen = set()
        for item in queries:
            if item not in seen:
                deduped.append(item)
                seen.add(item)
        return deduped

    def _normalize_place_name(self, value: Optional[str]) -> str:
        if not value:
            return ""
        normalized = value.strip().lower()
        for suffix in ("市", "区", "县", "省", "自治区", "特别行政区", "自治州", "地区"):
            if normalized.endswith(suffix):
                normalized = normalized[: -len(suffix)]
        return normalized.replace(" ", "")

    def _select_best_result(self, city: str, results: List[dict]) -> Optional[dict]:
        if not results:
            return None

        target = self._normalize_place_name(city)
        english_alias = OPENMETEO_CITY_ALIASES.get(city.strip(), "")
        english_target = self._normalize_place_name(english_alias)

        def score(item: dict) -> int:
            item_name = self._normalize_place_name(item.get("name"))
            admin1 = self._normalize_place_name(item.get("admin1"))
            admin2 = self._normalize_place_name(item.get("admin2"))
            feature_code = item.get("feature_code", "")
            country_code = item.get("country_code", "")

            total = 0
            if country_code == "CN":
                total += 50
            if feature_code in {"PPLC", "PPLA", "ADM1"}:
                total += 30
            elif feature_code == "PPL":
                total += 20

            if item_name == target:
                total += 120
            elif target and target in item_name:
                total += 60

            if target and (admin1 == target or admin2 == target):
                total += 120

            if english_target and item_name == english_target:
                total += 100
            elif english_target and (admin1 == english_target or admin2 == english_target):
                total += 100

            return total

        ranked = sorted(results, key=score, reverse=True)
        return ranked[0]

    def _fetch_forecast(self, lat: float, lon: float, start_date: str, travel_days: int) -> Optional[dict]:
        end_date = (
            datetime.strptime(start_date, "%Y-%m-%d") + timedelta(days=travel_days - 1)
        ).strftime("%Y-%m-%d")
        params = {
            "latitude": lat,
            "longitude": lon,
            "timezone": "auto",
            "daily": "weather_code,temperature_2m_max,temperature_2m_min,wind_speed_10m_max,wind_direction_10m_dominant",
            "start_date": start_date,
            "end_date": end_date,
        }
        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.get(self.FORECAST_URL, params=params)
                resp.raise_for_status()
            return resp.json()
        except Exception as e:
            print(f"Open-Meteo weather lookup failed: {str(e)}")
            return None

    def _travel_dates(self, start_date: str, travel_days: int) -> List[str]:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        return [(start + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(travel_days)]

    def _within_forecast_horizon(self, start_date: str) -> bool:
        target = datetime.strptime(start_date, "%Y-%m-%d").date()
        today = datetime.now().date()
        latest_supported = today + timedelta(days=self.MAX_FORECAST_DAYS - 1)
        return today <= target <= latest_supported

    def _unknown_weather(self, dates: List[str]) -> List[WeatherInfo]:
        return [
            WeatherInfo(
                date=d,
                day_weather="Unknown",
                night_weather="Unknown",
                day_temp=0,
                night_temp=0,
                wind_direction="",
                wind_power=""
            )
            for d in dates
        ]


class WeatherService:
    def __init__(self):
        settings = get_settings()
        self.provider_name = (settings.weather_provider or "openmeteo").lower()

        if self.provider_name == "openmeteo":
            self.provider: BaseWeatherProvider = OpenMeteoWeatherProvider()
        else:
            # Unsupported providers currently fall back to Open-Meteo.
            print(f"Unsupported weather provider {self.provider_name}; using openmeteo")
            self.provider = OpenMeteoWeatherProvider()
            self.provider_name = "openmeteo"

    def get_weather_for_trip(self, city: str, start_date: str, travel_days: int) -> List[WeatherInfo]:
        return self.provider.get_forecast(city, start_date, travel_days)

    def format_weather_for_planner(self, city: str, weather_info: List[WeatherInfo]) -> str:
        lines = [f"Weather for {city}, aligned to trip dates:"]
        for item in weather_info:
            if item.day_weather == "Unknown" and item.night_weather == "Unknown":
                lines.append(f"- {item.date}: forecast unavailable for this date.")
                continue
            lines.append(
                f"- {item.date}: daytime {item.day_weather}, nighttime {item.night_weather}, "
                f"{item.day_temp}°C / {item.night_temp}°C, {item.wind_direction} {item.wind_power}".strip()
            )
        return "\n".join(lines)


_weather_service: Optional[WeatherService] = None


def get_weather_service() -> WeatherService:
    global _weather_service
    if _weather_service is None:
        _weather_service = WeatherService()
    return _weather_service
