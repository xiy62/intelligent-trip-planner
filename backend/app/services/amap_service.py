"""Native AMap service and LangChain tool wrappers."""

from __future__ import annotations

import time
from datetime import date
from typing import Any, Dict, List, Optional

import httpx
from langchain_core.tools import StructuredTool

from ..config import get_settings
from ..models.schemas import Location, POIInfo, RouteInfo, WeatherInfo
from .weather_service import get_weather_service

AMAP_BASE_URL = "https://restapi.amap.com/v3"
AMAP_REQUEST_TIMEOUT_SECONDS = 15.0
AMAP_REQUEST_RETRIES = 3

_amap_service = None


class AmapService:
    """Thin native wrapper around the AMap Web APIs used by this project."""

    def __init__(self):
        settings = get_settings()
        if not settings.amap_api_key:
            raise ValueError("高德地图API Key未配置,请在.env文件中设置AMAP_API_KEY")
        self.api_key = settings.amap_api_key
        self.weather_service = get_weather_service()

    def health_summary(self) -> Dict[str, Any]:
        return {
            "provider": "amap_http",
            "tools": ["search_poi", "get_poi_detail", "plan_route", "geocode"],
        }

    def search_poi(self, keywords: str, city: str, citylimit: bool = True) -> List[POIInfo]:
        pois = self.search_poi_raw(keywords, city, citylimit=citylimit)
        return [self._to_poi_info(item) for item in pois]

    def search_poi_raw(
        self,
        keywords: str,
        city: str,
        citylimit: bool = True,
        page_size: int = 10,
    ) -> List[Dict[str, Any]]:
        data = self._request(
            "/place/text",
            {
                "keywords": keywords,
                "city": city,
                "citylimit": "true" if citylimit else "false",
                "offset": page_size,
                "extensions": "all",
            },
        )
        return data.get("pois", []) or []

    def get_weather(self, city: str) -> List[WeatherInfo]:
        today = date.today().strftime("%Y-%m-%d")
        return self.weather_service.get_weather_for_trip(city=city, start_date=today, travel_days=1)

    def plan_route(
        self,
        origin_address: str,
        destination_address: str,
        origin_city: Optional[str] = None,
        destination_city: Optional[str] = None,
        route_type: str = "walking",
    ) -> Dict[str, Any]:
        origin = self.geocode(origin_address, origin_city)
        destination = self.geocode(destination_address, destination_city)
        if origin is None or destination is None:
            return {}

        endpoint_map = {
            "walking": "/direction/walking",
            "driving": "/direction/driving",
            "transit": "/direction/transit/integrated",
        }
        endpoint = endpoint_map.get(route_type, "/direction/walking")
        params = {
            "origin": f"{origin.longitude},{origin.latitude}",
            "destination": f"{destination.longitude},{destination.latitude}",
        }
        if route_type == "transit" and (origin_city or destination_city):
            params["city"] = origin_city or destination_city or ""
        data = self._request(endpoint, params)
        route = data.get("route", {}) or {}

        distance = 0.0
        duration = 0
        description = ""
        if route_type == "transit":
            transits = route.get("transits", []) or []
            if transits:
                best = transits[0]
                distance = float(best.get("distance", 0) or 0)
                duration = int(best.get("duration", 0) or 0)
                description = best.get("cost", "") or "公共交通路线"
        else:
            paths = route.get("paths", []) or []
            if paths:
                best = paths[0]
                distance = float(best.get("distance", 0) or 0)
                duration = int(best.get("duration", 0) or 0)
                description = best.get("strategy", "") or f"{route_type}路线"

        return RouteInfo(
            distance=distance,
            duration=duration,
            route_type=route_type,
            description=description,
        ).model_dump()

    def geocode(self, address: str, city: Optional[str] = None) -> Optional[Location]:
        params = {"address": address}
        if city:
            params["city"] = city
        data = self._request("/geocode/geo", params)
        geocodes = data.get("geocodes", []) or []
        if not geocodes:
            return None
        location = geocodes[0].get("location", "")
        if "," not in location:
            return None
        longitude, latitude = location.split(",", 1)
        return Location(longitude=float(longitude), latitude=float(latitude))

    def get_poi_detail(self, poi_id: str) -> Dict[str, Any]:
        data = self._request("/place/detail", {"id": poi_id, "extensions": "all"})
        pois = data.get("pois", []) or []
        if not pois:
            return {}
        poi = pois[0]
        result = self._to_poi_info(poi).model_dump()
        result["raw"] = poi
        return result

    def get_langchain_tools(self) -> List[StructuredTool]:
        return [
            StructuredTool.from_function(
                name="amap_search_poi",
                description="Search POIs in a city using AMap. Returns a list of POI dictionaries.",
                func=self._tool_search_poi,
            )
        ]

    def _tool_search_poi(
        self,
        keywords: str,
        city: str,
        citylimit: bool = True,
        page_size: int = 10,
    ) -> List[Dict[str, Any]]:
        return self.search_poi_raw(keywords, city, citylimit=citylimit, page_size=page_size)

    def _to_poi_info(self, item: Dict[str, Any]) -> POIInfo:
        location = item.get("location", "")
        longitude, latitude = 0.0, 0.0
        if "," in location:
            lng_text, lat_text = location.split(",", 1)
            longitude = float(lng_text or 0)
            latitude = float(lat_text or 0)
        return POIInfo(
            id=str(item.get("id", "")),
            name=str(item.get("name", "")),
            type=str(item.get("type", "")),
            address=str(item.get("address", "")),
            location=Location(longitude=longitude, latitude=latitude),
            tel=item.get("tel") or None,
        )

    def _request(self, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        request_params = {"key": self.api_key, **params}
        last_error: Optional[Exception] = None
        for attempt in range(AMAP_REQUEST_RETRIES):
            try:
                with httpx.Client(timeout=AMAP_REQUEST_TIMEOUT_SECONDS) as client:
                    response = client.get(f"{AMAP_BASE_URL}{path}", params=request_params)
                    response.raise_for_status()
                    data = response.json()
                break
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc
                if attempt >= AMAP_REQUEST_RETRIES - 1:
                    raise
                time.sleep(0.5 * (attempt + 1))
        else:
            raise last_error or RuntimeError("AMap request failed")
        status = str(data.get("status", "0"))
        if status != "1":
            raise ValueError(data.get("info") or "AMap request failed")
        return data


def get_amap_service() -> AmapService:
    global _amap_service
    if _amap_service is None:
        _amap_service = AmapService()
    return _amap_service
