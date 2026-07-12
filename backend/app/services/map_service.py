"""Provider-neutral map service backed by Google Maps Platform."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import httpx
from langchain_core.tools import StructuredTool

from ..config import get_settings
from ..models.schemas import Location, POIInfo, RouteInfo

GOOGLE_PLACES_BASE_URL = "https://places.googleapis.com/v1"
GOOGLE_GEOCODE_URL = "https://maps.googleapis.com/maps/api/geocode/json"
GOOGLE_ROUTES_URL = "https://routes.googleapis.com/directions/v2:computeRoutes"
MAP_REQUEST_TIMEOUT_SECONDS = 15.0
MAP_REQUEST_RETRIES = 3

_map_service = None


class BaseMapProvider(ABC):
    """Provider-neutral map interface used by routes and LangGraph nodes."""

    @abstractmethod
    def health_summary(self) -> Dict[str, Any]:
        """Return provider and supported tool metadata."""

    @abstractmethod
    def search_poi(
        self,
        keywords: str,
        city: str,
        citylimit: bool = True,
        country_code: Optional[str] = None,
    ) -> List[POIInfo]:
        """Search places and return normalized POI records."""

    @abstractmethod
    def search_poi_raw(
        self,
        keywords: str,
        city: str,
        citylimit: bool = True,
        page_size: int = 10,
        country_code: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search places and return provider-neutral raw dictionaries."""

    @abstractmethod
    def geocode(self, address: str, city: Optional[str] = None) -> Optional[Location]:
        """Resolve an address to coordinates."""

    @abstractmethod
    def plan_route(
        self,
        origin_address: str,
        destination_address: str,
        origin_city: Optional[str] = None,
        destination_city: Optional[str] = None,
        route_type: str = "walking",
    ) -> Dict[str, Any]:
        """Return a normalized route summary."""

    @abstractmethod
    def get_poi_detail(self, poi_id: str) -> Dict[str, Any]:
        """Return provider-specific detail for a place ID."""

    @abstractmethod
    def get_photo_media_uri(self, photo_name: str, max_width_px: int = 800) -> str:
        """Return a temporary renderable image URI for a provider photo resource."""

    @abstractmethod
    def get_langchain_tools(self) -> List[StructuredTool]:
        """Return LangChain tools exposed to graph nodes."""


class GoogleMapsService(BaseMapProvider):
    """Google Maps Platform implementation for US-compatible place and route data."""

    def __init__(self):
        settings = get_settings()
        if not settings.google_maps_api_key:
            raise ValueError("GOOGLE_MAPS_API_KEY is not configured")
        self.api_key = settings.google_maps_api_key

    def health_summary(self) -> Dict[str, Any]:
        return {
            "provider": "google_maps",
            "tools": ["search_poi", "get_poi_detail", "plan_route", "geocode"],
        }

    def search_poi(
        self,
        keywords: str,
        city: str,
        citylimit: bool = True,
        country_code: Optional[str] = None,
    ) -> List[POIInfo]:
        return [
            self._to_poi_info(item)
            for item in self.search_poi_raw(
                keywords,
                city,
                citylimit,
                country_code=country_code,
            )
        ]

    def search_poi_raw(
        self,
        keywords: str,
        city: str,
        citylimit: bool = True,
        page_size: int = 10,
        country_code: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        query = f"{keywords} in {city}" if city else keywords
        payload: Dict[str, Any] = {
            "textQuery": query,
            "pageSize": max(1, min(page_size, 20)),
            "languageCode": "en",
        }
        region_code = self._normalize_region_code(country_code)
        if citylimit and region_code:
            payload["regionCode"] = region_code
        data = self._post_json(
            f"{GOOGLE_PLACES_BASE_URL}/places:searchText",
            payload,
            headers={
                "X-Goog-Api-Key": self.api_key,
                "X-Goog-FieldMask": (
                    "places.id,places.displayName,places.formattedAddress,"
                    "places.location,places.types,places.rating,places.userRatingCount,"
                    "places.nationalPhoneNumber,places.priceLevel,"
                    "places.googleMapsUri,places.websiteUri,places.photos"
                ),
            },
        )
        return [self._normalize_place(item) for item in data.get("places", []) or []]

    def geocode(self, address: str, city: Optional[str] = None) -> Optional[Location]:
        query = f"{address}, {city}" if city and city.lower() not in address.lower() else address
        data = self._get_json(
            GOOGLE_GEOCODE_URL,
            {"address": query, "key": self.api_key},
            google_status_key=True,
        )
        results = data.get("results", []) or []
        if not results:
            return None
        location = results[0].get("geometry", {}).get("location", {}) or {}
        lat = location.get("lat")
        lng = location.get("lng")
        if lat is None or lng is None:
            return None
        return Location(longitude=float(lng), latitude=float(lat))

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

        mode = self._to_google_travel_mode(route_type)
        data = self._post_json(
            GOOGLE_ROUTES_URL,
            {
                "origin": {
                    "location": {
                        "latLng": {
                            "latitude": origin.latitude,
                            "longitude": origin.longitude,
                        }
                    }
                },
                "destination": {
                    "location": {
                        "latLng": {
                            "latitude": destination.latitude,
                            "longitude": destination.longitude,
                        }
                    }
                },
                "travelMode": mode,
                "languageCode": "en-US",
                "units": "IMPERIAL",
            },
            headers={
                "X-Goog-Api-Key": self.api_key,
                "X-Goog-FieldMask": "routes.duration,routes.distanceMeters",
            },
        )
        routes = data.get("routes", []) or []
        if not routes:
            return {}
        route = routes[0]
        duration_seconds = self._parse_duration_seconds(route.get("duration"))
        return RouteInfo(
            distance=float(route.get("distanceMeters", 0) or 0),
            duration=duration_seconds,
            route_type=route_type,
            description=route.get("description") or f"{route_type} route",
        ).model_dump()

    def get_poi_detail(self, poi_id: str) -> Dict[str, Any]:
        data = self._get_json(
            f"{GOOGLE_PLACES_BASE_URL}/places/{poi_id}",
            {},
            headers={
                "X-Goog-Api-Key": self.api_key,
                "X-Goog-FieldMask": (
                    "id,displayName,formattedAddress,location,types,"
                    "rating,userRatingCount,nationalPhoneNumber,websiteUri,googleMapsUri,priceLevel,photos"
                ),
            },
        )
        result = self._to_poi_info(self._normalize_place(data)).model_dump()
        result["raw"] = data
        return result

    def get_photo_media_uri(self, photo_name: str, max_width_px: int = 800) -> str:
        normalized_width = max(1, min(int(max_width_px or 800), 1600))
        media_name = photo_name if photo_name.endswith("/media") else f"{photo_name}/media"
        data = self._get_json(
            f"{GOOGLE_PLACES_BASE_URL}/{media_name}",
            {
                "key": self.api_key,
                "maxWidthPx": normalized_width,
                "skipHttpRedirect": "true",
            },
        )
        photo_uri = data.get("photoUri")
        if not photo_uri:
            raise ValueError("Photo media URI was not returned")
        return str(photo_uri)

    def get_langchain_tools(self) -> List[StructuredTool]:
        return [
            StructuredTool.from_function(
                name="map_search_poi",
                description="Search places in a city using the configured map provider.",
                func=self._tool_search_poi,
            )
        ]

    def _tool_search_poi(
        self,
        keywords: str,
        city: str,
        citylimit: bool = True,
        page_size: int = 10,
        country_code: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        return self.search_poi_raw(
            keywords,
            city,
            citylimit=citylimit,
            page_size=page_size,
            country_code=country_code,
        )

    def _normalize_region_code(self, country_code: Optional[str]) -> str:
        normalized = (country_code or "US").strip().upper()
        if len(normalized) == 2 and normalized.isalpha():
            return normalized
        return ""

    def _normalize_place(self, item: Dict[str, Any]) -> Dict[str, Any]:
        display = item.get("displayName") or {}
        location = item.get("location") or {}
        types = item.get("types") or []
        photos = item.get("photos") or []
        photo_names = [
            str(photo.get("name"))
            for photo in photos
            if isinstance(photo, dict) and photo.get("name")
        ]
        first_photo_name = photo_names[0] if photo_names else ""
        return {
            "id": str(item.get("id", "")),
            "name": str(display.get("text") or item.get("name") or ""),
            "type": ", ".join(types[:3]) if isinstance(types, list) else str(types),
            "address": str(item.get("formattedAddress") or ""),
            "location": {
                "longitude": float(location.get("longitude", 0) or 0),
                "latitude": float(location.get("latitude", 0) or 0),
            },
            "tel": item.get("nationalPhoneNumber") or None,
            "rating": item.get("rating"),
            "user_rating_count": item.get("userRatingCount"),
            "maps_url": item.get("googleMapsUri") or None,
            "website_url": item.get("websiteUri") or None,
            "photo_names": photo_names,
            "image_url": self._photo_proxy_url(first_photo_name) if first_photo_name else None,
            "raw": item,
        }

    def _to_poi_info(self, item: Dict[str, Any]) -> POIInfo:
        location = item.get("location") or {}
        return POIInfo(
            id=str(item.get("id", "")),
            name=str(item.get("name", "")),
            type=str(item.get("type", "")),
            address=str(item.get("address", "")),
            location=Location(
                longitude=float(location.get("longitude", 0) or 0),
                latitude=float(location.get("latitude", 0) or 0),
            ),
            tel=item.get("tel") or None,
            rating=item.get("rating") if item.get("rating") is not None else None,
            image_url=item.get("image_url") or None,
            maps_url=item.get("maps_url") or None,
            website_url=item.get("website_url") or None,
        )

    def _photo_proxy_url(self, photo_name: str) -> str:
        from urllib.parse import quote

        return f"/api/map/photo?photo_name={quote(photo_name, safe='')}"

    def _to_google_travel_mode(self, route_type: str) -> str:
        normalized = (route_type or "").lower()
        if normalized in {"driving", "drive", "car", "taxi"}:
            return "DRIVE"
        if normalized in {"transit", "public transit", "subway", "bus"}:
            return "TRANSIT"
        if normalized in {"bicycling", "cycling", "bike"}:
            return "BICYCLE"
        return "WALK"

    def _parse_duration_seconds(self, value: Any) -> int:
        if isinstance(value, str) and value.endswith("s"):
            try:
                return int(float(value[:-1]))
            except ValueError:
                return 0
        if isinstance(value, (int, float)):
            return int(value)
        return 0

    def _get_json(
        self,
        url: str,
        params: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None,
        google_status_key: bool = False,
    ) -> Dict[str, Any]:
        return self._request_json("GET", url, params=params, headers=headers, google_status_key=google_status_key)

    def _post_json(
        self,
        url: str,
        payload: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        return self._request_json("POST", url, json_payload=payload, headers=headers)

    def _request_json(
        self,
        method: str,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        json_payload: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        google_status_key: bool = False,
    ) -> Dict[str, Any]:
        last_error: Optional[Exception] = None
        for attempt in range(MAP_REQUEST_RETRIES):
            try:
                with httpx.Client(timeout=MAP_REQUEST_TIMEOUT_SECONDS) as client:
                    if method == "GET":
                        response = client.get(url, params=params or {}, headers=headers)
                    else:
                        response = client.post(url, json=json_payload or {}, headers=headers)
                    response.raise_for_status()
                    data = response.json()
                break
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc
                if attempt >= MAP_REQUEST_RETRIES - 1:
                    raise
                time.sleep(0.5 * (attempt + 1))
        else:
            raise last_error or RuntimeError("Map provider request failed")

        if google_status_key:
            status = str(data.get("status", "OK"))
            if status not in {"OK", "ZERO_RESULTS"}:
                raise ValueError(data.get("error_message") or status)
        if isinstance(data.get("error"), dict):
            error = data["error"]
            raise ValueError(error.get("message") or "Map provider request failed")
        return data


def get_map_service() -> BaseMapProvider:
    """Return the configured map provider service."""
    global _map_service
    if _map_service is None:
        settings = get_settings()
        provider = (settings.map_provider or "google").lower()
        if provider != "google":
            raise ValueError(f"Unsupported map provider: {settings.map_provider}")
        _map_service = GoogleMapsService()
    return _map_service
