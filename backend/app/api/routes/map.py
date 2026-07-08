"""Map service API routes."""

from datetime import date

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import RedirectResponse

from ...models.schemas import POISearchResponse, RouteRequest, RouteResponse, WeatherResponse
from ...services.map_service import get_map_service
from ...services.weather_service import get_weather_service

router = APIRouter(prefix="/map", tags=["Map Services"])


@router.get("/poi", response_model=POISearchResponse, summary="Search POIs")
async def search_poi(
    keywords: str = Query(..., description="Search keywords", examples=["museum"]),
    city: str = Query(..., description="City", examples=["New York"]),
    citylimit: bool = Query(True, description="Restrict results to the requested city"),
    country_code: str = Query("US", min_length=2, max_length=2, description="2-letter region bias code"),
):
    """Search city POIs through the configured map provider."""
    try:
        pois = get_map_service().search_poi(keywords, city, citylimit, country_code=country_code)
        return POISearchResponse(success=True, message="POI search succeeded", data=pois)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"POI search failed: {exc}") from exc


@router.get("/weather", response_model=WeatherResponse, summary="Get city weather")
async def get_weather(city: str = Query(..., description="City name", examples=["New York"])):
    """Return weather for a city."""
    try:
        weather_info = get_weather_service().get_weather_for_trip(
            city=city,
            start_date=date.today().strftime("%Y-%m-%d"),
            travel_days=1,
        )
        return WeatherResponse(success=True, message="Weather lookup succeeded", data=weather_info)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Weather lookup failed: {exc}") from exc


@router.post("/route", response_model=RouteResponse, summary="Plan a route")
async def plan_route(request: RouteRequest):
    """Plan a route between two addresses."""
    try:
        route_info = get_map_service().plan_route(
            origin_address=request.origin_address,
            destination_address=request.destination_address,
            origin_city=request.origin_city,
            destination_city=request.destination_city,
            route_type=request.route_type,
        )
        return RouteResponse(success=True, message="Route planning succeeded", data=route_info)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Route planning failed: {exc}") from exc


@router.get("/photo", summary="Resolve a Google Places photo")
async def get_place_photo(
    photo_name: str = Query(..., description="Google Places photo resource name"),
    max_width_px: int = Query(800, ge=1, le=1600, description="Maximum image width"),
):
    """Resolve a Places photo resource through the backend so API keys stay server-side."""
    try:
        photo_uri = get_map_service().get_photo_media_uri(photo_name, max_width_px=max_width_px)
        return RedirectResponse(photo_uri)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"Photo lookup failed: {exc}") from exc


@router.get("/health", summary="Check map service health")
async def health_check():
    """Return map provider and tool metadata."""
    try:
        summary = get_map_service().health_summary()
        return {
            "status": "healthy",
            "service": "map-service",
            "provider": summary["provider"],
            "tools": summary["tools"],
        }
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Map service unavailable: {exc}") from exc
