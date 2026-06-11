"""Map service API routes."""

from fastapi import APIRouter, HTTPException, Query

from ...models.schemas import POISearchResponse, RouteRequest, RouteResponse, WeatherResponse
from ...services.amap_service import get_amap_service

router = APIRouter(prefix="/map", tags=["Map Services"])


@router.get("/poi", response_model=POISearchResponse, summary="Search POIs")
async def search_poi(
    keywords: str = Query(..., description="Search keywords", examples=["故宫"]),
    city: str = Query(..., description="City", examples=["北京"]),
    citylimit: bool = Query(True, description="Restrict results to the requested city"),
):
    """Search city POIs through AMap."""
    try:
        pois = get_amap_service().search_poi(keywords, city, citylimit)
        return POISearchResponse(success=True, message="POI search succeeded", data=pois)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"POI search failed: {exc}") from exc


@router.get("/weather", response_model=WeatherResponse, summary="Get city weather")
async def get_weather(city: str = Query(..., description="City name", examples=["北京"])):
    """Return weather for a city."""
    try:
        weather_info = get_amap_service().get_weather(city)
        return WeatherResponse(success=True, message="Weather lookup succeeded", data=weather_info)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Weather lookup failed: {exc}") from exc


@router.post("/route", response_model=RouteResponse, summary="Plan a route")
async def plan_route(request: RouteRequest):
    """Plan a route between two addresses."""
    try:
        route_info = get_amap_service().plan_route(
            origin_address=request.origin_address,
            destination_address=request.destination_address,
            origin_city=request.origin_city,
            destination_city=request.destination_city,
            route_type=request.route_type,
        )
        return RouteResponse(success=True, message="Route planning succeeded", data=route_info)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Route planning failed: {exc}") from exc


@router.get("/health", summary="Check map service health")
async def health_check():
    """Return map provider and tool metadata."""
    try:
        summary = get_amap_service().health_summary()
        return {
            "status": "healthy",
            "service": "map-service",
            "provider": summary["provider"],
            "tools": summary["tools"],
        }
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Map service unavailable: {exc}") from exc
