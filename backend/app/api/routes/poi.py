"""POI detail and image API routes."""

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ...services.map_service import get_map_service
from ...services.unsplash_service import get_unsplash_service

router = APIRouter(prefix="/poi", tags=["POI"])


class POIDetailResponse(BaseModel):
    """POI detail response."""

    success: bool
    message: str
    data: Optional[dict] = None


@router.get("/detail/{poi_id}", response_model=POIDetailResponse, summary="Get POI details")
async def get_poi_detail(poi_id: str):
    """Return map provider details for a POI ID."""
    try:
        result = get_map_service().get_poi_detail(poi_id)
        return POIDetailResponse(success=True, message="POI detail lookup succeeded", data=result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"POI detail lookup failed: {exc}") from exc


@router.get("/search", summary="Search POIs")
async def search_poi(keywords: str, city: str = "New York"):
    """Search POIs by keyword and city."""
    try:
        result = get_map_service().search_poi(keywords, city)
        return {"success": True, "message": "POI search succeeded", "data": result}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"POI search failed: {exc}") from exc


@router.get("/photo", summary="Get an attraction photo")
async def get_attraction_photo(name: str):
    """Return an Unsplash image URL for an attraction."""
    try:
        unsplash_service = get_unsplash_service()
        photo_url = unsplash_service.get_photo_url(f"{name} China landmark")
        if not photo_url:
            photo_url = unsplash_service.get_photo_url(name)
        return {
            "success": True,
            "message": "Photo lookup succeeded",
            "data": {"name": name, "photo_url": photo_url},
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Photo lookup failed: {exc}") from exc
