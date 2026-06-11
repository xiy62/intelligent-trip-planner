"""Unsplash image service."""

import requests
from typing import List, Optional
from ..config import get_settings

class UnsplashService:
    """Thin client for Unsplash image search."""

    def __init__(self):
        """Initialize the service from application settings."""
        settings = get_settings()
        self.access_key = settings.unsplash_access_key
        self.base_url = "https://api.unsplash.com"

    def search_photos(self, query: str, per_page: int = 5) -> List[dict]:
        """
        搜索图片

        Args:
            query: 搜索关键词
            per_page: 每页数量

        Returns:
            图片列表
        """
        try:
            url = f"{self.base_url}/search/photos"
            params = {
                "query": query,
                "per_page": per_page,
                "client_id": self.access_key
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            results = data.get("results", [])

            # Keep only the image metadata needed by the client.
            photos = []
            for photo in results:
                photos.append({
                    "id": photo.get("id"),
                    "url": photo.get("urls", {}).get("regular"),
                    "thumb": photo.get("urls", {}).get("thumb"),
                    "description": photo.get("description") or photo.get("alt_description"),
                    "photographer": photo.get("user", {}).get("name")
                })

            return photos

        except Exception as e:
            print(f"❌ Unsplash搜索失败: {str(e)}")
            return []

    def get_photo_url(self, query: str) -> Optional[str]:
        """
        获取单张图片URL

        Args:
            query: 搜索关键词

        Returns:
            图片URL
        """
        photos = self.search_photos(query, per_page=1)
        if photos:
            return photos[0].get("url")
        return None


# Shared service instance
_unsplash_service = None


def get_unsplash_service() -> UnsplashService:
    """Return the shared Unsplash service."""
    global _unsplash_service

    if _unsplash_service is None:
        _unsplash_service = UnsplashService()

    return _unsplash_service
