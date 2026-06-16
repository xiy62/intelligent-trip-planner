"""Tests for Google-backed map service behavior."""

from __future__ import annotations

import unittest

import httpx

from app.services import map_service
from app.services.map_service import GoogleMapsService


class FakePlacesResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {
            "places": [
                {
                    "id": "places/test-place",
                    "displayName": {"text": "Central Park"},
                    "formattedAddress": "New York, NY",
                    "location": {"latitude": 40.785091, "longitude": -73.968285},
                    "types": ["park", "tourist_attraction"],
                    "rating": 4.8,
                    "nationalPhoneNumber": "123",
                    "googleMapsUri": "https://maps.google.com/?cid=central",
                    "websiteUri": "https://www.centralparknyc.org/",
                    "photos": [{"name": "places/test-place/photos/photo-1"}],
                }
            ]
        }


class FakeRouteResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {"routes": [{"distanceMeters": 1200, "duration": "900s", "description": "Walk"}]}


class FakeGeocodeResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {
            "status": "OK",
            "results": [
                {"geometry": {"location": {"lat": 40.758, "lng": -73.9855}}}
            ],
        }


class FakePhotoResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {"photoUri": "https://lh3.googleusercontent.com/test-photo"}


class FakeClient:
    attempts = 0

    def __init__(self, timeout):
        self.timeout = timeout

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def post(self, url, json=None, headers=None):
        FakeClient.attempts += 1
        if FakeClient.attempts == 1:
            raise httpx.ConnectTimeout("timeout")
        if "computeRoutes" in url:
            return FakeRouteResponse()
        return FakePlacesResponse()

    def get(self, url, params=None, headers=None):
        if "places/test-place/photos/photo-1/media" in url:
            return FakePhotoResponse()
        return FakeGeocodeResponse()


class AlwaysTimeoutClient:
    attempts = 0

    def __init__(self, timeout):
        self.timeout = timeout

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def post(self, url, json=None, headers=None):
        AlwaysTimeoutClient.attempts += 1
        raise httpx.ConnectTimeout("persistent timeout")

    def get(self, url, params=None, headers=None):
        AlwaysTimeoutClient.attempts += 1
        raise httpx.ConnectTimeout("persistent timeout")


class ErrorResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {"error": {"message": "API key not valid"}}


class ErrorClient:
    def __init__(self, timeout):
        self.timeout = timeout

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def post(self, url, json=None, headers=None):
        return ErrorResponse()

    def get(self, url, params=None, headers=None):
        return ErrorResponse()


class GoogleMapsServiceTests(unittest.TestCase):
    def test_search_poi_retries_transient_timeout_and_normalizes_places(self):
        service = GoogleMapsService.__new__(GoogleMapsService)
        service.api_key = "test-key"
        original_client = map_service.httpx.Client
        original_sleep = map_service.time.sleep
        FakeClient.attempts = 0
        map_service.httpx.Client = FakeClient
        map_service.time.sleep = lambda _: None
        try:
            items = service.search_poi_raw("park", "New York")
        finally:
            map_service.httpx.Client = original_client
            map_service.time.sleep = original_sleep

        self.assertEqual(FakeClient.attempts, 2)
        self.assertEqual(items[0]["id"], "places/test-place")
        self.assertEqual(items[0]["name"], "Central Park")
        self.assertEqual(items[0]["location"]["longitude"], -73.968285)
        self.assertEqual(items[0]["maps_url"], "https://maps.google.com/?cid=central")
        self.assertEqual(items[0]["website_url"], "https://www.centralparknyc.org/")
        self.assertEqual(items[0]["photo_names"], ["places/test-place/photos/photo-1"])
        self.assertIn("/api/map/photo?photo_name=", items[0]["image_url"])

    def test_request_surfaces_persistent_timeout_after_retry_budget(self):
        service = GoogleMapsService.__new__(GoogleMapsService)
        service.api_key = "test-key"
        original_client = map_service.httpx.Client
        original_sleep = map_service.time.sleep
        AlwaysTimeoutClient.attempts = 0
        map_service.httpx.Client = AlwaysTimeoutClient
        map_service.time.sleep = lambda _: None
        try:
            with self.assertRaises(httpx.ConnectTimeout):
                service.search_poi_raw("park", "New York")
        finally:
            map_service.httpx.Client = original_client
            map_service.time.sleep = original_sleep

        self.assertEqual(AlwaysTimeoutClient.attempts, map_service.MAP_REQUEST_RETRIES)

    def test_request_surfaces_provider_error_response(self):
        service = GoogleMapsService.__new__(GoogleMapsService)
        service.api_key = "test-key"
        original_client = map_service.httpx.Client
        map_service.httpx.Client = ErrorClient
        try:
            with self.assertRaisesRegex(ValueError, "API key not valid"):
                service.search_poi_raw("park", "New York")
        finally:
            map_service.httpx.Client = original_client

    def test_geocode_and_route_are_normalized(self):
        service = GoogleMapsService.__new__(GoogleMapsService)
        service.api_key = "test-key"
        original_client = map_service.httpx.Client
        map_service.httpx.Client = FakeClient
        try:
            location = service.geocode("Times Square", "New York")
            route = service.plan_route("Times Square", "Central Park", "New York", "New York")
        finally:
            map_service.httpx.Client = original_client

        self.assertIsNotNone(location)
        self.assertEqual(route["distance"], 1200.0)
        self.assertEqual(route["duration"], 900)

    def test_health_summary_reports_google_provider(self):
        service = GoogleMapsService.__new__(GoogleMapsService)
        service.api_key = "test-key"

        summary = service.health_summary()

        self.assertEqual(summary["provider"], "google_maps")
        self.assertIn("search_poi", summary["tools"])

    def test_photo_media_uri_uses_google_photo_endpoint(self):
        service = GoogleMapsService.__new__(GoogleMapsService)
        service.api_key = "test-key"
        original_client = map_service.httpx.Client
        map_service.httpx.Client = FakeClient
        try:
            photo_uri = service.get_photo_media_uri("places/test-place/photos/photo-1")
        finally:
            map_service.httpx.Client = original_client

        self.assertEqual(photo_uri, "https://lh3.googleusercontent.com/test-photo")


if __name__ == "__main__":
    unittest.main()
