"""Tests for AMap service resilience behavior."""

from __future__ import annotations

import unittest

import httpx

from app.services import amap_service
from app.services.amap_service import AmapService


class FakeResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {"status": "1", "pois": []}


class FakeClient:
    attempts = 0

    def __init__(self, timeout):
        self.timeout = timeout

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def get(self, url, params):
        FakeClient.attempts += 1
        if FakeClient.attempts == 1:
            raise httpx.ConnectTimeout("timeout")
        return FakeResponse()


class AmapServiceTests(unittest.TestCase):
    def test_request_retries_transient_timeout(self):
        service = AmapService.__new__(AmapService)
        service.api_key = "test-key"
        original_client = amap_service.httpx.Client
        original_sleep = amap_service.time.sleep
        FakeClient.attempts = 0
        amap_service.httpx.Client = FakeClient
        amap_service.time.sleep = lambda _: None
        try:
            data = service._request("/place/text", {"keywords": "西湖", "city": "杭州"})
        finally:
            amap_service.httpx.Client = original_client
            amap_service.time.sleep = original_sleep

        self.assertEqual(data["status"], "1")
        self.assertEqual(FakeClient.attempts, 2)


if __name__ == "__main__":
    unittest.main()
