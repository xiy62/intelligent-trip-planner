"""Security-focused tests for RAG URL ingestion fetching.

The tests use fake resolvers and fake fetchers only. They must not perform DNS,
network, or external service calls.
"""

from __future__ import annotations

import socket
import unittest

from app.services.web_fetch_service import URLSafetyError, WebFetchError, WebFetchService, validate_url_safety


def public_resolver(host, *args, **kwargs):
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))]


def unresolved_resolver(host, *args, **kwargs):
    raise socket.gaierror("name or service not known")


class FakeHistoryItem:
    def __init__(self, url: str):
        self.url = url


class FakeNode:
    def __init__(self, html: str, text: str):
        self.html_content = html
        self._text = text

    def get_all_text(self):
        return self._text


class FakePage:
    def __init__(
        self,
        html: str,
        *,
        url: str = "https://example.com/guide",
        text: str | None = None,
        status: int = 200,
        history=None,
    ):
        self.html_content = html
        self.body = html.encode("utf-8")
        self.url = url
        self.status = status
        self.history = history or []
        self.encoding = "utf-8"
        self._text = text if text is not None else _strip_tags(html)

    def css(self, selector: str):
        if selector == "title":
            return []
        if selector in {"main", "article", "#main", '[role="main"]'} and "<main" in self.html_content:
            return [FakeNode(self.html_content, self._text)]
        return []

    def get_all_text(self):
        return self._text


class FakeStaticFetcher:
    page = None
    calls = []

    @classmethod
    def get(cls, url, **kwargs):
        cls.calls.append((url, kwargs))
        return cls.page


class FakeDynamicFetcher:
    page = None
    calls = []

    @classmethod
    def fetch(cls, url, **kwargs):
        cls.calls.append((url, kwargs))
        return cls.page


def _strip_tags(html: str) -> str:
    import re

    return re.sub(r"<[^>]+>", " ", html)


def long_travel_text() -> str:
    return (
        "This official travel guide describes museum routes, public transit access, "
        "neighborhood planning, safety notes, opening-hour checks, and accessible "
        "itinerary sequencing for visitors who need reliable destination context."
    )


class ValidateUrlSafetySecurityTests(unittest.TestCase):
    def assert_url_rejected(self, url: str, expected_reason: str, resolver=public_resolver) -> None:
        with self.assertRaises(URLSafetyError) as context:
            validate_url_safety(url, resolver=resolver)
        self.assertIn(expected_reason, str(context.exception))

    def test_rejects_non_http_file_scheme(self):
        self.assert_url_rejected("file:///etc/passwd", "Only http and https URLs are allowed")

    def test_rejects_missing_hostname(self):
        self.assert_url_rejected("https:///missing-host", "URL must include a hostname")

    def test_rejects_localhost_hostname(self):
        self.assert_url_rejected("http://localhost/admin", "Localhost URLs are not allowed")

    def test_rejects_loopback_ipv4(self):
        self.assert_url_rejected("http://127.0.0.1/admin", "Blocked URL address: 127.0.0.1")

    def test_rejects_unspecified_ipv4(self):
        self.assert_url_rejected("http://0.0.0.0/admin", "Blocked URL address: 0.0.0.0")

    def test_rejects_cloud_metadata_link_local_ipv4(self):
        self.assert_url_rejected(
            "http://169.254.169.254/latest/meta-data",
            "Blocked URL address: 169.254.169.254",
        )

    def test_rejects_private_ipv4(self):
        self.assert_url_rejected("http://192.168.1.10/admin", "Blocked URL address: 192.168.1.10")

    def test_rejects_private_ipv6(self):
        self.assert_url_rejected("http://[fd00::1]/admin", "Blocked URL address: fd00::1")

    def test_rejects_unresolved_hostname_with_fake_resolver(self):
        self.assert_url_rejected(
            "https://unresolved.example/guide",
            "URL hostname could not be resolved: unresolved.example",
            resolver=unresolved_resolver,
        )

    def test_accepts_public_hostname_with_fake_resolver(self):
        validate_url_safety("https://example.com/guide", resolver=public_resolver)


class WebFetchServiceSecurityTests(unittest.TestCase):
    def setUp(self):
        FakeStaticFetcher.page = None
        FakeStaticFetcher.calls = []
        FakeDynamicFetcher.page = None
        FakeDynamicFetcher.calls = []

    def build_service(self, **overrides) -> WebFetchService:
        defaults = {
            "resolver": public_resolver,
            "static_fetcher": FakeStaticFetcher,
            "dynamic_fetcher": FakeDynamicFetcher,
            "min_meaningful_text_chars": 40,
        }
        defaults.update(overrides)
        return WebFetchService(**defaults)

    def test_redirect_history_to_private_ip_is_rejected(self):
        text = long_travel_text()
        FakeStaticFetcher.page = FakePage(
            f"<html><body><main><p>{text}</p></main></body></html>",
            history=[FakeHistoryItem("https://example.com/start"), FakeHistoryItem("http://10.0.0.5/internal")],
            text=text,
        )
        service = self.build_service()

        with self.assertRaisesRegex(URLSafetyError, "Blocked URL address: 10.0.0.5"):
            service.fetch("https://example.com/start")

    def test_final_redirect_to_private_ip_is_rejected(self):
        text = long_travel_text()
        FakeStaticFetcher.page = FakePage(
            f"<html><body><main><p>{text}</p></main></body></html>",
            url="http://172.16.0.10/internal",
            text=text,
        )
        service = self.build_service()

        with self.assertRaisesRegex(URLSafetyError, "Blocked URL address: 172.16.0.10"):
            service.fetch("https://example.com/start")

    def test_oversized_html_is_rejected_before_extraction(self):
        text = long_travel_text()
        FakeStaticFetcher.page = FakePage(
            f"<html><body><main><p>{text}</p></main></body></html>",
            text=text,
        )
        service = self.build_service(max_html_bytes=20)

        with self.assertRaisesRegex(WebFetchError, "Fetched HTML exceeds maximum allowed size"):
            service.fetch("https://example.com/large")

    def test_too_little_meaningful_text_is_rejected_after_static_and_dynamic_fetch(self):
        filler = "<!-- padding keeps this out of the empty-body branch -->" * 3
        FakeStaticFetcher.page = FakePage(
            f"<html><body><main><p>tiny</p></main>{filler}</body></html>",
            text="tiny",
        )
        FakeDynamicFetcher.page = FakePage(
            f"<html><body><main><p>still tiny</p></main>{filler}</body></html>",
            text="still tiny",
        )
        service = self.build_service(min_meaningful_text_chars=40)

        with self.assertRaisesRegex(WebFetchError, "Fetched page is not usable: too little visible text"):
            service.fetch("https://example.com/tiny")

        self.assertEqual(len(FakeStaticFetcher.calls), 1)
        self.assertEqual(len(FakeDynamicFetcher.calls), 1)

    def test_app_shell_detection_triggers_dynamic_fallback(self):
        dynamic_text = long_travel_text()
        shell_text = "Loading travel guide shell with route planning placeholder"
        FakeStaticFetcher.page = FakePage(
            '<html><body><p>Loading travel guide shell with route planning placeholder</p>'
            '<div id="app"></div><script src="/app.js"></script><script src="/chunk.js"></script></body></html>',
            text=shell_text,
        )
        FakeDynamicFetcher.page = FakePage(
            f"<html><body><main><p>{dynamic_text}</p></main></body></html>",
            text=dynamic_text,
        )
        service = self.build_service(min_meaningful_text_chars=40)

        result = service.fetch("https://example.com/app-shell")

        self.assertEqual(result.fetch_mode, "dynamic")
        self.assertTrue(result.dynamic_fallback_used)
        self.assertEqual(len(FakeDynamicFetcher.calls), 1)
        self.assertIn("Static fetch unusable: JavaScript app shell", result.warnings)
        self.assertIn("official travel guide", result.extracted_markdown)

    def test_app_shell_remains_rejected_when_dynamic_fetch_is_also_an_app_shell(self):
        shell = (
            "<html><body><p>Loading travel guide shell with route planning placeholder</p>"
            '<div id="root"></div><script src="/bundle.js"></script><script src="/chunk.js"></script></body></html>'
        )
        shell_text = "Loading travel guide shell with route planning placeholder"
        FakeStaticFetcher.page = FakePage(shell, text=shell_text)
        FakeDynamicFetcher.page = FakePage(shell, text=shell_text)
        service = self.build_service(min_meaningful_text_chars=40)

        with self.assertRaisesRegex(WebFetchError, "Fetched page is not usable: JavaScript app shell"):
            service.fetch("https://example.com/app-shell")


if __name__ == "__main__":
    unittest.main()
