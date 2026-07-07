"""Single-page web fetching and extraction for RAG ingestion."""

from __future__ import annotations

import ipaddress
import logging
import re
import socket
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

MAX_WEB_HTML_BYTES = 5 * 1024 * 1024
FETCH_TIMEOUT_SECONDS = 20
MIN_MEANINGFUL_TEXT_CHARS = 120

BLOCKED_HOSTNAMES = {
    "localhost",
    "localhost.localdomain",
    "metadata.google.internal",
}

ACCESS_DENIED_PATTERNS = [
    re.compile(r"access\s+denied", re.I),
    re.compile(r"forbidden", re.I),
    re.compile(r"captcha", re.I),
    re.compile(r"cloudflare", re.I),
    re.compile(r"cf-challenge", re.I),
    re.compile(r"enable\s+javascript", re.I),
    re.compile(r"javascript\s+is\s+required", re.I),
]

LOW_VALUE_TAG_PATTERNS = [
    r"script",
    r"style",
    r"noscript",
    r"svg",
    r"picture",
    r"figure",
    r"video",
    r"iframe",
    r"nav",
    r"footer",
    r"header",
    r"form",
    r"aside",
]

LOW_VALUE_VOID_TAG_PATTERNS = [
    r"img",
    r"source",
    r"button",
]

LOW_VALUE_ATTR_PATTERN = re.compile(
    r"(cookie|newsletter|subscribe|social|share|advert|advertisement|"
    r"\bad\b|promo|promotion|sponsor|breadcrumb|related|recommend|modal|popup)",
    re.I,
)


class URLSafetyError(ValueError):
    """Raised when a submitted URL is unsafe for server-side fetching."""


class WebFetchError(RuntimeError):
    """Raised when a page cannot be fetched or extracted."""


class WebFetchResult(BaseModel):
    requested_url: str
    final_url: str
    page_title: Optional[str] = None
    raw_html: str
    extracted_markdown: str
    fetch_mode: str
    fetched_at: datetime
    warnings: list[str] = Field(default_factory=list)
    dynamic_fallback_used: bool = False
    fetch_duration_ms: float = 0.0


@dataclass
class WebFetchService:
    """Fetch one submitted URL and normalize it into Markdown for RAG review."""

    timeout_seconds: int = FETCH_TIMEOUT_SECONDS
    max_html_bytes: int = MAX_WEB_HTML_BYTES
    min_meaningful_text_chars: int = MIN_MEANINGFUL_TEXT_CHARS
    resolver: Callable[..., Iterable[Any]] = socket.getaddrinfo
    static_fetcher: Any = None
    dynamic_fetcher: Any = None
    _warnings: list[str] = field(default_factory=list)

    def fetch(self, url: str, css_selector: str = "") -> WebFetchResult:
        self._warnings = []
        started = time.perf_counter()
        validate_url_safety(url, resolver=self.resolver)

        page = self._fetch_static(url)
        final_url = str(getattr(page, "url", url) or url)
        self._validate_response_urls([url, *self._history_urls(page), final_url])
        unusable_reason = self._unusable_reason(page, css_selector)
        fetch_mode = "static"
        dynamic_used = False

        if unusable_reason:
            self._warnings.append(f"Static fetch unusable: {unusable_reason}")
            page = self._fetch_dynamic(final_url)
            final_url = str(getattr(page, "url", final_url) or final_url)
            self._validate_response_urls([final_url, *self._history_urls(page)])
            dynamic_reason = self._unusable_reason(page, css_selector)
            if dynamic_reason:
                raise WebFetchError(f"Fetched page is not usable: {dynamic_reason}")
            fetch_mode = "dynamic"
            dynamic_used = True

        raw_html = self._raw_html(page)
        self._check_size(raw_html)
        selected_html, selected_text = self._select_content(page, css_selector)
        cleaned_html = clean_web_html(selected_html)
        markdown = html_to_markdown(cleaned_html)

        if self._meaningful_text_len(markdown) < self.min_meaningful_text_chars:
            fallback_html = clean_web_html(raw_html)
            fallback_markdown = html_to_markdown(fallback_html)
            if self._meaningful_text_len(fallback_markdown) > self._meaningful_text_len(markdown):
                self._warnings.append("Used full-page fallback because selected extraction was too small.")
                markdown = fallback_markdown

        if self._meaningful_text_len(markdown) < self.min_meaningful_text_chars:
            raise WebFetchError("Extracted Markdown is too small to create a useful draft.")

        elapsed_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "RAG URL ingestion fetched requested_url=%s final_url=%s mode=%s duration_ms=%.1f "
            "raw_html_size=%s markdown_size=%s dynamic_fallback=%s warnings=%s",
            url,
            final_url,
            fetch_mode,
            elapsed_ms,
            len(raw_html.encode("utf-8")),
            len(markdown.encode("utf-8")),
            dynamic_used,
            len(self._warnings),
        )

        return WebFetchResult(
            requested_url=url,
            final_url=final_url,
            page_title=self._page_title(page),
            raw_html=raw_html,
            extracted_markdown=markdown,
            fetch_mode=fetch_mode,
            fetched_at=datetime.now(timezone.utc).replace(microsecond=0),
            warnings=list(self._warnings),
            dynamic_fallback_used=dynamic_used,
            fetch_duration_ms=round(elapsed_ms, 3),
        )

    def _fetch_static(self, url: str):
        fetcher = self.static_fetcher
        if fetcher is None:
            from scrapling.fetchers import Fetcher

            fetcher = Fetcher
        try:
            return fetcher.get(url, timeout=self.timeout_seconds, follow_redirects="safe")
        except TypeError:
            return fetcher.get(url, timeout=self.timeout_seconds)
        except Exception as exc:
            raise WebFetchError(f"Static fetch failed: {exc}") from exc

    def _fetch_dynamic(self, url: str):
        fetcher = self.dynamic_fetcher
        if fetcher is None:
            from scrapling.fetchers import DynamicFetcher

            fetcher = DynamicFetcher
        try:
            return fetcher.fetch(
                url,
                timeout=self.timeout_seconds,
                network_idle=True,
                headless=True,
            )
        except Exception as exc:
            raise WebFetchError(f"Dynamic fetch failed: {exc}") from exc

    def _validate_response_urls(self, urls: Iterable[str]) -> None:
        for item in urls:
            if item:
                validate_url_safety(str(item), resolver=self.resolver)

    def _history_urls(self, page: Any) -> list[str]:
        urls = []
        for item in getattr(page, "history", []) or []:
            url = getattr(item, "url", None)
            if url:
                urls.append(str(url))
        return urls

    def _unusable_reason(self, page: Any, css_selector: str) -> str:
        status = getattr(page, "status", 200) or 200
        if not 200 <= int(status) < 300:
            return f"HTTP status {status}"

        raw_html = self._raw_html(page)
        self._check_size(raw_html)
        visible_text = self._visible_text(page)
        if css_selector and not self._select_matches(page, css_selector):
            return f"CSS selector did not match: {css_selector}"
        if len(raw_html.strip()) < 100:
            return "empty body"
        if self._meaningful_text_len(visible_text) < self.min_meaningful_text_chars:
            return "too little visible text"
        if self._looks_like_app_shell(raw_html, visible_text):
            return "JavaScript app shell"
        if any(pattern.search(raw_html) or pattern.search(visible_text) for pattern in ACCESS_DENIED_PATTERNS):
            return "access denied or JavaScript-required page"
        return ""

    def _select_content(self, page: Any, css_selector: str) -> tuple[str, str]:
        if css_selector:
            matches = self._select_matches(page, css_selector)
            if not matches:
                raise WebFetchError(f"CSS selector did not match: {css_selector}")
            html = "\n".join(str(getattr(match, "html_content", "")) for match in matches)
            text = "\n".join(self._node_text(match) for match in matches)
            return html, text

        candidates = []
        selectors = [
            ("article", 3500),
            (".post-content", 3300),
            (".entry-content", 3300),
            (".article-content", 3200),
            (".single-card-content", 2500),
            ("main", 1600),
            ("#main", 1600),
            ('[role="main"]', 1400),
            ('[class*="post-content"]', 1200),
            ('[class*="entry-content"]', 1200),
            ('[class*="article-content"]', 1200),
        ]
        for selector, priority in selectors:
            for node in self._select_matches(page, selector):
                html = str(getattr(node, "html_content", ""))
                text = self._node_text(node)
                meaningful = self._meaningful_text_len(text)
                if meaningful < self.min_meaningful_text_chars:
                    continue
                density = meaningful / max(len(html), 1)
                score = min(meaningful, 9000) + int(density * 6000) + priority - int(len(html) * 0.003)
                candidates.append((score, selector, html, text))
        if not candidates:
            self._warnings.append("No main/article region found; used body fallback.")
            return self._raw_html(page), self._visible_text(page)

        candidates.sort(key=lambda item: item[0], reverse=True)
        _, selector, html, text = candidates[0]
        if selector in {"main", "#main", '[role="main"]'}:
            self._warnings.append(f"Used broad content selector {selector}; review extracted text for page chrome.")
        return html, text

    def _select_matches(self, page: Any, selector: str):
        try:
            return list(page.css(selector))
        except Exception as exc:
            raise WebFetchError(f"Invalid or unsupported CSS selector: {selector}") from exc

    def _raw_html(self, page: Any) -> str:
        html = getattr(page, "html_content", "") or ""
        if html:
            return str(html)
        body = getattr(page, "body", b"") or b""
        if isinstance(body, bytes):
            return body.decode(getattr(page, "encoding", "utf-8") or "utf-8", errors="replace")
        return str(body)

    def _visible_text(self, page: Any) -> str:
        try:
            return str(page.get_all_text())
        except Exception:
            return re.sub(r"<[^>]+>", " ", self._raw_html(page))

    def _node_text(self, node: Any) -> str:
        try:
            return str(node.get_all_text())
        except Exception:
            return re.sub(r"<[^>]+>", " ", str(getattr(node, "html_content", "")))

    def _page_title(self, page: Any) -> Optional[str]:
        try:
            titles = list(page.css("title"))
            if titles:
                title = titles[0].get_all_text().strip()
                return title or None
        except Exception:
            return None
        return None

    def _check_size(self, html: str) -> None:
        if len((html or "").encode("utf-8")) > self.max_html_bytes:
            raise WebFetchError("Fetched HTML exceeds maximum allowed size")

    def _meaningful_text_len(self, text: str) -> int:
        return len(re.sub(r"\s+", "", text or ""))

    def _looks_like_app_shell(self, html: str, text: str) -> bool:
        app_root = re.search(r"<div[^>]+id=[\"'](?:app|root|__next)[\"'][^>]*>\s*</div>", html, re.I)
        script_count = len(re.findall(r"<script\b", html, re.I))
        return bool(app_root and script_count > 0 and self._meaningful_text_len(text) < 300)


def validate_url_safety(url: str, *, resolver: Callable[..., Iterable[Any]] = socket.getaddrinfo) -> None:
    parsed = urlparse((url or "").strip())
    if parsed.scheme not in {"http", "https"}:
        raise URLSafetyError("Only http and https URLs are allowed")
    if not parsed.hostname:
        raise URLSafetyError("URL must include a hostname")

    host = parsed.hostname.strip().lower().rstrip(".")
    if host in BLOCKED_HOSTNAMES or host.endswith(".localhost"):
        raise URLSafetyError("Localhost URLs are not allowed")

    addresses = _resolve_host_addresses(host, resolver=resolver)
    if not addresses:
        raise URLSafetyError("URL hostname could not be resolved")
    for address in addresses:
        _reject_blocked_ip(address)


def _resolve_host_addresses(host: str, *, resolver: Callable[..., Iterable[Any]]) -> list[ipaddress._BaseAddress]:
    try:
        ip = ipaddress.ip_address(host)
        return [ip]
    except ValueError:
        pass

    addresses = []
    try:
        infos = resolver(host, None, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise URLSafetyError(f"URL hostname could not be resolved: {host}") from exc

    for info in infos:
        sockaddr = info[4]
        if not sockaddr:
            continue
        try:
            addresses.append(ipaddress.ip_address(sockaddr[0]))
        except ValueError:
            continue
    return addresses


def _reject_blocked_ip(address: ipaddress._BaseAddress) -> None:
    if (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
    ):
        raise URLSafetyError(f"Blocked URL address: {address}")
    if str(address) == "169.254.169.254":
        raise URLSafetyError("Cloud metadata endpoint is not allowed")


def clean_web_html(html: str) -> str:
    cleaned = html or ""
    for tag in LOW_VALUE_TAG_PATTERNS:
        cleaned = re.sub(
            rf"<{tag}\b[^>]*>.*?</{tag}>",
            "",
            cleaned,
            flags=re.I | re.S,
        )
    for tag in LOW_VALUE_VOID_TAG_PATTERNS:
        cleaned = re.sub(rf"<{tag}\b[^>]*?/?>", "", cleaned, flags=re.I | re.S)
    cleaned = re.sub(
        r"<([a-z0-9]+)\b([^>]*(?:class|id|aria-label|role)\s*=\s*[\"'][^\"']*"
        + LOW_VALUE_ATTR_PATTERN.pattern
        + r"[^\"']*[\"'][^>]*)>.*?</\1>",
        "",
        cleaned,
        flags=re.I | re.S,
    )
    cleaned = re.sub(r"<!--.*?-->", "", cleaned, flags=re.S)
    return cleaned.strip()


def html_to_markdown(html: str) -> str:
    try:
        from markitdown import MarkItDown
    except Exception as exc:
        raise WebFetchError("markitdown is required for HTML conversion") from exc

    temp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".html", encoding="utf-8", delete=False) as handle:
            document = html or ""
            if "<html" not in document.lower():
                document = f"<!doctype html><html><body>{document}</body></html>"
            handle.write(document)
            temp_path = Path(handle.name)
        result = MarkItDown().convert(str(temp_path))
        markdown = getattr(result, "text_content", "") or getattr(result, "markdown", "") or str(result)
        return normalize_markdown(markdown)
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink()


def normalize_markdown(value: str) -> str:
    text = (value or "").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


_web_fetch_service: Optional[WebFetchService] = None


def get_web_fetch_service() -> WebFetchService:
    global _web_fetch_service
    if _web_fetch_service is None:
        _web_fetch_service = WebFetchService()
    return _web_fetch_service
