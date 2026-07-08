"""Tests for human-in-the-loop RAG ingestion utilities."""

from __future__ import annotations

import json
import socket
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from app.api.main import app
from app.api.routes import rag_ingestion as rag_ingestion_routes
from app.services.rag_ingestion import (
    RAGPrefillSuggestion,
    RAGPrefillSuggestionItem,
    RAGIngestionJobStore,
    RAGValidatedSuggestion,
    SourceManifestEntry,
    apply_prefill_suggestions,
    approved_drafts,
    build_draft_document,
    build_ai_prefill_source_packet,
    extract_readable_text,
    load_manifest,
    merge_knowledge_docs,
    quote_in_section,
    read_draft,
    split_markdown_sections,
    validate_prefill_suggestions,
    write_raw_source_text,
    write_uploaded_source,
    write_draft,
)
from app.services.rag_service import KnowledgeDocument, TravelRAGService
from app.services import web_fetch_service
from app.services.web_fetch_service import URLSafetyError, WebFetchResult, WebFetchService, validate_url_safety


def public_resolver(host, *args, **kwargs):
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))]


def private_resolver(host, *args, **kwargs):
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.1", 443))]


class FakePage:
    def __init__(self, html: str, url: str = "https://example.com/page", status: int = 200, history=None):
        from scrapling.parser import Adaptor

        self._adaptor = Adaptor(content=html, url=url)
        self.html_content = html
        self.body = html.encode("utf-8")
        self.url = url
        self.status = status
        self.history = history or []
        self.encoding = "utf-8"

    def css(self, selector: str):
        return self._adaptor.css(selector)

    def get_all_text(self):
        return self._adaptor.get_all_text()


class FakeStaticFetcher:
    page = None
    error = None

    @classmethod
    def get(cls, url, **kwargs):
        if cls.error:
            raise cls.error
        return cls.page


class FakeDynamicFetcher:
    page = None
    called = False

    @classmethod
    def fetch(cls, url, **kwargs):
        cls.called = True
        return cls.page


class RAGIngestionTests(unittest.TestCase):
    def test_manifest_parsing_and_required_fields(self):
        payload = [
            {
                "source_id": "nyc-source",
                "country": "US",
                "city": "New York",
                "source_url": "https://www.nyctourism.com/",
                "source_type": "official_tourism_portal",
                "title": "New York Source",
                "theme": ["museums"],
                "poi_names": ["The Metropolitan Museum of Art"],
                "district": "Manhattan",
                "language": "en",
            }
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = Path(tmpdir) / "manifest.json"
            manifest.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            entries = load_manifest(manifest)

        self.assertEqual(entries[0].source_id, "nyc-source")
        self.assertEqual(entries[0].city, "New York")

    def test_extract_readable_text_ignores_scripts_and_navigation(self):
        html = """
        <html>
          <body>
            <nav>Menu should disappear</nav>
              <script>console.log("hidden")</script>
            <main>
              <h1>Museum visit guide</h1>
              <p>The Met fits a museum-focused itinerary.</p>
              <p>Reserve enough time for nearby galleries.</p>
            </main>
          </body>
        </html>
        """
        text = extract_readable_text(html)

        self.assertIn("Museum visit guide", text)
        self.assertIn("Reserve enough time", text)
        self.assertNotIn("console.log", text)
        self.assertNotIn("Menu should disappear", text)

    def test_draft_generation_roundtrip(self):
        entry = SourceManifestEntry(
            source_id="nyc-source",
            country="US",
            city="New York",
            source_url="https://www.nyctourism.com/",
            source_type="official_tourism_portal",
            title="New York Source",
            theme=["museums"],
            poi_names=["The Metropolitan Museum of Art"],
            district="Manhattan",
            language="en",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            draft = build_draft_document(
                entry=entry,
                extracted_text="The Met fits a museum-focused itinerary. Reserve enough time.",
                raw_html_path=root / "source.html",
                raw_text_path=root / "source.txt",
                fetched_at="2026-05-06T00:00:00+00:00",
            )
            draft_path = root / "draft.json"
            write_draft(draft_path, draft)
            loaded = read_draft(draft_path)

        self.assertEqual(loaded.review_status, "draft")
        self.assertEqual(loaded.doc_id, "new_york-nyc-source")
        self.assertEqual(loaded.last_verified_at, "2026-05-06")
        self.assertIn("The Met fits a museum-focused itinerary", loaded.content)

    def test_unapproved_draft_is_not_promoted(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            draft = build_draft_document(
                entry=SourceManifestEntry(
                    source_id="nyc-source",
                    country="US",
                    city="New York",
                    source_url="https://www.nyctourism.com/",
                    source_type="official_tourism_portal",
                    title="New York Source",
                ),
                extracted_text="The Met fits a museum-focused itinerary.",
                raw_html_path=root / "source.html",
                raw_text_path=root / "source.txt",
            )
            draft_path = root / "draft.json"
            write_draft(draft_path, draft)
            approved = approved_drafts([draft_path])

        self.assertEqual(approved, [])

    def test_approved_doc_validates_and_loads_from_knowledge_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            knowledge_file = root / "knowledge" / "us" / "new_york.json"
            doc = self._knowledge_doc("doc-a")
            promoted = merge_knowledge_docs(knowledge_file=knowledge_file, docs=[doc])
            service = TravelRAGService(knowledge_root=knowledge_file.parent, persist_directory=root / "index")
            loaded = service.load_knowledge_docs()

        self.assertEqual(promoted, 1)
        self.assertEqual(loaded[0].doc_id, "doc-a")

    def test_duplicate_doc_id_is_skipped_unless_overwrite(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            knowledge_file = root / "knowledge" / "new_york.json"
            first = self._knowledge_doc("doc-a", title="first")
            second = self._knowledge_doc("doc-a", title="second")

            self.assertEqual(merge_knowledge_docs(knowledge_file=knowledge_file, docs=[first]), 1)
            self.assertEqual(merge_knowledge_docs(knowledge_file=knowledge_file, docs=[second]), 0)
            payload = json.loads(knowledge_file.read_text(encoding="utf-8"))
            self.assertEqual(payload[0]["title"], "first")

            self.assertEqual(merge_knowledge_docs(knowledge_file=knowledge_file, docs=[second], overwrite=True), 1)
            payload = json.loads(knowledge_file.read_text(encoding="utf-8"))
            self.assertEqual(payload[0]["title"], "second")

    def test_upload_size_limit_is_50mb(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(ValueError, "50MB or smaller"):
                write_uploaded_source(
                    filename="large-guide.pdf",
                    content=b"x" * (50 * 1024 * 1024 + 1),
                    country="US",
                    city="New York",
                    source_id="large-guide",
                    upload_root=Path(tmpdir),
                )

    def test_upload_update_approve_promote_and_rebuild_api(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            original_draft_root = rag_ingestion_routes.DEFAULT_DRAFT_ROOT
            original_knowledge_root = rag_ingestion_routes.DEFAULT_KNOWLEDGE_ROOT
            original_job_getter = rag_ingestion_routes.get_rag_ingestion_job_store
            original_rag_getter = rag_ingestion_routes.get_rag_service
            job_store = RAGIngestionJobStore(root / "jobs.sqlite3")

            class FakeRAGService:
                def __init__(self):
                    self.rebuilt = False

                def ensure_index(self, force_rebuild: bool = False):
                    self.rebuilt = force_rebuild

            fake_rag = FakeRAGService()
            rag_ingestion_routes.DEFAULT_DRAFT_ROOT = root / "drafts"
            rag_ingestion_routes.DEFAULT_KNOWLEDGE_ROOT = root / "knowledge"
            rag_ingestion_routes.get_rag_ingestion_job_store = lambda: job_store
            rag_ingestion_routes.get_rag_service = lambda: fake_rag
            try:
                client = TestClient(app)
                upload_response = client.post(
                    "/api/rag-ingestion/uploads",
                    data={
                        "source_id": "nyc-test-source",
                        "country": "US",
                        "city": "New York",
                        "source_url": "https://example.com/nyc-guide",
                        "source_type": "official_tourism_portal",
                        "title": "NYC Test Guide",
                        "theme": "museums,parks",
                        "poi_names": "Central Park,Metropolitan Museum of Art",
                        "language": "en",
                    },
                    files={
                        "file": (
                            "nyc-guide.md",
                            b"# NYC Guide\nCentral Park and museums work well together.",
                            "text/markdown",
                        )
                    },
                )
                self.assertEqual(upload_response.status_code, 200, upload_response.text)
                draft_detail = upload_response.json()["data"]
                draft_id = draft_detail["draft_id"]
                self.assertIn("Central Park", draft_detail["extracted_text"])

                draft = draft_detail["draft"]
                draft["planning_tips"] = ["Pair museums with Central Park."]
                draft["content"] = "Central Park and museums work well together."
                update_response = client.put(
                    f"/api/rag-ingestion/drafts/{draft_id}",
                    json=draft,
                )
                self.assertEqual(update_response.status_code, 200, update_response.text)
                self.assertEqual(update_response.json()["data"]["draft"]["planning_tips"], ["Pair museums with Central Park."])

                approve_response = client.post(
                    f"/api/rag-ingestion/drafts/{draft_id}/approve",
                    json={"reviewer": "yxj", "review_notes": "Reviewed source text."},
                )
                self.assertEqual(approve_response.status_code, 200, approve_response.text)
                self.assertEqual(approve_response.json()["data"]["draft"]["review_status"], "approved")

                promote_response = client.post(
                    "/api/rag-ingestion/promote",
                    json={"country": "US", "overwrite": False},
                )
                self.assertEqual(promote_response.status_code, 200, promote_response.text)
                self.assertEqual(promote_response.json()["data"]["promoted"], 1)
                self.assertEqual(promote_response.json()["data"]["skipped_existing"], 0)
                knowledge_file = root / "knowledge" / "us" / "new_york.json"
                self.assertTrue(knowledge_file.exists())

                list_response = client.get("/api/rag-ingestion/drafts", params={"country": "US"})
                self.assertEqual(list_response.status_code, 200, list_response.text)
                self.assertTrue(list_response.json()["data"][0]["promoted"])
                self.assertEqual(list_response.json()["data"][0]["corpus_status"], "promoted")

                repeat_promote_response = client.post(
                    "/api/rag-ingestion/promote",
                    json={"country": "US", "overwrite": False},
                )
                self.assertEqual(repeat_promote_response.status_code, 200, repeat_promote_response.text)
                self.assertEqual(repeat_promote_response.json()["data"]["promoted"], 0)
                self.assertEqual(repeat_promote_response.json()["data"]["skipped_existing"], 1)

                rebuild_response = client.post("/api/rag-ingestion/index/rebuild")
                self.assertEqual(rebuild_response.status_code, 200, rebuild_response.text)
                job_id = rebuild_response.json()["data"]["job_id"]
                job_response = client.get(f"/api/rag-ingestion/jobs/{job_id}")
                self.assertEqual(job_response.status_code, 200, job_response.text)
                self.assertEqual(job_response.json()["data"]["status"], "succeeded")
                self.assertTrue(fake_rag.rebuilt)
            finally:
                rag_ingestion_routes.DEFAULT_DRAFT_ROOT = original_draft_root
                rag_ingestion_routes.DEFAULT_KNOWLEDGE_ROOT = original_knowledge_root
                rag_ingestion_routes.get_rag_ingestion_job_store = original_job_getter
                rag_ingestion_routes.get_rag_service = original_rag_getter

    def test_url_ingestion_creates_draft_and_saves_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            original_draft_root = rag_ingestion_routes.DEFAULT_DRAFT_ROOT
            original_web_getter = rag_ingestion_routes.get_web_fetch_service
            original_raw_writer = rag_ingestion_routes.write_raw_source_text
            original_extracted_writer = rag_ingestion_routes.write_extracted_text

            class FakeWebService:
                def fetch(self, url: str, css_selector: str = ""):
                    return WebFetchResult(
                        requested_url=url,
                        final_url=url,
                        page_title="NYC Guide",
                        raw_html="<html><main><h1>NYC Guide</h1></main></html>",
                        extracted_markdown="# NYC Guide\n\nCentral Park works well with subway access.",
                        fetch_mode="static",
                        fetched_at=datetime(2026, 6, 17, tzinfo=timezone.utc),
                    )

            rag_ingestion_routes.DEFAULT_DRAFT_ROOT = root / "drafts"
            rag_ingestion_routes.get_web_fetch_service = lambda: FakeWebService()
            rag_ingestion_routes.write_raw_source_text = lambda **kwargs: write_raw_source_text(
                **kwargs,
                upload_root=root / "uploads",
            )
            rag_ingestion_routes.write_extracted_text = lambda **kwargs: original_extracted_writer(
                **kwargs,
                extracted_root=root / "extracted",
            )
            try:
                client = TestClient(app)
                response = client.post(
                    "/api/rag-ingestion/urls",
                    json={
                        "source_id": "nyc-url-source",
                        "country": "US",
                        "city": "New York",
                        "source_url": "https://example.com/nyc",
                        "source_type": "official_tourism_portal",
                        "title": "NYC URL Guide",
                        "theme": ["parks"],
                        "poi_names": ["Central Park"],
                        "language": "en",
                    },
                )
                self.assertEqual(response.status_code, 200, response.text)
                data = response.json()["data"]
                draft = data["draft"]
                self.assertEqual(draft["review_status"], "draft")
                self.assertTrue(Path(draft["raw_html_path"]).exists())
                self.assertTrue(Path(draft["raw_text_path"]).exists())
                self.assertIn("Central Park", data["extracted_text"])
            finally:
                rag_ingestion_routes.DEFAULT_DRAFT_ROOT = original_draft_root
                rag_ingestion_routes.get_web_fetch_service = original_web_getter
                rag_ingestion_routes.write_raw_source_text = original_raw_writer
                rag_ingestion_routes.write_extracted_text = original_extracted_writer

    def test_validate_url_rejects_unsafe_targets(self):
        validate_url_safety("https://example.com/page", resolver=public_resolver)
        with self.assertRaises(URLSafetyError):
            validate_url_safety("file:///etc/passwd", resolver=public_resolver)
        with self.assertRaises(URLSafetyError):
            validate_url_safety("http://localhost/test", resolver=public_resolver)
        with self.assertRaises(URLSafetyError):
            validate_url_safety("http://127.0.0.1/test", resolver=public_resolver)
        with self.assertRaises(URLSafetyError):
            validate_url_safety("http://10.0.0.1/test", resolver=public_resolver)
        with self.assertRaises(URLSafetyError):
            validate_url_safety("http://169.254.169.254/latest/meta-data", resolver=public_resolver)
        with self.assertRaises(URLSafetyError):
            validate_url_safety("https://example.com/private", resolver=private_resolver)

    def test_web_fetch_static_success_and_css_selector(self):
        FakeStaticFetcher.page = FakePage(
            """
            <html><body><nav>Navigation</nav><main><h1>Central Park</h1>
            <p>Allow two hours.</p><aside>Newsletter signup</aside></main></body></html>
            """
        )
        FakeStaticFetcher.error = None
        FakeDynamicFetcher.called = False
        service = WebFetchService(
            resolver=public_resolver,
            static_fetcher=FakeStaticFetcher,
            dynamic_fetcher=FakeDynamicFetcher,
            min_meaningful_text_chars=10,
        )

        result = service.fetch("https://example.com/park", css_selector="main")

        self.assertEqual(result.fetch_mode, "static")
        self.assertIn("Central Park", result.extracted_markdown)
        self.assertIn("Allow two hours", result.extracted_markdown)
        self.assertNotIn("Navigation", result.extracted_markdown)
        self.assertFalse(FakeDynamicFetcher.called)

    def test_web_fetch_prefers_article_over_body_and_removes_media_noise(self):
        FakeStaticFetcher.page = FakePage(
            """
            <html><body>
              <nav>Skip navigation Visitors Guide</nav>
              <main>
                <div>Book your trip</div>
                <article>
                  <h1>Chicago Guide</h1>
                  <img src="/hero.jpg" alt="Hero image should not enter markdown">
                  <p>Central Park and Chicago Riverwalk are useful itinerary anchors.</p>
                  <p>Allow two hours for the riverfront route.</p>
                </article>
                <aside>Subscribe to our newsletter.</aside>
              </main>
              <footer>Copyright footer</footer>
            </body></html>
            """
        )
        FakeStaticFetcher.error = None
        FakeDynamicFetcher.called = False
        service = WebFetchService(
            resolver=public_resolver,
            static_fetcher=FakeStaticFetcher,
            dynamic_fetcher=FakeDynamicFetcher,
            min_meaningful_text_chars=10,
        )

        result = service.fetch("https://example.com/chicago")

        self.assertIn("Chicago Guide", result.extracted_markdown)
        self.assertIn("Allow two hours", result.extracted_markdown)
        self.assertNotIn("Skip navigation", result.extracted_markdown)
        self.assertNotIn("Hero image", result.extracted_markdown)
        self.assertNotIn("newsletter", result.extracted_markdown.lower())

    def test_web_fetch_dynamic_fallback_for_app_shell(self):
        FakeStaticFetcher.page = FakePage(
            '<html><body><div id="app"></div><script src="/app.js"></script></body></html>'
        )
        FakeStaticFetcher.error = None
        FakeDynamicFetcher.called = False
        FakeDynamicFetcher.page = FakePage(
            "<html><body><main><h1>Rendered Guide</h1><p>Use public transit and allow two hours.</p></main></body></html>"
        )
        service = WebFetchService(
            resolver=public_resolver,
            static_fetcher=FakeStaticFetcher,
            dynamic_fetcher=FakeDynamicFetcher,
            min_meaningful_text_chars=10,
        )

        result = service.fetch("https://example.com/rendered")

        self.assertEqual(result.fetch_mode, "dynamic")
        self.assertTrue(result.dynamic_fallback_used)
        self.assertTrue(FakeDynamicFetcher.called)
        self.assertIn("Rendered Guide", result.extracted_markdown)

    def test_web_fetch_redirect_to_private_target_is_rejected(self):
        FakeStaticFetcher.page = FakePage(
            "<html><body><main><p>Valid travel content with enough useful words for a draft.</p></main></body></html>",
            url="http://10.0.0.1/private",
        )
        service = WebFetchService(
            resolver=public_resolver,
            static_fetcher=FakeStaticFetcher,
            dynamic_fetcher=FakeDynamicFetcher,
            min_meaningful_text_chars=10,
        )

        with self.assertRaises(URLSafetyError):
            service.fetch("https://example.com/redirect")

    def test_web_fetch_timeout_and_oversized_response(self):
        FakeStaticFetcher.error = TimeoutError("timeout")
        service = WebFetchService(
            resolver=public_resolver,
            static_fetcher=FakeStaticFetcher,
            dynamic_fetcher=FakeDynamicFetcher,
        )
        with self.assertRaisesRegex(Exception, "Static fetch failed"):
            service.fetch("https://example.com/timeout")

        FakeStaticFetcher.error = None
        FakeStaticFetcher.page = FakePage("<html><body>" + ("x" * 200) + "</body></html>")
        tiny_service = WebFetchService(
            resolver=public_resolver,
            static_fetcher=FakeStaticFetcher,
            dynamic_fetcher=FakeDynamicFetcher,
            max_html_bytes=20,
            min_meaningful_text_chars=10,
        )
        with self.assertRaisesRegex(Exception, "maximum allowed size"):
            tiny_service.fetch("https://example.com/large")

    def test_ai_prefill_returns_unsaved_suggestions_from_mocked_llm(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            original_draft_root = rag_ingestion_routes.DEFAULT_DRAFT_ROOT
            original_llm_getter = rag_ingestion_routes.get_llm

            class FakeMessage:
                content = json.dumps(
                    {
                        "suggestions": [
                            {
                                "field": "poi_names",
                                "value": "Central Park",
                                "source_quote": "Central Park and nearby museums work well together.",
                                "section_id": "section-001",
                                "section_heading": "Overview",
                                "time_sensitive": False,
                                "confidence": 0.9,
                            },
                            {
                                "field": "planning_tips",
                                "value": "Avoid overscheduling; leave room for spontaneous NYC stops.",
                                "source_quote": "Central Park and nearby museums work well together.",
                                "section_id": "section-001",
                                "section_heading": "Overview",
                                "time_sensitive": False,
                                "confidence": 0.85,
                            }
                        ],
                        "warnings": ["Review source-specific safety claims manually."],
                    }
                )

            class FakeLLM:
                def invoke(self, prompt: str):
                    self.prompt = prompt
                    return FakeMessage()

            fake_llm = FakeLLM()
            rag_ingestion_routes.DEFAULT_DRAFT_ROOT = root / "drafts"
            rag_ingestion_routes.get_llm = lambda: fake_llm
            try:
                raw_text = root / "raw.md"
                raw_text.write_text(
                    "Central Park and nearby museums work well together.\n\n"
                    "NYC rewards spontaneity and does not require a jam-packed schedule.",
                    encoding="utf-8",
                )
                draft = build_draft_document(
                    entry=SourceManifestEntry(
                        source_id="nyc-ai-source",
                        country="US",
                        city="New York",
                        source_url="https://example.com/nyc-ai",
                        source_type="official_tourism_portal",
                        title="NYC AI Source",
                    ),
                    extracted_text=raw_text.read_text(encoding="utf-8"),
                    raw_html_path=root / "source.md",
                    raw_text_path=raw_text,
                )
                draft_path = rag_ingestion_routes.DEFAULT_DRAFT_ROOT / "us" / "new-york" / "new-york-nyc-ai-source.json"
                write_draft(draft_path, draft)

                client = TestClient(app)
                response = client.post(f"/api/rag-ingestion/drafts/{draft.doc_id}/ai-prefill")

                self.assertEqual(response.status_code, 200, response.text)
                data = response.json()["data"]
                self.assertEqual(data["suggested_draft"]["review_status"], "draft")
                self.assertIn("Central Park", data["suggested_draft"]["poi_names"])
                self.assertEqual(data["suggestions"][0]["field"], "poi_names")
                self.assertEqual(data["accepted_suggestion_count"], 2)
                self.assertGreater(data["source_char_count"], 0)
                self.assertGreater(data["used_char_count"], 0)

                persisted = read_draft(draft_path)
                self.assertEqual(persisted.poi_names, [])
                self.assertNotIn("Avoid overscheduling", persisted.planning_tips)
            finally:
                rag_ingestion_routes.DEFAULT_DRAFT_ROOT = original_draft_root
                rag_ingestion_routes.get_llm = original_llm_getter

    def test_ai_prefill_requires_existing_extracted_text(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            original_draft_root = rag_ingestion_routes.DEFAULT_DRAFT_ROOT
            rag_ingestion_routes.DEFAULT_DRAFT_ROOT = root / "drafts"
            try:
                draft = build_draft_document(
                    entry=SourceManifestEntry(
                        source_id="missing-text",
                        country="US",
                        city="New York",
                        source_url="https://example.com/missing",
                        source_type="official_tourism_portal",
                        title="Missing Text",
                    ),
                    extracted_text="Temporary text",
                    raw_html_path=root / "source.md",
                    raw_text_path=root / "does-not-exist.md",
                )
                draft_path = rag_ingestion_routes.DEFAULT_DRAFT_ROOT / "us" / "new-york" / "new-york-missing-text.json"
                write_draft(draft_path, draft)

                client = TestClient(app)
                response = client.post(f"/api/rag-ingestion/drafts/{draft.doc_id}/ai-prefill")

                self.assertEqual(response.status_code, 400, response.text)
                self.assertIn("Extracted text file does not exist", response.text)
            finally:
                rag_ingestion_routes.DEFAULT_DRAFT_ROOT = original_draft_root

    def test_ai_prefill_compacts_long_source_text(self):
        draft = build_draft_document(
            entry=SourceManifestEntry(
                source_id="long-source",
                country="US",
                city="New York",
                source_url="https://example.com/long",
                source_type="official_tourism_portal",
                title="Long NYC Source",
                poi_names=["Central Park"],
            ),
            extracted_text="Central Park is useful for an outdoor itinerary.",
            raw_html_path=Path("source.md"),
            raw_text_path=Path("source.md"),
        )
        extracted = "\n\n".join(
            [f"Navigation item {i}" for i in range(20)]
            + [
                f"Central Park itinerary paragraph {i} works well with museum visits, subway access, and walking routes."
                for i in range(50)
            ]
            + [
                f"Unrelated boilerplate paragraph {i} about accounts, member portals, and website navigation."
                for i in range(200)
            ]
        )

        packet = build_ai_prefill_source_packet(
            draft=draft,
            extracted_text=extracted,
            max_chars=600,
        )

        self.assertLess(packet.used_char_count, packet.source_char_count)
        self.assertIn("Central Park", packet.source_packet)
        self.assertTrue(packet.warnings)

    def test_prefill_cleanup_sections_and_relevance_filtering_for_short_source(self):
        draft = build_draft_document(
            entry=SourceManifestEntry(
                source_id="short-source",
                country="US",
                city="New York",
                source_url="https://example.com/short",
                source_type="official_tourism_portal",
                title="Short NYC Source",
                poi_names=["Central Park"],
            ),
            extracted_text="Central Park is useful.",
            raw_html_path=Path("source.md"),
            raw_text_path=Path("source.md"),
        )
        source = """
        Header repeated
        Header repeated
        Header repeated

        Cookie settings and privacy policy.

        # Getting There
        Central Park is near subway access.
        - Allow two hours.
        - Wheelchair accessible.

        Subscribe to our newsletter for more deals.

        # Los Angeles
        This unrelated destination has beaches and celebrity tours.
        """

        packet = build_ai_prefill_source_packet(draft=draft, extracted_text=source, max_chars=1200)

        self.assertIn("Getting There", packet.source_packet)
        self.assertIn("Allow two hours", packet.source_packet)
        self.assertNotIn("Cookie settings", packet.source_packet)
        self.assertNotIn("newsletter", packet.source_packet.lower())
        self.assertNotIn("Header repeated", packet.source_packet)
        self.assertTrue(packet.selected_sections)
        self.assertLess(packet.selected_section_count if hasattr(packet, "selected_section_count") else len(packet.selected_sections), len(packet.sections))

    def test_split_markdown_sections_preserves_headings_and_lists(self):
        sections = split_markdown_sections("# Getting There\nTake transit.\n- Allow two hours.\n- Closed Mondays.")

        self.assertEqual(sections[0].heading, "Getting There")
        self.assertIn("- Allow two hours.", sections[0].text)
        self.assertIn("Closed Mondays", sections[0].text)

    def test_evidence_validation_accepts_and_rejects_suggestions(self):
        draft = build_draft_document(
            entry=SourceManifestEntry(
                source_id="evidence-source",
                country="US",
                city="New York",
                source_url="https://example.com/evidence",
                source_type="official_tourism_portal",
                title="Evidence Source",
            ),
            extracted_text="Temporary text",
            raw_html_path=Path("source.md"),
            raw_text_path=Path("source.md"),
        )
        section = split_markdown_sections("# Hours\nCentral Park is closed Mondays and wheelchair accessible.")[0]
        suggestion = RAGPrefillSuggestion(
            suggestions=[
                RAGPrefillSuggestionItem(
                    field="planning_tips",
                    value="Central Park is closed Mondays.",
                    source_quote="Central Park is closed Mondays",
                    section_id=section.section_id,
                    section_heading=section.heading,
                ),
                RAGPrefillSuggestionItem(
                    field="planning_tips",
                    value="Central Park has free private tours every day.",
                    source_quote="fabricated quote",
                    section_id=section.section_id,
                    section_heading=section.heading,
                ),
                RAGPrefillSuggestionItem(
                    field="planning_tips",
                    value="Central Park is closed Mondays.",
                    source_quote="Central Park is closed Mondays",
                    section_id=section.section_id,
                    section_heading=section.heading,
                ),
            ]
        )

        validated = validate_prefill_suggestions(
            draft=draft,
            suggestion=suggestion,
            selected_sections=[section],
        )

        self.assertEqual(validated[0].status, "accepted")
        self.assertTrue(validated[0].time_sensitive)
        self.assertEqual(validated[1].status, "rejected")
        self.assertEqual(validated[1].reason, "source_quote_not_found")
        self.assertEqual(validated[2].status, "rejected")
        self.assertEqual(validated[2].reason, "duplicate_suggestion")

    def test_quote_validation_accepts_markdown_link_source_text(self):
        section = split_markdown_sections(
            "# Navy Pier\n"
            "**[Big Bus Tours](https://example.com/bus)** offers a convenient "
            "hop-on, hop-off sightseeing experience that is perfect for first-time visitors."
        )[0]

        self.assertTrue(
            quote_in_section(
                "Big Bus Tours offers a convenient hop-on, hop-off sightseeing experience",
                section.text,
            )
        )

    def test_prefill_validation_splits_atomic_list_values(self):
        draft = build_draft_document(
            entry=SourceManifestEntry(
                source_id="chicago-list-source",
                country="US",
                city="Chicago",
                source_url="https://example.com/chicago",
                source_type="official_tourism_portal",
                title="Chicago List Source",
            ),
            extracted_text="Temporary text",
            raw_html_path=Path("source.md"),
            raw_text_path=Path("source.md"),
        )
        section = split_markdown_sections(
            "# Observation Decks\n"
            "Chicago has two observation deck experiences: Skydeck at Willis Tower with The Ledge, "
            "and TILT at 360 CHICAGO at 875 N. Michigan Ave. These decks provide skyline views."
        )[0]
        suggestion = RAGPrefillSuggestion(
            suggestions=[
                RAGPrefillSuggestionItem(
                    field="poi_names",
                    value="Skydeck at Willis Tower; The Ledge; 360 CHICAGO; TILT",
                    source_quote="Skydeck at Willis Tower with The Ledge, and TILT at 360 CHICAGO",
                    section_id=section.section_id,
                    section_heading=section.heading,
                ),
                RAGPrefillSuggestionItem(
                    field="poi_names",
                    value="Unrelated Museum",
                    source_quote="Skydeck at Willis Tower with The Ledge, and TILT at 360 CHICAGO",
                    section_id=section.section_id,
                    section_heading=section.heading,
                ),
                RAGPrefillSuggestionItem(
                    field="theme",
                    value="observation decks, skyline views",
                    source_quote="observation deck experiences",
                    section_id=section.section_id,
                    section_heading=section.heading,
                ),
            ]
        )

        validated = validate_prefill_suggestions(
            draft=draft,
            suggestion=suggestion,
            selected_sections=[section],
        )
        accepted_values = [item.value for item in validated if item.status == "accepted"]
        rejected_values = {item.value: item.reason for item in validated if item.status == "rejected"}
        merged = apply_prefill_suggestions(draft=draft, suggestions=validated)

        self.assertIn("Skydeck at Willis Tower", accepted_values)
        self.assertIn("The Ledge", accepted_values)
        self.assertIn("360 CHICAGO", accepted_values)
        self.assertIn("TILT", accepted_values)
        self.assertEqual(rejected_values["Unrelated Museum"], "list_value_not_supported")
        self.assertIn("observation decks", merged.theme)
        self.assertIn("skyline views", merged.theme)
        self.assertEqual(merged.poi_names, ["Skydeck at Willis Tower", "The Ledge", "360 CHICAGO", "TILT"])

    def test_prefill_merge_only_accepted_and_preserves_existing_values(self):
        draft = build_draft_document(
            entry=SourceManifestEntry(
                source_id="merge-source",
                country="US",
                city="New York",
                source_url="https://example.com/merge",
                source_type="official_tourism_portal",
                title="Merge Source",
                theme=["parks"],
            ),
            extracted_text="Existing content.",
            raw_html_path=Path("source.md"),
            raw_text_path=Path("source.md"),
        )
        draft.content = "Reviewer-entered content."
        suggestions = [
            RAGValidatedSuggestion(
                field="theme",
                value="parks",
                source_quote="parks",
                section_id="section-001",
                status="accepted",
            ),
            RAGValidatedSuggestion(
                field="planning_tips",
                value="Allow two hours.",
                source_quote="Allow two hours.",
                section_id="section-001",
                status="accepted",
            ),
            RAGValidatedSuggestion(
                field="planning_tips",
                value="Book now for limited time deals.",
                source_quote="Book now.",
                section_id="section-001",
                status="rejected",
            ),
            RAGValidatedSuggestion(
                field="content",
                value="Accepted source-backed content.",
                source_quote="Accepted source-backed content.",
                section_id="section-001",
                status="accepted",
            ),
        ]

        merged = apply_prefill_suggestions(draft=draft, suggestions=suggestions)

        self.assertEqual(merged.theme, ["parks"])
        self.assertEqual(merged.planning_tips, ["Allow two hours."])
        self.assertNotIn("Book now", merged.planning_tips)
        self.assertIn("Reviewer-entered content.", merged.content)
        self.assertIn("Accepted source-backed content.", merged.content)
        self.assertEqual(merged.review_status, "draft")

    def _knowledge_doc(self, doc_id: str, title: str = "title") -> KnowledgeDocument:
        return KnowledgeDocument(
            doc_id=doc_id,
            country="US",
            city="New York",
            district="Manhattan",
            theme=["museums"],
            poi_names=["The Metropolitan Museum of Art"],
            best_for=[],
            recommended_duration="",
            seasonality=[],
            transport_advice=[],
            planning_tips=[],
            source_type="official_tourism_portal",
            source_url="https://www.nyctourism.com/",
            language="en",
            last_verified_at="2026-05-06",
            title=title,
            content="The Met fits a museum-focused itinerary.",
        )


if __name__ == "__main__":
    unittest.main()
