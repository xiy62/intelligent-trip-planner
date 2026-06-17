"""Tests for human-in-the-loop RAG ingestion utilities."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from app.api.main import app
from app.api.routes import rag_ingestion as rag_ingestion_routes
from app.services.rag_ingestion import (
    RAGIngestionJobStore,
    SourceManifestEntry,
    approved_drafts,
    build_draft_document,
    build_ai_prefill_source_packet,
    extract_readable_text,
    load_manifest,
    merge_knowledge_docs,
    read_draft,
    write_uploaded_source,
    write_draft,
)
from app.services.rag_service import KnowledgeDocument, TravelRAGService


class RAGIngestionTests(unittest.TestCase):
    def test_manifest_parsing_and_required_fields(self):
        payload = [
            {
                "source_id": "beijing-source",
                "country": "CN",
                "city": "北京",
                "source_url": "https://example.com/beijing",
                "source_type": "official_tourism_portal",
                "title": "Beijing Source",
                "theme": ["历史文化"],
                "poi_names": ["故宫博物院"],
                "district": "东城区",
                "language": "zh",
            }
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = Path(tmpdir) / "manifest.json"
            manifest.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            entries = load_manifest(manifest)

        self.assertEqual(entries[0].source_id, "beijing-source")
        self.assertEqual(entries[0].city, "北京")

    def test_extract_readable_text_ignores_scripts_and_navigation(self):
        html = """
        <html>
          <body>
            <nav>Menu should disappear</nav>
            <script>console.log("hidden")</script>
            <main>
              <h1>故宫参观指南</h1>
              <p>故宫适合历史文化主题行程。</p>
              <p>建议预留充足参观时间。</p>
            </main>
          </body>
        </html>
        """
        text = extract_readable_text(html)

        self.assertIn("故宫参观指南", text)
        self.assertIn("建议预留充足参观时间", text)
        self.assertNotIn("console.log", text)
        self.assertNotIn("Menu should disappear", text)

    def test_draft_generation_roundtrip(self):
        entry = SourceManifestEntry(
            source_id="beijing-source",
            city="北京",
            source_url="https://example.com/beijing",
            source_type="official_tourism_portal",
            title="Beijing Source",
            theme=["历史文化"],
            poi_names=["故宫博物院"],
            district="东城区",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            draft = build_draft_document(
                entry=entry,
                extracted_text="故宫适合历史文化主题行程。建议预留充足时间。",
                raw_html_path=root / "source.html",
                raw_text_path=root / "source.txt",
                fetched_at="2026-05-06T00:00:00+00:00",
            )
            draft_path = root / "draft.json"
            write_draft(draft_path, draft)
            loaded = read_draft(draft_path)

        self.assertEqual(loaded.review_status, "draft")
        self.assertEqual(loaded.doc_id, "beijing-beijing-source")
        self.assertEqual(loaded.last_verified_at, "2026-05-06")

    def test_unapproved_draft_is_not_promoted(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            draft = build_draft_document(
                entry=SourceManifestEntry(
                    source_id="beijing-source",
                    city="北京",
                    source_url="https://example.com/beijing",
                    source_type="official_tourism_portal",
                    title="Beijing Source",
                ),
                extracted_text="故宫适合历史文化主题行程。",
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
            knowledge_file = root / "knowledge" / "china" / "beijing.json"
            doc = self._knowledge_doc("doc-a")
            promoted = merge_knowledge_docs(knowledge_file=knowledge_file, docs=[doc])
            service = TravelRAGService(knowledge_root=knowledge_file.parent, persist_directory=root / "index")
            loaded = service.load_knowledge_docs()

        self.assertEqual(promoted, 1)
        self.assertEqual(loaded[0].doc_id, "doc-a")

    def test_duplicate_doc_id_is_skipped_unless_overwrite(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            knowledge_file = root / "knowledge" / "beijing.json"
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

    def test_ai_prefill_returns_unsaved_suggestions_from_mocked_llm(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            original_draft_root = rag_ingestion_routes.DEFAULT_DRAFT_ROOT
            original_llm_getter = rag_ingestion_routes.get_llm

            class FakeMessage:
                content = json.dumps(
                    {
                        "content": "Central Park pairs well with nearby museums and flexible solo travel.",
                        "theme": ["parks", "solo travel"],
                        "poi_names": ["Central Park", "The Metropolitan Museum of Art"],
                        "best_for": ["solo travelers", "first-time visitors"],
                        "recommended_duration": "Half day",
                        "seasonality": ["Works year-round with weather-dependent outdoor pacing."],
                        "transport_advice": ["Use subway lines near Midtown or the Upper West Side."],
                        "planning_tips": ["Avoid overscheduling; leave room for spontaneous NYC stops."],
                        "field_evidence": [
                            {
                                "field": "planning_tips",
                                "suggestion": "Avoid overscheduling.",
                                "evidence": "NYC rewards spontaneity and does not require a jam-packed schedule.",
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
                self.assertEqual(data["field_evidence"][0]["field"], "planning_tips")
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

        packet, source_count, used_count, warnings = build_ai_prefill_source_packet(
            draft=draft,
            extracted_text=extracted,
            max_chars=600,
        )

        self.assertLess(used_count, source_count)
        self.assertIn("Central Park", packet)
        self.assertTrue(warnings)

    def _knowledge_doc(self, doc_id: str, title: str = "title") -> KnowledgeDocument:
        return KnowledgeDocument(
            doc_id=doc_id,
            country="CN",
            city="北京",
            district="东城区",
            theme=["历史文化"],
            poi_names=["故宫博物院"],
            best_for=[],
            recommended_duration="",
            seasonality=[],
            transport_advice=[],
            planning_tips=[],
            source_type="official_tourism_portal",
            source_url="https://example.com/beijing",
            language="zh",
            last_verified_at="2026-05-06",
            title=title,
            content="故宫适合历史文化主题行程。",
        )


if __name__ == "__main__":
    unittest.main()
