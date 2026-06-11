"""Tests for human-in-the-loop RAG ingestion utilities."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app.services.rag_ingestion import (
    SourceManifestEntry,
    approved_drafts,
    build_draft_document,
    extract_readable_text,
    load_manifest,
    merge_knowledge_docs,
    read_draft,
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
