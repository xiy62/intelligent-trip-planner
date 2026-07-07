"""Tests for local and Chroma-backed RAG retrieval."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.models.schemas import TripRequest
from app.services.rag_service import TravelRAGService


class FakeEmbeddings(Embeddings):
    def _embed(self, text: str) -> list[float]:
        bucket = [0.0, 0.0, 0.0, 0.0]
        for idx, char in enumerate(text):
            bucket[idx % 4] += float(ord(char) % 97)
        return bucket

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def build_request(city: str, preferences: list[str]) -> TripRequest:
    return TripRequest(
        city=city,
        start_date="2026-06-01",
        end_date="2026-06-03",
        travel_days=3,
        transportation="公共交通",
        accommodation="经济型酒店",
        preferences=preferences,
        free_text_input="",
    )


class RAGServiceTests(unittest.TestCase):
    def test_local_chunks_match_request_preferences(self):
        service = TravelRAGService()
        chunks = service.retrieve_local_chunks(build_request("北京", ["历史文化"]))
        self.assertTrue(chunks)
        self.assertEqual(chunks[0].metadata["rag_backend"], "local_lightweight")

    def test_default_corpus_loads_us_knowledge_documents(self):
        service = TravelRAGService()
        doc_ids = {doc.doc_id for doc in service.load_knowledge_docs()}

        self.assertIn("new-york-museum-mile-central-park-001", doc_ids)
        self.assertIn("san-francisco-waterfront-wharf-001", doc_ids)
        self.assertIn("chicago-millennium-park-loop-001", doc_ids)

    def test_chroma_roundtrip_returns_city_chunk(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            persist_dir = Path(tmpdir) / "index"
            service = TravelRAGService(
                persist_directory=persist_dir,
                embedding_function=FakeEmbeddings(),
            )
            service.ensure_index(force_rebuild=True)
            chunks = service.retrieve_chroma_chunks(
                build_request("上海", ["美食", "城市漫步"]),
                attraction_candidates=[],
                k=3,
            )
            self.assertTrue(chunks)
            self.assertEqual(chunks[0].metadata["rag_backend"], "chroma_retrieval")
            self.assertIn(chunks[0].metadata["city"], {"上海"})

    def test_chroma_roundtrip_returns_new_york_chunk(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            persist_dir = Path(tmpdir) / "index"
            service = TravelRAGService(
                persist_directory=persist_dir,
                embedding_function=FakeEmbeddings(),
            )
            service.ensure_index(force_rebuild=True)
            chunks = service.retrieve_chroma_chunks(
                TripRequest(
                    city="New York",
                    start_date="2026-07-01",
                    end_date="2026-07-02",
                    travel_days=2,
                    transportation="Public transit",
                    accommodation="Mid-range hotel",
                    preferences=["Museums", "Food"],
                    free_text_input="Keep museum visits realistic and group nearby neighborhoods.",
                ),
                attraction_candidates=[],
                k=4,
            )

            self.assertTrue(chunks)
            self.assertEqual(chunks[0].metadata["rag_backend"], "chroma_retrieval")
            self.assertEqual(chunks[0].metadata["city"], "New York")
            self.assertEqual(chunks[0].metadata["language"], "en")

    def test_rerank_and_dedup_prefers_unique_matching_documents(self):
        service = TravelRAGService(embedding_function=FakeEmbeddings())
        request = TripRequest(
            city="New York",
            start_date="2026-07-01",
            end_date="2026-07-02",
            travel_days=2,
            transportation="subway and walking",
            accommodation="mid-range hotel",
            preferences=["food", "parks"],
            free_text_input="I want Brooklyn food and park time.",
        )
        docs = [
            Document(
                page_content="General New York sightseeing.",
                metadata={
                    "chunk_id": "doc-a-overview",
                    "doc_id": "doc-a",
                    "city": "New York",
                    "theme": "classic sightseeing",
                    "poi_names": "Times Square",
                    "language": "en",
                    "title": "General",
                },
            ),
            Document(
                page_content="Another chunk from the same generic document.",
                metadata={
                    "chunk_id": "doc-a-planning",
                    "doc_id": "doc-a",
                    "city": "New York",
                    "theme": "classic sightseeing",
                    "poi_names": "Times Square",
                    "language": "en",
                    "title": "General Planning",
                },
            ),
            Document(
                page_content="Brooklyn food neighborhoods and waterfront walking.",
                metadata={
                    "chunk_id": "doc-b-overview",
                    "doc_id": "doc-b",
                    "city": "New York",
                    "theme": "food,neighborhoods,citywalk",
                    "poi_names": "Brooklyn,DUMBO",
                    "language": "en",
                    "title": "Brooklyn Food",
                },
            ),
            Document(
                page_content="Central Park and relaxed outdoor time.",
                metadata={
                    "chunk_id": "doc-c-overview",
                    "doc_id": "doc-c",
                    "city": "New York",
                    "theme": "parks,outdoors",
                    "poi_names": "Central Park",
                    "language": "en",
                    "title": "Parks",
                },
            ),
        ]

        selected = service._rerank_and_dedup_docs(
            request=request,
            docs=docs,
            attraction_candidates=[],
            k=3,
        )
        selected_doc_ids = [metadata["doc_id"] for _, metadata in selected]

        self.assertEqual(len(selected_doc_ids), len(set(selected_doc_ids)))
        self.assertIn("doc-b", selected_doc_ids)
        self.assertIn("doc-c", selected_doc_ids)
        self.assertTrue(all("rerank_score" in metadata for _, metadata in selected))
        self.assertTrue(all("vector_rank" in metadata for _, metadata in selected))
        self.assertEqual([metadata["dedup_rank"] for _, metadata in selected], [1, 2, 3])

    def test_rerank_packs_multiple_sections_from_selected_document(self):
        service = TravelRAGService(embedding_function=FakeEmbeddings())
        request = TripRequest(
            city="Chicago",
            start_date="2026-07-01",
            end_date="2026-07-02",
            travel_days=2,
            transportation="rideshare and walking",
            accommodation="boutique hotel",
            preferences=["food", "nightlife"],
            free_text_input="Keep West Loop dinner and return route simple.",
        )
        docs = [
            Document(
                page_content="West Loop food overview.",
                metadata={
                    "chunk_id": "west-loop-overview",
                    "doc_id": "west-loop",
                    "city": "Chicago",
                    "theme": "food,nightlife",
                    "poi_names": "West Loop,Fulton Market",
                    "language": "en",
                    "section": "overview",
                    "title": "West Loop",
                },
            ),
            Document(
                page_content="Use rideshare or transit planning for the return trip after dinner.",
                metadata={
                    "chunk_id": "west-loop-transport",
                    "doc_id": "west-loop",
                    "city": "Chicago",
                    "theme": "food,nightlife",
                    "poi_names": "West Loop,Fulton Market",
                    "language": "en",
                    "section": "transport",
                    "title": "West Loop",
                },
            ),
            Document(
                page_content="For food requests, make West Loop or Fulton Market the dining anchor.",
                metadata={
                    "chunk_id": "west-loop-planning",
                    "doc_id": "west-loop",
                    "city": "Chicago",
                    "theme": "food,nightlife",
                    "poi_names": "West Loop,Fulton Market",
                    "language": "en",
                    "section": "planning_tips",
                    "title": "West Loop",
                },
            ),
            Document(
                page_content="Chicago river architecture overview.",
                metadata={
                    "chunk_id": "river-overview",
                    "doc_id": "river",
                    "city": "Chicago",
                    "theme": "architecture",
                    "poi_names": "Chicago Riverwalk",
                    "language": "en",
                    "section": "overview",
                    "title": "River",
                },
            ),
        ]

        selected = service._rerank_and_dedup_docs(
            request=request,
            docs=docs,
            attraction_candidates=[],
            k=2,
        )

        self.assertEqual(len(selected), 2)
        first_doc, first_metadata = selected[0]
        self.assertEqual(first_metadata["doc_id"], "west-loop")
        self.assertEqual(first_metadata["packed_section_count"], 3)
        self.assertEqual(
            set(first_metadata["sections"]),
            {"overview", "transport", "planning_tips"},
        )
        self.assertIn("return trip after dinner", first_doc.page_content)
        self.assertIn("dining anchor", first_doc.page_content)
