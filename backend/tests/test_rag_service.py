"""Tests for local and Chroma-backed RAG retrieval."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

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
