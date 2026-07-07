"""Tests for RAG ranking ablation metrics."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

from scripts.benchmark_rag_ranking_ablation import (
    claim_quote_recall_at_k,
    duplicate_doc_rate,
    missing_claim_ids_at_k,
    ndcg_at_k,
    normalize_claim_text,
    recall_at_k,
    reciprocal_rank,
    section_recall_at_k,
    summarize,
)


class RAGRankingAblationTests(unittest.TestCase):
    def test_recall_mrr_and_ndcg_score_ranked_results(self):
        retrieved = ["doc-x", "doc-b", "doc-a", "doc-c"]
        expected = ["doc-a", "doc-b"]

        self.assertEqual(recall_at_k(retrieved, expected), 1.0)
        self.assertEqual(reciprocal_rank(retrieved, expected), 0.5)
        self.assertGreater(ndcg_at_k(retrieved, expected), 0.6)
        self.assertLess(ndcg_at_k(retrieved, expected), 1.0)

    def test_ndcg_does_not_reward_duplicate_relevant_doc_multiple_times(self):
        retrieved = ["doc-a", "doc-a", "doc-a", "doc-a"]
        expected = ["doc-a", "doc-b"]

        self.assertLess(ndcg_at_k(retrieved, expected), 1.0)

    def test_unlabeled_requests_are_excluded_from_label_metrics(self):
        self.assertIsNone(recall_at_k(["doc-a"], []))
        self.assertIsNone(reciprocal_rank(["doc-a"], []))
        self.assertIsNone(ndcg_at_k(["doc-a"], []))

    def test_duplicate_doc_rate_counts_repeated_documents(self):
        self.assertEqual(duplicate_doc_rate(["doc-a", "doc-a", "doc-b", "doc-c"]), 0.25)
        self.assertEqual(duplicate_doc_rate(["doc-a", "doc-b"]), 0.0)

    def test_section_recall_requires_exact_doc_and_section(self):
        docs = [
            SimpleNamespace(metadata={"doc_id": "doc-a", "section": "overview"}),
            SimpleNamespace(metadata={"doc_id": "doc-a", "section": "transport"}),
        ]
        expected = [{"doc_id": "doc-a", "section": "transport"}]

        self.assertEqual(section_recall_at_k(docs, expected), 1.0)
        self.assertEqual(
            section_recall_at_k(docs[:1], expected),
            0.0,
        )

    def test_section_recall_does_not_reward_duplicates(self):
        docs = [
            SimpleNamespace(metadata={"doc_id": "doc-a", "section": "transport"}),
            SimpleNamespace(metadata={"doc_id": "doc-a", "section": "transport"}),
        ]
        expected = [
            {"doc_id": "doc-a", "section": "transport"},
            {"doc_id": "doc-b", "section": "transport"},
        ]

        self.assertEqual(section_recall_at_k(docs, expected), 0.5)

    def test_section_recall_counts_packed_sections(self):
        docs = [
            SimpleNamespace(
                metadata={
                    "doc_id": "doc-a",
                    "section": "overview",
                    "sections": ["overview", "transport", "seasonality"],
                }
            )
        ]
        expected = [
            {"doc_id": "doc-a", "section": "transport"},
            {"doc_id": "doc-a", "section": "seasonality"},
        ]

        self.assertEqual(section_recall_at_k(docs, expected), 1.0)

    def test_claim_quote_recall_accepts_whitespace_normalized_quotes(self):
        docs = [
            SimpleNamespace(
                metadata={"doc_id": "doc-a", "section": "transport"},
                page_content="Doc A\nTransport advice:\n- Use   the train\n  back to the hotel.",
            )
        ]
        claims = [
            {
                "claim_id": "claim-a",
                "doc_id": "doc-a",
                "section": "transport",
                "evidence_quote": "use the train back to the hotel.",
            }
        ]

        self.assertEqual(normalize_claim_text(" Use   THE train\nback "), "use the train back")
        self.assertEqual(claim_quote_recall_at_k(docs, claims), 1.0)
        self.assertEqual(missing_claim_ids_at_k(docs, claims), [])

    def test_claim_quote_recall_accepts_packed_section_metadata(self):
        docs = [
            SimpleNamespace(
                metadata={
                    "doc_id": "doc-a",
                    "section": "overview",
                    "sections": ["overview", "transport"],
                },
                page_content="### overview\nDoc A\n\n### transport\nUse the train back to the hotel.",
            )
        ]
        claims = [
            {
                "claim_id": "packed-transport",
                "doc_id": "doc-a",
                "section": "transport",
                "evidence_quote": "Use the train back to the hotel.",
            }
        ]

        self.assertEqual(claim_quote_recall_at_k(docs, claims), 1.0)

    def test_claim_quote_recall_rejects_wrong_section_and_fabricated_quote(self):
        docs = [
            SimpleNamespace(
                metadata={"doc_id": "doc-a", "section": "overview"},
                page_content="Use the train back to the hotel.",
            )
        ]
        claims = [
            {
                "claim_id": "wrong-section",
                "doc_id": "doc-a",
                "section": "transport",
                "evidence_quote": "Use the train back to the hotel.",
            },
            {
                "claim_id": "fabricated",
                "doc_id": "doc-a",
                "section": "overview",
                "evidence_quote": "Take a ferry back to the hotel.",
            },
        ]

        self.assertEqual(claim_quote_recall_at_k(docs, claims), 0.0)
        self.assertEqual(
            missing_claim_ids_at_k(docs, claims),
            ["wrong-section", "fabricated"],
        )

    def test_unlabeled_section_and_claim_metrics_return_none(self):
        docs = [SimpleNamespace(metadata={"doc_id": "doc-a", "section": "overview"}, page_content="text")]

        self.assertIsNone(section_recall_at_k(docs, []))
        self.assertIsNone(claim_quote_recall_at_k(docs, []))

    def test_summary_reports_mode_deltas(self):
        results = [
            {
                "modes": {
                    "vector_only": {
                        "recall_at_4": 0.0,
                        "hit": False,
                        "mrr": 0.0,
                        "ndcg_at_4": 0.0,
                        "section_recall_at_4": 0.0,
                        "section_hit": False,
                        "claim_quote_recall_at_4": 0.0,
                        "claim_hit": False,
                        "missing_claim_ids": ["claim-a"],
                        "duplicate_doc_rate": 0.25,
                        "unique_doc_count": 3,
                    },
                    "metadata_rerank_no_dedup": {
                        "recall_at_4": 1.0,
                        "hit": True,
                        "mrr": 1.0,
                        "ndcg_at_4": 1.0,
                        "section_recall_at_4": 1.0,
                        "section_hit": True,
                        "claim_quote_recall_at_4": 0.5,
                        "claim_hit": True,
                        "missing_claim_ids": ["claim-b"],
                        "duplicate_doc_rate": 0.25,
                        "unique_doc_count": 3,
                    },
                    "metadata_rerank_dedup": {
                        "recall_at_4": 1.0,
                        "hit": True,
                        "mrr": 1.0,
                        "ndcg_at_4": 1.0,
                        "section_recall_at_4": 1.0,
                        "section_hit": True,
                        "claim_quote_recall_at_4": 1.0,
                        "claim_hit": True,
                        "missing_claim_ids": [],
                        "duplicate_doc_rate": 0.0,
                        "unique_doc_count": 4,
                    },
                }
            }
        ]

        result = summarize(results)

        self.assertEqual(result["modes"]["metadata_rerank_dedup"]["recall_at_4"], 1.0)
        self.assertEqual(result["delta"]["dedup_vs_vector_recall_at_4"], 1.0)
        self.assertEqual(result["delta"]["dedup_vs_no_dedup_duplicate_rate"], -0.25)
        self.assertEqual(result["delta"]["dedup_vs_vector_section_recall_at_4"], 1.0)
        self.assertEqual(result["delta"]["dedup_vs_vector_claim_quote_recall_at_4"], 1.0)


if __name__ == "__main__":
    unittest.main()
