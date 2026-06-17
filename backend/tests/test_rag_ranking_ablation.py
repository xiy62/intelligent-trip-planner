"""Tests for RAG ranking ablation metrics."""

from __future__ import annotations

import unittest

from scripts.benchmark_rag_ranking_ablation import (
    duplicate_doc_rate,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
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

    def test_summary_reports_mode_deltas(self):
        results = [
            {
                "modes": {
                    "vector_only": {
                        "recall_at_4": 0.0,
                        "hit": False,
                        "mrr": 0.0,
                        "ndcg_at_4": 0.0,
                        "duplicate_doc_rate": 0.25,
                        "unique_doc_count": 3,
                    },
                    "metadata_rerank_no_dedup": {
                        "recall_at_4": 1.0,
                        "hit": True,
                        "mrr": 1.0,
                        "ndcg_at_4": 1.0,
                        "duplicate_doc_rate": 0.25,
                        "unique_doc_count": 3,
                    },
                    "metadata_rerank_dedup": {
                        "recall_at_4": 1.0,
                        "hit": True,
                        "mrr": 1.0,
                        "ndcg_at_4": 1.0,
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


if __name__ == "__main__":
    unittest.main()
