"""Compare RAG retrieval ranking strategies on labeled benchmark requests."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.models.langgraph_state import AttractionCandidate
from app.models.schemas import TripRequest
from app.services.rag_service import TravelRAGService, get_rag_service
from scripts.benchmark_trip_planners import BenchmarkCase, load_benchmark_cases


K = 4
FETCH_K = 12


def doc_id_for_item(item: Any) -> str:
    metadata = getattr(item, "metadata", item if isinstance(item, dict) else {})
    return str(metadata.get("doc_id", ""))


def compact_doc(doc: Any, *, rank: int, mode: str) -> Dict[str, Any]:
    metadata = dict(getattr(doc, "metadata", {}))
    return {
        "rank": rank,
        "mode": mode,
        "chunk_id": metadata.get("chunk_id", ""),
        "doc_id": metadata.get("doc_id", ""),
        "title": metadata.get("title", ""),
        "city": metadata.get("city", ""),
        "theme": metadata.get("theme", ""),
        "source_url": metadata.get("source_url", ""),
        "vector_rank": metadata.get("vector_rank", rank),
        "rerank_score": metadata.get("rerank_score"),
        "rerank_reasons": metadata.get("rerank_reasons", []),
        "dedup_rank": metadata.get("dedup_rank"),
    }


def recall_at_k(retrieved_doc_ids: Sequence[str], expected_doc_ids: Sequence[str], k: int = K) -> float | None:
    expected = {doc_id for doc_id in expected_doc_ids if doc_id}
    if not expected:
        return None
    retrieved = {doc_id for doc_id in retrieved_doc_ids[:k] if doc_id}
    return len(expected & retrieved) / len(expected)


def reciprocal_rank(retrieved_doc_ids: Sequence[str], expected_doc_ids: Sequence[str], k: int = K) -> float | None:
    expected = {doc_id for doc_id in expected_doc_ids if doc_id}
    if not expected:
        return None
    for index, doc_id in enumerate(retrieved_doc_ids[:k], start=1):
        if doc_id in expected:
            return 1.0 / index
    return 0.0


def ndcg_at_k(retrieved_doc_ids: Sequence[str], expected_doc_ids: Sequence[str], k: int = K) -> float | None:
    expected = {doc_id for doc_id in expected_doc_ids if doc_id}
    if not expected:
        return None
    seen_relevant_doc_ids: set[str] = set()
    gains: List[float] = []
    for doc_id in retrieved_doc_ids[:k]:
        if doc_id in expected and doc_id not in seen_relevant_doc_ids:
            gains.append(1.0)
            seen_relevant_doc_ids.add(doc_id)
        else:
            gains.append(0.0)
    dcg = sum(gain / math.log2(index + 2) for index, gain in enumerate(gains))
    ideal_relevant_count = min(len(expected), k)
    ideal_dcg = sum(1.0 / math.log2(index + 2) for index in range(ideal_relevant_count))
    return dcg / ideal_dcg if ideal_dcg else 0.0


def duplicate_doc_rate(retrieved_doc_ids: Sequence[str]) -> float:
    doc_ids = [doc_id for doc_id in retrieved_doc_ids if doc_id]
    if not doc_ids:
        return 0.0
    return (len(doc_ids) - len(set(doc_ids))) / len(doc_ids)


def score_and_sort_without_dedup(
    service: TravelRAGService,
    *,
    request: TripRequest,
    docs: List[Any],
    attraction_candidates: List[AttractionCandidate],
    k: int,
) -> List[Tuple[Any, Dict[str, Any]]]:
    request_terms = service._request_terms(request, attraction_candidates)
    scored_docs = []
    for vector_rank, doc in enumerate(docs, start=1):
        metadata = dict(doc.metadata)
        score, reasons = service._score_rag_doc(
            request=request,
            request_terms=request_terms,
            doc=doc,
            metadata=metadata,
            vector_rank=vector_rank,
        )
        metadata["vector_rank"] = vector_rank
        metadata["rerank_score"] = round(score, 4)
        metadata["rerank_reasons"] = reasons
        scored_docs.append((score, vector_rank, doc, metadata))
    scored_docs.sort(key=lambda item: (-item[0], item[1]))
    return [(doc, metadata) for _, _, doc, metadata in scored_docs[:k]]


def docs_from_ranked_pairs(ranked_pairs: Iterable[Tuple[Any, Dict[str, Any]]]) -> List[Any]:
    docs: List[Any] = []
    for doc, metadata in ranked_pairs:
        doc.metadata = metadata
        docs.append(doc)
    return docs


def retrieve_raw_vector_docs(
    service: TravelRAGService,
    request: TripRequest,
    *,
    attraction_candidates: List[AttractionCandidate],
    fetch_k: int,
) -> List[Any]:
    service.ensure_index()
    query = service.build_query(request, attraction_candidates)
    vectorstore = service._get_vectorstore()
    try:
        return vectorstore.similarity_search(
            query,
            k=fetch_k,
            filter={"city": request.city},
        )
    except Exception:
        return []


def evaluate_mode(
    *,
    mode: str,
    docs: List[Any],
    expected_doc_ids: Sequence[str],
) -> Dict[str, Any]:
    retrieved_doc_ids = [doc_id_for_item(doc) for doc in docs]
    recall = recall_at_k(retrieved_doc_ids, expected_doc_ids)
    mrr = reciprocal_rank(retrieved_doc_ids, expected_doc_ids)
    ndcg = ndcg_at_k(retrieved_doc_ids, expected_doc_ids)
    return {
        "mode": mode,
        "retrieved_doc_ids": retrieved_doc_ids[:K],
        "retrieved_docs": [
            compact_doc(doc, rank=index, mode=mode)
            for index, doc in enumerate(docs[:K], start=1)
        ],
        "recall_at_4": recall,
        "hit": None if recall is None else recall > 0,
        "mrr": mrr,
        "ndcg_at_4": ndcg,
        "duplicate_doc_rate": duplicate_doc_rate(retrieved_doc_ids[:K]),
        "unique_doc_count": len({doc_id for doc_id in retrieved_doc_ids[:K] if doc_id}),
    }


def evaluate_case(
    service: TravelRAGService,
    case: BenchmarkCase,
    *,
    fetch_k: int = FETCH_K,
    k: int = K,
) -> Dict[str, Any]:
    attraction_candidates: List[AttractionCandidate] = []
    raw_docs = retrieve_raw_vector_docs(
        service,
        case.request,
        attraction_candidates=attraction_candidates,
        fetch_k=fetch_k,
    )
    vector_docs = raw_docs[:k]
    rerank_no_dedup_docs = docs_from_ranked_pairs(
        score_and_sort_without_dedup(
            service,
            request=case.request,
            docs=list(raw_docs),
            attraction_candidates=attraction_candidates,
            k=k,
        )
    )
    rerank_dedup_docs = docs_from_ranked_pairs(
        service._rerank_and_dedup_docs(
            request=case.request,
            docs=list(raw_docs),
            attraction_candidates=attraction_candidates,
            k=k,
        )
    )
    return {
        "request": case.request.model_dump(),
        "expected_rag_doc_ids": case.expected_rag_doc_ids,
        "expected_rag_themes": case.expected_rag_themes,
        "benchmark_note": case.benchmark_note,
        "modes": {
            "vector_only": evaluate_mode(
                mode="vector_only",
                docs=vector_docs,
                expected_doc_ids=case.expected_rag_doc_ids,
            ),
            "metadata_rerank_no_dedup": evaluate_mode(
                mode="metadata_rerank_no_dedup",
                docs=rerank_no_dedup_docs,
                expected_doc_ids=case.expected_rag_doc_ids,
            ),
            "metadata_rerank_dedup": evaluate_mode(
                mode="metadata_rerank_dedup",
                docs=rerank_dedup_docs,
                expected_doc_ids=case.expected_rag_doc_ids,
            ),
        },
    }


def mean_or_none(values: List[float | None]) -> float | None:
    compact = [value for value in values if value is not None]
    if not compact:
        return None
    return round(statistics.fmean(compact), 4)


def summarize_mode(results: List[Dict[str, Any]], mode: str) -> Dict[str, Any]:
    mode_entries = [result["modes"][mode] for result in results]
    labeled_entries = [entry for entry in mode_entries if entry["recall_at_4"] is not None]
    hits = [1.0 if entry["hit"] else 0.0 for entry in labeled_entries]
    return {
        "labeled_request_count": len(labeled_entries),
        "retrieval_hit_rate": round(statistics.fmean(hits), 4) if hits else None,
        "recall_at_4": mean_or_none([entry["recall_at_4"] for entry in mode_entries]),
        "mrr": mean_or_none([entry["mrr"] for entry in mode_entries]),
        "ndcg_at_4": mean_or_none([entry["ndcg_at_4"] for entry in mode_entries]),
        "duplicate_doc_rate": round(
            statistics.fmean(entry["duplicate_doc_rate"] for entry in mode_entries),
            4,
        )
        if mode_entries
        else 0.0,
        "avg_unique_doc_count": round(
            statistics.fmean(entry["unique_doc_count"] for entry in mode_entries),
            4,
        )
        if mode_entries
        else 0.0,
    }


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    modes = ["vector_only", "metadata_rerank_no_dedup", "metadata_rerank_dedup"]
    by_mode = {mode: summarize_mode(results, mode) for mode in modes}
    return {
        "request_count": len(results),
        "modes": by_mode,
        "delta": {
            "dedup_vs_vector_recall_at_4": round(
                (by_mode["metadata_rerank_dedup"]["recall_at_4"] or 0.0)
                - (by_mode["vector_only"]["recall_at_4"] or 0.0),
                4,
            ),
            "dedup_vs_vector_mrr": round(
                (by_mode["metadata_rerank_dedup"]["mrr"] or 0.0)
                - (by_mode["vector_only"]["mrr"] or 0.0),
                4,
            ),
            "dedup_vs_no_dedup_duplicate_rate": round(
                by_mode["metadata_rerank_dedup"]["duplicate_doc_rate"]
                - by_mode["metadata_rerank_no_dedup"]["duplicate_doc_rate"],
                4,
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAG ranking ablation benchmark.")
    parser.add_argument(
        "--dataset",
        default="benchmarks/trip_requests.us_rag_benchmark.json",
        help="Path to labeled benchmark dataset relative to backend/ or absolute path.",
    )
    parser.add_argument(
        "--output",
        default="benchmarks/results/rag_ranking_ablation.json",
        help="Path to output JSON relative to backend/ or absolute path.",
    )
    parser.add_argument("--fetch-k", type=int, default=FETCH_K)
    parser.add_argument("--k", type=int, default=K)
    parser.add_argument("--rebuild-index", action="store_true")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    output_path = Path(args.output)
    if not dataset_path.is_absolute():
        dataset_path = ROOT / dataset_path
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    service = get_rag_service()
    service.ensure_index(force_rebuild=args.rebuild_index)
    cases = load_benchmark_cases(dataset_path)
    results = [
        evaluate_case(service, case, fetch_k=args.fetch_k, k=args.k)
        for case in cases
    ]
    output = {
        "dataset": str(dataset_path),
        "fetch_k": args.fetch_k,
        "k": args.k,
        "summary": summarize(results),
        "results": results,
    }
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2))
    print(f"\nSaved RAG ranking ablation results to {output_path}")


if __name__ == "__main__":
    main()
