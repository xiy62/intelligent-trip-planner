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


def section_key_for_item(item: Any) -> tuple[str, str]:
    metadata = getattr(item, "metadata", item if isinstance(item, dict) else {})
    return str(metadata.get("doc_id", "")), str(metadata.get("section", ""))


def section_keys_for_item(item: Any) -> set[tuple[str, str]]:
    metadata = getattr(item, "metadata", item if isinstance(item, dict) else {})
    doc_id = str(metadata.get("doc_id", ""))
    if not doc_id:
        return set()
    raw_sections = metadata.get("sections") or [metadata.get("section", "")]
    if isinstance(raw_sections, str):
        raw_sections = [raw_sections]
    return {
        (doc_id, str(section))
        for section in raw_sections
        if section
    }


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
        "section": metadata.get("section", ""),
        "sections": metadata.get("sections", []),
        "packed_section_count": metadata.get("packed_section_count", 1),
        "packed_chunk_ids": metadata.get("packed_chunk_ids", []),
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


def expected_section_keys(expected_sections: Sequence[Dict[str, str]]) -> set[tuple[str, str]]:
    return {
        (str(item.get("doc_id", "")), str(item.get("section", "")))
        for item in expected_sections
        if item.get("doc_id") and item.get("section")
    }


def section_recall_at_k(
    retrieved_items: Sequence[Any],
    expected_sections: Sequence[Dict[str, str]],
    k: int = K,
) -> float | None:
    expected = expected_section_keys(expected_sections)
    if not expected:
        return None
    retrieved = set().union(
        *(
            section_keys_for_item(item)
            for item in retrieved_items[:k]
        )
    )
    return len(expected & retrieved) / len(expected)


def item_has_section(item: Any, expected_section: str) -> bool:
    if not expected_section:
        return True
    metadata = dict(getattr(item, "metadata", {}))
    raw_sections = metadata.get("sections") or [metadata.get("section", "")]
    if isinstance(raw_sections, str):
        raw_sections = [raw_sections]
    return expected_section in {str(section) for section in raw_sections if section}


def normalize_claim_text(value: str) -> str:
    return " ".join(value.lower().split())


def claim_id_for_label(label: Dict[str, str], index: int) -> str:
    return str(label.get("claim_id") or f"claim-{index + 1}")


def claim_matches_doc(label: Dict[str, str], doc: Any) -> bool:
    quote = normalize_claim_text(str(label.get("evidence_quote", "")))
    if not quote:
        return False

    metadata = dict(getattr(doc, "metadata", {}))
    expected_doc_id = str(label.get("doc_id", ""))
    expected_section = str(label.get("section", ""))
    if expected_doc_id and str(metadata.get("doc_id", "")) != expected_doc_id:
        return False
    if not item_has_section(doc, expected_section):
        return False

    return quote in normalize_claim_text(str(getattr(doc, "page_content", "")))


def missing_claim_ids_at_k(
    retrieved_items: Sequence[Any],
    expected_claims: Sequence[Dict[str, str]],
    k: int = K,
) -> List[str]:
    labeled_claims = [
        (claim_id_for_label(label, index), label)
        for index, label in enumerate(expected_claims)
        if label.get("evidence_quote")
    ]
    missing: List[str] = []
    top_items = retrieved_items[:k]
    for claim_id, label in labeled_claims:
        if not any(claim_matches_doc(label, item) for item in top_items):
            missing.append(claim_id)
    return missing


def claim_quote_recall_at_k(
    retrieved_items: Sequence[Any],
    expected_claims: Sequence[Dict[str, str]],
    k: int = K,
) -> float | None:
    labeled_count = sum(1 for label in expected_claims if label.get("evidence_quote"))
    if not labeled_count:
        return None
    missing_count = len(missing_claim_ids_at_k(retrieved_items, expected_claims, k=k))
    return (labeled_count - missing_count) / labeled_count


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
    expected_sections: Sequence[Dict[str, str]],
    expected_claims: Sequence[Dict[str, str]],
    k: int = K,
) -> Dict[str, Any]:
    top_docs = docs[:k]
    retrieved_doc_ids = [doc_id_for_item(doc) for doc in top_docs]
    recall = recall_at_k(retrieved_doc_ids, expected_doc_ids, k=k)
    mrr = reciprocal_rank(retrieved_doc_ids, expected_doc_ids, k=k)
    ndcg = ndcg_at_k(retrieved_doc_ids, expected_doc_ids, k=k)
    section_recall = section_recall_at_k(top_docs, expected_sections, k=k)
    claim_recall = claim_quote_recall_at_k(top_docs, expected_claims, k=k)
    missing_claim_ids = missing_claim_ids_at_k(top_docs, expected_claims, k=k)
    return {
        "mode": mode,
        "retrieved_doc_ids": retrieved_doc_ids,
        "retrieved_docs": [
            compact_doc(doc, rank=index, mode=mode)
            for index, doc in enumerate(top_docs, start=1)
        ],
        "recall_at_4": recall,
        "hit": None if recall is None else recall > 0,
        "mrr": mrr,
        "ndcg_at_4": ndcg,
        "section_recall_at_4": section_recall,
        "section_hit": None if section_recall is None else section_recall > 0,
        "claim_quote_recall_at_4": claim_recall,
        "claim_hit": None if claim_recall is None else claim_recall > 0,
        "missing_claim_ids": missing_claim_ids,
        "duplicate_doc_rate": duplicate_doc_rate(retrieved_doc_ids),
        "unique_doc_count": len({doc_id for doc_id in retrieved_doc_ids if doc_id}),
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
        "expected_rag_sections": case.expected_rag_sections,
        "expected_rag_claims": case.expected_rag_claims,
        "benchmark_note": case.benchmark_note,
        "modes": {
            "vector_only": evaluate_mode(
                mode="vector_only",
                docs=vector_docs,
                expected_doc_ids=case.expected_rag_doc_ids,
                expected_sections=case.expected_rag_sections,
                expected_claims=case.expected_rag_claims,
                k=k,
            ),
            "metadata_rerank_no_dedup": evaluate_mode(
                mode="metadata_rerank_no_dedup",
                docs=rerank_no_dedup_docs,
                expected_doc_ids=case.expected_rag_doc_ids,
                expected_sections=case.expected_rag_sections,
                expected_claims=case.expected_rag_claims,
                k=k,
            ),
            "metadata_rerank_dedup": evaluate_mode(
                mode="metadata_rerank_dedup",
                docs=rerank_dedup_docs,
                expected_doc_ids=case.expected_rag_doc_ids,
                expected_sections=case.expected_rag_sections,
                expected_claims=case.expected_rag_claims,
                k=k,
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
    section_entries = [entry for entry in mode_entries if entry["section_recall_at_4"] is not None]
    section_hits = [1.0 if entry["section_hit"] else 0.0 for entry in section_entries]
    claim_entries = [entry for entry in mode_entries if entry["claim_quote_recall_at_4"] is not None]
    claim_hits = [1.0 if entry["claim_hit"] else 0.0 for entry in claim_entries]
    missing_claim_ids = sorted(
        {
            claim_id
            for entry in mode_entries
            for claim_id in entry.get("missing_claim_ids", [])
        }
    )
    return {
        "labeled_request_count": len(labeled_entries),
        "retrieval_hit_rate": round(statistics.fmean(hits), 4) if hits else None,
        "recall_at_4": mean_or_none([entry["recall_at_4"] for entry in mode_entries]),
        "mrr": mean_or_none([entry["mrr"] for entry in mode_entries]),
        "ndcg_at_4": mean_or_none([entry["ndcg_at_4"] for entry in mode_entries]),
        "section_labeled_request_count": len(section_entries),
        "section_hit_rate": round(statistics.fmean(section_hits), 4) if section_hits else None,
        "section_recall_at_4": mean_or_none([entry["section_recall_at_4"] for entry in mode_entries]),
        "claim_labeled_request_count": len(claim_entries),
        "claim_hit_rate": round(statistics.fmean(claim_hits), 4) if claim_hits else None,
        "claim_quote_recall_at_4": mean_or_none([entry["claim_quote_recall_at_4"] for entry in mode_entries]),
        "missing_claim_ids": missing_claim_ids,
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
            "dedup_vs_vector_section_recall_at_4": round(
                (by_mode["metadata_rerank_dedup"]["section_recall_at_4"] or 0.0)
                - (by_mode["vector_only"]["section_recall_at_4"] or 0.0),
                4,
            ),
            "dedup_vs_vector_claim_quote_recall_at_4": round(
                (by_mode["metadata_rerank_dedup"]["claim_quote_recall_at_4"] or 0.0)
                - (by_mode["vector_only"]["claim_quote_recall_at_4"] or 0.0),
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
