"""RAG services for local lightweight knowledge and Chroma-backed retrieval."""

from __future__ import annotations

import json
import os
import re
from datetime import date
from types import SimpleNamespace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from pydantic import BaseModel, Field

from ..config import get_settings
from ..models.langgraph_state import AttractionCandidate, RAGChunk
from ..models.schemas import TripRequest

RAG_STOPWORDS = {
    "a",
    "an",
    "and",
    "around",
    "back",
    "day",
    "days",
    "for",
    "hotel",
    "i",
    "in",
    "itinerary",
    "me",
    "near",
    "of",
    "on",
    "or",
    "plan",
    "route",
    "the",
    "to",
    "with",
    "without",
}

MAX_PACKED_SECTIONS_PER_DOC = 4
MAX_PACKED_CONTEXT_CHARS_PER_DOC = 6000
DEFAULT_STALE_AFTER_DAYS = 365
OFFICIAL_SOURCE_BOOST = 0.08
MISSING_SOURCE_URL_PENALTY = 0.08
STALE_SOURCE_PENALTY = 0.1


DEFAULT_LOCAL_KNOWLEDGE = [
    {
        "chunk_id": "travel-style-history-culture",
        "source": "local_travel_playbook",
        "title": "History and Culture Planning Guidance",
        "content": "For history and culture trips, prioritize museums, landmarks, and walkable districts. Avoid packing far-apart attractions into the same day.",
        "keywords": ["history", "culture", "museums", "landmarks", "历史文化", "博物馆", "古迹"],
    },
    {
        "chunk_id": "travel-style-food",
        "source": "local_travel_playbook",
        "title": "Food-Focused Planning Guidance",
        "content": "For food-focused trips, place lunch and dinner near dense neighborhoods or markets to reduce unnecessary cross-city travel.",
        "keywords": ["food", "restaurants", "markets", "citywalk", "美食", "小吃", "餐饮", "城市漫步"],
    },
    {
        "chunk_id": "travel-style-nature",
        "source": "local_travel_playbook",
        "title": "Outdoor and Nature Planning Guidance",
        "content": "For outdoor trips, consider weather, daylight, and physical load. Keep rainy-day outdoor stops shorter and add indoor alternatives.",
        "keywords": ["nature", "outdoors", "parks", "hiking", "photos", "自然风光", "公园", "徒步", "拍照"],
    },
    {
        "chunk_id": "transport-transit",
        "source": "local_travel_playbook",
        "title": "Public Transit Planning Guidance",
        "content": "For public-transit trips, group nearby attractions by day to reduce transfers, delays, and backtracking.",
        "keywords": ["public transit", "subway", "bus", "公共交通", "地铁", "公交"],
    },
    {
        "chunk_id": "accommodation-budget",
        "source": "local_travel_playbook",
        "title": "Budget Accommodation Guidance",
        "content": "For budget accommodation, prefer hotels near major transit hubs or the densest first/last-day sightseeing area.",
        "keywords": ["budget hotel", "hotel", "inn", "hostel", "经济型酒店", "快捷酒店", "民宿"],
    },
]


class KnowledgeDocument(BaseModel):
    """Canonical travel knowledge document used to build the RAG index."""

    doc_id: str
    country: str
    city: str
    district: str = ""
    theme: List[str] = Field(default_factory=list)
    poi_names: List[str] = Field(default_factory=list)
    best_for: List[str] = Field(default_factory=list)
    recommended_duration: str = ""
    seasonality: List[str] = Field(default_factory=list)
    transport_advice: List[str] = Field(default_factory=list)
    planning_tips: List[str] = Field(default_factory=list)
    source_type: str
    source_url: str
    language: str = "zh"
    last_verified_at: str = ""
    title: str
    content: str


class TravelRAGService:
    """Provides local baseline chunks and real Chroma-backed retrieval."""

    def __init__(
        self,
        knowledge_root: Optional[Path] = None,
        persist_directory: Optional[Path] = None,
        collection_name: str = "travel_knowledge",
        embedding_model: str = "text-embedding-3-small",
        local_corpus: Optional[List[Dict[str, Any]]] = None,
        embedding_function=None,
    ):
        backend_root = Path(__file__).resolve().parents[2]
        self.knowledge_root = knowledge_root or backend_root / "data" / "knowledge"
        self.persist_directory = persist_directory or backend_root / "data" / "index" / "chroma"
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.local_corpus = local_corpus or DEFAULT_LOCAL_KNOWLEDGE
        self._embedding_function = embedding_function
        self._vectorstore = None

    def retrieve_chunks(
        self,
        request: TripRequest,
        *,
        rag_mode: str = "local_lightweight",
        attraction_candidates: Optional[List[AttractionCandidate]] = None,
        k: int = 4,
    ) -> List[RAGChunk]:
        if rag_mode == "chroma_retrieval":
            return self.retrieve_chroma_chunks(request, attraction_candidates=attraction_candidates or [], k=k)
        return self.retrieve_local_chunks(request)

    def retrieve_local_chunks(self, request: TripRequest) -> List[RAGChunk]:
        request_terms = {
            request.city,
            request.transportation,
            request.accommodation,
            *(request.preferences or []),
        }
        chunks: List[RAGChunk] = []
        for item in self.local_corpus:
            keywords = set(item.get("keywords", []))
            score = len(request_terms & keywords)
            if score > 0:
                chunks.append(
                    RAGChunk(
                        chunk_id=str(item["chunk_id"]),
                        source=str(item["source"]),
                        title=str(item["title"]),
                        content=str(item["content"]),
                        metadata={"match_score": score, "keywords": list(keywords), "rag_backend": "local_lightweight"},
                    )
                )
        if not chunks:
            chunks.append(self._default_chunk(request.city, backend="local_lightweight"))
        return sorted(chunks, key=lambda c: c.metadata.get("match_score", 0), reverse=True)[:4]

    def retrieve_chroma_chunks(
        self,
        request: TripRequest,
        *,
        attraction_candidates: List[AttractionCandidate],
        k: int = 4,
    ) -> List[RAGChunk]:
        if not self.has_index():
            self.ensure_index()

        query = self.build_query(request, attraction_candidates)
        vectorstore = self._get_vectorstore()
        try:
            docs = vectorstore.similarity_search(
                query,
                k=max(k * 3, 12),
                filter={"city": request.city},
            )
        except Exception:
            docs = []

        chunks: List[RAGChunk] = []
        ranked_docs = self._rerank_and_dedup_docs(
            request=request,
            docs=docs,
            attraction_candidates=attraction_candidates,
            k=k,
        )
        for index, (doc, metadata) in enumerate(ranked_docs):
            chunk_id = str(metadata.get("chunk_id") or f"{metadata.get('doc_id', 'doc')}-{index}")
            chunks.append(
                RAGChunk(
                    chunk_id=chunk_id,
                    source=str(metadata.get("source_type", "official_travel_knowledge")),
                    title=str(metadata.get("title", metadata.get("doc_id", "travel-knowledge"))),
                    content=doc.page_content,
                    metadata={**metadata, "rag_backend": "chroma_retrieval"},
                )
            )
        if not chunks:
            return [self._default_chunk(request.city, backend="chroma_fallback_local")]
        return chunks

    def build_query(
        self,
        request: TripRequest,
        attraction_candidates: Optional[Iterable[AttractionCandidate]] = None,
    ) -> str:
        terms: List[str] = [request.city, request.transportation, request.accommodation, f"{request.travel_days} days"]
        terms.extend(request.preferences or [])
        if attraction_candidates:
            names = [candidate.name for candidate in list(attraction_candidates)[:3] if candidate.name]
            terms.extend(names)
        if request.free_text_input:
            terms.append(request.free_text_input)
        return " ".join(term for term in terms if term)

    def _rerank_and_dedup_docs(
        self,
        *,
        request: TripRequest,
        docs: List[Any],
        attraction_candidates: List[AttractionCandidate],
        k: int,
    ) -> List[tuple[Any, Dict[str, Any]]]:
        request_terms = self._request_terms(request, attraction_candidates)
        scored_docs = []
        for vector_rank, doc in enumerate(docs, start=1):
            metadata = dict(doc.metadata)
            score, reasons = self._score_rag_doc(
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
        return self._pack_ranked_docs(scored_docs, k=k)

    def _pack_ranked_docs(
        self,
        scored_docs: List[tuple[float, int, Any, Dict[str, Any]]],
        *,
        k: int,
    ) -> List[tuple[Any, Dict[str, Any]]]:
        grouped: Dict[str, List[tuple[float, int, Any, Dict[str, Any]]]] = {}
        for score, vector_rank, doc, metadata in scored_docs:
            group_key = self._doc_group_key(metadata, vector_rank)
            grouped.setdefault(group_key, []).append((score, vector_rank, doc, metadata))

        selected_groups: List[tuple[float, int, str]] = []
        seen_groups = set()
        for score, vector_rank, _, metadata in scored_docs:
            group_key = self._doc_group_key(metadata, vector_rank)
            if group_key in seen_groups:
                continue
            seen_groups.add(group_key)
            selected_groups.append((score, vector_rank, group_key))
            if len(selected_groups) >= k:
                break

        selected: List[tuple[Any, Dict[str, Any]]] = []
        for _, _, group_key in selected_groups:
            packed = self._pack_doc_group(grouped[group_key])
            if packed is not None:
                selected.append(packed)

        for dedup_rank, (_, metadata) in enumerate(selected, start=1):
            metadata["dedup_rank"] = dedup_rank
        return selected

    def _doc_group_key(self, metadata: Dict[str, Any], vector_rank: int) -> str:
        doc_id = str(metadata.get("doc_id", ""))
        if doc_id:
            return f"doc:{doc_id}"
        chunk_id = str(metadata.get("chunk_id", ""))
        if chunk_id:
            return f"chunk:{chunk_id}"
        return f"anonymous:{vector_rank}"

    def _pack_doc_group(
        self,
        group_entries: List[tuple[float, int, Any, Dict[str, Any]]],
    ) -> tuple[Any, Dict[str, Any]] | None:
        if not group_entries:
            return None

        ordered = sorted(group_entries, key=lambda item: (-item[0], item[1]))
        packed_entries: List[tuple[float, int, Any, Dict[str, Any]]] = []
        seen_chunk_ids = set()
        total_chars = 0

        for score, vector_rank, doc, metadata in ordered:
            chunk_id = str(metadata.get("chunk_id", ""))
            if chunk_id and chunk_id in seen_chunk_ids:
                continue
            section_text = self._format_packed_section(doc, metadata)
            projected_chars = total_chars + len(section_text)
            if (
                packed_entries
                and projected_chars > MAX_PACKED_CONTEXT_CHARS_PER_DOC
            ):
                continue
            packed_entries.append((score, vector_rank, doc, metadata))
            total_chars += len(section_text)
            if chunk_id:
                seen_chunk_ids.add(chunk_id)
            if len(packed_entries) >= MAX_PACKED_SECTIONS_PER_DOC:
                break

        primary_score, primary_vector_rank, primary_doc, primary_metadata = packed_entries[0]
        packed_metadata = dict(primary_metadata)
        sections = [
            str(metadata.get("section", ""))
            for _, _, _, metadata in packed_entries
            if metadata.get("section")
        ]
        chunk_ids = [
            str(metadata.get("chunk_id", ""))
            for _, _, _, metadata in packed_entries
            if metadata.get("chunk_id")
        ]
        packed_text = "\n\n".join(
            self._format_packed_section(doc, metadata)
            for _, _, doc, metadata in packed_entries
        )

        if len(packed_entries) > 1:
            doc_id = str(packed_metadata.get("doc_id", "doc"))
            packed_metadata["chunk_id"] = f"{doc_id}-evidence-packet"
        packed_metadata["section"] = sections[0] if sections else packed_metadata.get("section", "")
        packed_metadata["sections"] = sections
        packed_metadata["packed_section_count"] = len(packed_entries)
        packed_metadata["packed_chunk_ids"] = chunk_ids
        packed_metadata["packed_vector_ranks"] = [
            vector_rank for _, vector_rank, _, _ in packed_entries
        ]
        packed_metadata["packed_rerank_scores"] = [
            round(score, 4) for score, _, _, _ in packed_entries
        ]
        packed_metadata["rerank_score"] = round(primary_score, 4)
        packed_metadata["vector_rank"] = primary_vector_rank

        return self._new_document_like(primary_doc, packed_text, packed_metadata), packed_metadata

    def _format_packed_section(self, doc: Any, metadata: Dict[str, Any]) -> str:
        section = str(metadata.get("section", "")).strip()
        label = section or "section"
        return f"### {label}\n{str(doc.page_content).strip()}"

    def _new_document_like(self, source_doc: Any, page_content: str, metadata: Dict[str, Any]) -> Any:
        try:
            from langchain_core.documents import Document

            return Document(page_content=page_content, metadata=metadata)
        except Exception:
            return SimpleNamespace(page_content=page_content, metadata=metadata)

    def _score_rag_doc(
        self,
        *,
        request: TripRequest,
        request_terms: set[str],
        doc: Any,
        metadata: Dict[str, Any],
        vector_rank: int,
    ) -> tuple[float, List[str]]:
        score = max(0.0, 1.0 - ((vector_rank - 1) * 0.04))
        reasons = [f"vector_rank:{vector_rank}"]

        if metadata.get("city") == request.city:
            score += 0.3
            reasons.append("city_exact")

        request_language = "en" if request.city.isascii() else "zh"
        if metadata.get("language") == request_language:
            score += 0.1
            reasons.append(f"language:{request_language}")

        source_adjustment, source_reasons = self._source_quality_adjustments(metadata)
        score += source_adjustment
        reasons.extend(source_reasons)

        theme_terms = self._tokenize(str(metadata.get("theme", "")))
        theme_overlap = request_terms & theme_terms
        if theme_overlap:
            score += min(0.75, 0.25 * len(theme_overlap))
            reasons.append(f"theme_overlap:{','.join(sorted(theme_overlap))}")

        poi_terms = self._tokenize(str(metadata.get("poi_names", "")))
        poi_overlap = request_terms & poi_terms
        if poi_overlap:
            score += min(0.45, 0.15 * len(poi_overlap))
            reasons.append(f"poi_overlap:{','.join(sorted(poi_overlap))}")

        content_terms = self._tokenize(
            f"{metadata.get('title', '')} {doc.page_content}"
        )
        content_overlap = request_terms & content_terms
        if content_overlap:
            score += min(0.5, 0.05 * len(content_overlap))
            reasons.append(f"content_overlap:{','.join(sorted(content_overlap)[:8])}")

        return score, reasons

    def _source_quality_adjustments(self, metadata: Dict[str, Any]) -> tuple[float, List[str]]:
        adjustment = 0.0
        reasons: List[str] = []

        source_type = str(metadata.get("source_type", "")).strip().lower()
        if self._is_official_source_type(source_type):
            adjustment += OFFICIAL_SOURCE_BOOST
            reasons.append(f"source_quality:official:+{OFFICIAL_SOURCE_BOOST:.2f}")

        source_url = str(metadata.get("source_url", "")).strip()
        if not source_url:
            adjustment -= MISSING_SOURCE_URL_PENALTY
            reasons.append(f"source_quality:missing_source_url:-{MISSING_SOURCE_URL_PENALTY:.2f}")

        verified_at = self._parse_verified_date(metadata.get("last_verified_at"))
        stale_after_days = self._stale_after_days()
        if verified_at is not None:
            age_days = (date.today() - verified_at).days
            if age_days > stale_after_days:
                adjustment -= STALE_SOURCE_PENALTY
                reasons.append(f"source_quality:stale:{age_days}d:-{STALE_SOURCE_PENALTY:.2f}")

        return adjustment, reasons

    def _is_official_source_type(self, source_type: str) -> bool:
        official_markers = (
            "official",
            "government",
            "tourism_portal",
            "tourism_board",
            "visitor_bureau",
        )
        return any(marker in source_type for marker in official_markers)

    def _stale_after_days(self) -> int:
        raw_value = os.getenv("RAG_STALE_AFTER_DAYS", str(DEFAULT_STALE_AFTER_DAYS))
        try:
            parsed = int(raw_value)
        except ValueError:
            return DEFAULT_STALE_AFTER_DAYS
        return parsed if parsed > 0 else DEFAULT_STALE_AFTER_DAYS

    def _parse_verified_date(self, value: Any) -> Optional[date]:
        if value is None:
            return None
        raw_value = str(value).strip()
        if not raw_value:
            return None
        try:
            return date.fromisoformat(raw_value[:10])
        except ValueError:
            return None

    def _request_terms(
        self,
        request: TripRequest,
        attraction_candidates: List[AttractionCandidate],
    ) -> set[str]:
        values = [
            request.city,
            request.transportation,
            request.accommodation,
            request.free_text_input,
            *(request.preferences or []),
            *(candidate.name for candidate in attraction_candidates[:5] if candidate.name),
        ]
        return self._tokenize(" ".join(value for value in values if value))

    def _tokenize(self, value: str) -> set[str]:
        terms = set()
        for token in re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]{2,}", value.lower()):
            if len(token) < 2 or token in RAG_STOPWORDS:
                continue
            terms.add(token)
        return terms

    def has_index(self) -> bool:
        return self.persist_directory.exists() and any(self.persist_directory.iterdir())

    def ensure_index(self, force_rebuild: bool = False) -> None:
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        if self.has_index() and not force_rebuild:
            return

        try:
            from langchain_chroma import Chroma
            from langchain_core.documents import Document
        except Exception as exc:
            raise RuntimeError("Chroma dependencies are not installed") from exc

        embeddings = self._get_embeddings()
        if force_rebuild and self.persist_directory.exists():
            for child in self.persist_directory.iterdir():
                if child.is_file():
                    child.unlink()
                else:
                    import shutil

                    shutil.rmtree(child)
            self.persist_directory.mkdir(parents=True, exist_ok=True)

        documents: List[Document] = []
        for doc in self.load_knowledge_docs():
            for section_name, section_text in self._iter_document_sections(doc):
                if not section_text.strip():
                    continue
                metadata = {
                    "chunk_id": f"{doc.doc_id}-{section_name}",
                    "doc_id": doc.doc_id,
                    "country": doc.country,
                    "city": doc.city,
                    "district": doc.district,
                    "theme": ",".join(doc.theme),
                    "poi_names": ",".join(doc.poi_names),
                    "source_type": doc.source_type,
                    "source_url": doc.source_url,
                    "language": doc.language,
                    "section": section_name,
                    "last_verified_at": doc.last_verified_at,
                    "title": doc.title,
                }
                documents.append(Document(page_content=section_text, metadata=metadata))

        if not documents:
            raise RuntimeError(f"No knowledge documents found under {self.knowledge_root}")

        if force_rebuild:
            try:
                existing = Chroma(
                    collection_name=self.collection_name,
                    persist_directory=str(self.persist_directory),
                    embedding_function=embeddings,
                )
                existing.delete_collection()
            except Exception:
                pass

        self._vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name=self.collection_name,
            persist_directory=str(self.persist_directory),
        )

    def load_knowledge_docs(self) -> List[KnowledgeDocument]:
        documents: List[KnowledgeDocument] = []
        if not self.knowledge_root.exists():
            return documents
        for path in sorted(self.knowledge_root.rglob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                documents.extend(KnowledgeDocument(**item) for item in payload)
            else:
                documents.append(KnowledgeDocument(**payload))
        return documents

    def _iter_document_sections(self, doc: KnowledgeDocument) -> Iterable[tuple[str, str]]:
        overview = doc.content.strip()
        planning = "\n".join(f"- {tip}" for tip in doc.planning_tips if tip).strip()
        transport = "\n".join(f"- {tip}" for tip in doc.transport_advice if tip).strip()
        seasonality = "\n".join(f"- {tip}" for tip in doc.seasonality if tip).strip()

        header = [
            f"City: {doc.city}",
            f"District: {doc.district}" if doc.district else "",
            f"Themes: {', '.join(doc.theme)}" if doc.theme else "",
            f"Best for: {', '.join(doc.best_for)}" if doc.best_for else "",
            f"Recommended duration: {doc.recommended_duration}" if doc.recommended_duration else "",
        ]
        header_text = "\n".join(line for line in header if line)

        if overview:
            yield "overview", f"{doc.title}\n{header_text}\n\n{overview}".strip()
        if planning:
            yield "planning_tips", f"{doc.title}\nPlanning tips:\n{planning}".strip()
        if transport:
            yield "transport", f"{doc.title}\nTransport advice:\n{transport}".strip()
        if seasonality:
            yield "seasonality", f"{doc.title}\nSeasonality:\n{seasonality}".strip()

    def _default_chunk(self, city: str, backend: str) -> RAGChunk:
        return RAGChunk(
            chunk_id=f"default-{city}",
            source="local_travel_playbook",
            title="General Itinerary Planning Guidance",
            content="Group nearby attractions on the same day, keep daily pacing realistic, and use weather to balance indoor and outdoor activities.",
            metadata={"match_score": 0, "city": city, "rag_backend": backend},
        )

    def _get_embeddings(self):
        if self._embedding_function is not None:
            return self._embedding_function

        try:
            from langchain_openai import OpenAIEmbeddings
        except Exception as exc:
            raise RuntimeError("langchain-openai is not installed") from exc

        settings = get_settings()
        api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or settings.openai_api_key
        base_url = os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL") or settings.openai_base_url
        return OpenAIEmbeddings(
            model=self.embedding_model,
            api_key=api_key,
            base_url=base_url,
        )

    def _get_vectorstore(self):
        if self._vectorstore is not None:
            return self._vectorstore

        try:
            from langchain_chroma import Chroma
        except Exception as exc:
            raise RuntimeError("Chroma dependencies are not installed") from exc

        self._vectorstore = Chroma(
            collection_name=self.collection_name,
            persist_directory=str(self.persist_directory),
            embedding_function=self._get_embeddings(),
        )
        return self._vectorstore


_rag_service: Optional[TravelRAGService] = None


def get_rag_service() -> TravelRAGService:
    global _rag_service
    if _rag_service is None:
        _rag_service = TravelRAGService()
    return _rag_service
