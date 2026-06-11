"""RAG services for local lightweight knowledge and Chroma-backed retrieval."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from pydantic import BaseModel, Field

from ..config import get_settings
from ..models.langgraph_state import AttractionCandidate, RAGChunk
from ..models.schemas import TripRequest


DEFAULT_LOCAL_KNOWLEDGE = [
    {
        "chunk_id": "travel-style-history-culture",
        "source": "local_travel_playbook",
        "title": "历史文化偏好规划建议",
        "content": "历史文化类行程优先安排博物馆、古迹和步行友好的片区，避免把远距离景点塞进同一天。",
        "keywords": ["历史文化", "博物馆", "古迹"],
    },
    {
        "chunk_id": "travel-style-food",
        "source": "local_travel_playbook",
        "title": "美食偏好规划建议",
        "content": "美食偏好行程适合把午餐和晚餐嵌入热门街区，减少跨城移动，优先安排在高密度商圈附近。",
        "keywords": ["美食", "小吃", "餐饮", "城市漫步", "citywalk"],
    },
    {
        "chunk_id": "travel-style-nature",
        "source": "local_travel_playbook",
        "title": "自然风光偏好规划建议",
        "content": "自然风光类行程应优先关注天气与体力负荷，晴天安排长时间户外活动，阴雨天缩短户外停留。",
        "keywords": ["自然风光", "公园", "徒步", "拍照"],
    },
    {
        "chunk_id": "transport-transit",
        "source": "local_travel_playbook",
        "title": "公共交通行程建议",
        "content": "公共交通行程应尽量减少跨区跳跃，把相近景点聚合在同一天，降低换乘成本和延误风险。",
        "keywords": ["公共交通", "地铁", "公交"],
    },
    {
        "chunk_id": "accommodation-budget",
        "source": "local_travel_playbook",
        "title": "经济型住宿建议",
        "content": "经济型住宿推荐优先靠近主要交通枢纽或第一天/最后一天景点密集片区，以平衡预算和通勤时间。",
        "keywords": ["经济型酒店", "快捷酒店", "民宿"],
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
        collection_name: str = "travel_knowledge_cn",
        embedding_model: str = "text-embedding-3-small",
        local_corpus: Optional[List[Dict[str, Any]]] = None,
        embedding_function=None,
    ):
        backend_root = Path(__file__).resolve().parents[2]
        self.knowledge_root = knowledge_root or backend_root / "data" / "knowledge" / "china"
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
            docs = vectorstore.similarity_search(query, k=k, filter={"city": request.city})
            if not docs:
                docs = vectorstore.similarity_search(query, k=k)
        except Exception:
            docs = []

        chunks: List[RAGChunk] = []
        for index, doc in enumerate(docs):
            metadata = dict(doc.metadata)
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
        terms: List[str] = [request.city, request.transportation, request.accommodation, f"{request.travel_days}天"]
        terms.extend(request.preferences or [])
        if attraction_candidates:
            names = [candidate.name for candidate in list(attraction_candidates)[:3] if candidate.name]
            terms.extend(names)
        if request.free_text_input:
            terms.append(request.free_text_input)
        return " ".join(term for term in terms if term)

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
            f"城市: {doc.city}",
            f"片区: {doc.district}" if doc.district else "",
            f"主题: {', '.join(doc.theme)}" if doc.theme else "",
            f"适合人群: {', '.join(doc.best_for)}" if doc.best_for else "",
            f"推荐时长: {doc.recommended_duration}" if doc.recommended_duration else "",
        ]
        header_text = "\n".join(line for line in header if line)

        if overview:
            yield "overview", f"{doc.title}\n{header_text}\n\n{overview}".strip()
        if planning:
            yield "planning_tips", f"{doc.title}\n行程建议:\n{planning}".strip()
        if transport:
            yield "transport", f"{doc.title}\n交通建议:\n{transport}".strip()
        if seasonality:
            yield "seasonality", f"{doc.title}\n季节建议:\n{seasonality}".strip()

    def _default_chunk(self, city: str, backend: str) -> RAGChunk:
        return RAGChunk(
            chunk_id=f"default-{city}",
            source="local_travel_playbook",
            title="默认行程规划建议",
            content="优先把相近景点放在同一天，控制每天景点数量，并结合天气安排室内外活动。",
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
