"""Human-in-the-loop ingestion utilities for travel RAG knowledge."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable, List, Optional

from pydantic import BaseModel, Field, HttpUrl, field_validator

from .rag_service import KnowledgeDocument

CITY_SLUGS = {
    "北京": "beijing",
    "上海": "shanghai",
    "杭州": "hangzhou",
    "广州": "guangzhou",
}


class SourceManifestEntry(BaseModel):
    """Source page metadata used to create review-ready RAG drafts."""

    source_id: str
    country: str = "CN"
    city: str
    source_url: HttpUrl
    source_type: str
    title: str
    theme: List[str] = Field(default_factory=list)
    poi_names: List[str] = Field(default_factory=list)
    district: str = ""
    language: str = "zh"
    best_for: List[str] = Field(default_factory=list)
    recommended_duration: str = ""

    @field_validator("source_id", "city", "source_type", "title")
    @classmethod
    def non_empty_text(cls, value: str) -> str:
        value = (value or "").strip()
        if not value:
            raise ValueError("field cannot be empty")
        return value


class DraftKnowledgeDocument(KnowledgeDocument):
    """Reviewable draft that can be promoted only after human approval."""

    review_status: str = "draft"
    reviewer: str = ""
    review_notes: str = ""
    source_id: str = ""
    raw_html_path: str = ""
    raw_text_path: str = ""
    fetched_at: str = ""

    def to_knowledge_document(self) -> KnowledgeDocument:
        return KnowledgeDocument(
            doc_id=self.doc_id,
            country=self.country,
            city=self.city,
            district=self.district,
            theme=self.theme,
            poi_names=self.poi_names,
            best_for=self.best_for,
            recommended_duration=self.recommended_duration,
            seasonality=self.seasonality,
            transport_advice=self.transport_advice,
            planning_tips=self.planning_tips,
            source_type=self.source_type,
            source_url=self.source_url,
            language=self.language,
            last_verified_at=self.last_verified_at,
            title=self.title,
            content=self.content,
        )


class _ReadableTextParser(HTMLParser):
    ignored_tags = {"script", "style", "noscript", "svg", "header", "footer", "nav"}
    block_tags = {"p", "div", "section", "article", "main", "li", "br", "h1", "h2", "h3", "h4"}

    def __init__(self) -> None:
        super().__init__()
        self._ignored_depth = 0
        self._parts: List[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in self.ignored_tags:
            self._ignored_depth += 1
        if tag in self.block_tags:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.ignored_tags and self._ignored_depth > 0:
            self._ignored_depth -= 1
        if tag in self.block_tags:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._ignored_depth:
            return
        text = data.strip()
        if text:
            self._parts.append(text)

    def text(self) -> str:
        raw = " ".join(self._parts)
        raw = re.sub(r"[ \t\r\f\v]+", " ", raw)
        raw = re.sub(r"\n\s+", "\n", raw)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        return "\n".join(lines)


def slugify(value: str) -> str:
    """Return a filesystem-safe slug while preserving readable CJK text."""
    value = (value or "").strip().lower()
    value = re.sub(r"[^\w\u4e00-\u9fff.-]+", "-", value)
    value = re.sub(r"-{2,}", "-", value).strip("-")
    return value or "item"


def city_slug(city: str) -> str:
    return CITY_SLUGS.get(city, slugify(city))


def load_manifest(path: Path) -> List[SourceManifestEntry]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("manifest must be a JSON list")
    return [SourceManifestEntry(**item) for item in payload]


def extract_readable_text(html: str) -> str:
    parser = _ReadableTextParser()
    parser.feed(html)
    return parser.text()


def make_doc_id(entry: SourceManifestEntry) -> str:
    return f"{city_slug(entry.city)}-{slugify(entry.source_id)}"


def build_draft_document(
    *,
    entry: SourceManifestEntry,
    extracted_text: str,
    raw_html_path: Path,
    raw_text_path: Path,
    fetched_at: Optional[str] = None,
) -> DraftKnowledgeDocument:
    now = fetched_at or datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    content = summarize_content(extracted_text)
    return DraftKnowledgeDocument(
        doc_id=make_doc_id(entry),
        country=entry.country,
        city=entry.city,
        district=entry.district,
        theme=entry.theme,
        poi_names=entry.poi_names,
        best_for=entry.best_for,
        recommended_duration=entry.recommended_duration,
        seasonality=[],
        transport_advice=[],
        planning_tips=[],
        source_type=entry.source_type,
        source_url=str(entry.source_url),
        language=entry.language,
        last_verified_at=now[:10],
        title=entry.title,
        content=content,
        review_status="draft",
        reviewer="",
        review_notes="",
        source_id=entry.source_id,
        raw_html_path=str(raw_html_path),
        raw_text_path=str(raw_text_path),
        fetched_at=now,
    )


def summarize_content(text: str, max_chars: int = 1200) -> str:
    text = re.sub(r"\s+", " ", (text or "")).strip()
    if len(text) <= max_chars:
        return text
    clipped = text[:max_chars].rsplit("。", 1)[0].strip()
    if len(clipped) < max_chars * 0.5:
        clipped = text[:max_chars].strip()
    return clipped


def read_draft(path: Path) -> DraftKnowledgeDocument:
    return DraftKnowledgeDocument(**json.loads(path.read_text(encoding="utf-8")))


def write_draft(path: Path, draft: DraftKnowledgeDocument) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(draft.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def approved_drafts(paths: Iterable[Path]) -> List[DraftKnowledgeDocument]:
    drafts = []
    for path in paths:
        draft = read_draft(path)
        if draft.review_status == "approved":
            drafts.append(draft)
    return drafts


def merge_knowledge_docs(
    *,
    knowledge_file: Path,
    docs: List[KnowledgeDocument],
    overwrite: bool = False,
) -> int:
    existing_payload = []
    if knowledge_file.exists():
        existing_payload = json.loads(knowledge_file.read_text(encoding="utf-8"))
        if not isinstance(existing_payload, list):
            existing_payload = [existing_payload]

    by_id = {item["doc_id"]: item for item in existing_payload}
    promoted = 0
    for doc in docs:
        item = doc.model_dump()
        doc_id = item["doc_id"]
        if doc_id in by_id and not overwrite:
            continue
        by_id[doc_id] = item
        promoted += 1

    knowledge_file.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(by_id.values(), key=lambda item: item["doc_id"])
    knowledge_file.write_text(json.dumps(ordered, ensure_ascii=False, indent=2), encoding="utf-8")
    return promoted
