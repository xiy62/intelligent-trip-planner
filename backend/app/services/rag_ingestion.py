"""Human-in-the-loop ingestion utilities for travel RAG knowledge."""

from __future__ import annotations

import json
import re
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from html import unescape
from pathlib import Path
from typing import Iterable, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, HttpUrl, field_validator

from .rag_service import KnowledgeDocument

CITY_SLUGS = {
    "北京": "beijing",
    "上海": "shanghai",
    "杭州": "hangzhou",
    "广州": "guangzhou",
    "New York": "new_york",
    "San Francisco": "san_francisco",
    "Chicago": "chicago",
}

COUNTRY_SLUGS = {
    "CN": "china",
    "US": "us",
}

BACKEND_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_UPLOAD_ROOT = BACKEND_ROOT / "data" / "uploads"
DEFAULT_EXTRACTED_ROOT = BACKEND_ROOT / "data" / "extracted"
DEFAULT_DRAFT_ROOT = BACKEND_ROOT / "data" / "drafts"
DEFAULT_KNOWLEDGE_ROOT = BACKEND_ROOT / "data" / "knowledge"
DEFAULT_JOB_DB_PATH = BACKEND_ROOT / "data" / "ingestion" / "rag_ingestion_jobs.sqlite3"
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
AI_PREFILL_SOURCE_CHAR_LIMIT = 45_000

TRAVEL_RELEVANCE_TERMS = {
    "accessibility",
    "activity",
    "admission",
    "affordable",
    "architecture",
    "attraction",
    "bike",
    "budget",
    "bus",
    "city",
    "class",
    "cultural",
    "dining",
    "district",
    "duration",
    "event",
    "family",
    "ferry",
    "food",
    "free",
    "gallery",
    "guide",
    "hotel",
    "itinerary",
    "landmark",
    "market",
    "museum",
    "neighborhood",
    "park",
    "restaurant",
    "route",
    "safety",
    "season",
    "shopping",
    "show",
    "subway",
    "ticket",
    "tour",
    "transit",
    "transport",
    "travel",
    "visitor",
    "walk",
}

NOISE_LINE_PATTERNS = [
    re.compile(r"^https?://", re.I),
    re.compile(r"^\d+\s*/\s*\d+$"),
    re.compile(r"^\d{4}/\d{1,2}/\d{1,2}\s+\d{1,2}:\d{2}$"),
    re.compile(r"^skip to main content$", re.I),
    re.compile(r"^now in .+$", re.I),
    re.compile(r"^things to do$", re.I),
    re.compile(r"^eat\s*&\s*drink$", re.I),
    re.compile(r"^where to stay$", re.I),
    re.compile(r"^maps\s*&\s*guides$", re.I),
    re.compile(r"^business in .+$", re.I),
]


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


class RAGPrefillEvidence(BaseModel):
    """Short review-only evidence for one AI-filled field."""

    field: str
    suggestion: str
    evidence: str


class RAGPrefillSuggestion(BaseModel):
    """Structured LLM output for reviewable RAG draft suggestions."""

    content: str
    theme: List[str] = Field(default_factory=list)
    poi_names: List[str] = Field(default_factory=list)
    best_for: List[str] = Field(default_factory=list)
    recommended_duration: str = ""
    seasonality: List[str] = Field(default_factory=list)
    transport_advice: List[str] = Field(default_factory=list)
    planning_tips: List[str] = Field(default_factory=list)
    field_evidence: List[RAGPrefillEvidence] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


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


def country_slug(country: str) -> str:
    return COUNTRY_SLUGS.get((country or "").upper(), slugify(country))


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


def draft_path_for(
    *,
    draft_root: Path,
    country: str,
    city: str,
    doc_id: str,
) -> Path:
    return draft_root / country_slug(country) / city_slug(city) / f"{slugify(doc_id)}.json"


def find_draft_path(draft_root: Path, draft_id: str) -> Path:
    safe_id = slugify(draft_id)
    for path in sorted(draft_root.rglob("*.json")):
        if path.stem == safe_id:
            return path
        try:
            if read_draft(path).doc_id == draft_id:
                return path
        except Exception:
            continue
    raise FileNotFoundError(f"draft_id does not exist: {draft_id}")


def list_draft_paths(
    *,
    draft_root: Path,
    country: Optional[str] = None,
    city: Optional[str] = None,
    review_status: Optional[str] = None,
) -> List[Path]:
    base = draft_root / country_slug(country) if country else draft_root
    paths = sorted(base.rglob("*.json")) if base.exists() else []
    if not city and not review_status:
        return paths
    selected = []
    for path in paths:
        try:
            draft = read_draft(path)
        except Exception:
            continue
        if city and draft.city != city:
            continue
        if review_status and draft.review_status != review_status:
            continue
        selected.append(path)
    return selected


def extract_uploaded_source(source_path: Path) -> str:
    suffix = source_path.suffix.lower()
    if suffix in {".md", ".markdown", ".txt"}:
        return source_path.read_text(encoding="utf-8")
    if suffix == ".pdf":
        try:
            from markitdown import MarkItDown
        except Exception as exc:
            raise RuntimeError("markitdown is required for PDF extraction") from exc
        try:
            result = MarkItDown().convert(str(source_path))
        except Exception as exc:
            raise RuntimeError(f"PDF extraction failed: {exc}") from exc
        text = getattr(result, "text_content", "") or str(result)
        if not text.strip():
            raise RuntimeError("PDF extraction produced no text")
        return text
    raise ValueError("unsupported source file type")


def write_uploaded_source(
    *,
    filename: str,
    content: bytes,
    country: str,
    city: str,
    source_id: str,
    upload_root: Path = DEFAULT_UPLOAD_ROOT,
) -> Path:
    suffix = Path(filename).suffix.lower()
    if suffix not in {".pdf", ".md", ".markdown", ".txt"}:
        raise ValueError("Only .pdf, .md, and .txt uploads are supported")
    if len(content) > MAX_UPLOAD_BYTES:
        raise ValueError("Uploaded file must be 50MB or smaller")
    target_dir = upload_root / country_slug(country) / city_slug(city)
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / f"{slugify(source_id)}{suffix}"
    target_path.write_bytes(content)
    return target_path


def write_extracted_text(
    *,
    extracted_text: str,
    country: str,
    city: str,
    source_id: str,
    extracted_root: Path = DEFAULT_EXTRACTED_ROOT,
) -> Path:
    target_dir = extracted_root / country_slug(country) / city_slug(city)
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / f"{slugify(source_id)}.md"
    target_path.write_text(extracted_text, encoding="utf-8")
    return target_path


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


def build_ai_prefill_source_packet(
    *,
    draft: DraftKnowledgeDocument,
    extracted_text: str,
    max_chars: int = AI_PREFILL_SOURCE_CHAR_LIMIT,
) -> tuple[str, int, int, List[str]]:
    """Return cleaned source text compact enough for an LLM prefill prompt."""
    source_char_count = len(extracted_text or "")
    paragraphs = clean_source_paragraphs(extracted_text)
    warnings: List[str] = []

    if not paragraphs:
        return "", source_char_count, 0, ["No usable source paragraphs remained after cleaning."]

    cleaned_text = "\n\n".join(paragraphs)
    if len(cleaned_text) <= max_chars:
        return cleaned_text, source_char_count, len(cleaned_text), warnings

    selected = select_relevant_paragraphs(draft=draft, paragraphs=paragraphs, max_chars=max_chars)
    packet = "\n\n".join(selected).strip()
    warnings.append(
        f"Source text was compacted from {len(cleaned_text)} cleaned characters to {len(packet)} characters."
    )
    return packet, source_char_count, len(packet), warnings


def clean_source_paragraphs(text: str) -> List[str]:
    """Remove common PDF/web boilerplate while preserving reviewable source paragraphs."""
    text = unescape(text or "").replace("\r", "\n")
    raw_lines = text.splitlines()
    paragraphs: List[str] = []
    current: List[str] = []
    seen_paragraphs = set()
    seen_lines: dict[str, int] = {}

    def flush() -> None:
        if not current:
            return
        paragraph = re.sub(r"\s+", " ", " ".join(current)).strip()
        current.clear()
        if len(paragraph) < 25:
            return
        key = paragraph.lower()
        if key in seen_paragraphs:
            return
        seen_paragraphs.add(key)
        paragraphs.append(paragraph)

    for raw_line in raw_lines:
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            flush()
            continue
        if is_noise_line(line):
            flush()
            continue
        line_key = line.lower()
        seen_lines[line_key] = seen_lines.get(line_key, 0) + 1
        if seen_lines[line_key] > 2:
            continue
        current.append(line)
    flush()
    return paragraphs


def is_noise_line(line: str) -> bool:
    if line in {".", "|", "•"}:
        return True
    if len(line) <= 2:
        return True
    return any(pattern.search(line) for pattern in NOISE_LINE_PATTERNS)


def select_relevant_paragraphs(
    *,
    draft: DraftKnowledgeDocument,
    paragraphs: List[str],
    max_chars: int,
) -> List[str]:
    terms = metadata_terms(draft)
    selected_indexes = set()
    scored = []
    for index, paragraph in enumerate(paragraphs):
        paragraph_terms = set(tokenize_for_prefill(paragraph))
        score = len(paragraph_terms & terms)
        score += min(4, len(paragraph_terms & TRAVEL_RELEVANCE_TERMS))
        if index < 3:
            score += 2
        scored.append((score, index, paragraph))

    for _, index, _ in sorted(scored, key=lambda item: (-item[0], item[1])):
        selected_indexes.add(index)
        candidate = "\n\n".join(paragraphs[i] for i in sorted(selected_indexes))
        if len(candidate) >= max_chars:
            break

    selected: List[str] = []
    total = 0
    for index in sorted(selected_indexes):
        paragraph = paragraphs[index]
        if total + len(paragraph) + 2 > max_chars and selected:
            continue
        selected.append(paragraph)
        total += len(paragraph) + 2
        if total >= max_chars:
            break
    return selected


def metadata_terms(draft: DraftKnowledgeDocument) -> set[str]:
    values = [
        draft.country,
        draft.city,
        draft.district,
        draft.title,
        draft.recommended_duration,
        *draft.theme,
        *draft.poi_names,
        *draft.best_for,
    ]
    return set(tokenize_for_prefill(" ".join(item for item in values if item))) | TRAVEL_RELEVANCE_TERMS


def tokenize_for_prefill(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z'-]{2,}", (text or "").lower())


def apply_prefill_suggestion(
    *,
    draft: DraftKnowledgeDocument,
    suggestion: RAGPrefillSuggestion,
) -> DraftKnowledgeDocument:
    """Merge LLM suggestions into a draft without changing approval metadata."""
    payload = draft.model_dump()
    for field in [
        "content",
        "theme",
        "poi_names",
        "best_for",
        "recommended_duration",
        "seasonality",
        "transport_advice",
        "planning_tips",
    ]:
        value = getattr(suggestion, field)
        if isinstance(value, list):
            merged = merge_text_lists(payload.get(field, []), value)
            payload[field] = merged
        elif value:
            payload[field] = value.strip()
    payload["review_status"] = draft.review_status
    payload["reviewer"] = draft.reviewer
    payload["review_notes"] = draft.review_notes
    return DraftKnowledgeDocument(**payload)


def merge_text_lists(existing: Iterable[str], suggested: Iterable[str]) -> List[str]:
    merged: List[str] = []
    seen = set()
    for value in [*existing, *suggested]:
        item = str(value or "").strip()
        key = item.lower()
        if not item or key in seen:
            continue
        seen.add(key)
        merged.append(item)
    return merged


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


@dataclass
class PromoteResult:
    scanned: int
    approved: int
    promoted: int
    skipped_existing: int


def promote_approved_drafts(
    *,
    draft_root: Path = DEFAULT_DRAFT_ROOT,
    knowledge_root: Path = DEFAULT_KNOWLEDGE_ROOT,
    country: str = "US",
    overwrite: bool = False,
) -> PromoteResult:
    grouped: dict[str, list[KnowledgeDocument]] = {}
    scanned = 0
    approved = 0
    for path in sorted((draft_root / country_slug(country)).rglob("*.json")):
        scanned += 1
        draft = read_draft(path)
        if draft.review_status != "approved":
            continue
        approved += 1
        grouped.setdefault(city_slug(draft.city), []).append(draft.to_knowledge_document())

    promoted = 0
    before_existing = 0
    for city_dir, docs in grouped.items():
        knowledge_file = knowledge_root / country_slug(country) / f"{city_dir}.json"
        before_existing += count_existing_doc_ids(knowledge_file, [doc.doc_id for doc in docs])
        promoted += merge_knowledge_docs(
            knowledge_file=knowledge_file,
            docs=docs,
            overwrite=overwrite,
        )
    skipped_existing = 0 if overwrite else before_existing
    return PromoteResult(
        scanned=scanned,
        approved=approved,
        promoted=promoted,
        skipped_existing=skipped_existing,
    )


def count_existing_doc_ids(knowledge_file: Path, doc_ids: Iterable[str]) -> int:
    if not knowledge_file.exists():
        return 0
    payload = json.loads(knowledge_file.read_text(encoding="utf-8"))
    items = payload if isinstance(payload, list) else [payload]
    existing = {item.get("doc_id") for item in items if isinstance(item, dict)}
    return sum(1 for doc_id in doc_ids if doc_id in existing)


def is_draft_promoted(
    *,
    draft: DraftKnowledgeDocument,
    knowledge_root: Path = DEFAULT_KNOWLEDGE_ROOT,
) -> bool:
    knowledge_file = knowledge_root / country_slug(draft.country) / f"{city_slug(draft.city)}.json"
    return count_existing_doc_ids(knowledge_file, [draft.doc_id]) > 0


class RAGIngestionJobStore:
    """SQLite-backed local job store for ingestion admin tasks."""

    def __init__(self, db_path: Path = DEFAULT_JOB_DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def create_job(self, job_type: str, message: str = "") -> dict:
        now = time.time()
        job_id = str(uuid4())
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO rag_ingestion_jobs (
                    job_id, job_type, status, started_at, finished_at, message, error, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (job_id, job_type, "queued", 0.0, 0.0, message, "", now),
            )
        return self.get_job(job_id)

    def update_job(
        self,
        job_id: str,
        *,
        status: str,
        message: str = "",
        error: str = "",
        started_at: Optional[float] = None,
        finished_at: Optional[float] = None,
    ) -> dict:
        current = self.get_job(job_id)
        if current is None:
            raise KeyError(job_id)
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE rag_ingestion_jobs
                SET status = ?, message = ?, error = ?, started_at = ?, finished_at = ?
                WHERE job_id = ?
                """,
                (
                    status,
                    message,
                    error,
                    current["started_at"] if started_at is None else started_at,
                    current["finished_at"] if finished_at is None else finished_at,
                    job_id,
                ),
            )
        return self.get_job(job_id)

    def get_job(self, job_id: str) -> Optional[dict]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM rag_ingestion_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        return dict(row) if row is not None else None

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS rag_ingestion_jobs (
                    job_id TEXT PRIMARY KEY,
                    job_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at REAL NOT NULL DEFAULT 0,
                    finished_at REAL NOT NULL DEFAULT 0,
                    message TEXT NOT NULL DEFAULT '',
                    error TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn


_job_store: Optional[RAGIngestionJobStore] = None


def get_rag_ingestion_job_store() -> RAGIngestionJobStore:
    global _job_store
    if _job_store is None:
        _job_store = RAGIngestionJobStore()
    return _job_store
