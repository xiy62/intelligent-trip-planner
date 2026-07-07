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
    re.compile(r"^skip navigation$", re.I),
    re.compile(r"^visitors guide$", re.I),
    re.compile(r"^book your trip$", re.I),
    re.compile(r"^now in .+$", re.I),
    re.compile(r"^things to do$", re.I),
    re.compile(r"^eat\s*&\s*drink$", re.I),
    re.compile(r"^where to stay$", re.I),
    re.compile(r"^maps\s*&\s*guides$", re.I),
    re.compile(r"^business in .+$", re.I),
]

LOW_VALUE_CONTENT_PATTERNS = [
    re.compile(r"\bcookie(s)?\b.*\b(accept|settings|policy)\b", re.I),
    re.compile(r"\bprivacy policy\b|\bterms (of use|and conditions)\b", re.I),
    re.compile(r"\bnewsletter\b|\bsign up\b|\bsubscribe\b", re.I),
    re.compile(r"\bfollow us\b|\bshare this\b|\bsocial media\b", re.I),
    re.compile(r"\bsponsored\b|\badvertisement\b|\bpaid partnership\b", re.I),
    re.compile(r"\ball rights reserved\b|\bcopyright\b", re.I),
    re.compile(r"\btable of contents\b", re.I),
    re.compile(r"\bread more\b|\brelated articles\b|\byou may also like\b", re.I),
    re.compile(r"\bprime early deals\b|\bprime day deals\b|\bsave \d+%|\bexpedia\b", re.I),
]

PROMOTIONAL_PATTERNS = [
    re.compile(r"\b(unforgettable|world[- ]class|must[- ]see|hidden gem|iconic experience)\b", re.I),
    re.compile(r"\bbook now\b|\bclick here\b|\blimited time\b", re.I),
]

USEFUL_SHORT_FACT_PATTERNS = [
    re.compile(r"\bclosed\s+\w+", re.I),
    re.compile(r"\breservations?\s+(required|recommended)\b", re.I),
    re.compile(r"\bwheelchair accessible\b", re.I),
    re.compile(r"\ballow\s+\w+\s+(hour|hours|minutes)\b", re.I),
    re.compile(r"\b\d+\s*(hours?|minutes?)\b", re.I),
]

TIME_SENSITIVE_PATTERNS = [
    re.compile(r"\b(open|closed|hours?|daily|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.I),
    re.compile(r"[$€£¥]\s*\d+|\b(price|prices|ticket|admission|fee|free)\b", re.I),
    re.compile(r"\b(schedule|timetable|depart(s|ure)?|reservation|required|temporary|restriction|visa)\b", re.I),
    re.compile(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b|\b\d{1,2}:\d{2}\b", re.I),
]

PREFILL_LIST_FIELDS = {
    "theme",
    "poi_names",
    "best_for",
    "seasonality",
    "transport_advice",
    "planning_tips",
}

PREFILL_SCALAR_FIELDS = {"content", "recommended_duration"}
PREFILL_ALLOWED_FIELDS = PREFILL_LIST_FIELDS | PREFILL_SCALAR_FIELDS


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


class RAGSourceSection(BaseModel):
    """A deterministic source section passed to the AI prefill prompt."""

    section_id: str
    heading: str = ""
    text: str
    start_offset: int
    end_offset: int


class RAGSectionDecision(BaseModel):
    """Inspectable deterministic relevance decision for one source section."""

    section_id: str
    heading: str = ""
    selected: bool
    score: int
    reasons: List[str] = Field(default_factory=list)


class RAGPrefillSuggestionItem(BaseModel):
    """One evidence-backed LLM suggestion for a draft field."""

    field: str
    value: str
    source_quote: str
    section_id: str
    section_heading: str = ""
    time_sensitive: bool = False
    confidence: Optional[float] = None


class RAGPrefillSuggestion(BaseModel):
    """Structured LLM output for evidence-backed RAG draft suggestions."""

    suggestions: List[RAGPrefillSuggestionItem] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class RAGValidatedSuggestion(RAGPrefillSuggestionItem):
    """A suggestion after deterministic quote and quality validation."""

    status: str = "review_required"
    reason: str = ""


class RAGPrefillSourcePacket(BaseModel):
    """Prepared source packet and deterministic section decisions for AI prefill."""

    source_packet: str
    source_char_count: int
    used_char_count: int
    sections: List[RAGSourceSection] = Field(default_factory=list)
    selected_sections: List[RAGSourceSection] = Field(default_factory=list)
    section_decisions: List[RAGSectionDecision] = Field(default_factory=list)
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


def write_raw_source_text(
    *,
    filename: str,
    text: str,
    country: str,
    city: str,
    source_id: str,
    upload_root: Path = DEFAULT_UPLOAD_ROOT,
) -> Path:
    """Write a text-based raw source artifact using upload path conventions."""
    suffix = Path(filename).suffix.lower() or ".html"
    target_dir = upload_root / country_slug(country) / city_slug(city)
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / f"{slugify(source_id)}{suffix}"
    target_path.write_text(text or "", encoding="utf-8")
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
    content = summarize_content(clean_source_text(extracted_text))
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
) -> RAGPrefillSourcePacket:
    """Return selected source sections compact enough for an LLM prefill prompt."""
    source_char_count = len(extracted_text or "")
    cleaned_text = clean_source_text(extracted_text)
    sections = split_markdown_sections(cleaned_text)
    warnings: List[str] = []

    if not sections:
        return RAGPrefillSourcePacket(
            source_packet="",
            source_char_count=source_char_count,
            used_char_count=0,
            warnings=["No usable source sections remained after cleaning."],
        )

    selected_sections, decisions = select_relevant_sections(
        draft=draft,
        sections=sections,
        max_chars=max_chars,
    )

    if not selected_sections:
        return RAGPrefillSourcePacket(
            source_packet="",
            source_char_count=source_char_count,
            used_char_count=0,
            sections=sections,
            section_decisions=decisions,
            warnings=["No relevant source sections remained after filtering."],
        )

    if sum(len(section.text) for section in sections) > sum(len(section.text) for section in selected_sections):
        warnings.append(
            f"Selected {len(selected_sections)} of {len(sections)} source sections for AI prefill."
        )

    packet = "\n\n".join(
        f"[{section.section_id}] {section.heading or 'Untitled section'}\n{section.text.strip()}"
        for section in selected_sections
    )
    return RAGPrefillSourcePacket(
        source_packet=packet,
        source_char_count=source_char_count,
        used_char_count=len(packet),
        sections=sections,
        selected_sections=selected_sections,
        section_decisions=decisions,
        warnings=warnings,
    )


def clean_source_text(text: str) -> str:
    """Remove common PDF/web boilerplate while preserving useful Markdown structure."""
    text = unescape(text or "").replace("\r", "\n")
    raw_lines = text.splitlines()
    cleaned_lines: List[str] = []
    seen_lines: dict[str, int] = {}

    for raw_line in raw_lines:
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            if cleaned_lines and cleaned_lines[-1] != "":
                cleaned_lines.append("")
            continue
        line = strip_markdown_inline_noise(line)
        if not line:
            continue
        if is_noise_line(line):
            continue
        if is_low_value_text(line) and not is_useful_short_fact(line):
            continue
        line_key = line.lower()
        seen_lines[line_key] = seen_lines.get(line_key, 0) + 1
        if seen_lines[line_key] > 2:
            continue
        if (
            len(line) < 25
            and not is_heading_line(line)
            and not is_useful_short_fact(line)
            and not contains_cjk(line)
        ):
            continue
        cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines)
    paragraphs = []
    seen_paragraphs = set()
    for paragraph in re.split(r"\n{2,}", cleaned):
        normalized = re.sub(r"\s+", " ", paragraph).strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen_paragraphs:
            continue
        seen_paragraphs.add(key)
        paragraphs.append(paragraph.strip())
    return "\n\n".join(paragraphs)


def clean_source_paragraphs(text: str) -> List[str]:
    """Compatibility wrapper returning cleaned paragraphs."""
    cleaned = clean_source_text(text)
    return [paragraph.strip() for paragraph in re.split(r"\n{2,}", cleaned) if paragraph.strip()]


def is_noise_line(line: str) -> bool:
    if re.fullmatch(r"!?\[[^\]]*\]\([^)]+\)", line):
        return True
    if line in {".", "|", "•"}:
        return True
    if len(line) <= 2:
        return True
    return any(pattern.search(line) for pattern in NOISE_LINE_PATTERNS)


def is_heading_line(line: str) -> bool:
    return bool(re.match(r"^#{1,6}\s+\S+", line))


def is_low_value_text(text: str) -> bool:
    return any(pattern.search(text or "") for pattern in LOW_VALUE_CONTENT_PATTERNS)


def is_promotional_text(text: str) -> bool:
    return any(pattern.search(text or "") for pattern in PROMOTIONAL_PATTERNS)


def is_useful_short_fact(text: str) -> bool:
    return any(pattern.search(text or "") for pattern in USEFUL_SHORT_FACT_PATTERNS)


def contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u3400-\u9fff]", text or ""))


def strip_markdown_inline_noise(line: str) -> str:
    """Drop media-only Markdown and keep link text for evidence matching."""
    value = line or ""
    value = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", value)
    value = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", value)
    value = re.sub(r"</?[^>]+>", " ", value)
    value = re.sub(r"[*_`]+", "", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def split_markdown_sections(text: str, max_section_chars: int = 3500) -> List[RAGSourceSection]:
    """Split Markdown into heading-aware source sections with offsets."""
    text = (text or "").strip()
    if not text:
        return []

    heading_matches = list(re.finditer(r"(?m)^(#{1,6})\s+(.+?)\s*$", text))
    raw_sections: List[tuple[str, str, int, int]] = []
    if heading_matches:
        for index, match in enumerate(heading_matches):
            start = match.start()
            end = heading_matches[index + 1].start() if index + 1 < len(heading_matches) else len(text)
            heading = match.group(2).strip()
            raw_sections.append((heading, text[start:end].strip(), start, end))
        if heading_matches[0].start() > 0:
            prefix = text[: heading_matches[0].start()].strip()
            if prefix:
                raw_sections.insert(0, ("Overview", prefix, 0, heading_matches[0].start()))
    else:
        offset = 0
        for paragraph in re.split(r"\n{2,}", text):
            start = text.find(paragraph, offset)
            end = start + len(paragraph)
            offset = end
            raw_sections.append(("Overview", paragraph.strip(), start, end))

    sections: List[RAGSourceSection] = []
    for heading, body, start, end in raw_sections:
        if not body.strip():
            continue
        chunks = split_oversized_section(body, max_section_chars=max_section_chars)
        running_start = start
        for chunk in chunks:
            chunk_start = text.find(chunk, running_start)
            if chunk_start < 0:
                chunk_start = running_start
            chunk_end = chunk_start + len(chunk)
            sections.append(
                RAGSourceSection(
                    section_id=f"section-{len(sections) + 1:03d}",
                    heading=heading,
                    text=chunk.strip(),
                    start_offset=chunk_start,
                    end_offset=chunk_end,
                )
            )
            running_start = chunk_end
    return sections


def split_oversized_section(text: str, max_section_chars: int) -> List[str]:
    if len(text) <= max_section_chars:
        return [text]
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0
    for paragraph in re.split(r"\n{2,}", text):
        if current and current_len + len(paragraph) > max_section_chars:
            chunks.append("\n\n".join(current))
            current = []
            current_len = 0
        current.append(paragraph)
        current_len += len(paragraph) + 2
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def select_relevant_sections(
    *,
    draft: DraftKnowledgeDocument,
    sections: List[RAGSourceSection],
    max_chars: int,
) -> tuple[List[RAGSourceSection], List[RAGSectionDecision]]:
    terms = metadata_terms(draft)
    scored: List[tuple[int, int, RAGSourceSection, list[str]]] = []
    for index, section in enumerate(sections):
        score, reasons = score_source_section(draft=draft, section=section, metadata_terms=terms)
        scored.append((score, index, section, reasons))

    selected: List[RAGSourceSection] = []
    selected_ids = set()
    total = 0
    for score, _, section, _ in sorted(scored, key=lambda item: (-item[0], item[1])):
        if score < 2 and not is_useful_short_fact(section.text):
            continue
        if total + len(section.text) > max_chars and selected:
            continue
        selected.append(section)
        selected_ids.add(section.section_id)
        total += len(section.text) + len(section.heading) + 16
        if total >= max_chars:
            break

    selected.sort(key=lambda section: section.start_offset)
    decisions = [
        RAGSectionDecision(
            section_id=section.section_id,
            heading=section.heading,
            selected=section.section_id in selected_ids,
            score=score,
            reasons=reasons,
        )
        for score, _, section, reasons in scored
    ]
    return selected, decisions


def score_source_section(
    *,
    draft: DraftKnowledgeDocument,
    section: RAGSourceSection,
    metadata_terms: set[str],
) -> tuple[int, List[str]]:
    text = f"{section.heading} {section.text}"
    terms = set(tokenize_for_prefill(text))
    score = 0
    reasons: List[str] = []

    overlap = terms & metadata_terms
    if overlap:
        score += min(8, len(overlap))
        reasons.append(f"metadata_overlap:{','.join(sorted(overlap)[:6])}")
    travel_overlap = terms & TRAVEL_RELEVANCE_TERMS
    if travel_overlap:
        score += min(6, len(travel_overlap))
        reasons.append(f"travel_terms:{','.join(sorted(travel_overlap)[:6])}")
    if draft.city and draft.city.lower() in text.lower():
        score += 4
        reasons.append("city_match")
    if any(poi.lower() in text.lower() for poi in draft.poi_names if poi):
        score += 4
        reasons.append("poi_match")
    if is_useful_short_fact(text):
        score += 3
        reasons.append("useful_short_fact")
    if is_low_value_text(text):
        score -= 5
        reasons.append("low_value_pattern")
    if is_promotional_text(text):
        score -= 3
        reasons.append("promotional_pattern")
    if len(section.text) < 40 and not is_useful_short_fact(section.text):
        score -= 2
        reasons.append("too_short")

    return score, reasons


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


def validate_prefill_suggestions(
    *,
    draft: DraftKnowledgeDocument,
    suggestion: RAGPrefillSuggestion,
    selected_sections: List[RAGSourceSection],
) -> List[RAGValidatedSuggestion]:
    """Validate model-provided quotes and classify suggestions."""
    sections = {section.section_id: section for section in selected_sections}
    seen = set()
    validated: List[RAGValidatedSuggestion] = []
    for item in suggestion.suggestions:
        field = item.field.strip()
        values = split_prefill_list_values(field, item.value)
        source_quote = item.source_quote.strip()
        if not values:
            values = [""]

        for value in values:
            status = "accepted"
            reason = "validated"
            section = sections.get(item.section_id)

            if field not in PREFILL_ALLOWED_FIELDS:
                status, reason = "rejected", "unsupported_field"
            elif not value:
                status, reason = "rejected", "empty_value"
            elif not source_quote:
                status, reason = "rejected", "missing_source_quote"
            elif is_low_value_text(value) or is_promotional_text(value):
                status, reason = "rejected", "low_value_or_promotional"
            elif conflicts_with_city(value, draft.city) or conflicts_with_city(source_quote, draft.city):
                status, reason = "rejected", "city_conflict"
            elif section is None:
                status, reason = "rejected", "unknown_section"
            elif not quote_in_section(source_quote, section.text):
                status, reason = "rejected", "source_quote_not_found"
            elif field in PREFILL_LIST_FIELDS and not list_value_supported_by_evidence(
                field=field,
                value=value,
                source_quote=source_quote,
                section_text=section.text,
            ):
                status, reason = "rejected", "list_value_not_supported"

            duplicate_key = (field.lower(), normalize_for_quote_match(value))
            if duplicate_key in seen and status == "accepted":
                status, reason = "rejected", "duplicate_suggestion"
            seen.add(duplicate_key)

            time_sensitive = item.time_sensitive or is_time_sensitive_text(f"{value} {source_quote}")
            validated.append(
                RAGValidatedSuggestion(
                    field=field,
                    value=value,
                    source_quote=source_quote,
                    section_id=item.section_id,
                    section_heading=item.section_heading
                    or sections.get(item.section_id, RAGSourceSection(section_id="", text="", start_offset=0, end_offset=0)).heading,
                    time_sensitive=time_sensitive,
                    confidence=item.confidence,
                    status=status,
                    reason=reason,
                )
            )
    return validated


def split_prefill_list_values(field: str, raw_value: str) -> List[str]:
    value = (raw_value or "").strip()
    if not value:
        return []
    if field not in PREFILL_LIST_FIELDS:
        return [value]
    if field not in {"theme", "poi_names", "best_for"}:
        return [value]

    normalized = re.sub(r"^[\s\-*•]+", "", value)
    parts = [
        part.strip(" \t\r\n-•")
        for part in re.split(r"\s*(?:;|\n|\r|\u2022|\s+\|\s+)\s*", normalized)
        if part.strip(" \t\r\n-•")
    ]

    if len(parts) == 1 and field in {"theme", "best_for"} and "," in value:
        comma_parts = [part.strip() for part in value.split(",") if part.strip()]
        if 1 < len(comma_parts) <= 8 and all(len(part.split()) <= 6 for part in comma_parts):
            parts = comma_parts

    cleaned: List[str] = []
    seen = set()
    for part in parts:
        part = re.sub(r"^(and|or)\s+", "", part, flags=re.I)
        part = re.sub(r"\s+", " ", part).strip(" .")
        if not part:
            continue
        key = normalize_for_quote_match(part)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(part)
    return cleaned


def list_value_supported_by_evidence(
    *,
    field: str,
    value: str,
    source_quote: str,
    section_text: str,
) -> bool:
    if field in {"seasonality", "transport_advice", "planning_tips"}:
        return True

    normalized_value = normalize_for_quote_match(value)
    evidence = normalize_for_quote_match(f"{source_quote}\n{section_text}")
    if not normalized_value:
        return False
    if normalized_value in evidence:
        return True

    value_tokens = distinctive_tokens(normalized_value)
    if not value_tokens:
        return False
    evidence_tokens = set(distinctive_tokens(evidence))

    if field == "poi_names":
        return len(value_tokens) >= 2 and sum(1 for token in value_tokens if token in evidence_tokens) == len(value_tokens)

    overlap = sum(1 for token in value_tokens if token in evidence_tokens)
    return overlap / len(value_tokens) >= 0.5


def distinctive_tokens(value: str) -> List[str]:
    weak = {
        "and",
        "the",
        "for",
        "with",
        "from",
        "travelers",
        "visitors",
        "visitor",
        "travel",
        "chicago",
        "new",
        "york",
    }
    tokens = []
    for token in re.findall(r"[a-z0-9]+", value.lower()):
        if len(token) <= 2 or token in weak:
            continue
        if token.endswith("ies") and len(token) > 4:
            token = f"{token[:-3]}y"
        elif token.endswith("s") and len(token) > 4:
            token = token[:-1]
        tokens.append(token)
    return tokens


def apply_prefill_suggestions(
    *,
    draft: DraftKnowledgeDocument,
    suggestions: Iterable[RAGValidatedSuggestion],
) -> DraftKnowledgeDocument:
    """Merge accepted suggestions into an unsaved draft without approval changes."""
    payload = draft.model_dump()
    for suggestion in suggestions:
        if suggestion.status != "accepted":
            continue
        field = suggestion.field
        value = suggestion.value.strip()
        if field in PREFILL_LIST_FIELDS:
            payload[field] = merge_text_lists(payload.get(field, []), [value])
        elif field == "recommended_duration":
            if not payload.get(field):
                payload[field] = value
        elif field == "content":
            existing = str(payload.get("content", "") or "").strip()
            if existing and normalize_for_quote_match(value) not in normalize_for_quote_match(existing):
                payload["content"] = f"{existing}\n\n{value}"
            elif not existing:
                payload["content"] = value
    payload["review_status"] = draft.review_status
    payload["reviewer"] = draft.reviewer
    payload["review_notes"] = draft.review_notes
    return DraftKnowledgeDocument(**payload)


def apply_prefill_suggestion(
    *,
    draft: DraftKnowledgeDocument,
    suggestion: RAGPrefillSuggestion,
) -> DraftKnowledgeDocument:
    """Backward-compatible wrapper for callers that still pass raw suggestions."""
    validated = [
        RAGValidatedSuggestion(
            field=item.field,
            value=item.value,
            source_quote=item.source_quote,
            section_id=item.section_id,
            section_heading=item.section_heading,
            time_sensitive=item.time_sensitive,
            confidence=item.confidence,
            status="accepted",
            reason="legacy_wrapper",
        )
        for item in suggestion.suggestions
    ]
    return apply_prefill_suggestions(draft=draft, suggestions=validated)


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


def normalize_for_quote_match(value: str) -> str:
    plain = markdown_to_plain_match_text(value)
    plain = re.sub(r"[^\w\s'-]+", " ", plain, flags=re.UNICODE)
    return re.sub(r"\s+", " ", plain.strip()).lower()


def quote_in_section(quote: str, section_text: str) -> bool:
    normalized_quote = normalize_for_quote_match(quote)
    normalized_section = normalize_for_quote_match(section_text)
    if not normalized_quote:
        return False
    if normalized_quote in normalized_section:
        return True
    return approximate_quote_in_section(normalized_quote, normalized_section)


def markdown_to_plain_match_text(value: str) -> str:
    text = value or ""
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"^#{1,6}\s+", " ", text, flags=re.M)
    text = re.sub(r"[*_`>]+", " ", text)
    text = re.sub(r"</?[^>]+>", " ", text)
    return unescape(text)


def approximate_quote_in_section(normalized_quote: str, normalized_section: str) -> bool:
    quote_tokens = [token for token in normalized_quote.split() if len(token) > 2]
    if len(quote_tokens) < 6:
        return False
    section_tokens = normalized_section.split()
    section_token_set = set(section_tokens)
    covered = sum(1 for token in quote_tokens if token in section_token_set)
    if covered / len(quote_tokens) < 0.78:
        return False

    # Preserve some ordering signal so unrelated bags of common travel words do not pass.
    search_start = 0
    ordered_hits = 0
    for token in quote_tokens:
        try:
            index = section_tokens.index(token, search_start)
        except ValueError:
            continue
        ordered_hits += 1
        search_start = index + 1
    return ordered_hits / len(quote_tokens) >= 0.55


def is_time_sensitive_text(text: str) -> bool:
    return any(pattern.search(text or "") for pattern in TIME_SENSITIVE_PATTERNS)


def conflicts_with_city(text: str, city: str) -> bool:
    if not city:
        return False
    known_cities = set(CITY_SLUGS.keys())
    normalized = (text or "").lower()
    for known_city in known_cities:
        if known_city.lower() == city.lower():
            continue
        if known_city.lower() in normalized:
            return True
    return False


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
