"""Local admin API for human-in-the-loop RAG ingestion."""

from __future__ import annotations

import time
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Query, UploadFile
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, ValidationError

from ...prompts.rag_ingestion import build_rag_prefill_prompt
from ..dependencies import require_admin_token
from ...services.rag_ingestion import (
    DEFAULT_DRAFT_ROOT,
    DEFAULT_KNOWLEDGE_ROOT,
    DraftKnowledgeDocument,
    RAGPrefillSuggestion,
    SourceManifestEntry,
    apply_prefill_suggestions,
    build_ai_prefill_source_packet,
    build_draft_document,
    city_slug,
    country_slug,
    draft_path_for,
    extract_uploaded_source,
    find_draft_path,
    get_rag_ingestion_job_store,
    is_draft_promoted,
    list_draft_paths,
    promote_approved_drafts,
    read_draft,
    slugify,
    validate_prefill_suggestions,
    write_draft,
    write_extracted_text,
    write_raw_source_text,
    write_uploaded_source,
)
from ...services.llm_service import get_llm
from ...services.rag_service import get_rag_service
from ...services.web_fetch_service import URLSafetyError, WebFetchError, get_web_fetch_service

router = APIRouter(
    prefix="/rag-ingestion",
    tags=["RAG Ingestion"],
    dependencies=[Depends(require_admin_token)],
)
logger = logging.getLogger(__name__)


class RAGApproveRequest(BaseModel):
    reviewer: str = "local-admin"
    review_notes: str = ""


class RAGPromoteRequest(BaseModel):
    country: str = "US"
    overwrite: bool = False


class RAGUrlIngestionRequest(BaseModel):
    source_id: str
    country: str = "US"
    city: str
    source_url: str
    source_type: str = "official_tourism_portal"
    title: str
    theme: list[str] = []
    poi_names: list[str] = []
    district: str = ""
    language: str = "en"
    best_for: list[str] = []
    recommended_duration: str = ""
    css_selector: str = ""


def draft_summary(path: Path, draft: DraftKnowledgeDocument) -> dict:
    promoted = is_draft_promoted(draft=draft, knowledge_root=DEFAULT_KNOWLEDGE_ROOT)
    return {
        "draft_id": draft.doc_id,
        "doc_id": draft.doc_id,
        "country": draft.country,
        "city": draft.city,
        "title": draft.title,
        "source_type": draft.source_type,
        "source_url": draft.source_url,
        "review_status": draft.review_status,
        "promoted": promoted,
        "corpus_status": "promoted" if promoted else "not_promoted",
        "reviewer": draft.reviewer,
        "updated_path": str(path),
        "fetched_at": draft.fetched_at,
    }


def draft_detail(path: Path) -> dict:
    draft = read_draft(path)
    extracted_text = ""
    raw_text_path = Path(draft.raw_text_path) if draft.raw_text_path else None
    if raw_text_path and raw_text_path.exists():
        extracted_text = raw_text_path.read_text(encoding="utf-8")
    return {
        **draft_summary(path, draft),
        "draft": draft.model_dump(),
        "extracted_text": extracted_text,
    }


@router.post("/uploads", summary="Upload a source file and create a reviewable RAG draft")
async def upload_rag_source(
    file: UploadFile = File(...),
    source_id: str = Form(...),
    country: str = Form("US"),
    city: str = Form(...),
    source_url: str = Form(...),
    source_type: str = Form("official_tourism_portal"),
    title: str = Form(...),
    theme: str = Form(""),
    poi_names: str = Form(""),
    district: str = Form(""),
    language: str = Form("en"),
    best_for: str = Form(""),
    recommended_duration: str = Form(""),
):
    """Create a draft from an uploaded PDF, Markdown, or text file."""
    try:
        entry = SourceManifestEntry(
            source_id=source_id,
            country=country,
            city=city,
            source_url=source_url,
            source_type=source_type,
            title=title,
            theme=csv_items(theme),
            poi_names=csv_items(poi_names),
            district=district,
            language=language,
            best_for=csv_items(best_for),
            recommended_duration=recommended_duration,
        )
        content = await file.read()
        source_path = write_uploaded_source(
            filename=file.filename or f"{slugify(source_id)}.txt",
            content=content,
            country=entry.country,
            city=entry.city,
            source_id=entry.source_id,
        )
        extracted_text = extract_uploaded_source(source_path)
        extracted_path = write_extracted_text(
            extracted_text=extracted_text,
            country=entry.country,
            city=entry.city,
            source_id=entry.source_id,
        )
        draft = build_draft_document(
            entry=entry,
            extracted_text=extracted_text,
            raw_html_path=source_path,
            raw_text_path=extracted_path,
        )
        path = draft_path_for(
            draft_root=DEFAULT_DRAFT_ROOT,
            country=draft.country,
            city=draft.city,
            doc_id=draft.doc_id,
        )
        write_draft(path, draft)
        return {"success": True, "data": draft_detail(path)}
    except (ValueError, RuntimeError, ValidationError) as exc:
        logger.warning("RAG upload rejected: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected RAG upload failure")
        raise HTTPException(status_code=500, detail=f"Upload ingestion failed: {exc}") from exc


@router.post("/urls", summary="Fetch one URL and create a reviewable RAG draft")
async def ingest_rag_url(request: RAGUrlIngestionRequest):
    """Create a draft from one explicitly submitted webpage URL."""
    started = time.perf_counter()
    try:
        entry = SourceManifestEntry(
            source_id=request.source_id,
            country=request.country,
            city=request.city,
            source_url=request.source_url,
            source_type=request.source_type,
            title=request.title,
            theme=request.theme,
            poi_names=request.poi_names,
            district=request.district,
            language=request.language,
            best_for=request.best_for,
            recommended_duration=request.recommended_duration,
        )
        fetch_result = get_web_fetch_service().fetch(
            str(entry.source_url),
            css_selector=request.css_selector.strip(),
        )
        source_path = write_raw_source_text(
            filename=f"{slugify(entry.source_id)}.html",
            text=fetch_result.raw_html,
            country=entry.country,
            city=entry.city,
            source_id=entry.source_id,
        )
        extracted_path = write_extracted_text(
            extracted_text=fetch_result.extracted_markdown,
            country=entry.country,
            city=entry.city,
            source_id=entry.source_id,
        )
        draft = build_draft_document(
            entry=entry,
            extracted_text=fetch_result.extracted_markdown,
            raw_html_path=source_path,
            raw_text_path=extracted_path,
            fetched_at=fetch_result.fetched_at.isoformat(),
        )
        path = draft_path_for(
            draft_root=DEFAULT_DRAFT_ROOT,
            country=draft.country,
            city=draft.city,
            doc_id=draft.doc_id,
        )
        write_draft(path, draft)
        elapsed_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "RAG URL ingestion draft created source_id=%s requested_url=%s final_url=%s "
            "fetch_mode=%s duration_ms=%.1f raw_html_size=%s markdown_size=%s warnings=%s",
            entry.source_id,
            fetch_result.requested_url,
            fetch_result.final_url,
            fetch_result.fetch_mode,
            elapsed_ms,
            len(fetch_result.raw_html.encode("utf-8")),
            len(fetch_result.extracted_markdown.encode("utf-8")),
            len(fetch_result.warnings),
        )
        return {
            "success": True,
            "data": {
                **draft_detail(path),
                "fetch": fetch_result.model_dump(mode="json", exclude={"raw_html", "extracted_markdown"}),
            },
        }
    except (ValueError, RuntimeError, ValidationError, URLSafetyError, WebFetchError) as exc:
        logger.warning("RAG URL ingestion rejected: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected RAG URL ingestion failure")
        raise HTTPException(status_code=500, detail=f"URL ingestion failed: {exc}") from exc


@router.get("/drafts", summary="List reviewable RAG drafts")
async def list_rag_drafts(
    country: Optional[str] = Query(default=None),
    city: Optional[str] = Query(default=None),
    review_status: Optional[str] = Query(default=None),
):
    """Return draft summaries for the local admin console."""
    try:
        drafts = [
            draft_summary(path, read_draft(path))
            for path in list_draft_paths(
                draft_root=DEFAULT_DRAFT_ROOT,
                country=country,
                city=city,
                review_status=review_status,
            )
        ]
        return {"success": True, "data": drafts}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to list drafts: {exc}") from exc


@router.get("/drafts/{draft_id}", summary="Get one RAG draft")
async def get_rag_draft(draft_id: str):
    """Return one draft plus extracted text preview."""
    try:
        path = find_draft_path(DEFAULT_DRAFT_ROOT, draft_id)
        return {"success": True, "data": draft_detail(path)}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read draft: {exc}") from exc


@router.put("/drafts/{draft_id}", summary="Update a RAG draft")
async def update_rag_draft(draft_id: str, draft: DraftKnowledgeDocument):
    """Validate and persist a structured draft edit."""
    try:
        path = find_draft_path(DEFAULT_DRAFT_ROOT, draft_id)
        existing = read_draft(path)
        if draft.doc_id != existing.doc_id:
            raise ValueError("doc_id cannot be changed after draft creation")
        write_draft(path, draft)
        return {"success": True, "data": draft_detail(path)}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to update draft: {exc}") from exc


@router.post("/drafts/{draft_id}/approve", summary="Approve a reviewed RAG draft")
async def approve_rag_draft(draft_id: str, request: RAGApproveRequest):
    """Mark an existing draft as approved after human review."""
    try:
        path = find_draft_path(DEFAULT_DRAFT_ROOT, draft_id)
        draft = read_draft(path)
        draft.review_status = "approved"
        draft.reviewer = request.reviewer.strip() or "local-admin"
        draft.review_notes = request.review_notes.strip()
        write_draft(path, draft)
        return {"success": True, "data": draft_detail(path)}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to approve draft: {exc}") from exc


@router.post("/drafts/{draft_id}/ai-prefill", summary="Generate AI suggestions for a RAG draft")
async def ai_prefill_rag_draft(draft_id: str):
    """Return unsaved LLM suggestions for structured RAG draft fields."""
    try:
        path = find_draft_path(DEFAULT_DRAFT_ROOT, draft_id)
        draft = read_draft(path)
        if not draft.raw_text_path:
            raise ValueError("Draft has no extracted text path")
        raw_text_path = Path(draft.raw_text_path)
        if not raw_text_path.exists():
            raise ValueError("Extracted text file does not exist")
        extracted_text = raw_text_path.read_text(encoding="utf-8")
        if not extracted_text.strip():
            raise ValueError("Extracted text is empty")

        source_packet = build_ai_prefill_source_packet(
            draft=draft,
            extracted_text=extracted_text,
        )
        if not source_packet.source_packet.strip():
            raise ValueError("No usable source text remained after cleaning")

        parser = PydanticOutputParser(pydantic_object=RAGPrefillSuggestion)
        prompt = build_rag_prefill_prompt(draft=draft, source_packet=source_packet.source_packet, parser=parser)
        response = get_llm().invoke(prompt)
        content = getattr(response, "content", response)
        suggestion = parser.parse(str(content))
        validated = validate_prefill_suggestions(
            draft=draft,
            suggestion=suggestion,
            selected_sections=source_packet.selected_sections,
        )
        suggested_draft = apply_prefill_suggestions(draft=draft, suggestions=validated)
        accepted_count = sum(1 for item in validated if item.status == "accepted")
        review_required_count = sum(1 for item in validated if item.status == "review_required")
        rejected_count = sum(1 for item in validated if item.status == "rejected")
        time_sensitive_count = sum(1 for item in validated if item.time_sensitive and item.status == "accepted")
        logger.info(
            "RAG AI prefill processed draft_id=%s source_chars=%s sections=%s selected=%s "
            "discarded=%s accepted=%s review_required=%s rejected=%s",
            draft_id,
            source_packet.source_char_count,
            len(source_packet.sections),
            len(source_packet.selected_sections),
            len(source_packet.sections) - len(source_packet.selected_sections),
            accepted_count,
            review_required_count,
            rejected_count,
        )
        time_sensitive_warning = (
            [f"{time_sensitive_count} accepted suggestion(s) contain time-sensitive source-backed information."]
            if time_sensitive_count
            else []
        )

        return {
            "success": True,
            "data": {
                "suggested_draft": suggested_draft.model_dump(),
                "suggestions": [item.model_dump() for item in validated],
                "warnings": [*source_packet.warnings, *suggestion.warnings, *time_sensitive_warning],
                "source_char_count": source_packet.source_char_count,
                "used_char_count": source_packet.used_char_count,
                "section_count": len(source_packet.sections),
                "selected_section_count": len(source_packet.selected_sections),
                "discarded_section_count": len(source_packet.sections) - len(source_packet.selected_sections),
                "accepted_suggestion_count": accepted_count,
                "review_required_suggestion_count": review_required_count,
                "rejected_suggestion_count": rejected_count,
            },
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ValueError, RuntimeError, ValidationError) as exc:
        logger.warning("RAG AI prefill rejected: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected RAG AI prefill failure")
        raise HTTPException(status_code=500, detail=f"AI prefill failed: {exc}") from exc


@router.post("/promote", summary="Promote approved drafts into the RAG knowledge corpus")
async def promote_rag_drafts(request: RAGPromoteRequest):
    """Merge approved drafts into the production knowledge corpus."""
    try:
        result = promote_approved_drafts(
            draft_root=DEFAULT_DRAFT_ROOT,
            knowledge_root=DEFAULT_KNOWLEDGE_ROOT,
            country=request.country,
            overwrite=request.overwrite,
        )
        return {
            "success": True,
            "data": {
                "country": country_slug(request.country),
                "scanned": result.scanned,
                "approved": result.approved,
                "promoted": result.promoted,
                "skipped_existing": result.skipped_existing,
            },
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to promote drafts: {exc}") from exc


@router.post("/index/rebuild", summary="Start an async Chroma index rebuild")
async def rebuild_rag_index(background_tasks: BackgroundTasks):
    """Start an asynchronous Chroma rebuild and return a local job ID."""
    try:
        store = get_rag_ingestion_job_store()
        job = store.create_job("rag_index_rebuild", "Queued Chroma rebuild")
        background_tasks.add_task(run_rebuild_job, job["job_id"])
        return {"success": True, "data": job}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to queue rebuild: {exc}") from exc


@router.get("/jobs/{job_id}", summary="Get ingestion job status")
async def get_rag_ingestion_job(job_id: str):
    """Return async job status."""
    job = get_rag_ingestion_job_store().get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job_id does not exist")
    return {"success": True, "data": job}


def run_rebuild_job(job_id: str) -> None:
    store = get_rag_ingestion_job_store()
    try:
        store.update_job(
            job_id,
            status="running",
            started_at=time.time(),
            message="Rebuilding Chroma index",
        )
        get_rag_service().ensure_index(force_rebuild=True)
        store.update_job(
            job_id,
            status="succeeded",
            finished_at=time.time(),
            message="Chroma index rebuilt",
        )
    except Exception as exc:
        store.update_job(
            job_id,
            status="failed",
            finished_at=time.time(),
            message="Chroma index rebuild failed",
            error=f"{exc.__class__.__name__}: {exc}",
        )


def csv_items(value: str) -> list[str]:
    return [item.strip() for item in (value or "").split(",") if item.strip()]
