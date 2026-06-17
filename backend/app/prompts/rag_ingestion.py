"""Prompts for LLM-assisted RAG draft prefill."""

from __future__ import annotations

from langchain_core.output_parsers import PydanticOutputParser

from ..services.rag_ingestion import DraftKnowledgeDocument, RAGPrefillSuggestion


def build_rag_prefill_prompt(
    *,
    draft: DraftKnowledgeDocument,
    source_packet: str,
    parser: PydanticOutputParser,
) -> str:
    """Build a prompt that asks the LLM to produce reviewable RAG draft suggestions."""
    return f"""You are helping prepare a source-backed travel knowledge draft for a RAG index.

Your job is to extract only information supported by the source text. You are not approving the document.
A human reviewer will inspect your output before saving, approving, and promoting it.

Existing draft metadata:
- doc_id: {draft.doc_id}
- country: {draft.country}
- city: {draft.city}
- title: {draft.title}
- source_url: {draft.source_url}
- source_type: {draft.source_type}
- existing themes: {', '.join(draft.theme) if draft.theme else 'none'}
- existing POI names: {', '.join(draft.poi_names) if draft.poi_names else 'none'}
- existing best_for: {', '.join(draft.best_for) if draft.best_for else 'none'}

Instructions:
- Fill travel-planning fields for the same city only.
- Keep content concise but useful for retrieval. Remove navigation, page numbers, ads, timestamps, and boilerplate.
- Prefer concrete itinerary-relevant facts: neighborhoods, POIs, route grouping, duration, transport advice, budget/free activities, safety/accessibility notes, and timing advice.
- Do not invent opening hours, closures, ticket policies, or reservation requirements unless directly supported.
- `field_evidence` must include short source-backed snippets for the most important suggestions.
- Evidence snippets must be short review aids, not long copied passages.
- If source text is weak, noisy, incomplete, or too narrow, add warnings instead of guessing.

Source text:
```text
{source_packet}
```

Return structured JSON only.

{parser.get_format_instructions()}
"""
