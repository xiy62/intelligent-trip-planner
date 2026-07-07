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
- Return a list of evidence-backed suggestions. Each suggestion must target one field.
- Allowed fields: content, theme, poi_names, best_for, recommended_duration, seasonality, transport_advice, planning_tips.
- For list-like fields, prefer multiple atomic suggestions instead of one combined string.
  - theme: one concise category per suggestion, such as "first-time visitors", "lakefront attractions", "neighborhoods", "dining", or "theatre".
  - poi_names: one attraction, neighborhood, museum, venue, route, or named experience per suggestion. Do not combine several POIs in one value.
  - best_for: one audience/use case per suggestion, such as "families" or "culture travelers".
  - seasonality, transport_advice, and planning_tips: one concrete fact or tip per suggestion.
- When the source clearly supports themes, POI names, or best_for values, include them. These fields help retrieval and review.
- Extract only actionable, trip-planning-relevant information for the same city.
- Useful content includes concrete attraction descriptions, neighborhoods, route grouping, transport, parking, duration, accessibility, safety, seasonality, dining, accommodation, reservation requirements, and supported opening-hours or admission information.
- Do not include generic praise, slogans, marketing copy, newsletter text, author biography, copyright, social sharing text, duplicated statements, unrelated destinations, vague history with no trip-planning value, or unsupported assumptions.
- Prefer returning no suggestion over returning a weak, generic, repetitive, or poorly supported suggestion.
- Do not fill fields merely because they exist.
- Do not infer common facts without evidence.
- Do not convert vague prose into specific claims.
- Do not invent prices, opening hours, closures, schedules, duration, transport details, or reservation rules.
- Leave fields empty when evidence is insufficient.
- Every suggestion must include a short exact source_quote copied from the supplied source section.
- Every suggestion must include the section_id and section_heading from the source packet.
- Mark time_sensitive=true when the suggestion involves opening hours, prices, closures, transport schedules, event dates, visa rules, reservation policies, temporary restrictions, or business operating status.
- If source text is weak, noisy, incomplete, or too narrow, add warnings instead of guessing.

Selected source sections:
```text
{source_packet}
```

Return structured JSON only.

{parser.get_format_instructions()}
"""
