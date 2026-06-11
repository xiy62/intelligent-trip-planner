"""Generate review-ready RAG knowledge drafts from source manifests."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.rag_ingestion import (  # noqa: E402
    build_draft_document,
    city_slug,
    extract_readable_text,
    load_manifest,
    slugify,
    write_draft,
)


def fetch_html(url: str) -> str:
    headers = {
        "User-Agent": "IntelligentTripPlannerRAGIngestion/1.0",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    with httpx.Client(timeout=20.0, follow_redirects=True, headers=headers) as client:
        response = client.get(url)
        response.raise_for_status()
        return response.text


def main() -> None:
    parser = argparse.ArgumentParser(description="Create human-reviewable RAG knowledge drafts.")
    parser.add_argument(
        "--manifest",
        default="data/sources/china_travel_sources.sample.json",
        help="Path to source manifest relative to backend/ or absolute.",
    )
    parser.add_argument(
        "--raw-root",
        default="data/raw/china",
        help="Directory for raw HTML/text output relative to backend/ or absolute.",
    )
    parser.add_argument(
        "--draft-root",
        default="data/drafts/china",
        help="Directory for draft JSON output relative to backend/ or absolute.",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    raw_root = Path(args.raw_root)
    draft_root = Path(args.draft_root)
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    if not raw_root.is_absolute():
        raw_root = ROOT / raw_root
    if not draft_root.is_absolute():
        draft_root = ROOT / draft_root

    entries = load_manifest(manifest_path)
    for entry in entries:
        city_dir = city_slug(entry.city)
        source_slug = slugify(entry.source_id)
        html = fetch_html(str(entry.source_url))
        text = extract_readable_text(html)
        if not text:
            raise RuntimeError(f"No readable text extracted for {entry.source_id}")

        raw_city_root = raw_root / city_dir
        raw_city_root.mkdir(parents=True, exist_ok=True)
        raw_html_path = raw_city_root / f"{source_slug}.html"
        raw_text_path = raw_city_root / f"{source_slug}.txt"
        raw_html_path.write_text(html, encoding="utf-8")
        raw_text_path.write_text(text, encoding="utf-8")

        draft = build_draft_document(
            entry=entry,
            extracted_text=text,
            raw_html_path=raw_html_path,
            raw_text_path=raw_text_path,
        )
        draft_path = draft_root / city_dir / f"{source_slug}.json"
        write_draft(draft_path, draft)
        print(f"Wrote draft: {draft_path}")


if __name__ == "__main__":
    main()
