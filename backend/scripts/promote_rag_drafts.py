"""Promote human-approved RAG drafts into the production knowledge corpus."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.rag_ingestion import (  # noqa: E402
    city_slug,
    merge_knowledge_docs,
    read_draft,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote approved RAG drafts to knowledge corpus.")
    parser.add_argument(
        "--draft-root",
        default="data/drafts/china",
        help="Directory containing draft JSON files relative to backend/ or absolute.",
    )
    parser.add_argument(
        "--knowledge-root",
        default="data/knowledge/china",
        help="Production knowledge directory relative to backend/ or absolute.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing docs with the same doc_id.",
    )
    args = parser.parse_args()

    draft_root = Path(args.draft_root)
    knowledge_root = Path(args.knowledge_root)
    if not draft_root.is_absolute():
        draft_root = ROOT / draft_root
    if not knowledge_root.is_absolute():
        knowledge_root = ROOT / knowledge_root

    grouped = defaultdict(list)
    scanned = 0
    approved = 0
    for path in sorted(draft_root.rglob("*.json")):
        scanned += 1
        draft = read_draft(path)
        if draft.review_status != "approved":
            continue
        approved += 1
        grouped[city_slug(draft.city)].append(draft.to_knowledge_document())

    promoted = 0
    for city_dir, docs in grouped.items():
        promoted += merge_knowledge_docs(
            knowledge_file=knowledge_root / f"{city_dir}.json",
            docs=docs,
            overwrite=args.overwrite,
        )

    print(f"Scanned drafts: {scanned}")
    print(f"Approved drafts: {approved}")
    print(f"Promoted docs: {promoted}")
    if approved and promoted == 0:
        print("No docs promoted; approved docs may already exist. Use --overwrite to replace them.")


if __name__ == "__main__":
    main()
