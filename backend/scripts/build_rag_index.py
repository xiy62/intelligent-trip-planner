"""Build the local Chroma travel-knowledge index for RAG retrieval."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.rag_service import get_rag_service


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the travel knowledge Chroma index.")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the index from scratch.")
    args = parser.parse_args()

    rag_service = get_rag_service()
    rag_service.ensure_index(force_rebuild=args.rebuild)
    print(f"Built RAG index under {rag_service.persist_directory}")


if __name__ == "__main__":
    main()
