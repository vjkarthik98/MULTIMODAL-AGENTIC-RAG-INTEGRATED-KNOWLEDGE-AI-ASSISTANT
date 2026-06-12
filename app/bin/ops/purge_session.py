"""One-shot cleanup: drop all chunks for a single session_id from
Qdrant (text + vision collections) and BM25 index.

Usage:
    python -m scripts.purge_session user_1
"""
from __future__ import annotations

import sys

from app.core.infra_registry import infra
from app.utils.logger import get_logger

logger = get_logger(__name__)


def main(session_id: str) -> int:
    if not session_id:
        print("usage: python -m scripts.purge_session <session_id>")
        return 2

    vs = infra.get_vector_store()
    if vs is None:
        print("vector store unavailable")
        return 1
    vs.delete_by_session(session_id)
    print(f"qdrant: deleted points for session_id={session_id}")

    bm25 = infra.get_bm25()
    if bm25 is not None:
        removed = bm25.purge_by_session(session_id)
        print(f"bm25:   purged {removed} docs for session_id={session_id}")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else ""))
