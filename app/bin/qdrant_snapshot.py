#!/usr/bin/env python3
"""
Manual, on-demand Qdrant collection snapshots — MLOps rollback point for the
KB's actual ingested data, which otherwise has zero point-in-time recovery
(see app/vectorstore/qdrant_store.py's create_snapshot/list_snapshots/
recover_snapshot, added 2026-08-21 alongside this script).

Deliberately NOT a scheduled job: at this project's scale a cron snapshot
would just accumulate storage cost for windows nobody asked to protect. Run
this yourself before a risky operation (bulk re-ingest, a schema change,
testing a new chunker) and you have a real rollback point; skip it and
Qdrant behaves exactly as it always has.

Usage:
    python -m app.bin.qdrant_snapshot create                  # both collections
    python -m app.bin.qdrant_snapshot create --collection text_collection
    python -m app.bin.qdrant_snapshot list
    python -m app.bin.qdrant_snapshot restore text_collection <snapshot_name>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_project_root))

try:
    from dotenv import load_dotenv

    load_dotenv(str(_project_root / ".env"))
except ImportError:
    pass


def _get_store():
    from app.core.infra_registry import infra

    store = infra.get_vector_store()
    if store is None:
        sys.exit(
            "QDRANT_UNAVAILABLE: could not obtain a QdrantVectorStore instance "
            "(check QDRANT_URL/QDRANT_API_KEY and connectivity)."
        )
    return store


def _cmd_create(args: argparse.Namespace) -> None:
    store = _get_store()
    created = store.create_snapshot(collection_name=args.collection)
    if not created:
        sys.exit("No snapshot created — see logged errors above.")
    print("Created:")
    for collection, name in created.items():
        print(f"  {collection} -> {name}")


def _cmd_list(args: argparse.Namespace) -> None:
    store = _get_store()
    snaps = store.list_snapshots(collection_name=args.collection)
    for collection, entries in snaps.items():
        print(f"\n{collection} ({len(entries)} snapshot(s)):")
        for e in sorted(entries, key=lambda x: x.get("creation_time") or "", reverse=True):
            size_mb = round((e.get("size") or 0) / (1024 * 1024), 2)
            print(f"  {e['name']}  created={e.get('creation_time')}  size={size_mb}MB")


def _cmd_restore(args: argparse.Namespace) -> None:
    confirm = input(
        f"This will REPLACE all current contents of '{args.collection}' with "
        f"snapshot '{args.snapshot_name}'. This cannot be undone. Type the "
        f"collection name to confirm: "
    )
    if confirm.strip() != args.collection:
        sys.exit("Confirmation did not match collection name — aborted, nothing changed.")
    store = _get_store()
    ok = store.recover_snapshot(args.collection, args.snapshot_name)
    if ok:
        print(f"Restored {args.collection} from {args.snapshot_name}.")
    else:
        sys.exit("Restore failed — see logged errors above.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_create = sub.add_parser("create", help="Create a snapshot of one or both collections")
    p_create.add_argument("--collection", default=None, help="Collection name (default: both)")
    p_create.set_defaults(func=_cmd_create)

    p_list = sub.add_parser("list", help="List existing snapshots")
    p_list.add_argument("--collection", default=None, help="Collection name (default: both)")
    p_list.set_defaults(func=_cmd_list)

    p_restore = sub.add_parser("restore", help="Restore a collection from a snapshot (DESTRUCTIVE)")
    p_restore.add_argument("collection", help="Collection name to restore")
    p_restore.add_argument("snapshot_name", help="Snapshot name (see 'list')")
    p_restore.set_defaults(func=_cmd_restore)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
