"""Image citation locators must survive the BM25 path and RRF fusion.

Symptom: some queries showed an image source as a bare filename chip with no
caption/title, while the same image cited correctly for other queries. Two
independent causes, both covered here:

  1. BM25 never carried `image_title` (the citation-grade chart title) out of
     a chunk's structure — the same class of bug already fixed for XLSX
     `sheet_name` and audio `speaker`/timestamps.
  2. RRF fusion keyed the same chunk from both retrievers to one entry and
     kept the FIRST writer's metadata. BM25 is fused before dense, so any
     chunk BM25 also matched had its richer Qdrant metadata discarded — which
     is exactly why it was query-dependent.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from app.bm25.base_bm25 import BaseBM25
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.hybrid_retriever import HybridRetriever

_IMAGE_STRUCTURE = {
    "chunk_id": "c1",
    "image_title": "Apple vs. S&P 500 cumulative return",
    "image_type": "line_chart",
    "asset_path": "/data/u1/knowledge_base/aapl-20240928_g2.jpg",
    "caption": "A line chart comparing cumulative five-year returns. " * 12,
}


def _image_doc():
    doc = MagicMock()
    doc.structure = dict(_IMAGE_STRUCTURE)
    doc.modality = "image"
    doc.subtype = "image_caption"
    doc.source = "aapl-20240928_g2.jpg"
    doc.source_type = "image"
    doc.chunk_id = "c1"
    doc.page = None
    return doc


# ── BM25 metadata ─────────────────────────────────────────────────────────────

class TestBM25CarriesImageLocators:
    """`_metadata` is what the citation layer actually reads. Both BM25
    implementations build it from `doc.structure`, so the fields exist on
    already-indexed documents — they were simply dropped on the way out."""

    def test_base_bm25_metadata_has_image_title(self):
        meta = BaseBM25._metadata(None, _image_doc())
        assert meta["image_title"] == _IMAGE_STRUCTURE["image_title"]
        assert meta["image_type"] == "line_chart"
        assert meta["asset_path"] == _IMAGE_STRUCTURE["asset_path"]

    def test_bm25_retriever_metadata_has_image_title(self):
        meta = BM25Retriever._metadata(None, _image_doc())
        assert meta["image_title"] == _IMAGE_STRUCTURE["image_title"]
        assert meta["image_type"] == "line_chart"
        assert meta["asset_path"] == _IMAGE_STRUCTURE["asset_path"]

    def test_caption_is_still_carried_separately(self):
        # The chip shows the title; the caption stays available but is NOT a
        # substitute for it (it's a multi-paragraph analysis dump).
        meta = BaseBM25._metadata(None, _image_doc())
        assert meta["caption"] == _IMAGE_STRUCTURE["caption"]
        assert meta["image_title"] != meta["caption"]

    def test_non_image_chunk_gets_none_not_a_crash(self):
        doc = MagicMock()
        doc.structure = {"chunk_id": "c9"}
        doc.modality = "pdf"
        doc.subtype = None
        doc.source = "apple_10k.pdf"
        doc.source_type = "pdf"
        doc.chunk_id = "c9"
        doc.page = 12
        for meta in (BaseBM25._metadata(None, doc), BM25Retriever._metadata(None, doc)):
            assert meta["image_title"] is None
            assert meta["page"] == 12

    def test_from_payload_round_trips_image_title(self):
        # The rebuild-from-Qdrant path (index reconstruction) must keep it too.
        from app.bm25.base_bm25 import BM25Document as BaseDoc
        from app.retrieval.bm25_retriever import BM25Document as RetrieverDoc

        payload = {
            "text": "chart caption",
            "modality": "image",
            "source": "aapl-20240928_g2.jpg",
            "image_title": "Apple vs. S&P 500 cumulative return",
            "asset_path": "/data/u1/aapl.jpg",
        }
        for cls in (BaseDoc, RetrieverDoc):
            obj = cls.from_payload(payload)
            assert obj.structure["image_title"] == payload["image_title"]
            assert obj.structure["asset_path"] == payload["asset_path"]


# ── RRF fusion metadata merge ─────────────────────────────────────────────────

def _retriever() -> HybridRetriever:
    return HybridRetriever(bm25=MagicMock(), vector_store=MagicMock(), embedder=MagicMock())


class TestFusionMergesMetadata:

    def test_dense_only_field_survives_a_bm25_first_fusion(self):
        r = _retriever()
        combined: dict = {}
        text = "Apple's cumulative return chart"
        bm25_hit = {
            "text": text,
            "metadata": {"chunk_id": "c1", "source": "aapl.jpg", "modality": "image"},
        }
        dense_hit = {
            "text": text,
            "metadata": {
                "chunk_id": "c1",
                "source": "aapl.jpg",
                "modality": "image",
                "image_title": "Apple vs. S&P 500 cumulative return",
                "asset_path": "/data/u1/aapl.jpg",
            },
        }
        r._fuse(combined, [bm25_hit], 0.5, "bm25")
        r._fuse(combined, [dense_hit], 0.5, "dense")

        assert len(combined) == 1, "same chunk must stay one fused entry"
        merged = next(iter(combined.values()))
        assert merged["metadata"]["image_title"] == "Apple vs. S&P 500 cumulative return"
        assert merged["metadata"]["asset_path"] == "/data/u1/aapl.jpg"
        assert merged["sources"] == {"bm25", "dense"}

    def test_merge_never_overwrites_a_resolved_value(self):
        target = {"image_title": "From dense", "page": 3}
        HybridRetriever._merge_missing_metadata(target, {"image_title": "From bm25", "page": 99})
        assert target == {"image_title": "From dense", "page": 3}

    def test_merge_fills_only_missing_or_none(self):
        target = {"image_title": None, "source": "a.jpg"}
        HybridRetriever._merge_missing_metadata(
            target, {"image_title": "Chart", "asset_path": "/a.jpg", "source": "other.jpg"}
        )
        assert target["image_title"] == "Chart"
        assert target["asset_path"] == "/a.jpg"
        assert target["source"] == "a.jpg", "a populated value is never replaced"

    def test_merge_ignores_none_from_the_other_side(self):
        target = {"image_title": "Chart"}
        HybridRetriever._merge_missing_metadata(target, {"image_title": None, "page": None})
        assert target["image_title"] == "Chart"
        assert target.get("page") is None

    def test_fusion_still_sums_scores(self):
        r = _retriever()
        combined: dict = {}
        hit = {"text": "shared chunk", "metadata": {"chunk_id": "c1"}}
        r._fuse(combined, [dict(hit)], 0.5, "bm25")
        single = next(iter(combined.values()))["score"]
        r._fuse(combined, [dict(hit)], 0.5, "dense")
        assert next(iter(combined.values()))["score"] > single

    def test_missing_embedding_is_filled_from_the_other_retriever(self):
        r = _retriever()
        combined: dict = {}
        r._fuse(combined, [{"text": "t", "metadata": {"chunk_id": "c1"}}], 0.5, "bm25")
        r._fuse(
            combined,
            [{"text": "t", "metadata": {"chunk_id": "c1"}, "embedding": [0.1, 0.2]}],
            0.5,
            "dense",
        )
        assert next(iter(combined.values()))["embedding"] == [0.1, 0.2]
