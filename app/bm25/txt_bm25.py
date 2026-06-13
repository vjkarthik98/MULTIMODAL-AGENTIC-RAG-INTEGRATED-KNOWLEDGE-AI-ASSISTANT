"""txt_bm25.py — BM25 index for plain-text / transcript chunks.

Implements the main index plus three speaker sub-indexes required by the
MAGIK spec (Phase 3.1):
  - bm25_txt_ceo.pkl  → CEO speaker turns only
  - bm25_txt_cfo.pkl  → CFO speaker turns only
  - bm25_txt_qa.pkl   → Q&A section turns only

Sub-indexes are built/updated automatically alongside the main index whenever
AUDIO_SPEAKER_SUBINDEX_ENABLED is True in config.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, List, Optional

from rank_bm25 import BM25Plus

from app.bm25.base_bm25 import BaseBM25, _INDEX_VERSION
from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

_ROLE_CEO = {"ceo", "chief executive officer", "chief executive", "president & ceo"}
_ROLE_CFO = {"cfo", "chief financial officer", "chief financial"}
_QA_SECTIONS = {"qa_session", "q&a", "question and answer", "qa", "questions and answers"}


def _speaker_role_tag(doc: Any) -> Optional[str]:
    """Return 'ceo', 'cfo', 'qa', or None based on speaker/call_section metadata."""
    s = getattr(doc, "structure", {}) or {}

    call_section = (s.get("call_section") or "").strip().lower().replace("_", " ")
    if any(qa in call_section for qa in _QA_SECTIONS):
        return "qa"

    chunk_type = (s.get("chunk_type") or "").strip().lower()
    if "qa" in chunk_type or "question" in chunk_type:
        return "qa"
    if s.get("is_question"):
        return "qa"

    role = (s.get("speaker_role") or "").strip().lower()
    speaker = (s.get("speaker") or s.get("speaker_name") or "").strip().lower()
    combined = f"{role} {speaker}"

    if any(r in combined for r in _ROLE_CFO):
        return "cfo"
    if any(r in combined for r in _ROLE_CEO):
        return "ceo"
    return None


class TxtBM25(BaseBM25):
    """BM25 index for .txt documents and earnings call transcripts.

    Enrichment over base:
    - section_title repeated ×2 (already in base ×2 — add once more for TXT)
    - FLS amplification: forward-looking chunks get guidance/outlook tokens ×2
    - Speaker prefix: "speaker {role}" token for transcript queries
    - Three speaker sub-indexes (CEO/CFO/QA) when sub-indexing is enabled
    """

    modality = "txt"

    def __init__(self, user_id: Optional[str] = None) -> None:
        super().__init__(user_id=user_id)
        # Speaker sub-indexes — each is a lightweight (corpus, BM25Plus) pair
        self._sub_ceo: List[List[str]] = []
        self._sub_cfo: List[List[str]] = []
        self._sub_qa:  List[List[str]] = []
        self._bm25_ceo: Optional[BM25Plus] = None
        self._bm25_cfo: Optional[BM25Plus] = None
        self._bm25_qa:  Optional[BM25Plus] = None
        self._sub_docs_ceo: List[Any] = []
        self._sub_docs_cfo: List[Any] = []
        self._sub_docs_qa:  List[Any] = []

    def _build_indexed_text(self, doc: Any) -> str:
        s = getattr(doc, "structure", {}) or {}
        parts: List[str] = list(self._base_text(doc))

        # Extra section_title boost for TXT (base already added ×2; add once more)
        section_title = (s.get("section_title") or "").strip()
        if section_title:
            parts.append(section_title)

        # Speaker prefix for transcript speaker-turn chunks
        speaker = (s.get("speaker") or s.get("speaker_name") or "").strip()
        role    = (s.get("speaker_role") or "").strip()
        if speaker:
            parts.append(f"speaker {speaker}")
        if role:
            parts.append(f"speaker role {role}")

        # FLS amplification: forward-looking statements are high-value for guidance queries
        if s.get("is_forward_looking"):
            fls_tokens = "guidance outlook forecast projection expects anticipated"
            parts.append(fls_tokens)
            parts.append(fls_tokens)

        # Call section token
        call_section = (s.get("call_section") or "").strip()
        if call_section:
            parts.append(call_section.replace("_", " "))

        return " ".join(parts)

    # ── Sub-index path helpers ────────────────────────────────────────────────

    def _sub_path(self, role_tag: str, user_id: Optional[str] = None) -> Path:
        from app.utils.paths import user_data_dir
        uid = user_id or self._loaded_user_id or "default"
        return user_data_dir(uid) / "bm25_index" / f"txt_{role_tag}.pkl"

    # ── Override add_documents to also populate sub-indexes ──────────────────

    def add_documents(
        self,
        documents: List[Any],
        session_id: str = "",
        user_id: Optional[str] = None,
    ) -> None:
        # Main index via parent
        super().add_documents(documents, session_id=session_id, user_id=user_id)

        if not getattr(settings, "AUDIO_SPEAKER_SUBINDEX_ENABLED", False):
            return

        for doc in documents:
            tag = _speaker_role_tag(doc)
            if tag is None:
                continue
            tokens = self.tokenize(self._build_indexed_text(doc))
            if not tokens:
                continue
            if tag == "ceo":
                self._sub_ceo.append(tokens)
                self._sub_docs_ceo.append(doc)
            elif tag == "cfo":
                self._sub_cfo.append(tokens)
                self._sub_docs_cfo.append(doc)
            elif tag == "qa":
                self._sub_qa.append(tokens)
                self._sub_docs_qa.append(doc)

        self._rebuild_sub_indexes(user_id)

    def _rebuild_sub_indexes(self, user_id: Optional[str] = None) -> None:
        for tag, corpus, docs, attr_bm25, attr_docs, attr_sub in [
            ("ceo", self._sub_ceo, self._sub_docs_ceo,
             "_bm25_ceo", "_sub_docs_ceo", "_sub_ceo"),
            ("cfo", self._sub_cfo, self._sub_docs_cfo,
             "_bm25_cfo", "_sub_docs_cfo", "_sub_cfo"),
            ("qa",  self._sub_qa,  self._sub_docs_qa,
             "_bm25_qa",  "_sub_docs_qa",  "_sub_qa"),
        ]:
            if not corpus:
                continue
            bm25 = BM25Plus(corpus, k1=1.5, b=0.75)
            setattr(self, attr_bm25, bm25)
            path = self._sub_path(tag, user_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            with open(tmp, "wb") as f:
                pickle.dump({
                    "index_version": _INDEX_VERSION,
                    "documents": getattr(self, attr_docs),
                    "tokenized_corpus": corpus,
                }, f, protocol=pickle.HIGHEST_PROTOCOL)
            tmp.replace(path)
            logger.info(event="bm25_sub_index_saved", tag=tag, docs=len(corpus))

    def search_sub(
        self,
        query: str,
        role_tag: str,
        top_k: int = 10,
    ) -> List[Any]:
        """Search a speaker sub-index (role_tag: 'ceo'|'cfo'|'qa').

        Returns list of (doc, score) pairs, sorted descending.
        Falls back to main index search if sub-index is empty.
        """
        bm25 = getattr(self, f"_bm25_{role_tag}", None)
        docs = getattr(self, f"_sub_docs_{role_tag}", [])
        if bm25 is None or not docs:
            return self.search(query, top_k=top_k)

        tokens = self.tokenize(query)
        if not tokens:
            return []

        scores = bm25.get_scores(tokens)
        import numpy as _np
        if len(scores) == 0:
            return []
        ranked = _np.argsort(scores)[::-1][:top_k]
        return [(docs[i], float(scores[i])) for i in ranked if float(scores[i]) > 0]
