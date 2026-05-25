"""Lexical overlap fallback judge — used when GGUF model is unavailable.

Uses ROUGE-L + token overlap to approximate faithfulness / answer_relevancy.
All scores are clearly labelled as lexical_fallback so reports don't mislead.

This is NOT a substitute for a real LLM judge — it will not catch subtle hallucinations.
"""
from __future__ import annotations

import re
from typing import List, Optional


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())


def _rouge_l(hyp: List[str], ref: List[str]) -> float:
    """ROUGE-L (LCS-based F1) between two token lists."""
    if not hyp or not ref:
        return 0.0
    m, n = len(ref), len(hyp)
    # LCS via DP
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    prec = lcs / n if n else 0.0
    rec = lcs / m if m else 0.0
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def lexical_faithfulness(
    answer: str,
    contexts: List[str],
) -> float:
    """Approximate faithfulness: fraction of answer tokens covered by context tokens."""
    if not answer or not contexts:
        return 0.0
    ctx_tokens = set(_tokenize(" ".join(contexts)))
    ans_tokens = _tokenize(answer)
    if not ans_tokens:
        return 0.0
    covered = sum(1 for t in ans_tokens if t in ctx_tokens)
    return covered / len(ans_tokens)


def lexical_answer_relevancy(
    answer: str,
    question: str,
) -> float:
    """Approximate answer relevancy: ROUGE-L between answer and question."""
    return _rouge_l(_tokenize(answer), _tokenize(question))


def lexical_context_recall(
    contexts: List[str],
    reference_answer: str,
) -> float:
    """Approximate context recall: fraction of reference answer tokens in context."""
    if not contexts or not reference_answer:
        return 0.0
    ctx_tokens = set(_tokenize(" ".join(contexts)))
    ref_tokens = _tokenize(reference_answer)
    if not ref_tokens:
        return 0.0
    covered = sum(1 for t in ref_tokens if t in ctx_tokens)
    return covered / len(ref_tokens)
