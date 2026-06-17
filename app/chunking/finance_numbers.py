from __future__ import annotations

import hashlib
import re
from typing import Dict, List, Tuple

try:
    import tiktoken as _tiktoken
    _enc = _tiktoken.get_encoding("cl100k_base")

    def approx_tokens(text: str) -> int:
        return len(_enc.encode(text, disallowed_special=()))
except Exception:
    def approx_tokens(text: str) -> int:  # type: ignore[misc]
        return int(len(text.split()) * 1.3)


# Finance number pattern — these tokens must NEVER be split at internal commas or dots.
_FIN_RE = re.compile(
    r"""(?x)
    [$€£¥₹₽][\d,]+(?:\.\d+)?[BMKTbmkt]?                        # $1,234.56B  €2.3T
    | [\d,]+(?:\.\d+)?\s?(?:billion|million|trillion|thousand)  # 4.3 billion
    | -?\d[\d,]*(?:\.\d+)?%                                     # -4.2%  23.5%
    | -?\d[\d,]*(?:\.\d+)?\s+percent(?:age\s+point)?s?          # 2.8 percent / 1/4 percentage point
    | \d[\d,]*(?:\.\d+)?\s?bps                                  # 350 bps
    | \d[\d,]*(?:\.\d+)?\s+basis\s+points?                      # 100 basis points
    | \d/\d\s+percent(?:age\s+point)?s?                         # 1/4 percentage point (fraction)
    | \d[\d,]*(?:\.\d+)?[xX]                                    # 12.3x
    | Q[1-4]\s?FY?\s?\d{2,4}                                    # Q3FY25  Q3 2024
    | FY\s?\d{2,4}                                              # FY2024
    | H[12]\s?\d{4}                                             # H1 2024
    | \(\d[\d,]*(?:\.\d+)?\)                                    # (2,345) accounting negative
    """,
    re.IGNORECASE,
)

_SLOT = "\x02FIN_{n}\x03"


def protect(text: str) -> Tuple[str, Dict[str, str]]:
    """Replace finance numbers with stable single-token placeholders.

    Returns (protected_text, mapping) where mapping is placeholder→original.
    The placeholders survive any whitespace-based splitter because they contain
    no spaces, commas, or dots.
    """
    mapping: Dict[str, str] = {}
    counter = [0]

    def _sub(m: re.Match) -> str:
        key = _SLOT.format(n=counter[0])
        mapping[key] = m.group()
        counter[0] += 1
        return key

    return _FIN_RE.sub(_sub, text), mapping


def restore(text: str, mapping: Dict[str, str]) -> str:
    """Restore protected finance number placeholders to their original form."""
    for k, v in mapping.items():
        text = text.replace(k, v)
    return text


def deterministic_chunk_id(source: str, locator: str, idx: int) -> str:
    """sha256(source|locator|idx)[:16] — stable across re-ingestion runs."""
    raw = f"{source}|{locator}|{idx}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def extract_finance_entities(text: str) -> List[str]:
    """Lightweight regex extractor — returns unique finance entities in order."""
    return list(dict.fromkeys(_FIN_RE.findall(text)))
