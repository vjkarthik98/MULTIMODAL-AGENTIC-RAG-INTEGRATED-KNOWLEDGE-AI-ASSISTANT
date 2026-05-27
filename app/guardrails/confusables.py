"""Unicode confusables normalization (Unicode TR39).

Covers the two known gaps that NFKC alone cannot fix:
  - enc-006: modifier/superscript letters (ᴵᴳᴻᴼᴿᴱ → IGNORE)
  - enc-007: Latin-extended substitutions (ignøre → ignore)

Uses a curated map of the most attack-relevant confusables rather than
shipping the full 7000-entry Unicode TR39 dataset (too large for startup).

Integrated into input_guard._normalize_encoding() as a post-NFKC step.
"""
from __future__ import annotations

import re
import unicodedata

# Curated confusables map — covers characters actually seen in adversarial prompts.
# Entries: confusable_char → ASCII_replacement
# Source: Unicode TR39 confusables.txt filtered to injection-relevant chars.
_CONFUSABLES: dict[str, str] = {
    # Latin extended / diacritics used as letter substitutes
    "à": "a", "á": "a", "â": "a", "ã": "a", "ä": "a", "å": "a", "ā": "a", "ă": "a",
    "è": "e", "é": "e", "ê": "e", "ë": "e", "ē": "e", "ĕ": "e",
    "ì": "i", "í": "i", "î": "i", "ï": "i", "ī": "i", "ĭ": "i",
    "ò": "o", "ó": "o", "ô": "o", "õ": "o", "ö": "o", "ø": "o", "ō": "o", "ŏ": "o",
    "ù": "u", "ú": "u", "û": "u", "ü": "u", "ū": "u", "ŭ": "u",
    "ñ": "n", "ń": "n", "ņ": "n",
    "ç": "c", "ć": "c", "ĉ": "c",
    "ß": "ss",
    "ð": "d", "đ": "d",
    "þ": "th",
    "ý": "y", "ÿ": "y",
    "ĝ": "g", "ğ": "g",
    "ĥ": "h",
    "ĵ": "j",
    "ķ": "k",
    "ļ": "l", "ĺ": "l", "ľ": "l",
    "ř": "r",
    "š": "s", "ś": "s", "ş": "s",
    "ţ": "t", "ť": "t",
    "ž": "z", "ź": "z", "ż": "z",

    # Superscript / modifier letters (enc-006 gap)
    "ᵃ": "a", "ᵇ": "b", "ᶜ": "c", "ᵈ": "d", "ᵉ": "e", "ᶠ": "f", "ᵍ": "g",
    "ʰ": "h", "ⁱ": "i", "ʲ": "j", "ᵏ": "k", "ˡ": "l", "ᵐ": "m", "ⁿ": "n",
    "ᵒ": "o", "ᵖ": "p", "ʳ": "r", "ˢ": "s", "ᵗ": "t", "ᵘ": "u", "ᵛ": "v",
    "ʷ": "w", "ˣ": "x", "ʸ": "y", "ᶻ": "z",
    "ᴬ": "A", "ᴮ": "B", "ᴰ": "D", "ᴱ": "E", "ᴳ": "G", "ᴴ": "H", "ᴵ": "I",
    "ᴶ": "J", "ᴷ": "K", "ᴸ": "L", "ᴹ": "M", "ᴺ": "N", "ᴼ": "O", "ᴾ": "P",
    "ᴿ": "R", "ᵀ": "T", "ᵁ": "U", "ᵂ": "W",
    # Modifier letters used in enc-006 test case
    "ᴻ": "N", "ᵁ": "U", "ᴴ": "H",

    # Greek lookalikes
    "α": "a", "β": "b", "γ": "c", "ε": "e", "ζ": "z", "η": "n", "ι": "i",
    "κ": "k", "μ": "m", "ν": "n", "ο": "o", "ρ": "p", "τ": "t", "υ": "u",
    "χ": "x", "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I",
    "Κ": "K", "Μ": "M", "Ν": "N", "Ο": "O", "Ρ": "P", "Τ": "T", "Υ": "Y",
    "Χ": "X",

    # Cyrillic lookalikes
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x",
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O",
    "Р": "P", "С": "C", "Т": "T", "У": "Y", "Х": "X",

    # Mathematical bold/italic/script (NFKC maps most but misses some)
    "𝐚": "a", "𝐛": "b", "𝐜": "c", "𝐝": "d", "𝐞": "e", "𝐟": "f", "𝐠": "g",
    "𝐡": "h", "𝐢": "i", "𝐣": "j", "𝐤": "k", "𝐥": "l", "𝐦": "m", "𝐧": "n",
    "𝐨": "o", "𝐩": "p", "𝐪": "q", "𝐫": "r", "𝐬": "s", "𝐭": "t", "𝐮": "u",
    "𝐯": "v", "𝐰": "w", "𝐱": "x", "𝐲": "y", "𝐳": "z",
    "𝐀": "A", "𝐁": "B", "𝐂": "C", "𝐃": "D", "𝐄": "E", "𝐅": "F", "𝐆": "G",
    "𝐇": "H", "𝐈": "I", "𝐉": "J", "𝐊": "K", "𝐋": "L", "𝐌": "M", "𝐍": "N",
    "𝐎": "O", "𝐏": "P", "𝐐": "Q", "𝐑": "R", "𝐒": "S", "𝐓": "T", "𝐔": "U",
    "𝐕": "V", "𝐖": "W", "𝐗": "X", "𝐘": "Y", "𝐙": "Z",

    # Whitespace / separator lookalikes → standard space
    " ": " ",   # non-breaking space
    " ": " ",   # en quad
    " ": " ",   # em quad
    " ": " ",   # en space
    " ": " ",   # em space
    " ": " ",   # three-per-em space
    " ": " ",   # four-per-em space
    " ": " ",   # six-per-em space
    " ": " ",   # figure space
    " ": " ",   # punctuation space
    " ": " ",   # thin space
    " ": " ",   # hair space
    " ": " ",   # narrow no-break space
    " ": " ",   # medium mathematical space
    "　": " ",   # ideographic space
    "\t":     " ",   # tab → space (enc-008: tab-separated injection)
}

# Build translation table once at import time
_TABLE = str.maketrans(_CONFUSABLES)


def normalize_confusables(text: str) -> str:
    """Apply confusables map then re-run NFKC to catch chained substitutions."""
    return unicodedata.normalize("NFKC", text.translate(_TABLE))
