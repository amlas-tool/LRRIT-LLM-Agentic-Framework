from __future__ import annotations

import re
import unicodedata
from typing import Optional, Tuple

from lrrit_llm.evidence.schema import EvidencePack


_WS_RE = re.compile(r"\s+")
_HYPHEN_LINEBREAK_RE = re.compile(r"(\w)[\-\u2010\u2011]\s*\n\s*(\w)")
# Normalise curly quotes and NBSP
_TRANSLATION = str.maketrans({
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "’": "'", "‘": "'", "‚": "'", "‛": "'",
    "\u00A0": " ",
})


def _norm_for_match(s: str) -> str:
    """
    Tolerant normalisation for matching quotes to PDF-extracted text:
    - Unicode normalisation
    - normalise quotes
    - join hyphenated line breaks: 'trans-\\nfer' -> 'transfer'
    - collapse whitespace/newlines
    - lowercase
    """
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.translate(_TRANSLATION)
    s = s.replace("\r\n", "\n")
    s = _HYPHEN_LINEBREAK_RE.sub(r"\1\2", s)
    s = _WS_RE.sub(" ", s).strip().lower()
    return s

import re
from typing import List

_WORD_RE = re.compile(r"[a-z0-9']+")

def _content_tokens(s: str) -> List[str]:
    toks = _WORD_RE.findall(_norm_for_match(s))
    # remove very short tokens to reduce noise
    return [t for t in toks if len(t) >= 5]

def _token_overlap_score(quote: str, block: str, max_tokens: int = 12) -> int:
    qtoks = _content_tokens(quote)[:max_tokens]
    if not qtoks:
        return 0
    b = _norm_for_match(block)
    return sum(1 for t in qtoks if t in b)


def resolve_evidence_id_and_page(
    pack: EvidencePack,
    evidence_id: str,
    quote: str,
) -> Tuple[Optional[str], Optional[int]]:
    """
    Resolve (id, page) for an evidence item.

    Fast path:
      - If evidence_id matches a Text chunk_id or Table table_id, return its provenance.page.

    Repair path:
      - If evidence_id is wrong or missing, search for the quote in all Text chunks and Table fallbacks.
      - If a match is found, return the canonical id (e.g. "Text p03_c01" / "Table p13_t01") and correct page.

    Returns:
      (resolved_id, page) where either may be None if no match could be found.
    """
    evidence_id = (evidence_id or "").strip()
    quote = (quote or "").strip()

    # -------------------
    # 1) Fast path: resolve by ID (but validate quote)
    # -------------------
    if evidence_id:
        # Accept both "Text p01_c01" and bare "p01_c01"
        m = re.search(r"\b(Text|Table)\s+([pP]\d+_[ct]\d+)\b", evidence_id)
        if m:
            kind = m.group(1).lower()
            key = m.group(2)

            if kind == "text":
                for c in pack.text_chunks:
                    if c.chunk_id == key:
                        # Validate quote if provided
                        if quote:
                            if _norm_for_match(quote) in _norm_for_match(c.text):
                                return f"Text {c.chunk_id}", int(c.provenance.page)
                        else:
                            return f"Text {c.chunk_id}", int(c.provenance.page)

            if kind == "table":
                for t in pack.tables:
                    if t.table_id == key:
                        blob = t.text_fallback or ""
                        if quote:
                            if _norm_for_match(quote) in _norm_for_match(blob):
                                return f"Table {t.table_id}", int(t.provenance.page)
                        else:
                            return f"Table {t.table_id}", int(t.provenance.page)

        # If ID exists but quote does not match that block,
        # we DO NOT return here — fall through to repair path.

    # -------------------
    # 2) Repair path: resolve by quote
    # -------------------
    nq = _norm_for_match(quote)
    if not nq:
        return None, None

    # Search Text chunks first (usually preferred for narrative quotes)
    for c in pack.text_chunks:
        if nq in _norm_for_match(c.text):
            return f"Text {c.chunk_id}", int(c.provenance.page)

    # Then search Tables via text_fallback
    for t in pack.tables:
        blob = (t.text_fallback or "")
        if nq in _norm_for_match(blob):
            return f"Table {t.table_id}", int(t.provenance.page)

    return None, None
