from __future__ import annotations

import os
import json
import re
from typing import List, Dict, Any, Optional, Tuple

import pdfplumber

from lrrit_llm.evidence.render import (
    render_markdown_table,
    render_table_text_fallback,
)

# ---------------------------------------------------------------------------
# pdfplumber table settings
#
# NHS reports often come from Word -> PDF. Many tables have ruling lines,
# but some templates are "borderless" (text-positioned tables).
#
# We therefore try a "lines" strategy first, then fall back to "text".
# ---------------------------------------------------------------------------

TABLE_SETTINGS_LINES: Dict[str, Any] = {
    "vertical_strategy": "lines",
    "horizontal_strategy": "lines",
    "snap_tolerance": 3,
    "join_tolerance": 3,
    "edge_min_length": 20,
    "min_words_vertical": 2,
    "min_words_horizontal": 1,
    "intersection_tolerance": 3,
    "text_tolerance": 3,
}

TABLE_SETTINGS_TEXT: Dict[str, Any] = {
    "vertical_strategy": "text",
    "horizontal_strategy": "text",
    "snap_tolerance": 3,
    "join_tolerance": 3,
    "edge_min_length": 20,
    "min_words_vertical": 3,
    "min_words_horizontal": 1,
    "intersection_tolerance": 3,
    "text_tolerance": 3,
}


def extract_tables_from_pdf(
    pdf_path: str,
    report_id: str,
    out_dir: str,
    page_numbers: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    """
    Extract tables using pdfplumber in a way that is more robust to:
      - multi-line cells
      - headers that are split over multiple rows
      - "borderless" tables (text-positioned)

    Returns a list of table dicts suitable for build_evidence_pack().
    Also writes CSV / MD / JSON artefacts to disk.
    """
    tables_out: List[Dict[str, Any]] = []

    tables_dir = os.path.join(out_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)

    with pdfplumber.open(pdf_path) as pdf:
        pages = pdf.pages
        page_idxs = [p - 1 for p in page_numbers] if page_numbers else range(len(pages))
        
        # Track last header seen for a given table signature (to handle multi-page tables)
        last_header_by_sig: Dict[str, List[str]] = {}
        last_row_by_sig: Dict[str, List[str]] = {}


        for pi in page_idxs:
            page = pages[pi]
            page_no = pi + 1

            # Prefer find_tables() with explicit settings (better than page.extract_tables()).
            found, strategy_used = _find_tables_best_effort(page)
            if not found:
                continue

            for ti, table in enumerate(found, start=1):
                try:
                    grid = table.extract()
                except Exception:
                    continue

                if not grid or len(grid) < 2:
                    continue

                # Fix common PDF table artefacts:
                # - header split across multiple rows
                # - word fragments split across rows/columns ("Implemente d")
                # - continuation rows in the body
                header, rows = _fix_table_grid(grid)
                
                header, rows = _fix_table_grid(grid)

                # --- Multi-page continuation handling ---
                bbox = list(table.bbox) if getattr(table, "bbox", None) else None
                sig = _table_signature(len(header) if header else (len(rows[0]) if rows else 0), bbox)

                prev_header = last_header_by_sig.get(sig)

                # Case 1: Our inferred "header" looks like narrative text and body is tiny
                # (common when the table continues and headers are not repeated)
                if header and _looks_like_paragraph_row(header) and prev_header:
                    # Treat "header" as the first body row; reuse previous header
                    rows = [header] + rows
                    header = prev_header

                # Case 2: Header is repeated on next page → drop repeated header row
                if prev_header and header and _headers_similar(header, prev_header):
                    header = prev_header  # standardise
                    # If the first body row is also a repeated header fragment, drop it (rare, but happens)
                    if rows and _headers_similar(rows[0], prev_header):
                        rows = rows[1:]

                # Update remembered header
                if header:
                    last_header_by_sig[sig] = header
                
                # --- Cross-page row stitching ---
                # If we have a previous row for this table signature, and the first row on this page
                # is sparse (mostly empty), treat it as continuation and merge into previous row.
                if prev_header and sig in last_row_by_sig and rows:
                    first = rows[0]
                    if _row_is_sparse(first, max_filled=2):
                        merged = _merge_row_into_prev(last_row_by_sig[sig], first)
                        last_row_by_sig[sig] = merged
                        rows = rows[1:]  # consumed continuation row

                # After finalising rows, update last_row_by_sig with the last row on this page
                if rows:
                    last_row_by_sig[sig] = rows[-1]


                if not header and not rows:
                    continue

                table_id = f"p{page_no:02d}_t{ti:02d}"

                md = render_markdown_table(header, rows, max_rows=12)
                text_fallback = render_table_text_fallback(table_id, page_no, md)

                # Paths
                csv_path = os.path.join(tables_dir, f"{table_id}.csv")
                md_path = os.path.join(tables_dir, f"{table_id}.md")
                json_path = os.path.join(tables_dir, f"{table_id}.json")

                _write_csv(csv_path, header, rows)
                _write_text(md_path, md)

                meta = {
                    "report_id": report_id,
                    "table_id": table_id,
                    "page": page_no,
                    "extractor": "pdfplumber",
                    "strategy": strategy_used,
                    "bbox": list(table.bbox) if getattr(table, "bbox", None) else None,
                    "n_rows": len(rows),
                    "n_cols": len(header) if header else 0,
                    "notes": "header/body post-processed to merge split header rows and wrapped cells",
                }
                _write_json(json_path, meta)

                tables_out.append(
                    {
                        "page": page_no,
                        "extractor": "pdfplumber",
                        "table_id": table_id,
                        "header": header,
                        "rows": rows,
                        "csv_path": csv_path,
                        "md_path": md_path,
                        "json_path": json_path,
                        "text_fallback": text_fallback,
                        "title_hint": None,
                        "bbox": list(table.bbox) if getattr(table, "bbox", None) else None,
                        "confidence": None,
                        "notes": f"pdfplumber strategy={strategy_used}",
                    }
                )

    return tables_out


# ---------------------------------------------------------------------------
# Table detection
# ---------------------------------------------------------------------------

def _find_tables_best_effort(page) -> Tuple[List[Any], str]:
    """
    Try to find tables using a 'lines' strategy first, then fall back to 'text'.
    Returns (tables, strategy_used).
    """
    try:
        tables = page.find_tables(table_settings=TABLE_SETTINGS_LINES)
        if tables:
            return tables, "lines"
    except Exception:
        pass

    try:
        tables = page.find_tables(table_settings=TABLE_SETTINGS_TEXT)
        if tables:
            return tables, "text"
    except Exception:
        pass

    return [], "none"

def _table_signature(n_cols: int, bbox: Optional[List[float]]) -> str:
    """
    Create a coarse signature for identifying a table across pages.
    We mainly use column count + x-span (left/right) rounded.
    """
    if not bbox or len(bbox) != 4:
        return f"cols:{n_cols}|bbox:None"
    x0, y0, x1, y1 = bbox
    return f"cols:{n_cols}|x0:{round(x0, -1)}|x1:{round(x1, -1)}"



def _looks_like_paragraph_row(row: List[str]) -> bool:
    """
    Detect a row that is likely narrative text, not a header.
    Typical continuation rows contain long sentences with punctuation.
    """
    cells = [c for c in row if c]
    if not cells:
        return False

    # If any single cell is "sentence-like", treat as paragraph-like.
    for c in cells:
        words = c.split()
        if len(words) >= 12 and any(p in c for p in [".", ",", ";", ":", "’", "'", "(", ")"]):
            return True

    # Or overall average cell length too high for a header
    avg_len = sum(len(c) for c in cells) / len(cells)
    return avg_len > 35


def _jaccard_similarity(a: str, b: str) -> float:
    a_set = {t for t in re.split(r"\W+", (a or "").lower()) if t}
    b_set = {t for t in re.split(r"\W+", (b or "").lower()) if t}
    if not a_set and not b_set:
        return 1.0
    if not a_set or not b_set:
        return 0.0
    return len(a_set & b_set) / len(a_set | b_set)


def _headers_similar(h1: List[str], h2: List[str], threshold: float = 0.65) -> bool:
    """
    Determine if two header rows are essentially the same.
    Compare concatenated token sets (robust to spacing/wrapping).
    """
    s1 = " ".join([c for c in (h1 or []) if c]).strip()
    s2 = " ".join([c for c in (h2 or []) if c]).strip()
    return _jaccard_similarity(s1, s2) >= threshold

_UNFINISHED_RE = re.compile(r".*\b(to|and|or|for|of|with|until|via|by)\s*$", re.IGNORECASE)

def _row_is_sparse(row: List[str], max_filled: int = 2) -> bool:
    return sum(1 for c in row if c.strip()) <= max_filled

def _find_best_merge_col(prev_row: List[str]) -> int:
    # Prefer an unfinished-looking cell; otherwise the last non-empty cell.
    unfinished = [i for i,c in enumerate(prev_row) if c and _UNFINISHED_RE.match(c)]
    if unfinished:
        return unfinished[-1]
    nonempty = [i for i,c in enumerate(prev_row) if c]
    return nonempty[-1] if nonempty else 0

def _merge_row_into_prev(prev_row: List[str], cont_row: List[str]) -> List[str]:
    # Merge all non-empty fragments from cont_row into the chosen column in prev_row
    out = prev_row[:]
    merge_col = _find_best_merge_col(out)
    fragments = [c for c in cont_row if c]
    if fragments:
        out[merge_col] = (out[merge_col] + " " + " ".join(fragments)).strip() if out[merge_col] else " ".join(fragments)
    return out

# ---------------------------------------------------------------------------
# Grid post-processing
# ---------------------------------------------------------------------------

def _norm_cell(c: Any) -> str:
    return (c or "").replace("\n", " ").strip()


def _rectangularise(rows: List[List[str]]) -> List[List[str]]:
    n_cols = max((len(r) for r in rows), default=0)
    return [(r + [""] * (n_cols - len(r)))[:n_cols] for r in rows]


def _drop_near_empty_rows(rows: List[List[str]], empty_threshold: float = 0.85) -> List[List[str]]:
    """
    Drop rows that are mostly empty (common artefact when pdfplumber splits lines).
    """
    out: List[List[str]] = []
    for r in rows:
        if not r:
            continue
        empties = sum(1 for c in r if c == "")
        if len(r) > 0 and (empties / len(r)) >= empty_threshold:
            continue
        out.append(r)
    return out


def _looks_like_header_fragment_row(row: List[str]) -> bool:
    """
    Heuristic: header fragments tend to have many blanks and short tokens,
    and generally do not contain dates/numbers.
    """
    nonempty = [c for c in row if c]
    if not nonempty:
        return False
    blank_ratio = sum(1 for c in row if not c) / len(row)
    avg_len = sum(len(c) for c in nonempty) / len(nonempty)
    has_digits = any(any(ch.isdigit() for ch in c) for c in nonempty)
    return blank_ratio > 0.25 and avg_len < 28 and not has_digits


def _merge_rows_columnwise(rows: List[List[str]]) -> List[str]:
    """
    Merge multiple rows into one by concatenating non-empty cell fragments per column.
    Also repairs common "word split" artefacts like 'Implemente d' -> 'Implemented'.
    """
    if not rows:
        return []

    n_cols = max(len(r) for r in rows)
    merged = [""] * n_cols

    for r in rows:
        r = (r + [""] * (n_cols - len(r)))[:n_cols]
        for i, c in enumerate(r):
            c = _norm_cell(c)
            if c:
                merged[i] = (merged[i] + " " + c).strip() if merged[i] else c

    merged = [re.sub(r"\s+", " ", m).strip() for m in merged]

    # Fix wrapped-word splits: "Responsibilit y" -> "Responsibility"
    # Only apply when the trailing token is 1–2 lowercase letters.
    merged = [re.sub(r"\b([A-Za-z]{3,})\s+([a-z]{1,2})\b", r"\1\2", m) for m in merged]

    # Fix hyphenation: "moni-\ntoring" -> "monitoring"
    merged = [re.sub(r"(\w)-\s+(\w)", r"\1\2", m) for m in merged]

    return merged


def _merge_continuation_rows(body: List[List[str]], cont_fill_max: int = 2) -> List[List[str]]:
    """
    Merge "continuation" rows into the previous row. Continuation rows typically
    have very few filled cells because wrapped text got split into a new table row.
    """
    out: List[List[str]] = []
    for r in body:
        filled = [i for i, c in enumerate(r) if c]
        if out and len(filled) <= cont_fill_max:
            prev = out[-1]
            for i in filled:
                prev[i] = (prev[i] + " " + r[i]).strip() if prev[i] else r[i]
            out[-1] = prev
        else:
            out.append(r)
    return out


def _fix_table_grid(grid: List[List[Any]]) -> Tuple[List[str], List[List[str]]]:
    """
    Convert a raw pdfplumber grid into:
      - header: single merged header row
      - rows: body rows (with continuation rows stitched)

    Tuned for the failure mode where the header is split across 4–10 rows and
    words are split across rows (e.g., 'Implemente' + 'd').
    """
    # Normalise
    rows = [[_norm_cell(c) for c in (row or [])] for row in (grid or []) if row is not None]
    rows = _rectangularise(rows)
    rows = _drop_near_empty_rows(rows)

    if not rows:
        return [], []

    # Identify header fragment block (first up to 10 rows)
    header_rows: List[List[str]] = []
    body_start = 0
    for i, r in enumerate(rows[:10]):
        if _looks_like_header_fragment_row(r):
            header_rows.append(r)
            body_start = i + 1
        else:
            break

    if header_rows:
        header = _merge_rows_columnwise(header_rows)
        body = rows[body_start:]
    else:
        header = rows[0]
        body = rows[1:]

    # Ensure body is rectangularised to header width
    n_cols = len(header) if header else max((len(r) for r in body), default=0)
    body = [(r + [""] * (n_cols - len(r)))[:n_cols] for r in body]

    # Stitch continuation rows (common with wrapped cells)
    body = _merge_continuation_rows(body, cont_fill_max=2)

    # Drop near-empty rows again after stitching
    body = _drop_near_empty_rows(body, empty_threshold=0.90)

    return header, body


# ---------------------------------------------------------------------------
# File writers
# ---------------------------------------------------------------------------

def _write_csv(path: str, header: List[str], rows: List[List[str]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(_csv_escape(h) for h in header) + "\n")
        for r in rows:
            r = (r + [""] * (len(header) - len(r)))[:len(header)]
            f.write(",".join(_csv_escape(c) for c in r) + "\n")


def _write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _write_json(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _csv_escape(s: str) -> str:
    s = (s or "").replace('"', '""')
    if any(ch in s for ch in [",", "\n", '"']):
        return f'"{s}"'
    return s
