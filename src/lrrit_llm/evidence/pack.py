from __future__ import annotations

from typing import List, Dict, Any, Optional
import json

from lrrit_llm.evidence.schema import (
    EvidencePack,
    Provenance,
    TextChunk,
    TableEvidence,
    stable_hash,
    to_jsonable,
)


def build_evidence_pack(
    report_id: str,
    source_path: str,
    text_pages: List[Dict[str, Any]],
    tables: List[Dict[str, Any]],
    extractor_text_name: str = "pymupdf",
    metadata: Optional[Dict[str, Any]] = None,
) -> EvidencePack:
    """
    Build an EvidencePack from:
      - text_pages: [{ "page": int, "text": str }]
      - tables: list of dicts describing extracted tables (see below)

    The table dicts should include (minimum):
      {
        "page": int,
        "extractor": str,
        "table_id": str,
        "header": list[str] | None,
        "rows": list[list[str]],
        "csv_path": str,
        "md_path": str,
        "json_path": str,
        "text_fallback": str,
        "title_hint": str | None,
        "bbox": [x0,y0,x1,y1] | None,
        "confidence": float | None
      }
    """
    metadata = metadata or {}

    # --- Build text chunks (page-as-chunk baseline) ---
    text_chunks: List[TextChunk] = []
    for p in text_pages:
        page_no = int(p.get("page"))
        text = (p.get("text") or "").strip()
        if not text:
            continue

        chunk_id = f"p{page_no:02d}_c01"
        prov = Provenance(
            report_id=report_id,
            source_path=source_path,
            page=page_no,
            extractor=extractor_text_name,
        )
        text_hash = stable_hash({"text": text})
        text_chunks.append(
            TextChunk(
                chunk_id=chunk_id,
                provenance=prov,
                text=text,
                text_hash=text_hash,
            )
        )

    # --- Build table evidence ---
    table_evidence: List[TableEvidence] = []
    for t in tables:
        page_no = int(t.get("page"))
        extractor = t.get("extractor") or "pdfplumber"
        table_id = t.get("table_id") or f"p{page_no:02d}_t00"

        prov = Provenance(
            report_id=report_id,
            source_path=source_path,
            page=page_no,
            extractor=extractor,
            bbox=t.get("bbox"),
            confidence=t.get("confidence"),
            notes=t.get("notes"),
        )

        header = t.get("header")
        rows = t.get("rows") or []
        n_rows = len(rows)
        n_cols = len(header) if header else (len(rows[0]) if rows else 0)

        table_hash = stable_hash(
            {
                "table_id": table_id,
                "page": page_no,
                "extractor": extractor,
                "header": header,
                "rows": rows,
            }
        )

        table_evidence.append(
            TableEvidence(
                table_id=table_id,
                provenance=prov,
                title_hint=t.get("title_hint"),
                n_rows=n_rows,
                n_cols=n_cols,
                header=header,
                rows=rows,
                csv_path=t.get("csv_path") or "",
                md_path=t.get("md_path") or "",
                json_path=t.get("json_path") or "",
                table_hash=table_hash,
                text_fallback=t.get("text_fallback") or "",
            )
        )

    # --- Pack-level hash (audit/versioning) ---
    pack_hash = stable_hash(
        {
            "report_id": report_id,
            "source_path": source_path,
            "text_chunks": [c.text_hash for c in text_chunks],
            "tables": [te.table_hash for te in table_evidence],
            "metadata": metadata,
        }
    )

    pack = EvidencePack(
        report_id=report_id,
        source_path=source_path,
        text_chunks=text_chunks,
        tables=table_evidence,
        pack_hash=pack_hash,
        metadata=metadata,
    )

    return pack


def save_evidence_pack(pack: EvidencePack, out_path: str) -> None:
    """
    Serialize EvidencePack to JSON (human-inspectable, stable for audit).
    """
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(pack), f, ensure_ascii=False, indent=2)


def load_evidence_pack(in_path: str) -> EvidencePack:
    """
    Load an EvidencePack JSON file produced by save_evidence_pack.
    """
    with open(in_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    text_chunks = [
        TextChunk(
            chunk_id=chunk["chunk_id"],
            provenance=Provenance(**chunk["provenance"]),
            text=chunk["text"],
            text_hash=chunk["text_hash"],
        )
        for chunk in payload.get("text_chunks", [])
    ]

    tables = [
        TableEvidence(
            table_id=table["table_id"],
            provenance=Provenance(**table["provenance"]),
            title_hint=table.get("title_hint"),
            n_rows=table["n_rows"],
            n_cols=table["n_cols"],
            header=table.get("header"),
            rows=table.get("rows", []),
            csv_path=table.get("csv_path", ""),
            md_path=table.get("md_path", ""),
            json_path=table.get("json_path", ""),
            table_hash=table["table_hash"],
            text_fallback=table.get("text_fallback", ""),
        )
        for table in payload.get("tables", [])
    ]

    return EvidencePack(
        report_id=payload["report_id"],
        source_path=payload["source_path"],
        text_chunks=text_chunks,
        tables=tables,
        pack_hash=payload["pack_hash"],
        metadata=payload.get("metadata", {}),
    )
