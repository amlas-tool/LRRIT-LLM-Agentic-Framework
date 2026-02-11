from __future__ import annotations

import base64
from email.utils import quote
import re
import json
import base64
import html
from pathlib import Path
from datetime import datetime

# NHS-ish palette (approx)
NHS_BLUE = "#005EB8"
NHS_LIGHT_BLUE = "#E8F1FB"
NHS_DARK = "#0B0C0C"
NHS_GREY = "#F3F2F1"
NHS_RED = "#D5281B"
NHS_AMBER = "#FFB81C"
NHS_GREEN = "#007F3B"


def _file_url(p: str) -> str:
    # Produces a properly URL-encoded file:// URL (spaces handled)
    return Path(p).resolve().as_uri()


def _laj_badge(overall: str):
    overall = (overall or "").upper()
    col = _badge_colour("GOOD" if overall == "PASS" else "SOME" if overall == "WARN" else "LITTLE")
    return overall, col


def _laj_tooltip(laj_obj: dict) -> str:
    metrics = laj_obj.get("metrics") or []
    bad = [m["metric_id"] for m in metrics if (m.get("score") or "").upper() in ("FAIL","WARN")]
    if not bad:
        return "All metrics PASS"
    return "Flags: " + ", ".join(bad[:6])


_PAGE_RE = re.compile(r"p(\d{1,3})", re.IGNORECASE)


def _page_from_evidence_id(ev_id: str) -> int | None:
    if not ev_id:
        return None
    m = _PAGE_RE.search(ev_id)
    if not m:
        return None
    return int(m.group(1))


def _badge_colour(value: str) -> str:
    v = (value or "").upper().strip()
    if v in ("GOOD", "NO", "TRUE"):
        return NHS_GREEN
    if v in ("SOME",):
        return NHS_AMBER
    if v in ("LITTLE", "YES", "FALSE"):
        return NHS_RED
    return NHS_BLUE


def _esc(s: str) -> str:
    return html.escape(s or "")

def render_laj_details(laj: dict) -> str:
    metrics = laj.get("metrics") or []
    name_map = {
        "M1": "Rubric fidelity",
        "M2": "Evidence grounding",
        "M3": "Reasoning coherence",
        "M4": "Values alignment (PSIRF/LRRIT)",
        "M5": "Transparency & uncertainty",
        "M6": "Unsupported-claim risk",
    }

    # Build rows
    rows = []
    for m in metrics:
        mid = (m.get("metric_id") or "").upper()
        score = (m.get("score") or "").upper()
        note = m.get("notes") or ""
        label = name_map.get(mid, mid or "Metric")

        score_cls = (
            "laj-score-fail" if score == "FAIL"
            else "laj-score-warn" if score == "WARN"
            else "laj-score-pass"
        )

        rows.append(f"""
          <tr>
            <td class="laj-metric-name">
              {_esc(label)} <span class="{score_cls}">({_esc(score)})</span>
            </td>
            <td class="laj-metric-note">{_esc(note)}</td>
          </tr>
        """)

    # If no metrics, be explicit
    if not rows:
        rows.append("""
          <tr>
            <td class="laj-metric-name">No metrics returned</td>
            <td class="laj-metric-note"></td>
          </tr>
        """)

    return f"""
      <div class="laj-metrics-wrap">
        <table class="laj-metrics-table">
          <tbody>
            {''.join(rows)}
          </tbody>
        </table>
      </div>
    """



def render_html(report_dir: Path) -> Path:
    results_path = report_dir / "agent_results.json"
    pack_path = report_dir / "evidence_pack.json"
    laj_path = report_dir / "laj_results.json"
    laj_results = {}
    if laj_path.exists():
        laj_results = json.loads(laj_path.read_text(encoding="utf-8"))

    if not results_path.exists():
        raise FileNotFoundError(f"Missing: {results_path}")

    results = json.loads(results_path.read_text(encoding="utf-8"))

    # Model metadata (preferred in results["_meta"], with safe fallbacks)
    meta = results.get("_meta", {}) if isinstance(results, dict) else {}
    model_name = (meta.get("model") or meta.get("openai_model") or "").strip()
    
    if not model_name:
        # Some earlier runners may store this at top-level
        model_name = (results.get("model") or "").strip()
    if not model_name:
        # Last resort: try env var (renderer is often run in same env as runner)
        import os
        model_name = (os.environ.get("OPENAI_MODEL") or "unknown").strip()
    
    pdf_url = None
    meta = results.get("_meta", {}) if isinstance(results, dict) else {}
    if isinstance(meta, dict):
        if meta.get("pdf_url"):
            pdf_url = meta["pdf_url"]
        elif meta.get("pdf_path"):
            pdf_url = _file_url(meta["pdf_path"])


    pack = None
    if pack_path.exists():
        try:
            pack = json.loads(pack_path.read_text(encoding="utf-8"))
        except Exception:
            pack = None

    # Extract a few header fields if available
    report_id = report_dir.name
    source_path = (pack or {}).get("source_path", "")
    pack_hash = (pack or {}).get("pack_hash", "")
    chunk_count = len((pack or {}).get("text_chunks", []) or [])
    table_count = len((pack or {}).get("tables", []) or [])

    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Order agents by numeric id if possible and ignore non-agent keys (e.g. _meta)
    def sort_key(item):
        k = str(item[0]).lower()
        if k.startswith("d") and k[1:].isdigit():
            return (0, int(k[1:]))
        return (1, k)

    agent_items = []
    for k, v in (results or {}).items():
        if k == "_meta":
            continue
        if isinstance(v, dict) and (v.get("agent_id") or str(k).lower().startswith("d")):
            agent_items.append((k, v))
    agent_items = sorted(agent_items, key=sort_key)

    # Build summary table (one line per dimension, clickable to jump to card)
    summary_rows = []
    for key, obj in agent_items:
        agent_id = obj.get("agent_id", key)
        dim = obj.get("dimension", "")
        rating = obj.get("rating", "")
        uncertainty = obj.get("uncertainty", False)

        rating_col = _badge_colour(rating)
        uncert_col = _badge_colour("YES" if uncertainty else "NO")

        anchor = f"dim-{_esc(agent_id).lower()}"
        
        laj = laj_results.get(agent_id.lower()) or laj_results.get(agent_id) or {}
        overall = laj.get("overall", "")
        overall_txt, overall_col = _laj_badge(overall)

        laj_cell = '<span class="muted">—</span>'
        if overall_txt:
            laj_cell = (
                f'<span class="pill" style="background:{overall_col}" '
                f'title="{_esc(_laj_tooltip(laj))}">'
                f'{overall_txt}</span>'
            )


        summary_rows.append(f"""
        <tr class=\"summary-row\" onclick=\"location.href='#{anchor}'\" tabindex=\"0\" role=\"link\">
          <td class=\"mono\">{_esc(agent_id)}</td>
          <td>{_esc(dim)}</td>
          <td style=\"text-align: center;\"><span class=\"pill\" style=\"background:{rating_col}\">{_esc(rating)}</span></td>
          
          <td style=\"text-align: center;\">{laj_cell}</td>
        </tr>
          """
          )
     
        # <td><span class=\"pill\" style=\"background:{uncert_col}\">{'YES' if uncertainty else 'NO'}</span></td>
         

    cards_html = []
    for key, obj in agent_items:
        agent_id = obj.get("agent_id", key)
        dim = obj.get("dimension", "")
        rating = obj.get("rating", "")
        uncertainty = obj.get("uncertainty", False)
        rationale = obj.get("rationale", "")
        evidence = obj.get("evidence", []) or []

        rating_col = _badge_colour(rating)
        uncert_col = _badge_colour("YES" if uncertainty else "NO")

        # Evidence list
        #pdf_url = meta.get("pdf_url") or meta.get("pdf_path") or os.environ.get("LRRIT_PDF_URL", "")

        ev_rows = []
        if evidence:
            for e in evidence:
                eid = e.get("id", "")
                quote = e.get("quote", "")
                etype = e.get("evidence_type", "")

                et_col = _badge_colour(
                    "GOOD" if etype == "positive"
                    else "LITTLE" if etype == "negative"
                    else "SOME"
                )
                eid = e.get("id", "")
                ev_id_display = eid if eid else "Unresolved source"
                # Prefer explicit page from resolver; fallback to parsing id for older files.
                page = e.get("page")
                if page is None:
                    page = _page_from_evidence_id(eid)

                copy_b64 = base64.b64encode((quote or "").encode("utf-8")).decode("ascii")

                action_html = ""
                if pdf_url and page:
                    pdf_href = f"{pdf_url}#page={page}"
                    action_html = (
                        f'<a class="btn btn-compact" target="lrrit_pdf_tab" '
                        f'href="{_esc(pdf_href)}" '
                        f'data-copy="{copy_b64}" '
                        f'onclick="copyB64(this.dataset.copy);">Open report (page {page})</a>'
                    )
                else:
                    # No resolvable page means quote couldn't be located verbatim in the EvidencePack.
                    # (Often due to non-verbatim / normalisation / hallucinated fragments.)
                    action_html = '<span class="pill warn">Verbatim quote not found</span>'

                #print("DEBUG evidence:", eid, "page=", page, "pdf_url=", bool(pdf_url))
                laj = laj_results.get(agent_id.lower()) or laj_results.get(agent_id) or {}
                laj_overall = (laj.get("overall") or "").upper()
                tooltip = _esc(_laj_tooltip(laj))
                laj_html = ""
                if laj_overall:
                  overall_txt, overall_col = _laj_badge(laj_overall)
                  laj_html = f"""
                  <details class="laj-details">
                    <summary> 
                      <span class="pill" style="background:{overall_col}" title="{tooltip}">{agent_id}: {overall_txt}</span>
                      <span class="laj-summary-link">View detailed evaluation metrics for {agent_id}</span>
                    </summary>
                    {render_laj_details(laj)}
                  </details>
                  """               


                ev_rows.append(f"""
                  <div class="ev-row">
                    <div class="ev-meta">
                      <span class="pill" style="background:{et_col}">{_esc(etype or "evidence")}</span>
                      <span class="ev-id">{_esc(ev_id_display)}</span>
                    </div>

                    <div class="ev-main">
                      <div class="ev-quote">“{_esc(quote)}”</div>
                      <div class="ev-action">{action_html}</div>
                    </div>
                  </div>
                  """)
            else:
                ev_rows.append(f'<div class="muted">No more evidence quotes returned.</div>'
                                f'<h3>{agent_id} Task Evaluation</h3>{laj_html}')



        anchor = f"dim-{_esc(agent_id).lower()}"

        cards_html.append(f"""
        <section class="card" id="{anchor}">
          <div class="card-head">
            <div>
              <div class="agent-title">{_esc(agent_id)} — {_esc(dim)}</div>
              <div class="muted">Key: positive = supports dimension, negative = contrary/weakening evidence</div>
            </div>
            <div class="badges">
              <div class="badge">
                <div class="badge-label">Rating</div>
                <div class="pill" style="background:{rating_col}">{_esc(rating)}</div>
              </div>
            </div>
          </div>

          <div class="card-body">
            <h3>Rationale</h3>
            <p>{_esc(rationale)}</p>

            <h3>Evidence</h3>
            {''.join(ev_rows)}

            
          </div>
        </section>
        """)

              #         <div class="badge">
              #   <div class="badge-label">Uncertainty</div>
              #   <div class="pill" style="background:{uncert_col}">{'YES' if uncertainty else 'NO'}</div>
              # </div>

    html_out = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>LRRIT Agent Results — {html.escape(report_id)}</title>
  <style>
    :root {{
      --nhs-blue: {NHS_BLUE};
      --nhs-light: {NHS_LIGHT_BLUE};
      --nhs-dark: {NHS_DARK};
      --nhs-grey: {NHS_GREY};
    }}
    body {{
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      color: var(--nhs-dark);
      background: var(--nhs-grey);
    }}
    header {{
      background: var(--nhs-blue);
      color: white;
      padding: 20px 24px;
    }}
    .wrap {{
      max-width: 1100px;
      margin: 0 auto;
      padding: 18px 18px 40px 18px;
    }}
    .meta {{
      background: white;
      border-radius: 12px;
      padding: 14px 16px;
      margin-top: -18px;
      box-shadow: 0 2px 10px rgba(0,0,0,0.08);
      border-left: 6px solid var(--nhs-blue);
    }}
    .meta-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px 18px;
      margin-top: 8px;
    }}
    .mono {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
    }}
    .summary {{
      background: white;
      border-radius: 12px;
      padding: 14px 16px;
      margin-top: 16px;
      box-shadow: 0 2px 10px rgba(0,0,0,0.06);
      border: 1px solid #e6e6e6;
      border-left: 6px solid var(--nhs-blue);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 10px;
    }}
    th, td {{
      padding: 10px 8px;
      border-bottom: 1px solid #eee;
      vertical-align: top;
    }}
    th {{
      font-size: 12px;
      letter-spacing: 0.2px;
      text-transform: uppercase;
      opacity: 0.85;
    }}
    tr.summary-row {{
      cursor: pointer;
    }}
    tr.summary-row:hover {{
      background: #f7f7f7;
    }}
    .meta .k {{ font-weight: 700; }}
    .meta .v {{ word-break: break-all; }}
    .card {{
      background: white;
      border-radius: 12px;
      margin-top: 16px;
      box-shadow: 0 2px 10px rgba(0,0,0,0.06);
      overflow: hidden;
      border: 1px solid #e6e6e6;
    }}
    .card-head {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      padding: 14px 16px;
      background: var(--nhs-light);
      border-bottom: 1px solid #e6e6e6;
    }}
    .agent-title {{
      font-size: 18px;
      font-weight: 800;
      margin-bottom: 4px;
    }}
    .badges {{
      display: flex;
      gap: 12px;
      align-items: center;
    }}
    .badge-label {{
      text-transform: uppercase;
      text-align: center;
      font-size: 13px;
      opacity: 0.85;
      margin-bottom: 6px;
    }}
    .pill {{
      display: inline-block;
      color: white;
      font-weight: 600;
      border-radius: 999px;
      padding: 6px 10px;
      font-size: 12px;
      letter-spacing: 0.2px;
    }}
    .pill.warn {{ 
      background:#FFB81C; 
      font-weight: 500;
      font-size: 12px;
      color:#000; 
    }}
    .card-body {{
      padding: 14px 16px 18px 16px;
    }}
    h3 {{
      margin: 14px 0 6px 0;
      font-size: 14px;
      letter-spacing: 0.2px;
      text-transform: uppercase;
      opacity: 0.8;
    }}
    p {{
      margin: 0;
      line-height: 1.45;
    }}
    .muted {{
      opacity: 0.75;
      font-size: 12px;
    }}
    .ev-row {{
      border-left: 4px solid var(--nhs-blue);
      padding: 10px 10px;
      margin: 10px 0;
      background: #fafafa;
      border-radius: 8px;
    }}
    .ev-main{{
      display:flex;
      gap:12px;
      align-items:flex-start;
      justify-content:space-between;
      flex-wrap: nowrap;          /* key: don’t wrap action under quote */
    }}
    .ev-meta {{
      display: flex;
      gap: 12px;
      align-items: center;
      margin-bottom: 6px;
    }}
    .ev-id {{
      font-size: 12px;
      opacity: 0.8;
    }}
    .ev-quote {{
      font-size: 14px;
    }}
    .ev-actions {{
      margin-top: 8px;
      display: flex;
      gap: 8px;
      flex: 0 0 auto;
      white-space: nowrap;
    }}
    .btn {{
      border: 1px solid #ccc;
      background: #fff;
      padding: 5px 10px;
      border-radius: 10px;
      cursor: pointer;
      text-decoration: none;
      font-size: 0.9em;
      white-space: nowrap;   /* keep “Copy + open” or “Open PDF (page 2)” on one line */
      display: inline-flex;  /* better sizing */
    }}
    .pill-click{{ border:none; cursor:pointer; }}
    .laj-box{{
      border:1px solid rgba(255,255,255,0.12);
      border-radius:12px;
      padding:12px;
    }}
    .laj-header{{ display:flex; gap:10px; align-items:center; margin-bottom:8px; }}
    .laj-metric{{
      display:flex;
      justify-content:space-between;
      gap:12px;
      padding:6px 0;
      border-top:1px solid rgba(255,255,255,0.08);
    }}
    .laj-metrics-wrap {{ margin-top: 10px; }}

    .laj-metrics-table{{
      width: 100%;
      border-collapse: collapse;
    }}

    .laj-metrics-table td{{
      padding: 8px 0;
      border-top: 1px solid rgba(255,255,255,0.10);
      vertical-align: top;
    }}

    .laj-metric-name{{
      width: 38%;
      font-weight: 600;
      opacity: 0.95;
      padding-right: 16px;
    }}

    .laj-score{{
      font-weight: 500;
      opacity: 0.8;
    }}
    .laj-score-pass{{
      color: #1e7e34;     /* NHS green */
      font-weight: 500;
    }}

    .laj-score-warn{{
      color: #b36b00;     /* amber */
      font-weight: 500;
    }}

    .laj-score-fail{{
      color: {NHS_RED};     /* NHS red */
      font-weight: 700;
    }}

    .laj-metric-note{{
      width: 62%;
      text-align: right;     /* per your request */
      opacity: 0.9;
    }}
    .laj-left{{ display:flex; gap:8px; align-items:baseline; }}
    .laj-mid{{ font-weight:600; opacity:0.9; }}
    .laj-name{{ opacity:0.75; font-size:0.95em; }}
    .laj-right{{ display:flex; gap:10px; align-items:baseline; }}
    .laj-note{{ opacity:0.85; }}
    .btn:hover {{
      background: #f5f5f5;
    }}
    .laj-info{{
      cursor: pointer;
      opacity: 0.75;
      font-weight: 600;
      margin-left: 4px;
    }}
    .laj-details summary{{
      cursor: pointer;           /* hand cursor */
      user-select: none;         /* prevents text-selection cursor feel */
    }}

    .laj-details summary *{{
      user-select: none;
    }}

    .laj-info:hover{{
      opacity: 1.0;
      cursor: pointer;
    }}

    details.raw {{
      margin-top: 12px;
    }}
    pre {{
      background: #0b0c0c;
      color: #f5f5f5;
      padding: 12px;
      border-radius: 10px;
      overflow-x: auto;
      font-size: 12px;
    }}
    footer {{
      margin-top: 18px;
      font-size: 12px;
      opacity: 0.75;
    }}
    @media (max-width: 760px) {{
      .meta-grid {{ grid-template-columns: 1fr; }}
      .card-head {{ flex-direction: column; }}
      .badges {{ justify-content: flex-start; }}
    }}
  </style>
</head>
<script>
    async function copyText(text) {{
      try {{
        await navigator.clipboard.writeText(text);
      }} catch (e) {{
        const ta = document.createElement("textarea");
        ta.value = text;
        document.body.appendChild(ta);
        ta.select();
        document.execCommand("copy");
        document.body.removeChild(ta);
      }}
    }}
    
    function decodeB64Utf8(b64) {{
      try {{
        return decodeURIComponent(escape(atob(b64)));
      }} catch (e) {{
        return atob(b64);
      }}
    }}

    async function copyB64(b64) {{
      return copyText(decodeB64Utf8(b64));
    }}
  </script>
 <script>
    function toggleLaj(ev, agentId){{
      if(ev) ev.stopPropagation();
      const row = document.getElementById("lajrow-" + agentId);
      if(!row) {{ console.warn("No LaJ row for", agentId); return; }}
      row.style.display = (row.style.display === "none" || row.style.display === "") ? "table-row" : "none";
    }}
</script>


<body>
  <header>
    <div class="wrap">
      <div style="font-size:28px;font-weight:900;">LRRIT Agentic Assessment of Dimensions</div>
      <div style="opacity:0.9;margin-top:8px;">Report: {html.escape(report_id)} • Evaluation generated: {html.escape(now)}</div>
    </div>
  </header>

  <div class="wrap">
    <div class="meta">
      <div style="font-weight:900;font-size:14px;"></div>
      <div class="meta-grid">
        <div><span class="k">File path of report:</span> <span class="v">{html.escape(source_path)}</span></div>
        <div><span class="k">Large Language Model used:</span> <span class="v mono">{html.escape(model_name)}</span></div>
        <div id="pdf-status" style="display:none; margin: 10px 0; padding: 10px; border: 1px solid #f0c36d; background: #fff8e1; border-radius: 10px;">
          </div>
      </div>
      <footer>This file is stored locally. <p>NB.  
      The <b>open pdf</b> button copies the verbatim quote to the clipboard and opens the report at that page in a new tab. 
      You can then use search to find the context of the quote within the report. 
      <p><i>Caveat emptor:</i> long quotes may be split across lines, pages or the model may have added punctuation or changed the formatting. 
      In this case, delete parts of the quote until the search works. </footer>
    </div>

    <div class="summary" id="summary">
      <div style="font-weight:900;font-size:16px;">LRRIT Dimension Summary</div>
      <div class="muted" style="margin-top:6px;">Click a row to jump to the detailed section for that dimension.</div>
      <table aria-label="Dimension summary">
        <thead>
          <tr>
            <th>Agent</th>
            <th>Data Dimension</th>
            <th>Evidence Level Rating</th>
            <th>Agent Task <br>Evaluation</th>
          </tr>
        </thead>
        <tbody>
          {''.join(summary_rows)}
        </tbody>
      </table>
    </div>

    {''.join(cards_html)}
  </div>
 <script>
    let pdfWin = null;

    function setPdfStatus(msg) {{
      const el = document.getElementById("pdf-status");
      if (!el) return;
      el.textContent = msg || "";
      el.style.display = msg ? "block" : "none";
    }}

    function openPdf(url) {{
      // Reuse a single named window/tab so repeated clicks don't spawn new tabs
      const name = "lrrit_pdf_tab";

      // If we already have a handle and it isn't closed, reuse it
      if (pdfWin && !pdfWin.closed) {{
        pdfWin.location.href = url;
        pdfWin.focus();
        setPdfStatus("");  // clear any prior warning
        return;
      }}

      // Try to open (user gesture should allow this)
      pdfWin = window.open(url, name, "noopener");

      if (pdfWin) {{
        pdfWin.focus();
        setPdfStatus("");
      }} else {{
        // Popup blocked: keep report tab, show an inline instruction
        setPdfStatus("PDF tab blocked by browser. Please allow popups for this report, then click 'Open PDF' again.");
      }}
    }}
</script>

  <script>
    // Keyboard accessibility for clickable summary rows
    document.querySelectorAll('tr.summary-row').forEach(function(row) {{
      row.addEventListener('keydown', function(e) {{
        if (e.key === 'Enter' || e.key === ' ') {{
          e.preventDefault();
          row.click();
        }}
      }});
    }});
  </script>
</body>
</html>
"""
    out_path = report_dir / "agent_results.html"
    out_path.write_text(html_out, encoding="utf-8")
    return out_path


def main():
    # Default to your 'test' report directory
    report_dir = Path("data") / "processed" / "reports" / "SYNTHETIC AAR"

    # Allow override via env var or first CLI arg
    import os, sys
    if len(sys.argv) > 1:
        report_dir = Path(sys.argv[1])
    elif os.environ.get("LRRIT_REPORT_DIR"):
        report_dir = Path(os.environ["LRRIT_REPORT_DIR"])

    out = render_html(report_dir)
    print(f"Wrote: {out.resolve()}")


if __name__ == "__main__":
    main()
