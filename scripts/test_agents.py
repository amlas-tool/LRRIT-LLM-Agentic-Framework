from __future__ import annotations

import os
import json
from pathlib import Path

from lrrit_llm.evidence import pack
from lrrit_llm.ingest.pdf_text import extract_text_pages
from lrrit_llm.ingest.pdf_tables import extract_tables_from_pdf
from lrrit_llm.evidence.pack import build_evidence_pack, save_evidence_pack
from lrrit_llm.clients.openai_client import OpenAIChatClient

from lrrit_llm.agents.d1_compassion import D1CompassionAgent
from lrrit_llm.agents.d2_systems import D2SystemsApproachAgent
from lrrit_llm.agents.d3_learning_actions import D3LearningActionsAgent
from lrrit_llm.agents.d4_blame import D4BlameLanguageAgent
from lrrit_llm.agents.d5_local_rationality import D5LocalRationalityAgent
from lrrit_llm.agents.d6_counterfactuals import D6HindsightBiasAgent
from lrrit_llm.agents.d7_actions import D7ImprovementActionsAgent
from lrrit_llm.agents.d8_clarity import D8CommunicationQualityAgent

from lrrit_llm.laj.laj_meta import LaJMetaEvaluator
from lrrit_llm.laj.dimension_defs import DIMENSION_DEFS


def main():
    # ---- CONFIG ----
    pdf_path = Path(
        os.environ.get(
            "LRRIT_TEST_PDF",
            r"G:\My Drive\LLM projects\lrrit-llm\data\raw_pdfs\SYNTHETIC AAR.pdf",
        )
    )

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    report_id = pdf_path.stem
    out_dir = Path("data") / "processed" / "reports" / report_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- INGEST ----
    print(f"[1/4] Extracting text: {pdf_path}")
    text_pages = extract_text_pages(str(pdf_path))

    print(f"[2/4] Extracting tables -> {out_dir}")
    tables = extract_tables_from_pdf(
        pdf_path=str(pdf_path),
        report_id=report_id,
        out_dir=str(out_dir),
        page_numbers=None,
    )

    # ---- BUILD EVIDENCE PACK ----
    print("[3/4] Building EvidencePack")
    pack = build_evidence_pack(
        report_id=report_id,
        source_path=str(pdf_path),
        text_pages=text_pages,
        tables=tables,
        extractor_text_name="pymupdf",
        metadata={"note": "smoke_test"},
    )

    if pack is None:
        raise RuntimeError("build_evidence_pack returned None")

    pack_path = out_dir / "evidence_pack.json"
    save_evidence_pack(pack, str(pack_path))
    print(f"Saved EvidencePack: {pack_path}")

    # ---- RUN AGENTS ----
    print("[4/4] Running agents...")
    client = OpenAIChatClient(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0.0,
    )

    d1 = D1CompassionAgent(client)
    d2 = D2SystemsApproachAgent(client)
    d3 = D3LearningActionsAgent(client)
    d4 = D4BlameLanguageAgent(client)
    d5 = D5LocalRationalityAgent(client)
    d6 = D6HindsightBiasAgent(client)
    d7 = D7ImprovementActionsAgent(client)
    d8 = D8CommunicationQualityAgent(client)

    d1_out = d1.run(pack)
    d2_out = d2.run(pack)
    d3_out = d3.run(pack)
    d4_out = d4.run(pack)
    d5_out = d5.run(pack)
    d6_out = d6.run(pack)
    d7_out = d7.run(pack)
    d8_out = d8.run(pack)


    results = {"d1": d1_out, "d2": d2_out, "d3": d3_out, "d4": d4_out, "d5": d5_out, "d6": d6_out, "d7": d7_out, "d8": d8_out}
    results["_meta"] = {
        "model": os.environ.get("OPENAI_MODEL", "unknown"), # for record-keeping - needs changing for local clients
        "pdf_path": str(Path(pdf_path).resolve()),
    }

    results_path = out_dir / "agent_results.json"
    results_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Saved agent results: {results_path}")

    print(f"Running LaJ meta-evaluation on {results_path}")

    laj = LaJMetaEvaluator(client)

    laj_results = {}

    for key, agent_out in results.items():
        if not isinstance(agent_out, dict): 
            continue
        agent_id = agent_out.get("agent_id", "").strip()
        if agent_id in DIMENSION_DEFS:
            laj_results[agent_id.lower()] = laj.run(
                pack=pack,
                agent_output=agent_out,
                dimension_definition=DIMENSION_DEFS[agent_id],
                strict_quote_check=True,
            )

    #results["laj"] = laj_results

    laj_results_path = out_dir / "laj_results.json"
    laj_results_path.write_text(
    json.dumps(laj_results, ensure_ascii=False, indent=2),
    encoding="utf-8",
    )



if __name__ == "__main__":
    main()
