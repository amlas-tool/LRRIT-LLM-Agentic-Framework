from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

from lrrit_llm.ingest.pdf_text import extract_text_pages
from lrrit_llm.ingest.pdf_tables import extract_tables_from_pdf
from lrrit_llm.evidence.pack import build_evidence_pack, save_evidence_pack, load_evidence_pack

from lrrit_llm.clients.openai_client import OpenAIChatClient

# Agents
from lrrit_llm.agents.d1_compassion import D1CompassionAgent
from lrrit_llm.agents.d2_systems import D2SystemsApproachAgent
from lrrit_llm.agents.d3_learning_actions import D3LearningActionsAgent
from lrrit_llm.agents.d4_blame import D4BlameLanguageAgent
from lrrit_llm.agents.d5_local_rationality import D5LocalRationalityAgent
from lrrit_llm.agents.d6_counterfactuals import D6HindsightBiasAgent
from lrrit_llm.agents.d7_actions import D7ImprovementActionsAgent
from lrrit_llm.agents.d8_clarity import D8CommunicationQualityAgent

# LaJ + HTML
from lrrit_llm.laj.laj_meta import LaJMetaEvaluator
from render_results_html import render_html as render_results_html  # or your module path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="run_report",
        description="Run LRRIT-LLM pipeline on a single PDF: ingest -> agents -> optional LaJ -> optional HTML.",
    )
    p.add_argument("pdf", help="Path to PDF report")
    p.add_argument("--report-id", default=None, help="Override report_id (default: PDF stem)")
    p.add_argument("--out-dir", default=str(Path("data") / "processed" / "reports"),
                   help="Base output directory (default: data/processed/reports)")

    p.add_argument("--no-ingest", action="store_true",
                   help="Skip ingest/build EvidencePack; assumes evidence_pack.json exists in processed reports output dir")
    p.add_argument("--agents", action="store_true", help="Run dimension agents (default: yes unless --html-only)")
    p.add_argument("--laj", action="store_true", help="Run LaJ meta-evaluation (requires agent_results.json)")
    p.add_argument("--html", action="store_true", help="Render HTML report")
    p.add_argument("--html-only", action="store_true",
                   help="Only render HTML from existing JSON outputs (skips ingest/agents/LaJ)")

    p.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
                   help="OpenAI model (default: env OPENAI_MODEL or gpt-4o-mini)")
    p.add_argument("--temperature", type=float, default=0.0, help="LLM temperature (default: 0.0)")
    p.add_argument("--force", action="store_true",
               help="Re-run all steps even if outputs already exist")


    return p.parse_args()


def main() -> None:
    args = parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    report_id = args.report_id or pdf_path.stem
    base_out = Path(args.out_dir)
    report_out = base_out / report_id
    report_out.mkdir(parents=True, exist_ok=True)

    evidence_pack_path = report_out / "evidence_pack.json"
    agent_results_path = report_out / "agent_results.json"
    laj_results_path = report_out / "laj_results.json"
    html_path = report_out / "agent_results.html"

    have_pack = evidence_pack_path.exists()
    have_agents = agent_results_path.exists()
    have_laj = laj_results_path.exists()


    # Decide steps (default = do minimum needed)
    do_ingest = not args.no_ingest and not args.html_only
    do_agents = args.agents
    do_laj = args.laj
    do_html = args.html or args.html_only

    if args.html_only:
        do_ingest = False
        do_agents = False
        do_laj = False

    # If user asked for LaJ but didn't explicitly ask for agents:
    # only run agents if agent_results.json is missing (or --force)
    if do_laj and not args.agents:
        do_agents = (not have_agents) or args.force

    # Only ingest if we actually need a pack for agents or LaJ, and it's missing (or --force)
    need_pack_obj = do_agents or do_laj
    if need_pack_obj:
        do_ingest = (not have_pack) or args.force
    else:
        do_ingest = False


    pack = None
    # ---- INGEST ----
    if do_ingest:
        print(f"[1/5] Extracting text: {pdf_path}")
        text_pages = extract_text_pages(str(pdf_path))

        print(f"[2/5] Extracting tables -> {report_out}")
        tables = extract_tables_from_pdf(
            pdf_path=str(pdf_path),
            report_id=report_id,
            out_dir=str(report_out),
            page_numbers=None,
        )

        print("[3/5] Building EvidencePack")
        pack = build_evidence_pack(
            report_id=report_id,
            source_path=str(pdf_path),
            text_pages=text_pages,
            tables=tables,
            extractor_text_name="pymupdf",
            metadata={"note": "cli_run"},
        )
        save_evidence_pack(pack, str(evidence_pack_path))
        print(f"Saved EvidencePack: {evidence_pack_path}")
    else:
        if need_pack_obj and evidence_pack_path.exists():
            pack = load_evidence_pack(str(evidence_pack_path))
            print(f"[ingest] Skipped. Loaded EvidencePack from: {evidence_pack_path}")
        elif evidence_pack_path.exists():
            print(f"[ingest] Skipped. EvidencePack JSON exists at: {evidence_pack_path}")
        else:
            print("[ingest] Skipped, but no evidence_pack.json found.")

    if need_pack_obj and pack is None:
        raise RuntimeError(
            "This run requires an EvidencePack object, but ingest was skipped and no evidence_pack.json was found."
        )

    # ---- RUN AGENTS ----
    if do_agents:
        print("[4/5] Running agents (D1–D8)")
        client = OpenAIChatClient(model=args.model, temperature=args.temperature)

        agents = {
            "d1": D1CompassionAgent(client),
            "d2": D2SystemsApproachAgent(client),
            "d3": D3LearningActionsAgent(client),
            "d4": D4BlameLanguageAgent(client),
            "d5": D5LocalRationalityAgent(client),
            "d6": D6HindsightBiasAgent(client),
            "d7": D7ImprovementActionsAgent(client),
            "d8": D8CommunicationQualityAgent(client),
        }

        results: Dict[str, Any] = {}
        for key, agent in agents.items():
            results[key] = agent.run(pack)

        results["_meta"] = {"model": args.model, "pdf_path": str(pdf_path)}
        agent_results_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved agent results: {agent_results_path}")

    # ---- RUN LAJ ----
    if args.laj:
        if not agent_results_path.exists():
            raise RuntimeError("LaJ requires agent_results.json (run agents first).")

        agent_results = json.loads(agent_results_path.read_text(encoding="utf-8"))
        client = OpenAIChatClient(model=args.model, temperature=args.temperature)
        laj = LaJMetaEvaluator(model_client=client)
        laj_out = laj.run_all(pack=pack, agent_results=agent_results, strict_quote_check=True)

        laj_results_path.write_text(json.dumps(laj_out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved LaJ results: {laj_results_path}")

    # ---- RENDER HTML ----
    if args.html or args.html_only:
        if not agent_results_path.exists():
            raise RuntimeError("HTML render requires agent_results.json.")

        laj_results = None
        if laj_results_path.exists():
            laj_results = json.loads(laj_results_path.read_text(encoding="utf-8"))

        agent_results = json.loads(agent_results_path.read_text(encoding="utf-8"))

        html_path = render_results_html(report_out)
        print(f"Saved HTML report: {html_path}")



if __name__ == "__main__":
    main()
