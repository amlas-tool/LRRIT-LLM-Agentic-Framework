 
from __future__ import annotations

from typing import Dict, Any, List
from unittest import result

from lrrit_llm.evidence.schema import EvidencePack, TextChunk, TableEvidence
from lrrit_llm.evidence.resolve import resolve_evidence_id_and_page


import json


class D1CompassionAgent:
    """
    D1 – Compassionate Engagement

    Evaluates whether the learning response demonstrates compassionate
    engagement with people affected by the incident.
    """

    AGENT_ID = "D1"
    DIMENSION_NAME = "Compassionate engagement with people affected"
    PROMPT_FILE = "d1_compassion_prompt.txt"

    def __init__(self, model_client):
        """
        model_client: wrapper around OpenAI / local LLM.
        Must expose a .complete(prompt: str) -> str method.
        """
        self.model = model_client

    def run(self, pack: EvidencePack) -> Dict[str, Any]:
        """
        Run the agent on a single EvidencePack.
        Returns a structured result for LaJ + human comparison.
        """
        prompt = self._build_prompt(pack)
        raw_response = self.model.complete(prompt)

        # Parsing is deliberately simple for now.
        # You may later harden this with JSON schema enforcement.
        result = self._parse_response(raw_response)

        # -------------------------
        # NEW: resolve evidence page (and repair mis-attributed IDs where possible)
        # -------------------------
        result = self._add_pages_to_evidence(result, pack)

        if result["rating"] in ("GOOD", "SOME"):
            if not any(e.get("evidence_type") == "positive" for e in result["evidence"]):
                result["uncertainty"] = True # flag uncertainty if no positive evidence

        if result["rating"] == "LITTLE":
            if not result["evidence"]:
                result["uncertainty"] = True # flag uncertainty if no evidence at all


        return {
            "agent_id": self.AGENT_ID,
            "dimension": self.DIMENSION_NAME,
            "rating": result.get("rating"),
            "rationale": result.get("rationale"),
            "evidence": result.get("evidence", []),
            "missing_indicators": result.get("missing_indicators", []),
            "uncertainty": result.get("uncertainty", False),
            #"raw_output": raw_response,
        }

    # -------------------------
    # Prompt construction
    # -------------------------

    def _build_prompt(self, pack: EvidencePack) -> str:
        """
        Construct a conservative, evidence-grounded prompt.
        """
        prompt_body = self._load_prompt_body()
        evidence_blocks = []

        for chunk in pack.text_chunks:
            evidence_blocks.append(
                f"[Text {chunk.chunk_id} | page {chunk.provenance.page}]\n{chunk.text}"
            )

        for table in pack.tables:
            # Ensure tables are citeable by a stable ID and contain fallback text
            evidence_blocks.append(
                f"[Table {table.table_id} | page {table.provenance.page}]\n{table.text_fallback}"
            )

        evidence_text = "\n\n".join(evidence_blocks)

        return f"""
You are an expert reviewer applying the Learning Response Review and Improvement Tool (LRRIT). 
Apply the evidence type discriminators in the rubric to evaluate if the report meets the 
dimension below. Provide an overall evidence rating of GOOD, SOME, or LITTLE, with a detailed  
rationale backed up by verbatim quotes that support that rating. 

{prompt_body}

Return STRICT JSON ONLY (no markdown, no extra text). :

{{
  "rating": "GOOD" | "SOME" | "LITTLE",
  "rationale": "string",
  "evidence": [
    {{
      "id": "Text pXX_cYY" | "Table pXX_tYY",
      "quote": "verbatim excerpt from the evidence without trailing punctuation, <= 50 words",
      "evidence_type": "evidence_type from indicator prefix (e.g. SOME - Influence of engagement is implied but unclear: )" 
    }}
  ],
  "missing_indicators": ["<exact indicator text>"],
  }}



Evidence:
{evidence_text}

""".strip()
    
    def _load_prompt_body(self) -> str:
        from pathlib import Path

        prompt_path = Path(__file__).resolve().parents[1] / "prompts" / self.PROMPT_FILE
        data = prompt_path.read_bytes()
        for enc in ("utf-8", "cp1252", "latin-1"):
            try:
                return data.decode(enc).strip()
            except UnicodeDecodeError:
                continue
        return data.decode("utf-8", errors="replace").strip()    
    # -------------------------
    # Response parsing
    # -------------------------

    def _parse_response(self, text: str) -> Dict[str, Any]:
        """
        Parse strict JSON output. If the model returns extra text,
        attempt to recover the first JSON object.
        """
        text = text.strip()

        # Fast path: strict JSON
        try:
            obj = json.loads(text)
            return self._normalise_obj(obj)
        except Exception:
            pass

        # Recovery: extract first {...} block
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start:end+1]
            obj = json.loads(candidate)
            return self._normalise_obj(obj)

        raise ValueError("Agent did not return valid JSON.")
    def _normalise_obj(self, obj: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "rating": obj.get("rating"),
            "rationale": obj.get("rationale"),
            "evidence": obj.get("evidence", []) or [],
            "missing_indicators": obj.get("missing_indicators", []) or [],
        }

    # -------------------------
    # NEW: Evidence page enrichment
    # -------------------------

    def _add_pages_to_evidence(self, result: Dict[str, Any], pack: EvidencePack) -> Dict[str, Any]:
        evidence = result.get("evidence", []) or []
        if not evidence:
            return result

        enriched = []
        for e in evidence:
            eid = (e.get("id") or "").strip()
            quote = (e.get("quote") or "").strip()
            etype = (e.get("evidence_type") or "").strip()

            resolved_id, page = resolve_evidence_id_and_page(pack, eid, quote)

            # Prefer repaired ID if we found one (fixes misattribution)
            final_id = resolved_id or eid

            enriched.append({
                "id": final_id,
                "page": page,
                "quote": quote,
                "evidence_type": etype,
            })

        result["evidence"] = enriched
        return result









