from __future__ import annotations

import json
from typing import Dict, Any

from lrrit_llm.evidence.schema import EvidencePack
from lrrit_llm.evidence.resolve import resolve_evidence_id_and_page

class D5LocalRationalityAgent:
    """
    D5 – Local rationality

    Evaluates whether the learning response reconstructs how actions/decisions
    made sense to those involved at the time, given information available,
    uncertainty, constraints, priorities, and trade-offs.

    This is NOT merely "absence of hindsight" and NOT generic systems framing
    unless it is used to make the contemporaneous reasoning intelligible.
    """

    AGENT_ID = "D5"
    DIMENSION_NAME = "Local rationality"
    PROMPT_FILE = "d5_local_rationality_prompt.txt"

    # Optional cues used ONLY for post-hoc uncertainty checks (not decision logic)
    # These help catch mislabelling (e.g., calling something "positive" when it
    # contains no contemporaneous reasoning).
    LOCAL_RATIONALE_CUES = (
        "at the time", "based on", "given", "in the context", "initially",
        "working diagnosis", "appeared", "interpreted", "thought", "believed",
        "concern", "uncertain", "uncertainty", "ambigu", "limited information",
        "competing", "priority", "trade-off", "capacity", "availability",
        "handover", "pathway", "access", "resource", "workload", "pressure"
    )

    HINDSIGHT_CUES = (
        "should have", "should've", "failed to", "did not", "didn't",
        "obvious", "clearly", "in hindsight", "neglig", "incompet",
        "to blame", "fault"
    )

    COUNTERFACTUAL_CUES = (
    "no certainty", "cannot determine", "can't determine", "unclear whether",
    "we cannot determine", "no way of knowing"
    )
    
    REASSURANCE_CUES = (
        "timely", "appropriate", "good care", "managed well"
    )

    def __init__(self, model_client):
        self.model = model_client

    def run(self, pack: EvidencePack) -> Dict[str, Any]:
        prompt = self._build_prompt(pack)
        raw_response = self.model.complete(prompt)

        parsed = self._parse_response(raw_response)
        parsed = self._apply_guards(parsed)

        # -------------------------
        # NEW: resolve evidence page (and repair mis-attributed IDs where possible)
        # -------------------------
        parsed = self._add_pages_to_evidence(parsed, pack)

        return {
            "agent_id": self.AGENT_ID,
            "dimension": self.DIMENSION_NAME,
            "rating": parsed.get("rating"),
            "rationale": parsed.get("rationale"),
            "evidence": parsed.get("evidence", []),
            "uncertainty": parsed.get("uncertainty", False),
            #"raw_output": raw_response,
        }

    # -------------------------
    # Prompt construction
    # -------------------------

    def _build_prompt(self, pack: EvidencePack) -> str:
        
        prompt_body = self._load_prompt_body()
        evidence_blocks = []

        for chunk in pack.text_chunks:
            evidence_blocks.append(
                f"[Text {chunk.chunk_id} | page {chunk.provenance.page}]\n{chunk.text}"
            )

        for table in pack.tables:
            evidence_blocks.append(
                f"[Table {table.table_id} | page {table.provenance.page}]\n{table.text_fallback}"
            )

        evidence_text = "\n\n".join(evidence_blocks)

        return f"""
You are an expert reviewer applying the Learning Response Review and Improvement Tool (LRRIT). 

Apply the discriminators in the rubric to evaluate the evidence. Provide an overall evidence rating 
of GOOD, SOME, or LITTLE, with a rationale and verbatim quotes that support that rating. 

{prompt_body}

Return STRICT JSON ONLY (no markdown, no extra text):

{{
  "rating": "GOOD" | "SOME" | "LITTLE",
  "rationale": "string",
  "evidence": [
    {{
      "id": "Text pXX_cYY" | "Table pXX_tYY",
      "quote": "verbatim excerpt from the evidence without trailing punctuation, <= 25 words",
      "evidence_type": "evidence_type from indicator prefix (e.g. SOME - Influence of engagement is implied but unclear: )" 
    }}
  ],
}}


Evidence:
{evidence_text}

""".strip()
    
    def _load_prompt_body(self) -> str:
        from pathlib import Path

        prompt_path = Path(__file__).resolve().parents[1] / "prompts" / self.PROMPT_FILE
        return prompt_path.read_text(encoding="utf-8").strip()
 
    
    # -------------------------
    # JSON parsing
    # -------------------------

    def _parse_response(self, text: str) -> Dict[str, Any]:
        text = text.strip()

        try:
            obj = json.loads(text)
            return self._normalise_obj(obj)
        except Exception:
            pass

        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            obj = json.loads(text[start:end + 1])
            return self._normalise_obj(obj)

        raise ValueError("Agent did not return valid JSON.")

    def _normalise_obj(self, obj: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "rating": obj.get("rating"),
            "rationale": obj.get("rationale"),
            "evidence": obj.get("evidence", []) or [],
            "uncertainty": bool(obj.get("uncertainty", False)),
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

            # If we couldn't resolve a page at all, preserve but mark uncertain
            if page is None:
                result["uncertainty"] = True

            enriched.append({
                "id": final_id,
                "page": page,
                "quote": quote,
                "evidence_type": etype,
            })

        result["evidence"] = enriched
        return result


    # -------------------------
    # Guards (lightweight)
    # -------------------------

    def _apply_guards(self, result: Dict[str, Any]) -> Dict[str, Any]:
        rating = result.get("rating")
        evidence = result.get("evidence", []) or []

        # Must be auditable
        if not evidence:
            result["uncertainty"] = True
            return result

        # Rating consistency
        if rating == "GOOD" and not any(e.get("evidence_type") == "positive" for e in evidence):
            result["uncertainty"] = True

        if rating == "LITTLE" and not any(e.get("evidence_type") == "negative" for e in evidence):
            result["uncertainty"] = True

        # Polarity plausibility checks (only escalate uncertainty; do not silently relabel)
        for e in evidence:
            q = (e.get("quote") or "").lower()
            et = e.get("evidence_type")

            if et == "negative":
                # Counterfactual outcome-uncertainty is usually NOT valid negative evidence for D5.
                # It speaks to outcome attribution, not contemporaneous sense-making.
                if any(cue in q for cue in self.COUNTERFACTUAL_CUES):
                    result["uncertainty"] = True
                # Otherwise, many valid negatives are hindsight-ish; if no hindsight cue, mark uncertain.
                elif not any(cue in q for cue in self.HINDSIGHT_CUES):
                    result["uncertainty"] = True


            if et == "positive":
                # Reassurance alone is not local rationality.
                if any(cue in q for cue in self.REASSURANCE_CUES) and not any(cue in q for cue in self.LOCAL_RATIONALE_CUES):
                    result["uncertainty"] = True
                elif not any(cue in q for cue in self.LOCAL_RATIONALE_CUES):
                    result["uncertainty"] = True



            # Positive quotes should usually contain contemporaneous framing / constraints / uncertainty.
            if not any(cue in q for cue in self.LOCAL_RATIONALE_CUES):
                result["uncertainty"] = True

        return result


