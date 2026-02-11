from __future__ import annotations

import json
from typing import Dict, Any

from lrrit_llm.evidence.schema import EvidencePack
from lrrit_llm.evidence.resolve import resolve_evidence_id_and_page

class D6HindsightBiasAgent:
    """
    D6 – Avoidance of hindsight bias and inappropriate counterfactual certainty

    Evaluates whether the learning response reasons cautiously about outcomes
    and alternative actions, acknowledging uncertainty and avoiding definitive
    hindsight claims.
    """

    AGENT_ID = "D6"
    DIMENSION_NAME = "Avoidance of hindsight bias and counterfactual certainty"

    # Used only for guardrails / uncertainty escalation
    UNCERTAINTY_CUES = (
        "no certainty", "cannot determine", "can't determine", "unclear whether",
        "it is unclear", "unknown", "we cannot know", "may not have", "might not have"
    )

    DEFINITIVE_CUES = (
        "would have", "would've", "definitely", "clearly", "obviously",
        "inevitably", "certainly", "directly caused", "resulted in"
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

Dimension: D6 – Avoidance of hindsight bias and inappropriate counterfactual certainty.

Definition:
- This dimension assesses how cautiously the response reasons about outcomes and
  alternative actions after the event.
- It rewards explicit acknowledgement of uncertainty and penalises definitive,
  unsupported counterfactual claims.

Task:
- Identify how the response discusses what might have happened under different circumstances.
- Assess whether uncertainty is acknowledged and whether causal claims are proportionate.
- Base your judgement ONLY on the evidence provided.
- You must explain your rationale in detail, explaining why the specific evidence you cite supports your rationale.

Rating options:
- GOOD evidence: cautious counterfactual reasoning with explicit uncertainty
- SOME evidence: mixed cautious and overconfident counterfactual reasoning
- LITTLE evidence: strong hindsight bias or definitive unsupported causal claims

Return STRICT JSON ONLY (no markdown, no extra text, no extra text, no final period, full stop or punctuation):

{{
  "rating": "GOOD" | "SOME" | "LITTLE",
  "rationale": "string",
  "evidence": [
    {{
      "id": "Text pXX_cYY" | "Table pXX_tYY",
      "quote": "verbatim excerpt from evidence, <= 25 words",
      "evidence_type": "positive" | "negative"
    }}
  ],
  "uncertainty": true | false
}}

Rules:
- Every evidence item MUST include a verbatim quote (<= 25 words).
- evidence_type:
  - "positive" = explicit acknowledgement of uncertainty or cautious counterfactual reasoning.
  - "negative" = definitive hindsight claims or unsupported causal certainty.
- If rating is GOOD: include at least one positive evidence item.
- If rating is LITTLE: include at least one negative evidence item IF such text exists.
- For rating = SOME, it is acceptable to include both positive and negative evidence.
- If you cannot find any relevant excerpt to quote, set evidence to [] AND set uncertainty true.
- Do not invent causal claims. Do not paraphrase quotes.

Evidence:
{evidence_text}
""".strip()

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
    # Guards
    # -------------------------

    def _apply_guards(self, result: Dict[str, Any]) -> Dict[str, Any]:
        rating = result.get("rating")
        evidence = result.get("evidence", []) or []

        if not evidence:
            result["uncertainty"] = True
            return result

        if rating == "GOOD" and not any(e.get("evidence_type") == "positive" for e in evidence):
            result["uncertainty"] = True

        if rating == "LITTLE" and not any(e.get("evidence_type") == "negative" for e in evidence):
            result["uncertainty"] = True

        # Plausibility checks
        for e in evidence:
            q = (e.get("quote") or "").lower()
            et = e.get("evidence_type")

            if et == "positive":
                if not any(cue in q for cue in self.UNCERTAINTY_CUES):
                    result["uncertainty"] = True

            if et == "negative":
                if not any(cue in q for cue in self.DEFINITIVE_CUES):
                    result["uncertainty"] = True

        return result
 
