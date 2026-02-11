from __future__ import annotations

import json
from typing import Dict, Any

from lrrit_llm.evidence.schema import EvidencePack
from lrrit_llm.evidence.resolve import resolve_evidence_id_and_page

class D3LearningActionsAgent:
    """
    D3 – Quality and appropriateness of learning actions

    Evaluates whether the learning actions proposed are concrete, appropriate,
    and likely to reduce recurrence.
    """

    AGENT_ID = "D3"
    DIMENSION_NAME = "Quality and appropriateness of learning actions"

    # Optional cues (sanity checking only, not decision logic)
    ACTION_CUES = (
        "introduce", "implement", "update", "establish", "develop",
        "protocol", "pathway", "guideline", "process", "audit",
        "escalation", "handover", "governance", "standardise"
    )
    WEAK_ACTION_CUES = (
        "reflect", "reflection", "appraisal", "remind", "awareness",
        "education session", "training", "discuss"
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

Dimension: Quality and appropriateness of learning actions (D3).

Task:
- Assess whether the learning actions identified are appropriate, concrete,
  and likely to reduce recurrence.
- Focus on the actions proposed, not just the problems identified.
- Base your judgement ONLY on the evidence provided.


Rating options:
- GOOD evidence: clear, concrete, system-level learning actions
- SOME evidence: actions present but generic, mixed, or weakly specified
- LITTLE evidence: vague, individual-only, or absent learning actions

Return STRICT JSON ONLY (no markdown, no extra text):

{{
  "rating": "GOOD" | "SOME" | "LITTLE",
  "rationale": "string",
  "evidence": [
    {{
      "id": "Text pXX_cYY" | "Table pXX_tYY",
      "quote": "verbatim excerpt from evidence without trailing punction, <= 25 words",
      "evidence_type": "positive" | "negative"
    }}
  ],
  "uncertainty": true | false
}}

Rules:
- You must explain your rationale in the context of the evidence, explaining why the evidence you cite supports your rating.
- Every evidence item MUST be numbered and include a verbatim quote (<= 25 words).
- Evidence_type:
  - "positive" = concrete, actionable learning actions embedded in systems/processes.
  - "negative" = vague actions, individual reflection only, reminders, or absence of actions.
- If rating is GOOD: include at least one positive evidence item.
    - For GOOD, at least one evidence quote must describe a specific implementable change 
    (e.g., a rule, protocol, pathway, escalation mechanism), not merely 'linking to work' or 'discussion'.
- If rating is LITTLE: include at least one negative evidence item (if present).
- If no learning actions are stated at all, evidence may be [] but set uncertainty true.
- Do not invent actions. Do not paraphrase quotes.

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

        if rating == "GOOD":
            if not any(e.get("evidence_type") == "positive" for e in evidence):
                result["uncertainty"] = True

        if rating == "LITTLE":
            if not any(e.get("evidence_type") == "negative" for e in evidence):
                result["uncertainty"] = True

        return result

