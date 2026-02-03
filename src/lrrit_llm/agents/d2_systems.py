from __future__ import annotations

import json
from typing import Dict, Any

from lrrit_llm.evidence.schema import EvidencePack


class D2SystemsApproachAgent:
    """
    D2 – Systems approach to contributory factors

    Evaluates whether the response frames causes and improvements in terms of
    system/process contributors, not primarily individual fault or reminders.
    """

    AGENT_ID = "D2"
    DIMENSION_NAME = "Systems approach to contributory factors"

    # Optional lightweight cues for post-parse sanity checks (not decision logic)
    SYSTEM_CUES = (
        "system", "process", "pathway", "escalat", "handover", "workflow",
        "capacity", "staffing", "resource", "protocol", "governance",
        "communication", "interface", "coordination", "policy", "standard"
    )
    INDIVIDUAL_CUES = (
        "should have", "should've", "failed", "did not", "didn't",
        "be more vigilant", "remind", "ensure clinicians", "education"
    )

    def __init__(self, model_client):
        self.model = model_client

    def run(self, pack: EvidencePack) -> Dict[str, Any]:
        prompt = self._build_prompt(pack)
        raw_response = self.model.complete(prompt)

        parsed = self._parse_response(raw_response)
        parsed = self._apply_guards(parsed)

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
            # Ensure tables are citeable by a stable ID and contain fallback text
            evidence_blocks.append(
                f"[Table {table.table_id} | page {table.provenance.page}]\n{table.text_fallback}"
            )

        evidence_text = "\n\n".join(evidence_blocks)

        return f"""
You are an expert reviewer applying the Learning Response Review and Improvement Tool (LRRIT).

Dimension: Systems approach to contributory factors (D2).

Task:
- Judge whether the response analyses contributory factors using a systems/process perspective.
- Look for explicit system-level contributors (process design, escalation pathways, communication, capacity, governance).
- Look for improvement actions that change the system, not just individual reminders.
- Base your judgement ONLY on the evidence provided.

Rating options:
- GOOD evidence: clear systems framing + system-level actions
- SOME evidence: partial systems framing or mixed individual/system emphasis
- LITTLE evidence: mostly individual-centric framing; minimal systems analysis

Return STRICT JSON ONLY (no markdown, no extra text):

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
- Every evidence item MUST include a verbatim quote from the cited Text/Table block (<= 25 words).
- evidence_type:
  - "positive" = explicit systems/process framing or system-level interventions.
  - "negative" = primarily individual blame/reminders or absence of systems framing where expected.
- When citing negative evidence, explain why it weakens a systems approach (e.g. individual learning rather than system change)
- If rating is GOOD: include at least one positive evidence item.
- If rating is LITTLE: include at least one negative evidence item (if present). If not present, evidence may be [] but set uncertainty true.
- If no relevant excerpt exists to quote, set evidence to [] AND set uncertainty true.
- Do not invent quotes. Do not paraphrase quotes.

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
    # Guards (lightweight)
    # -------------------------

    def _apply_guards(self, result: Dict[str, Any]) -> Dict[str, Any]:
        rating = result.get("rating")
        evidence = result.get("evidence", []) or []

        # If no evidence, must be uncertain
        if not evidence:
            result["uncertainty"] = True
            return result

        # If claimed GOOD but no positive evidence item, force uncertainty
        if rating == "GOOD":
            if not any(e.get("evidence_type") == "positive" for e in evidence):
                result["uncertainty"] = True

        # If claimed LITTLE but no negative evidence item, force uncertainty
        if rating == "LITTLE":
            if not any(e.get("evidence_type") == "negative" for e in evidence):
                result["uncertainty"] = True

        return result

