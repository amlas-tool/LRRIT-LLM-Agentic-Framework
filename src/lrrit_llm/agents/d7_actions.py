from __future__ import annotations

import json
from typing import Dict, Any, List

from lrrit_llm.evidence.schema import EvidencePack
from lrrit_llm.evidence.resolve import resolve_evidence_id_and_page

class D7ImprovementActionsAgent:
    """
    D7 – Improvement actions (systems-focused, evidence-informed, collaboratively developed)

    Evaluates whether the report translates learning into credible actions that:
    - target system elements (pathways/workflows/roles/interfaces/IT/equipment),
    - are grounded in the analysis / contributory factors,
    - are collaboratively developed,
    - avoid "safety clutter" (generic extra checks/policies that don't improve work),
    - include ownership/monitoring arrangements.

    IMPORTANT (AAR proportionality):
    In AAR-style reports, actions may be absent and this should not automatically imply weak learning.
    If actions are genuinely absent, the agent should not force a "LITTLE" score purely for absence.
    """

    AGENT_ID = "D7"
    DIMENSION_NAME = "Improvement actions (systems-focused, evidence-informed, collaborative)"

    # Guardrail cue lists (lightweight; only used to set uncertainty, not to rewrite decisions)
    SYSTEM_ACTION_CUES = (
        "pathway", "workflow", "process", "care process", "handover", "escalat",
        "interface", "role", "responsibil", "IT", "system", "equipment",
        "staffing", "capacity", "transfer", "referral", "criteria", "protocol"
    )

    COLLAB_CUES = (
        "co-developed", "developed collaboratively", "with staff", "stakeholders",
        "multidisciplinary", "MDT", "agreed with", "co-designed", "engaged"
    )

    GOVERNANCE_CUES = (
        "owner", "responsible", "lead", "by", "due", "deadline",
        "monitor", "review", "audit", "track", "progress", "report to"
    )

    SAFETY_CLUTTER_CUES = (
        "remind", "retrain", "training", "education", "awareness", "reinforce",
        "re-circulate", "share learning", "update policy", "revise policy",
        "poster", "email", "reiterate", "compliance", "ensure staff"
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
        evidence_blocks: List[str] = []

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

Dimension: D7 – Improvement actions (systems-focused, evidence-informed, collaboratively developed).

Definition (what you are judging):
- Do the proposed safety actions / improvements / recommendations:
  • focus on system elements (IT, equipment, care processes/pathways, roles/interfaces), not individuals;
  • address the key contributory factors in the report;
  • avoid “safety clutter” (generic extra checks/policies/training that do not improve work-as-done);
  • show evidence of collaborative development with relevant staff/stakeholders;
  • include monitoring/ownership arrangements (who/how progress will be reviewed)?

IMPORTANT proportionality (AAR):
- In AAR reports, improvement actions may legitimately be absent. Absence alone should not automatically imply weak learning.
- Prioritise quotes from sections headed: "Improvement action plan", "Actions", "Recommendations", "Key learning points".
- If actions are absent, do NOT force a "LITTLE" rating purely because no actions are listed. Instead, explain the limitation and set uncertainty true.

Rating options:
- GOOD: system-focused actions, grounded in analysis, collaboratively developed, with ownership/monitoring
- SOME: genuine attempt but underdeveloped (weak linkage/collaboration/governance OR mix of system + individual actions)
- LITTLE: generic/compliance/individual-focused actions dominate; weak link to analysis; no rationale; no collaboration; no monitoring

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
- You must explain your rationale in the context of the evidence, explaining why the evidence you cite supports your rating.
- Every evidence item MUST include a verbatim quote (<= 25 words) from the cited Text/Table block.
- evidence_type:
  - "positive" = supports D7 (system-focused, linked to analysis, collaborative, governed/monitored).
  - "negative" = indicates weak D7 (generic/compliance/individual-focused actions, safety clutter, missing governance/collaboration, weak linkage).
- If rating is GOOD: include at least one positive evidence item.
- If rating is LITTLE: include at least one negative evidence item IF such text exists.
- If actions appear absent, you may return rating = SOME with evidence = [] and uncertainty = true (AAR conditionality).
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
    # Guards (lightweight)
    # -------------------------

    def _apply_guards(self, result: Dict[str, Any]) -> Dict[str, Any]:
        rating = result.get("rating")
        evidence = result.get("evidence", []) or []

        # If the model claims actions are present but provides no evidence, force uncertainty and downgrade rating.
        if not evidence:
            r = (result.get("rationale") or "").lower()
            if "actions are present" in r or "improvement actions are present" in r:
                result["uncertainty"] = True
                # Optional: force a conservative rating because it's not auditable
                result["rating"] = "SOME"

        # Rating consistency checks
        if rating == "GOOD" and not any(e.get("evidence_type") == "positive" for e in evidence):
            result["uncertainty"] = True

        if rating == "LITTLE" and not any(e.get("evidence_type") == "negative" for e in evidence):
            result["uncertainty"] = True

        # Plausibility checks (soft): do NOT relabel evidence, only escalate uncertainty.
        for e in evidence:
            q = (e.get("quote") or "").lower()
            et = e.get("evidence_type")

            if et == "positive":
                # At least one of: system-action / collaboration / governance should appear
                if not (
                    any(c in q for c in self.SYSTEM_ACTION_CUES)
                    or any(c in q for c in self.COLLAB_CUES)
                    or any(c in q for c in self.GOVERNANCE_CUES)
                ):
                    result["uncertainty"] = True

            if et == "negative":
                # Negatives often include safety-clutter / compliance cues
                if not any(c in q for c in self.SAFETY_CLUTTER_CUES):
                    # Still could be negative (e.g., "no monitoring described"), so just soften.
                    result["uncertainty"] = True

        return result
 
