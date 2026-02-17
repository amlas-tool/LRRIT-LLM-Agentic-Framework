from __future__ import annotations

import json
from typing import Dict, Any, List

from lrrit_llm.evidence.schema import EvidencePack
from lrrit_llm.evidence.resolve import resolve_evidence_id_and_page

class D8CommunicationQualityAgent:
    """
    D8 – Communication quality and usability of the learning response

    Evaluates whether the report is clearly written and usable:
    structure, readability, clarity of learning, accessibility of language,
    and whether key points are easy to extract.

    Notes:
    - This is NOT judging the clinical correctness; it is judging communication quality.
    - Evidence must be verbatim quotes showing clarity (positive) or confusion/vagueness/jargon (negative).
    """

    AGENT_ID = "D8"
    DIMENSION_NAME = "Communication quality and usability"

    # Guardrail cues only (soft checks)
    STRUCTURE_CUES = (
        "what happened", "summary", "immediate safety actions", "key learning points",
        "improvement action plan", "contributory", "actions", "recommendations"
    )
    VAGUE_CUES = (
        "appropriate", "timely", "good care", "managed well", "should", "ensure",
        "raise awareness", "as soon as possible"
    )
    JARGON_CUES = (
        "SDEC", "AMU", "M&M", "Datix", "CRP", "WCC", "NBM", "ED", "SHO", "QM"
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
        # print(f"Constructed prompt for {self.AGENT_ID} with {len(evidence_blocks)} evidence blocks.")
        # print(f"Prompt length (characters): {len(evidence_text)}")
        return f"""
You are an expert reviewer applying the Learning Response Review and Improvement Tool (LRRIT).

Dimension: D8 – Communication quality and usability of the learning response.

Definition:
- Judge whether the report is clearly communicated and usable:
  structure, readability, clarity of learning, and accessibility of language.
- This is about communication quality, not clinical correctness.

What to look for:
- Clear structure and signposting (sections like what happened / learning points / action plan)
- Clear and consistent terminology
- Learning and actions are easy to identify
- Avoids excessive vagueness ("appropriate", "timely") without explanation
- Avoids jargon/acronyms without explanation where it harms readability.
- You must explain your rationale in detail, explaining why the specific evidence you cite supports your rationale.

Rating options:
- GOOD: clear structure and readable narrative; learning/actions are easy to extract
- SOME: understandable but with issues (vagueness, jargon, weak signposting, inconsistencies)
- LITTLE: hard to follow; confusing structure/terminology; learning/actions hard to extract

Return STRICT JSON ONLY (no markdown, no extra text) with this schema:):

{{
  "rating": "GOOD" | "SOME" | "LITTLE",
  "rationale": "string",
  "evidence": [
    {{
      "id": "Text pXX_cYY" | "Table pXX_tYY",
      "quote": "verbatim excerpt from the evidence without trailing punctuation, <= 25 words",
      "evidence_type": "positive" | "negative"
    }}
  ],
  "uncertainty": true | false
}}

Rules:
- Every evidence item MUST include a verbatim quote (<= 25 words).
- Evidence_type:
  - "positive" = clear structure/signposting, explicit learning statements, accessible phrasing.
  - "negative" = vague/ambiguous language, jargon/acronyms harming clarity, confusing phrasing/structure.
    - Prefer negative evidence showing vagueness ('appropriate', 'timely'), unexplained acronyms/jargon, or 
    formatting/structure issues, rather than concise problem statements.
- If rating is GOOD: include at least one positive evidence item.
- If rating is LITTLE: include at least one negative evidence item IF such text exists.
- If you cannot find any relevant excerpt to quote, set evidence to [] AND set uncertainty true.
- Do not paraphrase quotes.

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

        if not evidence:
            result["uncertainty"] = True
            return result

        if rating == "GOOD" and not any(e.get("evidence_type") == "positive" for e in evidence):
            result["uncertainty"] = True
        if rating == "LITTLE" and not any(e.get("evidence_type") == "negative" for e in evidence):
            result["uncertainty"] = True

        # Soft plausibility checks
        for e in evidence:
            q = (e.get("quote") or "").lower()
            et = e.get("evidence_type")

            if et == "positive":
                # Structure cues often show up; if none, mark uncertain.
                if not any(cue in q for cue in self.STRUCTURE_CUES):
                    # Not required, but common; keep it soft.
                    pass

            if et == "negative":
                # Many negatives are vagueness or jargon; if none present, mark uncertain softly.
                if not (any(cue in q for cue in self.VAGUE_CUES) or any(cue.lower() in q for cue in self.JARGON_CUES)):
                    result["uncertainty"] = True

        return result

