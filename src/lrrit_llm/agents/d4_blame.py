from __future__ import annotations

import json
from typing import Dict, Any

from lrrit_llm.evidence.schema import EvidencePack


class D4BlameLanguageAgent:
    """
    D4 – Blame language avoided

    Evaluates whether the learning response avoids blame-oriented language
    and individual attribution of fault.
    """

    AGENT_ID = "D4"
    DIMENSION_NAME = "Blame language avoided"

# ---------------------------------------------------------------------
# D4: Blame language detection – design rationale
#
# This agent distinguishes between:
#   (a) SYSTEM / PROCESS FRAMING  → non-blaming (positive evidence)
#   (b) PERSON-ATTRIBUTING BLAME  → blame language (negative evidence)
#
# The model may sometimes label any mention of delay, error, or uncertainty
# as "blame". To prevent this, we apply lightweight post-hoc guards using:
#
#   1) BLAME_CUES:
#      Lexical indicators of judgement or fault attribution
#      (e.g. "failed to", "should have", "did not", "negligent").
#
#   2) PERSON_TOKENS:
#      References to individuals or teams (e.g. "staff", "team", "SHO",
#      "consultant", "they").
#
# A quote should ONLY be treated as blame-oriented ("negative") if it:
#   - contains explicit judgement / fault language (BLAME_CUES), OR
#   - attributes responsibility to a person or team (PERSON_TOKENS)
#
# Statements describing:
#   - delays,
#   - uncertainty,
#   - counterfactual outcomes,
#   - or system/process weaknesses
# WITHOUT attributing fault to people are NOT blame language and should
# be labelled as "positive" evidence for D4.
#
# If the model labels evidence as "negative" without satisfying either
# condition above, we mark the result as uncertain rather than silently
# correcting it. This preserves transparency and allows LaJ / humans to
# inspect borderline cases.
#
# See apply guards method below for implementation.
# ---------------------------------------------------------------------

    BLAME_CUES = (
        "failed", "should have", "should've", "did not", "didn't", "non-compliance",
        "neglig", "careless", "incompet", "to blame", "fault")
    
    PERSON_TOKENS = ("staff", "team", "sho", "doctor", "nurse", "consultant", "they", "he", "she", "we")

 
 
    def __init__(self, model_client):
        """
        model_client must expose:
          .complete(prompt: str) -> str
        """
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
                f"[Text {chunk.chunk_id} | page {chunk.provenance.page}]\n"
                f"{chunk.text}"
            )

        for table in pack.tables:
            evidence_blocks.append(table.text_fallback)

        evidence_text = "\n\n".join(evidence_blocks)

        return f"""
You are an expert reviewer applying the Learning Response Review and Improvement Tool (LRRIT).

Dimension: Blame language avoided.

Task:
- Assess whether the learning response avoids blame-oriented language.
- Focus on tone, framing, and attribution of responsibility.
- Base your judgement ONLY on the evidence provided.

Rating options:
- GOOD evidence
- SOME evidence
- LITTLE evidence

Important:
- Describing actions or errors does NOT automatically imply blame.
- Focus on language and attribution, not clinical correctness.
- Do not infer intent beyond the text.

Evidence:
{evidence_text}

Return STRICT JSON ONLY (no markdown, no extra text) with this schema:

{{
  "rating": "GOOD" | "SOME" | "LITTLE",
  "rationale": "string",
  "evidence": [
    {{
      "id": "Text pXX_cYY" | "Table pXX_tYY",
      "quote": "verbatim excerpt from the evidence, <= 25 words",
      "evidence_type": "positive" | "negative"
    }}
  ],
  "uncertainty": true | false
}}

Rules:
- Every evidence item MUST include a verbatim quote taken from the cited Text/Table block above (<= 25 words).
- For D4, label evidence_type as follows:
  - "positive" = neutral or systems/process framing; discusses issues without attributing fault to people.
  - "negative" = blame-oriented language that attributes fault to an individual or team, or uses judgemental descriptors about people.

- IMPORTANT: Systems/process critique is NOT blame.
  Examples that are NOT blame (label "positive"):
  - "No systematic way to ensure..."
  - "There is no standard process..."
  - "System factors contributed..."
  - "Escalation pathways were unclear..."

- Examples of blame language (label "negative"):
  - "X failed to..."
  - "The team did not..."
  - "Should have escalated..."
  - "Negligent / incompetent / careless..."
  - "Non-compliance with policy by staff..."
- Do NOT label a quote "negative" unless it clearly refers to actions/omissions of a person/team (not a system/process).

- If you find no blame-oriented language (no negative evidence), rate "GOOD".
- Use "SOME" only when both neutral/system framing AND at least one genuine blame-oriented statement are present.

- Otherwise label it "positive".
- If rating is GOOD: include at least one "positive" evidence item.
- If rating is LITTLE: include at least one "negative" evidence item.
- If rating is SOME: include one item of the most salient type; include both if mixed language exists.
- If you cannot find any relevant excerpt to quote, set evidence to [] AND set uncertainty to true.
- Do not invent quotes. Do not paraphrase quotes.
""".strip()

    # -------------------------
    # Response parsing
    # -------------------------

    def _parse_response(self, text: str) -> Dict[str, Any]:
        text = text.strip()

        # Fast path: strict JSON
        try:
            obj = json.loads(text)
            return self._normalise_obj(obj)
        except Exception:
            pass

        # Recovery: extract first JSON object
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start:end + 1]
            obj = json.loads(candidate)
            return self._normalise_obj(obj)

        raise ValueError("Agent did not return valid JSON.")

    def _normalise_obj(self, obj: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "rating": obj.get("rating"),
            "rationale": obj.get("rationale"),
            "evidence": obj.get("evidence", []) or [],
            "uncertainty": bool(obj.get("uncertainty", False)),
        }


# ---------------------------------------------------------------------
# How D4 guards are applied (_apply_guards)
#
# Purpose:
#   Provide a lightweight sanity-check over the model's JSON output.
#   We do NOT use these heuristics to decide the rating; we use them to
#   prevent overconfident mislabelling of evidence polarity.
#
# Steps:
#   1) If the model returns no evidence items:
#        - force uncertainty = True
#     Rationale: a rating without any quoted support is not auditable.
#
#   2) For each evidence item labelled evidence_type == "negative":
#        - lower-case the quote text
#        - compute:
#            has_blame_cue   = any(phrase in quote for phrase in BLAME_CUES)
#            has_person_ref  = any(token  in quote for token  in PERSON_TOKENS)
#        - if neither is present, then the quote is likely describing
#          systems/process issues or counterfactual uncertainty (not blame).
#          In that case we set uncertainty = True.
#
#     Important: we typically do NOT auto-change evidence_type from
#     "negative" to "positive". Marking uncertainty keeps the model output
#     transparent and lets LaJ/humans review borderline cases.
#
#   3) Rating consistency checks:
#        - If rating == "GOOD" but there are no "positive" evidence items,
#          set uncertainty = True.
#        - If rating == "LITTLE" but there are no "negative" evidence items,
#          set uncertainty = True.
#
# Output:
#   Returns the same result dict, possibly with uncertainty escalated to True.
# ---------------------------------------------------------------------


    def _apply_guards(self, result: Dict[str, Any]) -> Dict[str, Any]:
        rating = result.get("rating")
        evidence = result.get("evidence", []) or []

        if not evidence:
            result["uncertainty"] = True
            return result

        # Validate "negative" labels
        for e in evidence:
            if e.get("evidence_type") == "negative":
                q = (e.get("quote") or "").lower()
                has_cue = any(c in q for c in self.BLAME_CUES)          # blame cue present
                has_person = any(p in q for p in self.PERSON_TOKENS)    # person attribution present


                # If neither blame cue nor person attribution is present,
                # it is very likely mislabelled (system/uncertainty statements).
                if not has_cue and not has_person:
                    result["uncertainty"] = True

        # Rating consistency checks (as before)
        has_pos = any(e.get("evidence_type") == "positive" for e in evidence)
        has_neg = any(e.get("evidence_type") == "negative" for e in evidence)

        if rating == "GOOD" and not has_pos:
            result["uncertainty"] = True
        if rating == "LITTLE" and not has_neg:
            result["uncertainty"] = True

        return result
