 
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

LRRIT Dimension: People affected by incidents are compassionately engaged and meaningfully involved. 
The report includes the perspectives of those affected such as staff, patients, families and carers.

Core Definition:
Compassionate engagement means that affected people’s needs, experiences, and perspectives were 
sensitively elicited, understood, and responded to, and that those conducting the engagement had 
appropriate skills to ensure safe, respectful involvement. Meaningful involvement requires that 
these perspectives inform the learning response.  


Discriminators:
GOOD evidence: Does the report demonstrate engagement that both (a) shows that affected people’s 
needs or perspectives were sensitively elicited and understood by individuals with the appropriate 
skills, and (b) illustrates how these informed or shaped the learning response?

SOME evidence: Does the report provide evidence of any meaningful engagement with affected people, 
i.e., engagement that went beyond procedural notification and generated insight into their needs, 
perspectives, or experiences, even if this did not influence the learning response?

LITTLE evidence: 
Is there little to no evidence, you should report this saying "I looked through the report and 
found no evidence of compassionate engagement".

You must use the following indicators for evidence levels. For each indicator, you should look for 
evidence in the report and include verbatim quotes that demonstrate that indicator.

- GOOD
    - Meaningful engagement is clearly described: The report specifies how patients, families, carers, or staff were engaged.
    - Needs and perspectives are clearly articulated: The report presents what affected people said, needed, or emphasised.
    - Engagement was conducted by suitably skilled individuals: Involvement was carried out by people with appropriate roles and skills (e.g., family liaison, bereavement team, trained facilitator).
    - Engagement influenced the learning response: Needs or insights from those affected directly shaped analysis, decisions, or actions.
    - Restorative orientation is demonstrated: The learning response attends to what mattered to affected people and addresses their needs or concerns.

- SOME
    - Engagement is mentioned but minimally described: Affected people were contacted, consulted, or interviewed, but details are sparse.
    - Needs and perspectives are acknowledged but vague: The report refers to concerns or experiences without elaborating on them.
    - Influence of engagement is implied but unclear: Engagement may have informed the response, but no clear link is shown.
    - Restorative response is partial or unclear: The report recognises impact or distress but does not show how needs were explored or addressed.

- LITTLE
    - Engagement is absent or purely procedural: No meaningful engagement. Any reference is limited to formal notification (e.g., “family informed in line with policy”).
    - Needs or perspectives are not described: The report does not describe any needs, concerns, or perspectives of those affected.
    - Competency cannot be assessed: The report does not indicate who engaged with those affected, or whether any relevant skills were involved.
    - No evidence of influence on the learning response: Perspectives of those affected did not shape insights, decisions, or actions.


Return STRICT JSON ONLY (no markdown, no extra text, no final period, full stop or punctuation):

{{
  "rating": "GOOD" | "SOME" | "LITTLE",
  "rationale": "string",
  "evidence": [
    {{
      "id": "Text pXX_cYY" | "Table pXX_tYY",
      "quote": "verbatim excerpt from the evidence without trailing punctuation, <= 25 words",
      "evidence_type": "Rating" 
      "rubric_indicator": "Indicator description"
    }}
  ],
}}

Rules:

- Every evidence item MUST include:
- a verbatim quote taken from the evidence pack (<= 25 words)
- an evidence_type that maps to the specific indicator it supports (e.g., "GOOD - Needs and perspectives are clearly articulated")
- If no relevant excerpt exists, set evidence to [] and set uncertainty to true.
- Do not invent quotes. Do not paraphrase quotes. 

Evidence:
{evidence_text}

""".strip()
    
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
