from __future__ import annotations

import json
import re
import string

from dataclasses import dataclass
from sys import flags
from typing import Any, Dict, List, Optional, Tuple
from unittest import result

from lrrit_llm.evidence.schema import EvidencePack


# -------------------------
# Helpers: light normalisation for quote checks
# -------------------------

_WS_RE = re.compile(r"\s+")
# word tokens: letters/digits, allowing clinical abbreviations
_TOKEN_RE = re.compile(r"[a-z0-9]+")

# hyphenation at line breaks: "perfor-\nated" -> "perforated"
_HYPHEN_LINEBREAK_RE = re.compile(r"(\w)-\s+(\w)")

_CHUNK_RE = re.compile(r"(p\d{1,3}_(?:c|t)\d{1,3})", re.IGNORECASE)

def _extract_chunk_id(evidence_id: str) -> str | None:
    if not evidence_id:
        return None
    m = _CHUNK_RE.search(evidence_id)
    return m.group(1) if m else None

def _canon(s: str) -> str:
    """Whitespace + punctuation tolerant canonical form."""
    s = (s or "").strip()
    s = s.replace("\u00ad", "")  # soft hyphen
    s = s.replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", "'").replace("\u2019", "'")
    s = s.replace("\u2013", "-").replace("\u2014", "-")
    s = _HYPHEN_LINEBREAK_RE.sub(r"\1\2", s)  # dehyphenate across wraps
    s = s.lower()
    # Replace punctuation with spaces (so tokens remain)
    s = re.sub(rf"[{re.escape(string.punctuation)}]", " ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s

def _compact(s: str) -> str:
    """Ultra-tolerant form: remove all non-alphanumerics."""
    s = _canon(s)
    return re.sub(r"[^a-z0-9]+", "", s)

def _tokens(s: str) -> list[str]:
    return _TOKEN_RE.findall(_canon(s))

def _token_fuzzy_match(quote: str, block: str, min_ratio: float = 0.80, slack: int = 10) -> bool:
    qt = _tokens(quote)
    bt = _tokens(block)
    if len(qt) < 6:
        # if very short quote, token fuzzy is unreliable; rely on canon/compact
        return False
    win = len(qt) + slack
    # Sliding window: check in-order token match ratio
    for i in range(0, max(1, len(bt) - win + 1)):
        window = bt[i:i+win]
        # in-order match count
        j = 0
        hits = 0
        for tok in window:
            if j < len(qt) and tok == qt[j]:
                hits += 1
                j += 1
            if j == len(qt):
                break
        if hits / len(qt) >= min_ratio:
            return True
    return False

def quote_matches_block(quote: str, block: str) -> bool:
    if not quote or not block:
        return False
    q1, b1 = _canon(quote), _canon(block)
    if q1 and q1 in b1:
        return True
    q2, b2 = _compact(quote), _compact(block)
    if q2 and q2 in b2:
        return True
    return _token_fuzzy_match(quote, block)



# -------------------------
# Data model
# -------------------------

@dataclass
class LaJMetricResult:
    metric_id: str
    score: str              # PASS | WARN | FAIL
    notes: str              # short, actionable


class LaJMetaEvaluator:
    """
    LLM-as-Judge (LaJ) meta-evaluation layer.

    Judges the QUALITY of a dimension agent output using a structured metric basket.
    Does NOT re-evaluate the original report. Only uses the cited evidence blocks (and optional
    programmatic quote checks) to assess grounding and hallucination risk.

    Expected agent_output schema:
      {
        "agent_id": "D5",
        "dimension": "...",
        "rating": "GOOD|SOME|LITTLE",
        "rationale": "...",
        "evidence": [{"id": "...", "quote": "...", "evidence_type": "positive|negative"}],
        "uncertainty": true|false
      }

    Output:
      {
        "judge_id": "LaJ",
        "agent_id": "...",
        "dimension": "...",
        "overall": "PASS|WARN|FAIL",
        "metrics": [{"metric_id":"...", "score":"...", "notes":"..."}],
        "flags": {"missing_evidence": bool, "quote_mismatch": bool, "invalid_evidence_id": bool}
      }
    """

    JUDGE_ID = "LaJ"

    METRICS = [
        ("M1", "Rubric Fidelity"),
        ("M2", "Evidence Grounding"),
        ("M3", "Reasoning Quality & Internal Coherence"),
        ("M4", "Values Alignment (PSIRF/LRRIT)"),
        ("M5", "Transparency & Uncertainty Handling"),
        ("M6", "Hallucination Screening"),
    ]

    def __init__(self, model_client, temperature: float = 0.0):
        self.model = model_client
        self.temperature = temperature

    # -------------------------
    # Public API
    # -------------------------

    def run(
        self,
        pack: EvidencePack,
        agent_output: Dict[str, Any],
        dimension_definition: str,
        strict_quote_check: bool = True,
    ) -> Dict[str, Any]:
        """
        pack: EvidencePack for resolving evidence ids and optional quote verification
        agent_output: output dict of D1..D8 agent
        dimension_definition: short definition of the target dimension (what the agent should judge)
        strict_quote_check: if True, run programmatic quote verification (recommended)
        """
        agent_id = agent_output.get("agent_id", "UNKNOWN")
        dimension = agent_output.get("dimension", "UNKNOWN")

        # Programmatic checks (do not require LLM, do not re-review report)
        flags, evidence_context = self._build_evidence_context(pack, agent_output, strict_quote_check=strict_quote_check)

        prompt = self._build_prompt(
            agent_output=agent_output,
            dimension_definition=dimension_definition,
            evidence_context=evidence_context,
            flags=flags,
        )

        raw = self.model.complete(prompt)
        parsed = self._parse_response(raw)
        parsed = self._apply_guards(parsed, flags=flags)

        return {
            "judge_id": self.JUDGE_ID,
            "agent_id": agent_id,
            "dimension": dimension,
            "overall": parsed.get("overall"),
            "metrics": parsed.get("metrics", []),
            "flags": flags,
            #"raw_output": raw,
        }

    # -------------------------
    # Evidence resolution / quote checks
    # -------------------------

    def _resolve_block(self, pack: EvidencePack, evidence_id: str) -> Optional[str]:
        """
        Resolve an evidence_id like:
        - "Text p03_c01"
        - "Table p02_t01"
        to its corresponding raw text in the EvidencePack.
        """
        eid = (evidence_id or "").strip()
        if not eid:
            return None

        chunk_id = _extract_chunk_id(eid)
        if not chunk_id:
            return None

        # Decide whether it’s text chunk or table by the _c/_t marker
        if "_c" in chunk_id.lower():
            for c in pack.text_chunks:
                if c.chunk_id.lower() == chunk_id.lower():
                    return c.text

        if "_t" in chunk_id.lower():
            for t in pack.tables:
                if t.table_id.lower() == chunk_id.lower():
                    return t.text_fallback or ""

        return None


    def _build_evidence_context(
        self,
        pack: EvidencePack,
        agent_output: Dict[str, Any],
        strict_quote_check: bool,
    ) -> Tuple[Dict[str, bool], str]:
        evidence = agent_output.get("evidence", []) or []

        flags = {
            "missing_evidence": False,
            "invalid_evidence_id": False,
            "quote_mismatch": False,
        }

        if not evidence:
            flags["missing_evidence"] = True
            return flags, ""

        blocks: List[str] = []
        seen_ids: set[str] = set()

        for ev in evidence:
            eid = (ev.get("id") or "").strip()
            quote = (ev.get("quote") or "").strip()

            if not eid:
                flags["invalid_evidence_id"] = True
                continue

            block = self._resolve_block(pack, eid)
            if block is None:
                flags["invalid_evidence_id"] = True
                continue

            # Quote check: quote must appear in the referenced block (after light normalisation)
            if strict_quote_check and quote:
                if not quote_matches_block(quote, block):
                    flags["quote_mismatch"] = True


            if eid not in seen_ids:
                seen_ids.add(eid)
                # Provide only the referenced blocks (LaJ does not read full report)
                blocks.append(f"[{eid}]\n{block}")

        return flags, "\n\n".join(blocks)

    # -------------------------
    # Prompt construction
    # -------------------------

    def _build_prompt(
        self,
        agent_output: Dict[str, Any],
        dimension_definition: str,
        evidence_context: str,
        flags: Dict[str, bool],
    ) -> str:
        # Keep this concise: LaJ judges the agent output, not the report.
        agent_json = json.dumps(agent_output, ensure_ascii=False, indent=2)

        flags_json = json.dumps(flags, ensure_ascii=False, indent=2)

        metric_list = "\n".join([f"- {mid} {name}" for mid, name in self.METRICS])

        return f"""
You are an LLM-as-Judge (LaJ) meta-evaluator. Your job is to assess the QUALITY of a dimension-agent's output,
not to re-review the original report.

You MUST NOT introduce new evidence from outside the supplied evidence blocks.
You MUST NOT re-grade the report for the dimension; only judge whether the agent output is rubric-faithful,
well-grounded, coherent, values-aligned, transparent about uncertainty, and free of unsupported claims.

Target dimension definition (what the agent should be judging):
{dimension_definition}

Dimension agent output (JSON):
{agent_json}

Programmatic QA flags (JSON):
{flags_json}

Referenced evidence blocks ONLY (agent-cited):
{evidence_context if evidence_context else "[NO EVIDENCE BLOCKS PROVIDED]"}

Metric basket:
{metric_list}

Return STRICT JSON ONLY in the following schema:

{{
  "overall": "PASS" | "WARN" | "FAIL",
  "metrics": [
    {{
      "metric_id": "M1",
      "score": "PASS" | "WARN" | "FAIL",
      "notes": "short, actionable notes (<= 2 sentences)"
    }}
  ]
}}

Scoring guidance:
- PASS: clearly meets the metric
- WARN: partially meets; minor gaps
- FAIL: materially fails; unreliable

Rules:
- Provide ALL 6 metrics M1..M6 exactly once.
- Keep notes short and actionable.
- If programmatic flags indicate issues (missing evidence, invalid evidence id, quote mismatch), reflect this in M2/M6.
Hallucination Screening (M6) is ONLY about unsupported factual assertions about the report content.
- PASS if the rationale stays within what is supported by the provided evidence blocks, even if the critique is generic.
- WARN if evidence is thin but not demonstrably false.
- FAIL only if the rationale asserts facts that are not present in the provided evidence blocks, OR programmatic flags indicate invalid evidence IDs / unverifiable quotes.

Notes on “Hallucination”
Do NOT use M6 to penalise "insufficient specificity", "weak emphasis", or "could have cited more examples".
Those belong in M1/M3 (rubric fidelity / reasoning quality).
""".strip()

    # -------------------------
    # Parsing / guards
    # -------------------------

    def _parse_response(self, text: str) -> Dict[str, Any]:
        text = (text or "").strip()

        try:
            obj = json.loads(text)
            return self._normalise(obj)
        except Exception:
            pass

        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            obj = json.loads(text[start:end + 1])
            return self._normalise(obj)

        raise ValueError("LaJ did not return valid JSON.")

    def _normalise(self, obj: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "overall": obj.get("overall"),
            "metrics": obj.get("metrics", []) or [],
        }

    def _apply_guards(self, result: Dict[str, Any], flags: Dict[str, bool]) -> Dict[str, Any]:
        # Ensure all metrics present exactly once
        metrics = result.get("metrics", []) or []

        m_by_id = {m.get("metric_id"): m for m in (result.get("metrics") or []) if isinstance(m, dict)}
        required = [mid for mid, _ in self.METRICS]

        # Clamp M6 when programmatic grounding is clean (prevents LaJ misusing hallucination)
        if not flags.get("invalid_evidence_id") and not flags.get("quote_mismatch"):
            m6 = m_by_id.get("M6")
            if m6 and m6.get("score") == "FAIL":
                m6["score"] = "WARN"
                # Optional: nudge notes to be accurate
                m6["notes"] = "No programmatic grounding issues detected; treat as potential overreach rather than hallucination."

        result["metrics"] = [m_by_id[mid] for mid in required]

        # If severe programmatic issues, overall cannot be PASS
        severe = flags.get("invalid_evidence_id") or flags.get("quote_mismatch")
        if severe and result.get("overall") == "PASS":
            result["overall"] = "WARN"

        # If no evidence, M2 must at least WARN and M6 must at least WARN
        if flags.get("missing_evidence"):
            m2 = m_by_id.get("M2", {})
            m6 = m_by_id.get("M6", {})
            if m2.get("score") == "PASS":
                m2["score"] = "WARN"
            if m6.get("score") == "PASS":
                m6["score"] = "WARN"
            result["metrics"] = [m_by_id[mid] for mid in required]

        return result
