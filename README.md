# LRRIT-LLM
**Learning Response Review and Improvement Tool – LLM Agentic Prototype**

This repository contains a prototype **agentic framework** for applying Large Language Models (LLMs) to the *Learning Response Review and Improvement Tool (LRRIT)* used in healthcare safety governance.

The system ingests incident / learning response documents (PDFs), extracts structured evidence, and evaluates them across multiple LRRIT dimensions using **dimension-specific agents**, with outputs designed to be auditable, explainable, and suitable for human comparison.

---

## Purpose

The project explores whether LLMs can:

- Apply LRRIT dimensions consistently
- Provide **evidence-anchored judgements** (verbatim quotes)
- Distinguish between:
  - systems vs individual framing,
  - learning actions vs analysis,
  - blame vs non-blame language
- Support **human review**, not replace it

This codebase is intended for **research, prototyping, and governance evaluation**, not operational deployment.

---

## High-level architecture

1. **Ingest**
   - Extracts text and tables from PDF reports
2. **EvidencePack**
   - Normalises extracted content into an auditable structure
3. **Dimension-specific agents**
   - Each agent evaluates one LRRIT dimension only
   - Agents return structured JSON with ratings, rationale, evidence quotes, and uncertainty
4. **Presentation**
   - Results rendered as static HTML for human review

---

## Repository structure

```text
lrrit-llm/
│
├── src/
│   └── lrrit_llm/
│       ├── agents/                 # Dimension-specific agents
│       │   ├── d1_compassion.py    # D1: Compassionate engagement
│       │   ├── d2_systems.py       # D2: Systems approach
│       │   ├── d3_learning_actions.py
│       │   ├── d4_blame.py         # D4: Blame language avoided
│       │   └── *.md                # Agent design notes
│       │
│       ├── ingest/
│       │   ├── pdf_text.py         # Text extraction (PyMuPDF)
│       │   └── pdf_tables.py       # Table extraction
│       │
│       ├── evidence/
│       │   ├── schema.py           # EvidencePack data model
│       │   └── pack.py             # EvidencePack builder / serializer
│       │
│       └── clients/
│           └── openai_client.py    # LLM client wrapper
│
├── scripts/
│   ├── test_d1_d4.py               # Example runner (D1 + D4)
│   └── render_results_html.py      # Render agent results to HTML
│
├── data/
│   ├── raw_pdfs/                   # Input PDF reports
│   └── processed/
│       └── reports/
│           └── <report_id>/
│               ├── evidence_pack.json
│               ├── agent_results.json
│               └── agent_results.html
│
├── README.md
└── requirements.txt
```

---

## Key concepts

### EvidencePack
A structured, auditable container holding:

- text chunks (with provenance)
- extracted tables (when present)
- stable hashes for traceability

All agents operate **only** on the EvidencePack.

### Dimension-specific agents
Each agent:

- evaluates **one LRRIT dimension only**
- returns **strict JSON** (machine-parseable)
- cites **verbatim evidence quotes** (auditable)
- flags uncertainty explicitly

Implemented agents to date:

| Agent | Dimension |
|---|---|
| D1 | Compassionate engagement |
| D2 | Systems approach to contributory factors |
| D3 | Quality & appropriateness of learning actions |
| D4 | Blame language avoided |

### Evidence polarity
Each evidence item is labelled as:

- `positive` → supports the dimension judgement
- `negative` → weakens/contradicts the dimension judgement (or indicates absence/over-reliance on individual action depending on dimension)

This supports transparent human review and later LLM-as-Judge (LaJ) consistency checking.

---

## Installation

Python **3.10+** recommended.

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Configuration

Set your OpenAI API key (PowerShell example):

```powershell
setx OPENAI_API_KEY "sk-..."
```

Optional model override:

```powershell
setx OPENAI_MODEL "gpt-4o-mini"
```

---

## Running the pipeline

### 1) Add a PDF
Place a PDF in:

```text
data/raw_pdfs/test.pdf
```

The filename (without extension) becomes the report ID (`test`).

### 2) Run agents (example: D1 + D4)

From repository root:

```powershell
py .\scripts\test_d1_d4.py
```

Outputs will be saved to:

```text
data/processed/reports/test/
```

### 3) Render HTML report

```powershell
py .\scripts\render_results_html.py
```

<img width="1355" height="1850" alt="image" src="https://github.com/user-attachments/assets/4ae0a050-dde0-4af6-a1d5-a61da502fac1" />

---

## Outputs

| File | Purpose |
|---|---|
| `evidence_pack.json` | Auditable extracted evidence |
| `agent_results.json` | Structured agent outputs |
| `agent_results.html` | Human-readable report |

---

## Design principles

- Evidence-first reasoning (verbatim quotes)
- No silent inference
- Explicit uncertainty
- Systems ≠ people
- Human review is central

---

## Status

- ✔ EvidencePack ingestion stable (text + optional tables)
- ✔ D1–D4 agents implemented and calibrated
- ✔ HTML presentation layer
- ⏳ LLM-as-Judge (LaJ) meta-evaluation layer (planned)
- ⏳ Human–LLM comparison tooling (planned)

---

## Disclaimer

This project is a **research prototype**.
It must **not** be used for operational safety governance without formal validation, clinical oversight, and organisational approval.
