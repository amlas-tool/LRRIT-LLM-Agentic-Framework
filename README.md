# Learning Response Review and Improvement Tool – An LLM-based Agentic Prototype

This repository contains a prototype **agentic framework** for applying Large Language Models (LLMs) to the *Learning Response Review and Improvement Tool (LRRIT)* used in healthcare safety governance.

The code ingests incident / learning response documents (PDFs), extracts structured evidence, and evaluates them across multiple LRRIT dimensions using **dimension-specific LLM agents**, with outputs designed to be auditable, explainable, and suitable for human comparison.

The code implements a structured, multi-agent Large Language Model (LLM) evaluation pipeline to assess investigation reports against the LRRIT scoring rubric (v2) and associated PSIRF principles. The system operationalises eight defined LRRIT dimensions (D1–D8), each implemented as a specialist “reviewer” agent. These agents produce ratings, evidence-anchored rationales, and structured outputs. A secondary LLM-as-Judge (LaJ) layer then evaluates the quality of each agent’s reasoning against a defined meta-evaluation basket (rubric fidelity, grounding, coherence, values alignment, uncertainty handling, hallucination risk, rubric structure adherence).

The overall objective is to provide a reproducible, traceable, and auditable AI-assisted review tool that supports structured analysis of investigation reports while preserving explicit links to source text in the underlying PDF.

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

The original thinking behind the implementation was to have:

- Clean separation of dimensions (D1–D8) with no conceptual leakage
- Evidence-grounded outputs (verbatim, auditable, aligned with rubric)
- Appropriate uncertainty handling (especially in D5–D7)
- An agentic architecture that actually maps directly to the LRRIT rubric
- A rating system that can be explained to clinicians, safety scientists, and auditors.

Further explanation can be found below.

NB. This codebase is intended for **research, prototyping, and governance evaluation**, not operational deployment.

---

## High-level architecture
<img width="1644" height="878" alt="image" src="https://github.com/user-attachments/assets/5c65d686-c6bd-4512-95fa-ebb446be8152" />

The pipeline is organised into five conceptual stages, using the test_agents.py script run from the source root directory. Currently it uses a hardcoded pdf file to analyse:

- PDF ingestion and extraction
- EvidencePack construction
- Dimension agent evaluation (D1–D8)
- LLM-as-Judge (LaJ) meta-evaluation
- HTML report rendering
---

1. **PDF ingestion and extraction**
   - Extracts text and tables from PDF reports using open source pdf python libs.
2. **EvidencePack construction**
   - Normalises extracted content into an auditable structure. This is the ground truth in terms of what is passed to the LLM.
3. **Dimension-specific agent evaluation (D1–D8)**
   - Each agent evaluates one LRRIT dimension only using the evidence pack contents.
   - Agent prompts are kept separately for easier editing and contain the entire rubric (v2).
   - Agents return structured JSON with ratings, rationale, evidence quotes. The evidence should be grouped by the evidence type indicators from the rubric.
   - Each agent is assessed for its performance by the LLM (LLM-as-Judge or LaJ), using a basket of metrics for task evaluation to reduce the risk of hallucination, errors, misalignment etc. by the agents.
     - *NB. The LaJ does not assess the agents against the report, it assesses them against the rubric and the verbatim evidence retrieved by the agent.*
4. **LLM-as-Judge (LaJ) meta-evaluation**
   - Series of metrics that try to give a confidence score on whether the agent completed its task correctly and successfully.
   - Results in a PASS, WARN, FAIL grading.
6. **Presentation**
   - Separate script uses agent_results.json and and laj_results.json files to generate html presentation layer.
   - Results rendered as static HTML for human review
   - Dynamic drop down allows user to drill down into detail for evaluation metrics.

---

## Key Implementation Decisions & Trade-offs

- Deterministic Extraction: Prioritizes reproducibility over the flexibility of LLM-based vision extraction.
- JSON-First Output: Uses strict formatting with parsing to balance robustness and structure.
- Independent LaJ Layer: Separating agent evals from the initial agent call increases assurance.
- Smaller LLMs from OpenAI were used during development to keep costs down. However, performance was largely unsatisfactory.

Some design decisions to note:

First, verbatim quoting is mandatory. Agents must not paraphrase. This constraint reduces semantic drift and supports automated validation. 

Second, page resolution is not trusted to the agent. Agents may mis-assign chunk IDs. A helper module (resolve.py) validates and corrects ID–quote alignment against the EvidencePack. This includes tolerant matching (normalisation of punctuation, whitespace, line breaks). 

Third, agent outputs are stored independently in agent_results.json before any LaJ evaluation. This preserves a clean separation between primary review and meta-review.

---
### EvidencePack

All agents operate **only** on the EvidencePack.   

The ingestion stage parses the PDF using PyMuPDF (for text) and table extraction utilities. Extracted content is structured into an EvidencePack object and serialised to evidence_pack.json.

Each EvidencePack entry contains:

- chunk_id (e.g., p18_c01 or p13_t01)
- provenance metadata (page, extractor, source path)
- raw text
- text_hash (for integrity checks)

Chunks are page-scoped and relatively coarse. This simplifies downstream quote matching and makes traceability manageable. The EvidencePack design decision was deliberate:

- Agents do not operate directly on PDFs.
- All evaluation must reference structured, indexed text blocks.
- Verbatim quotes must be ≤25 words.
- Traceability must support PDF deep linking.

## PDF parsing and evidence extraction

PDF reports are ingested using open-source libraries (`PyMuPDF` for text extraction and `pdfplumber` for tables). We do this so that the text extraction from the reports is deterministic and not LLM dependent, otherwise what is extracted might change each time we change the main model. By using standard python libraries, we can better guarantee consistency. However, the performance of these libraries on real world data leaves much to be desired, and we may need to find an alternative mechanism, such as using the LLM to extract the text and tables.

Text is extracted page-by-page and normalised into traceable text chunks, while tables (where present) are extracted separately and preserved with fallback textual representations. All extracted content is wrapped in an **EvidencePack** with explicit provenance (source file, page number, extractor), ensuring that every agent judgement can be traced back to the original document. This isn't completely reliable.

---
### Dimension-specific agents
Each agent:

- evaluates **one LRRIT dimension only**
- returns **strict JSON** (machine-parseable)
- cites **verbatim evidence quotes** (auditable)
- flags uncertainty explicitly

Agents follow a consistent pattern:

- Prompt construction referencing the LRRIT rubric definition
- Requirement for verbatim quotes
- Structured JSON output schema
- Post-processing to normalise and validate output

Agent output schema:
```
{{
  "rating": "GOOD" | "SOME" | "LITTLE",
  "rationale": "string",
  "evidence": [
    {{
      "id": "Text pXX_cYY" | "Table pXX_tYY",
      "quote": "verbatim excerpt from the evidence without trailing punctuation, <= 50 words",
      "evidence_type": "evidence_type from indicator prefix (e.g. SOME - Influence of engagement is implied but unclear: )" 
    }}
  ],
  "missing_indicators": ["<exact indicator text>"],
  }}
```

Implemented data dimension agents:

| Agent | Dimension |
|---|---|
| D1 | Compassionate engagement |
| D2 | Systems approach to contributory factors |
| D3 | Quality & appropriateness of learning actions |
| D4 | Blame language avoided |
| D5 | Local rationality (why actions made sense at the time)|
| D6 | Counterfactuals: How outcomes & alternatives are reasoned about afterwards|
| D7 | Safety actions / recommendations|
| D8 | Communication quality and usability of the learning response|

---

### Evidence type and rating

The LRRIT guidance words on evidence - GOOD, SOME, LITTLE - are used to grade the dimensions with supporting rationale. 

The LRRIT scoring rubric v2 follows the same pattern for all dimensions: LRRIT dimension, core definition, discriminators and indicators for evidence levels. Each dimension's scoring rubric is maintained in a separate prompt text file which is loaded by the agent at runtime.

Evidence from each agent should be collated under a single grading, and then under each discriminator heading. The collation is done in the html presentation layer to prevent misalignment in agent outputs.

---

## LaJ evaluation metrics
The LLM-as-judge or LaJ asesses how well the agents have performed their task. It tries to see if the evidence they cite and their rationale could be erroneous, hallucinated or incoherent / misaligned. 

*It does not judge the agent outputs against the report in question.* That would be very expensive in terms of tokens and time.

Below are a basket of metrics that are currently unweighted. However, we may weight them to give us a combined single grade as a confidence metric. Please note, this part is _still under development_.

### LaJ metric basket
1. Rubric Fidelity
   - Does the rationale address the intended LRRIT judgement criteria for the dimension?
2. Evidence Grounding
   - Are claims in the rationale supported by the cited excerpts?
3. Reasoning Quality & Internal Coherence
   - Does the rationale logically support the rating without generic/circular statements?
4. Values Alignment (PSIRF/LRRIT)
   - Does the rationale reflect PSIRF/LRRIT values (systems thinking, compassion, local rationality, avoid blame/counterfactual misuse)?
5. Transparency & Uncertainty Handling
   - Is uncertainty signalled appropriately for mixed/ambiguous evidence?
6. Hallucination Screening (agent-output level)
   - Does the rationale introduce claims not supported by the supplied excerpts?
7. Rubric Structure Adherence
   - Has the agent followed the evidence discriminators specified in the rubric? It should not paraphrase or reword these.
  
The LaJ grades each of these metrics as follows
- PASS: clearly meets the metric, verbatim evidence backs up the rating.
- WARN: partially meets; minor gaps, evidence is acceptable, but minor issues such as changed punctuation in verbatim quotes.
- FAIL: materially fails; usually due to verbatim quote not found or hallucinated.

The LaJ assigns an overall grade to the agent's performance based on the same grades. However, this may change if we introduce weights into the metrics (especially no. 2 and 6).

Note that the LaJ oes NOT re-evaluate the original report. It only uses the cited evidence blocks (and optional programmatic quote checks) to assess grounding and hallucination risk.

Expected agent and laj output schema:
```
  Agent Output
      {
        "rating": "GOOD" | "SOME" | "LITTLE",
        "rationale": "string",
        "evidence": [
            {{
            "id": "Text pXX_cYY" | "Table pXX_tYY",
            "quote": "verbatim excerpt from the evidence without trailing punctuation, <= 50 words",
            "evidence_type": "evidence_type from indicator prefix (e.g. SOME - Influence of engagement is implied but unclear: )" 
            }}
        ],
        "missing_indicators": ["<exact indicator text>"],
      }

    LaJ Output:
      {
        "judge_id": "LaJ",
        "agent_id": "...",
        "dimension": "...",
        "overall": "PASS|WARN|FAIL",
        "metrics": [{"metric_id":"...", "score":"...", "notes":"..."}],
        "flags": {"missing_evidence": bool, "quote_mismatch": bool, "invalid_evidence_id": bool}
      }
```

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

### 2) Run agents 

From repository root:

```powershell
py .\scripts\test_agents.py
```
Example session is given below. Note, running with a model such 03-mini on a report over 50 pages in length can be very slow (5 mins). Currently no attempt is made to create parallel execution threads:
```
PS G:\My Drive\LLM projects\lrrit-llm> py .\scripts\test_agents.py
[1/4] Extracting text: G:\My Drive\LLM projects\lrrit-llm\data\raw_pdfs\investigation-report-202410-redcliffematernity-services.pdf
[2/4] Extracting tables -> data\processed\reports\investigation-report-202410-redcliffematernity-services
[3/4] Building EvidencePack
Saved EvidencePack: data\processed\reports\investigation-report-202410-redcliffematernity-services\evidence_pack.json
[4/4] Running agents...
Running agents on evidence pack using model: o3-mini
D1 completed, evidence of compassion : SOME ...
D2 completed, systems approach : GOOD ...
D3 completed, learning actions : GOOD ...
D4 completed, blame language : GOOD ...
D5 completed, local rationality : GOOD ...
D6 completed, hindsight bias : GOOD ...
D7 completed, improvement actions : GOOD ...
D8 completed, communication quality : GOOD ...
Saved agent results: data\processed\reports\investigation-report-202410-redcliffematernity-services\agent_results.json
Running LaJ meta-evaluation on data\processed\reports\investigation-report-202410-redcliffematernity-services\agent_results.json
D1 completed, rating: WARN...
D2 completed, rating: WARN...
D3 completed, rating: WARN...
D4 completed, rating: WARN...
D5 completed, rating: FAIL...
D6 completed, rating: FAIL...
D7 completed, rating: WARN...
D8 completed, rating: WARN...
PS G:\My Drive\LLM projects\lrrit-llm> py .\scripts\render_results_html.py
Wrote: G:\My Drive\LLM projects\lrrit-llm\data\processed\reports\investigation-report-202410-redcliffematernity-services\agent_results.html
PS G:\My Drive\LLM projects\lrrit-llm> 
```


The outputs will be saved to the report directory. Don't forget to generate the html report as follows.

### 3) Render HTML report

```powershell
py .\scripts\render_results_html.py
```
## Note on run_report.py 
```
script scripts\run_report.py
````
This script is unfinished and currently won't work. It was intended to give a better CLI with parameters, so that pdfs did not have to be hardcoded and you could run agents individually without re-running the pdf extraction processes. It was also meant to enable running the LaJ separately. Unfortunately it required too many changes to the existing code and was abandoned. However, with a bit more effort something like this would be a big help during development and testing. Alternatively, a web-based UI to load parameters and run agents / LaJ parts might be better for users.

---
## Example html output
A full example can be found here (**NB.** the `Open report` button **will not work**):

[https://raw.githack.com/amlas-tool/lrrit-llm/main/data/processed/reports/investigation-report-202410-redcliffematernity-services/agent_results.html]

---

<img width="1008" height="1261" alt="image" src="https://github.com/user-attachments/assets/ff581862-a743-4684-a074-42844b40d93b" />

---

## Outputs

| File | Purpose |
|---|---|
| `evidence_pack.json` | Auditable extracted evidence |
| `agent_results.json` | Structured agent outputs |
| `laj_results.json`   | Structured LaJ evals for each agent |
| `agent_results.html` | Human-readable report containing agent outputs and LaJ evaluations (generated by render_results_html.py)|

---

---
## Repository structure

```text
lrrit-llm/
│
├── src/
│   └── lrrit_llm/
│       ├── agents/                       # Dimension-specific agents
│           ├── profiles/                 # Agent profiles in mark down (reference only)
│       │      └── *.md                      # Agent design notes
│       │   ├── d1_compassion.py          # D1: Compassionate engagement
│       │   ├── d2_systems.py             # D2: Systems approach
│       │   ├── d3_learning_actions.py    # D3: Human error / learning actions
│       │   ├── d4_blame.py               # D4: Blame language avoided
│       │   ├── d5_local_rationality.py   # D5: Local rationality / reasoning
│       │   ├── d6_counterfactuals.py     # D6: Counterfactual reasoning
│       │   ├── d7_actions.py             # D7: Safety Actions / recs to take
│       │   ├── d8_clarity.py             # D8: Communication quality and usability
│       │
│       ├── ingest/
│       │   ├── pdf_text.py         # Text extraction (PyMuPDF)
│       │   └── pdf_tables.py       # Table extraction
│       │
│       ├── evidence/
│       │   ├── schema.py           # EvidencePack data model
│       │   └── pack.py             # EvidencePack builder / serializer
│       │ 
│       ├── laj/
│       │   ├── laj_meta.py           # LLM-as_Judge metrics and rules
│       │   ├── dimensions_defs.py    # summary of data dimensions
│       │   
│       ├── prompts/
│       │   ├── d1_prompt.txt        # text prompt loaded by agents
│       │   ├── d2_prompt.txt        # prompts follow rubric word for word
│       │   ├── d3_prompt.txt
│       │   ├── etc
│       │   
│       └── clients/
│           └── openai_client.py    # LLM client wrapper
│
├── scripts/
│   ├── test_agents.py              # Test script to run agents against a hardcoded pdf report
│   └── render_results_html.py      # Render agent results to HTML
│
├── data/
│   ├── raw_pdfs/                   # Input PDF reports
│   └── processed/
│       └── reports/
│           └── <report_id>/
│               ├── evidence_pack.json
│               ├── agent_results.json
│               ├── laj_results.json        # LaJ QA over D1–D8
│               └── agent_results.html # Generated by render_results.py
│
├── README.md
└── requirements.txt
```
## Status

- ✔ EvidencePack ingestion stable (text + optional tables)
- ✔ D1–D8 agents implemented and calibrated
- ✔ HTML presentation layer
- ✔ LLM-as-Judge (LaJ) meta-evaluation layer
- ⏳ Feature extraction for complexity analysis (planned)
- ⏳ Human–LLM comparison tooling (planned)

---

## Disclaimer

This project is a **research prototype**.
It must **not** be used for operational safety governance without formal validation, clinical oversight, and organisational approval.
