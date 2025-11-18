# 👤  Multi-Dimensional Candidate Matching System

## Overview

Multi-Dimensional Candidate Matching System is an end-to-end Retrieve-and-Rerank (RAG) hiring assistant that solves the classic "keyword matching" problem by combining semantic retrieval and deterministic, explainable scoring. Instead of matching on raw token overlap alone, the system indexes and evaluates candidates using both dense (semantic) and sparse (keyword) retrieval, then applies a multi-dimensional scoring framework to provide transparent, justifiable hiring recommendations.

**Problem Statement**

Traditional applicant screening relies heavily on keyword matching and brittle boolean logic. This produces two common failure modes: (1) false negatives when strong candidates use different terminology (e.g., "server-side developer" vs "backend engineer"), and (2) false positives when resumes contain keywords but lack the required depth or recency. Recruiters need a system that understands meaning, captures exact keyword requirements, and produces auditable, explainable recommendations.

**Solution Approach**

This project addresses the problem with a hybrid Retrieve-and-Rerank architecture:
- Use LLM-powered parsing (with Schema Injection) to convert unstructured resumes into validated JSON records (skills, roles, dates, education).
- Index structured records in both a dense vector store (ChromaDB) for semantic search and a BM25 index for exact keyword retrieval.
- Retrieve Top-K from both signals, merge results, and apply a deterministic 6-dimensional scoring engine that scores Must-Have, Important, Nice-to-Have, Experience, Recency, and Domain Fit.
- Produce an LLM-generated, human-readable explanation for each candidate based on the deterministic scores.

This approach balances recall (semantic matching) with precision (keyword matching) and adds transparency via deterministic scoring and explicit LLM explanations.

---

## Key Features

- **Intelligent Parsing:** LLM-based resume parsing with JSON Schema Injection to convert unstructured resumes (PDF/DOCX/TXT) into structured JSON (Skills, Education, Experience, Domain).
- **Hybrid Retrieval:** Dense vector search (ChromaDB + sentence-transformer embeddings) + BM25 keyword search for exact matches.
- **Multi-Dimensional Scoring:** Transparent 6-axis scoring (Must-Have, Important, Nice-to-Have, Years of Experience, Recency, Domain Fit) that produces a 0–100 overall score.
- **Explainable AI Reports:** Natural-language justification for each candidate showing Strengths, Gaps, and a Hiring Recommendation.
- **Streamlit UI:** Modern dashboard with a Dark Glassmorphism theme for recruiter-friendly interaction.
- **Deterministic Reranker:** Scoring logic is deterministic and adjustable via `config.py` weights.

---

## 🏗️ System Architecture

The system follows a Retrieve-and-Rerank flow:

- Ingest raw resumes and job descriptions.
- Extract text and clean content (PDF/DOCX/TXT → text).
- Use LLM (with Schema Injection) to structure resume text into strict JSON pydantic models.
- Index structured JSON into two retrieval layers: ChromaDB (dense vectors) and BM25 (sparse keyword index).
- On a Job Query, run both semantic and keyword retrieval, merge Top-K results, then apply the Scoring Engine to produce multi-dimensional scores.
- Finally, pass numeric scores and candidate facts to the LLM Explanation Layer for a human-readable report and present results in the Streamlit Dashboard.

---

## 🔄 Workflow (Detailed)

<!-- Add `diagram.png` here to visually summarize the pipeline. -->
<!-- Place your image file at repository root and name it `diagram.png`. -->
![Workflow diagram](https://github.com/adarshms444/Multi-Dimensional-Candidate-Matching-System-for-Any-Job-Posting/blob/main/deepseek_mermaid_20251118_e6490d.png)

**1. Ingestion (`utils.py`)**
- Extract text from uploaded files using `pdfplumber` (PDF) and `python-docx` (DOCX). Clean punctuation, normalize whitespace, and segment into sections (Education, Experience, Skills).

**2. Structuring (`llm_interface.py`)**
- Build prompts using JSON Schema Injection so the LLM must return structured JSON that conforms to the Pydantic models in `config.py`.
- Use an LLM endpoint (e.g., Mistral-7B-Instruct via OpenRouter) with `tenacity` for retries.
- Validate and coerce LLM output using Pydantic; fallback heuristics applied if the LLM output is invalid.

**3. Indexing & Embeddings (`retrieval.py`)**
- Generate embeddings with `sentence-transformers/all-MiniLM-L6-v2` (local) for privacy and speed.
- Index candidate embeddings in ChromaDB for dense retrieval.
- Build a BM25 index (e.g., `rank_bm25`) from tokenized skill and role text for exact keyword matching.

**4. Retrieval (Hybrid)**
- For each Job Description, produce a job embedding and run:
  - Semantic search in ChromaDB → Top-K semantic candidates.
  - BM25 keyword search → Top-K keyword matches.
- Merge and de-duplicate candidates into a Top-N set for reranking.

**5. Scoring (`scoring.py`)**
- For each retrieved candidate compute 6 dimension scores (0–100):
  - `must_have_skills` — binary/partial match scoring for required skills.
  - `important_skills` — weighted overlap for important skills.
  - `nice_to_have_skills` — bonus points for extra skills.
  - `experience_relevance` — ratio-based scoring vs required years.
  - `recency` — how recent the candidate's relevant experiences are.
  - `domain_match` — semantic similarity on domain/industry terms.
- Combine dimension scores using configurable weights in `config.py` to yield a final score.
- All scoring logic is deterministic and logged for auditability.

**6. Explanation & Presentation (`llm_interface.py` + `app.py`)**
- Send the dimension scores and candidate summary to the LLM Explanation Layer with a focused prompt asking for Strengths, Gaps, and a Final Recommendation.
- Display in the Streamlit app with a score breakdown, explanation, and links to the parsed structured JSON.

---

## 📂 Project Structure

```
candidate-matching-system/
├── app.py                # Streamlit UI
├── matching_system.py    # Orchestrator for end-to-end flow
├── llm_interface.py      # LLM prompts, retries, schema injection
├── retrieval.py          # ChromaDB + BM25 hybrid search
├── scoring.py            # 6-dimensional scoring logic
├── config.py             # Pydantic models + scoring weights
├── utils.py              # PDF/DOCX extraction helpers
├── requirements.txt      # Dependencies
├── arch.png              # Flowchart illustrating the architecture
├── End-End Jupyter notebook.ipynb  # Example notebook demonstrating end-to-end pipeline
└── README.md             # This file
```

---

## Configuration (Example)

Adjust scoring behavior in `config.py`. Example weight override to prioritize must-have skills:

```python
SCORING_WEIGHTS = {
    "must_have_skills": 0.45,
    "important_skills": 0.20,
    "nice_to_have_skills": 0.05,
    "experience_relevance": 0.10,
    "recency": 0.10,
    "domain_match": 0.10,
}
```

---

## Design Rationale

- Using both dense and sparse retrieval preserves recruiter desires for exact keyword matches (acronyms, tool names) while capturing semantic matches where labels differ ("server-side developer" vs "backend engineer").
- Schema Injection + Pydantic ensures structured, validated resume data even when using compact LLMs.
- Deterministic scoring produces an auditable ranking; LLM is used to generate human-readable justifications only (not to decide scores).
- Streamlit provides fast iteration and a polished recruiter-facing UI with low development overhead.

---

## Future Improvements

- Replace in-memory ChromaDB with persistent vector store for scale (Docker/AWS).
- Add UI model-selector (Mistral for low-cost, GPT-4 for high-accuracy paid option).
- Capture recruiter feedback (thumbs up/down) to adjust scoring weights automatically.

---

## Installation & Usage

Prerequisites:
- Python 3.8+
- OpenRouter API key (or other LLM endpoint)

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit UI:

```bash
streamlit run app.py
```

Usage:
- Enter your OpenRouter API key in the sidebar.
- Upload a Job Description (PDF/TXT/DOCX).
- Upload one or more Resumes (PDF/TXT/DOCX).
- Click "Start Matching" and review ranked candidates and explanations.

---
