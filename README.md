# Agentic Academic Advisor

An academic advising prototype that combines retrieval from university PDFs with SQLite tools for course prerequisites, student eligibility, and degree progress.

## Features

- BM25 retrieval over five NED University PDFs, with source and page metadata.
- Optional Pinecone dense search using OpenAI embeddings, Reciprocal Rank Fusion (RRF), and optional Sentence Transformers cross-encoder reranking.
- SQLite tools backed by 27 sample courses, 22 prerequisite mappings, and two synthetic student profiles.
- Query routing across document search, course lookup, and student tools. The uploaded implementation uses LangGraph when installed, with a sequential fallback when it is unavailable.
- Groq or OpenAI answer generation, with an evidence-based fallback when no LLM is configured.
- Jupyter notebooks and command-line evaluation with routing, source coverage, answer-term coverage, and latency metrics.

The database is synthetic demonstration data. Its course codes must not be treated as an official NED course catalogue. The documents contain historical information, including 2024 admission rules; this project is not a source of current admission advice.

## Setup

Use Python 3.10 or later. In Windows Command Prompt, from the repository root:

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
copy .env.example .env
```

On macOS or Linux, activate with `source .venv/bin/activate` and copy the template with `cp .env.example .env`.

Edit `.env` locally. For Groq generation, set `GROQ_API_KEY` and a model your account can access. The model name in the template is the project default, not a guarantee of ongoing availability. Leave unused API keys blank. No key is needed for SQLite tools or BM25-only evaluation.

`ENABLE_RERANKER=true` enables cross-encoder reranking for the advisor. The first use downloads model weights and requires network access. The retrieval notebook has its own `ENABLE_RERANKER` switch.

Pinecone search is enabled only when both Pinecone and OpenAI keys are present. It requires an already populated index compatible with the configured OpenAI embedding model. A Groq key does not enable OpenAI embeddings. Without dense search, RRF receives the BM25 ranking alone.

## Run

The sample academic database is included. To rebuild it from the synthetic fixtures:

```cmd
python scripts\seed_academic_db.py
```

To explore the notebooks:

```cmd
jupyter notebook
```

Run `notebooks/01_hybrid_retrieval_starter.ipynb`, then `notebooks/02_advisor_tools_and_graph.ipynb`.

Example questions:

- What are the prerequisites for CS402?
- Can student 1001 take CS402 next semester?
- How many credits does student 1001 still need for the degree?
- What is the admission form submission process?

## Evaluate

Run from the repository root. The evaluator uses the 24-question balanced set by default. Without `--use-llm`, it uses heuristic routing and fallback answers without LLM calls:

```cmd
python scripts\evaluate_advisor.py
```

For the 24-question mixed benchmark with full LLM generation:

```cmd
python scripts\evaluate_advisor.py --eval-file data\eval_questions_balanced_24.json --use-llm
```

`--output results.json` optionally saves the full answers and evidence. Results are printed by default, without creating experiment files.

| Metric | Meaning |
| --- | --- |
| Route exact match | Percentage of cases selecting exactly the expected tools |
| Source hit rate | Percentage of checked cases containing every expected source label |
| Expected source coverage | Individual expected-source matches across checked cases |
| Answer-term pass rate | Percentage of checked answers containing all required term groups |
| Average latency | Mean duration of an evaluated pipeline call |

Source-label matching does not verify passage relevance. Term coverage does not establish factual accuracy or citation correctness. Empty answer checks are reported as N/A. Results depend on the benchmark, LLM mode, model, retrieval configuration, and hardware. No evaluation score is claimed here without an accompanying run report.

## Repository Layout

```text
data/          University PDFs, sample SQLite database, evaluation questions
modules/       Advisor workflow, retrieval, structured tools, legacy backend helpers
notebooks/     Retrieval and advising demonstrations
scripts/       Database seeding and evaluation
routers/       Legacy backend endpoints
main.py        Inherited FastAPI application
```

The inherited FastAPI backend also needs MySQL, JWT, and cloud configuration; its frontend and database migrations are not included. The notebook and evaluation workflow uses the local SQLite database independently of those services. Generated FAISS indexes are omitted.

