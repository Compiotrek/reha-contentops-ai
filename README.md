# Reha ContentOps AI

Reha ContentOps AI is a minimal portfolio MVP for an AI-assisted rehabilitation content operations workflow.

## Problem

Rehabilitation content teams need a structured way to triage user feedback, identify content requests, notice safety-sensitive messages, and compare requests against existing exercise content. This project demonstrates the workflow without autonomous medical decisions.

## MVP Scope

- Load synthetic German feedback messages.
- Load a small normalized exercise database.
- Classify feedback into structured JSON with a mock keyword classifier or optional OpenAI-backed LLM classifier.
- Route safety and low-confidence cases to human review.
- Match content requests against existing exercises with placeholder metadata scoring.
- Generate review queue outputs.
- Generate a deterministic daily content ops report.

## Architecture Overview

- `data/`: synthetic input datasets.
- `prompts/`: prompts for optional LLM-backed workflow steps.
- `src/schemas.py`: Pydantic data contracts.
- `src/load_data.py`: CSV loading helpers.
- `src/classify.py`: deterministic mock classifier.
- `src/decide.py`: routing and match-status decision logic.
- `src/match.py`: placeholder exercise matching.
- `src/report.py`: deterministic Markdown report generation.
- `src/process.py`: end-to-end pipeline entry point.
- `tests/`: focused pytest coverage for routing logic.

## Current Implementation

The default version uses simple keyword rules and metadata scoring only. The optional LLM classifier calls the OpenAI API, validates the returned JSON against the existing Pydantic schema, and falls back to `needs_review` if parsing or validation fails. The project does not run vector search and does not generate or publish medical content.

If `OPENAI_MODEL` is not set, the LLM classifier falls back to `gpt-5.4-nano`, a low-cost model currently documented by OpenAI for simple high-volume tasks such as classification and data extraction.

## Environment Setup

Copy the example environment file:

```bash
cp .env.example .env
```

Set local values in `.env`:

```bash
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-5.4-nano
```

Never commit `.env`. The real `.env` file is ignored by Git.

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the pipeline with the default mock classifier:

```bash
python -m src.process --classifier mock
```

Run the mock classifier on only the first five messages:

```bash
python -m src.process --classifier mock --limit 5
```

Run the optional LLM classifier:

```bash
python -m src.process --classifier llm
```

Limit LLM runs to control API cost:

```bash
python -m src.process --classifier llm --limit 5
```

Run tests:

```bash
pytest
```

## Output Files

The pipeline writes:

- `outputs/processed_feedback.json`: full structured processed records.
- `outputs/review_queue.csv`: safety and low-confidence cases for human review.
- `outputs/daily_report.md`: deterministic content operations summary.

Generated output files are ignored by Git.

## Safety Principles

This is an AI-assisted content operations workflow, not an autonomous medical AI system. It does not generate medical content, publish content, or provide treatment advice. Safety-sensitive or uncertain cases are routed to human review.

The LLM classifier is constrained to return structured JSON and must not provide medical advice. Any invalid JSON, schema mismatch, changed message metadata, missing API key, or API failure returns a safe fallback classification with `labels = ["unclear"]`, `confidence = 0.0`, and `routing = "needs_review"`.

## Limitations

- Synthetic data only.
- Keyword-based mock classification by default.
- Optional LLM classification depends on OpenAI API availability.
- Placeholder metadata matching only.
- No clinical validation.
- No embedding model integration.
- No production authentication, storage, monitoring, or reviewer UI.

## Next Steps

- Add validation against `data/expected_labels.csv`.
- Replace placeholder matching with embedding-based retrieval.
- Add reviewer-facing output states and audit logs.
- Expand synthetic test cases and report checks.
