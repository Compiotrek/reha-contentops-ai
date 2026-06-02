# Reha ContentOps AI

Reha ContentOps AI is a minimal portfolio MVP for an AI-assisted rehabilitation content operations workflow.

## Problem

Rehabilitation content teams need a structured way to triage user feedback, identify content requests, notice safety-sensitive messages, and compare requests against existing exercise content. This project demonstrates the workflow without autonomous medical decisions or external AI calls.

## MVP Scope

- Load synthetic German feedback messages.
- Load a small normalized exercise database.
- Classify feedback into structured JSON with a mock keyword classifier.
- Route safety and low-confidence cases to human review.
- Match content requests against existing exercises with placeholder metadata scoring.
- Generate review queue outputs.
- Generate a deterministic daily content ops report.

## Architecture Overview

- `data/`: synthetic input datasets.
- `prompts/`: future LLM prompt placeholders.
- `src/schemas.py`: Pydantic data contracts.
- `src/load_data.py`: CSV loading helpers.
- `src/classify.py`: deterministic mock classifier.
- `src/decide.py`: routing and match-status decision logic.
- `src/match.py`: placeholder exercise matching.
- `src/report.py`: deterministic Markdown report generation.
- `src/process.py`: end-to-end pipeline entry point.
- `tests/`: focused pytest coverage for routing logic.

## Current Implementation

The current version uses simple keyword rules and metadata scoring only. It does not call external APIs, does not use an LLM, and does not run vector search. The mock classifier is intentionally small and replaceable so the project can later add a real classifier while keeping the same schema and routing boundaries.

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the pipeline:

```bash
python -m src.process
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

## Limitations

- Synthetic data only.
- Keyword-based mock classification.
- Placeholder metadata matching only.
- No clinical validation.
- No external LLM or embedding model integration.
- No production authentication, storage, monitoring, or reviewer UI.

## Next Steps

- Add a real LLM classifier behind the existing schema.
- Add validation against `data/expected_labels.csv`.
- Replace placeholder matching with embedding-based retrieval.
- Add reviewer-facing output states and audit logs.
- Expand synthetic test cases and report checks.
