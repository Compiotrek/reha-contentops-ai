# Reha ContentOps AI

Reha ContentOps AI is a minimal portfolio MVP for an AI-assisted rehabilitation content operations workflow.

## Problem

Rehabilitation content teams need a structured way to triage user feedback, identify content requests, notice safety-sensitive messages, and compare requests against existing exercise content. This project demonstrates the workflow without autonomous medical decisions.

## MVP Scope

- Load 180 synthetic German feedback messages.
- Load a 50-record synthetic normalized exercise database.
- Classify feedback into structured JSON with mock, LLM, ML, or hybrid ML+LLM classifier modes.
- Route safety and low-confidence cases to human review.
- Match content requests against existing approved exercises with placeholder metadata scoring or optional embedding-based RAG matching.
- Generate review queue outputs.
- Generate a deterministic daily content ops report.

## Architecture Overview

- `data/`: synthetic input datasets.
- `prompts/`: optional local prompt files, ignored by Git.
- `src/schemas.py`: Pydantic data contracts.
- `src/load_data.py`: CSV loading helpers.
- `src/classify.py`: deterministic mock classifier.
- `src/train_ml_classifier.py`: local TF-IDF classifier training.
- `src/evaluate_ml_classifier.py`: illustrative ML evaluation report generation.
- `src/decide.py`: routing and match-status decision logic.
- `src/match.py`: placeholder and embedding-based exercise matching.
- `src/report.py`: deterministic Markdown report generation.
- `src/process.py`: end-to-end pipeline entry point.
- `tests/`: focused pytest coverage for routing logic.

## Current Implementation

The default version uses simple keyword rules and placeholder metadata scoring. The optional LLM classifier calls the OpenAI API, validates the returned JSON against the existing Pydantic schema, and falls back to `needs_review` if parsing or validation fails. The experimental local ML classifier uses TF-IDF plus one-vs-rest logistic regression trained from synthetic expected labels. The optional embedding matcher uses OpenAI embeddings to compare content requests against existing approved exercise records, then combines vector similarity with simple metadata fit checks.

Embedding-based RAG matching is used only for content operations lookup. It does not validate clinical appropriateness, generate exercise instructions, or publish medical content.

All dataset records are synthetic and demo-oriented: 50 exercise records, 180 feedback messages, and expected labels for evaluation checks. The data is not clinically validated and must not be treated as real patient data or real treatment guidance.

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
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

Never commit `.env`. The real `.env` file is ignored by Git.

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the pipeline with the default mock classifier:

```bash
python -m src.process --classifier mock --matcher placeholder
```

Run the mock classifier and placeholder matcher on only the first five messages:

```bash
python -m src.process --classifier mock --matcher placeholder --limit 5
```

Run embedding-based RAG matching with the mock classifier:

```bash
python -m src.process --classifier mock --matcher embeddings --limit 5
```

Run the optional LLM classifier with embedding matching:

```bash
python -m src.process --classifier llm --matcher embeddings --limit 5
```

Use `--limit` during testing to control API cost.

## Hybrid ML + LLM Classification

The local ML classifier is a cheap cost-control layer. It uses TF-IDF features with one-vs-rest logistic regression and a separate binary safety classifier. It only auto-accepts high-confidence predictions with a sufficient top-2 margin. Low-confidence or ambiguous ML outputs abstain and route to review in `ml` mode.

In `hybrid` mode, the ML classifier runs first. Safety signals route directly to safety review. High-confidence ML predictions are accepted. Uncertain cases fall back to the LLM classifier. This is not a medical classifier and does not validate clinical appropriateness.

With limited synthetic data, ML evaluation is illustrative and experimental. Do not overinterpret the metrics.

Train the local model:

```bash
python -m src.train_ml_classifier
```

Generate an illustrative evaluation report:

```bash
python -m src.evaluate_ml_classifier
```

Run ML-only mode:

```bash
python -m src.process --classifier ml --matcher placeholder --limit 5
```

Run hybrid mode with embedding matching:

```bash
python -m src.process --classifier hybrid --matcher embeddings --limit 5
```

## Embedding Cache

Exercise embeddings are cached locally in `outputs/exercise_embeddings.json`. The cache is reused when the embedding model, exercise ID, and searchable exercise text hash match. Only missing or changed exercise embeddings are regenerated.

Query embeddings are generated per run and are not cached yet. If `OPENAI_EMBEDDING_MODEL` is not set, the matcher defaults to `text-embedding-3-small`.

If `OPENAI_API_KEY` is missing, use:

```bash
python -m src.process --classifier mock --matcher placeholder --limit 5
```

Run tests:

```bash
pytest
```

## FastAPI Service

Run the API locally:

```bash
uvicorn src.api:app --reload
```

Test health:

```bash
curl http://localhost:8000/health
```

Process one feedback message:

```bash
curl -X POST http://localhost:8000/process-feedback \
  -H "Content-Type: application/json" \
  -d '{"message_id":"msg_demo_001","user_message":"Die Knieübungen sind zu schwer. Ich hätte gern leichtere Varianten ohne Geräte.","classifier":"mock","matcher":"placeholder"}'
```

`classifier` and `matcher` are optional. Defaults are `mock` and `placeholder`.

n8n can call `POST /process-feedback` with an HTTP Request node and pass the feedback message plus optional classifier and matcher modes.

For routing, n8n should use an IF node that checks:

```text
review_required == true
```

Only those items should go to a human review queue. Items with `review_required == false` can be logged, aggregated, linked to existing content, or routed to support based on `final_action` and `priority`.

## Output Files

The pipeline writes:

- `outputs/processed_feedback.json`: full detailed JSON output for every processed message.
- `outputs/content_ops_decisions.csv`: one row per message with decision, match, action, review, and priority fields.
- `outputs/review_queue.csv`: only items that require human review.
- `outputs/daily_report.md`: deterministic aggregate content operations report.
- `outputs/exercise_embeddings.json`: local embedding cache for approved exercise records when embedding matching is used.
- `outputs/ml_evaluation.md`: illustrative local ML classifier evaluation report.

Generated output files are ignored by Git.

## Safety Principles

This is an AI-assisted content operations workflow, not an autonomous medical AI system. It does not generate medical content, publish content, or provide treatment advice. Safety-sensitive or uncertain cases are routed to human review.

The LLM classifier is constrained to return structured JSON and must not provide medical advice. Any invalid JSON, schema mismatch, changed message metadata, missing API key, or API failure returns a safe fallback classification with `labels = ["unclear"]`, `confidence = 0.0`, and `routing = "needs_review"`.

RAG matching is conservative: strong matches can be marked as existing content, mismatched but similar items become possible duplicates, and low-similarity single requests are tracked only. A single request is not treated as a content gap candidate.

The ML classifier is also conservative: uncertain predictions abstain, hybrid mode falls back to the LLM where available, and safety signals route to review.

## Limitations

- Synthetic data only.
- Demo/evaluation data is not clinically validated.
- Keyword-based mock classification by default.
- Experimental ML classifier trained only on synthetic labels.
- Optional LLM classification depends on OpenAI API availability.
- Embedding matching depends on OpenAI API availability.
- No clinical validation.
- No production authentication, storage, monitoring, or reviewer UI.

## Next Steps

- Add validation against `data/expected_labels.csv`.
- Add repeated-request clustering before any content gap workflow.
- Add reviewer-facing output states and audit logs.
- Expand synthetic test cases and report checks.
