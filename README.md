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
- Raise MVP content-gap alerts when repeated `track_only` content requests cluster together.
- Generate a deterministic daily content ops report.

## Architecture Overview

- `data/`: synthetic input datasets.
- `prompts/`: optional local prompt files, ignored by Git.
- `src/schemas.py`: Pydantic data contracts.
- `src/load_data.py`: CSV loading helpers.
- `src/classify.py`: classifier facade and hybrid orchestration.
- `src/metadata.py`: post-classification request metadata extraction.
- `src/train_ml_classifier.py`: local TF-IDF classifier training.
- `src/evaluate_ml_classifier.py`: illustrative ML evaluation report generation.
- `src/evaluate_pipeline.py`: held-out end-to-end outcome evaluation.
- `src/decide.py`: routing and match-status decision logic.
- `src/match.py`: placeholder and embedding-based exercise matching.
- `src/content_gap.py`: repeated track-only request clustering and alert generation.
- `src/report.py`: deterministic Markdown report generation.
- `src/process.py`: end-to-end pipeline entry point.
- `tests/`: focused pytest coverage for routing logic.

## Current Implementation

The default version uses simple keyword rules and placeholder metadata scoring. The optional LLM classifier calls the OpenAI API, validates the returned JSON against the existing Pydantic schema, and falls back to `needs_review` if parsing or validation fails. The experimental local ML classifier uses TF-IDF plus one-vs-rest logistic regression trained from synthetic expected labels. Request metadata extraction is a separate post-classification step, so ML or rule-gated classifications can still enrich fields such as body region, therapy goal, equipment, difficulty, and a short free-text request theme before matching. The optional embedding matcher uses OpenAI embeddings to compare content requests against existing approved exercise records, then combines vector similarity with metadata fit checks when metadata is available. Content-gap detection is implemented as an MVP alerting layer for repeated `track_only` requests, not as a production-grade semantic clustering system.

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

Evaluate the end-to-end pipeline against a held-out outcome set:

```bash
python -m src.evaluate_pipeline \
  --classifier hybrid \
  --matcher embeddings \
  --metadata-extractor llm \
  --input-csv data/heldout_eval_messages.csv \
  --expected-csv data/heldout_expected_outcomes.csv
```

This writes `outputs/pipeline_evaluation.md` and `outputs/pipeline_evaluation_details.csv`.

Run ML-only mode:

```bash
python -m src.process --classifier ml --matcher placeholder --limit 5
```

Run hybrid mode with embedding matching:

```bash
python -m src.process --classifier hybrid --matcher embeddings --limit 5
```

Use metadata-only LLM extraction after cheap ML/rule classification:

```bash
python -m src.process --classifier hybrid --matcher embeddings --metadata-extractor llm --limit 5
```

Use `--metadata-extractor none` to keep only metadata returned by the classifier. In `auto` mode, the pipeline uses LLM metadata extraction when an API key is available and falls back to deterministic metadata rules for mock or placeholder demos.

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

Process a batch through the backend with CSV text:

```bash
curl -X POST http://localhost:8000/process-feedback-batch \
  -H "Content-Type: application/json" \
  -d '{
    "csv_text": "message_id,user_message\nmsg_001,Bitte mehr Knieübungen ohne Geräte\nmsg_002,Video lädt nicht",
    "classifier": "mock",
    "matcher": "placeholder",
    "metadata_extractor": "rules",
    "write_output_files": true
  }'
```

The batch endpoint also accepts a JSON `messages` list instead of `csv_text`. The exercise database is loaded by the FastAPI backend from `data/exercises.csv`; callers do not send exercise records. If `write_output_files` is true, the backend writes the same files as the CLI pipeline into `outputs/`.

n8n can call `POST /process-feedback-batch` with an HTTP Request node and pass either CSV text or message records plus optional classifier, matcher, and metadata modes. n8n should not implement matching or load the exercise database itself.

An importable showcase workflow is available at:

```text
n8n/reha_contentops_showcase_workflow.json
```

Before running it, start the backend:

```bash
uvicorn src.api:app --port 8000
```

The workflow includes two clearly separated demo lanes:

- Batch lane: manual demo or Slack-style webhook input, backend batch processing, review-queue payload, content-gap ticket payload, and ops digest payload.
- Chat lane: browser chat webhook input, backend batch processing, short UI reply, plus the same review/gap/digest payloads for showcase visibility.

The imported n8n workflow hardcodes the backend URL for n8n running in Docker:

```text
http://host.docker.internal:8000/process-feedback-batch
```

Use different webhook URLs for batch CSV and the chat UI:

```text
Batch CSV: http://localhost:5678/webhook/reha-contentops-batch
Chat UI:   http://localhost:5678/webhook/reha-contentops-chat
```

Send a CSV file to the n8n batch webhook:

```bash
curl -X POST http://localhost:5678/webhook-test/reha-contentops-batch \
  -F "file=@slack_interview_demo_messages.csv" \
  -F "classifier=hybrid" \
  -F "matcher=embeddings" \
  -F "metadata_extractor=llm"
```

The chat UI should keep using `/webhook/reha-contentops-chat`. It sends a single message and expects a short `{ "reply": "..." }` response.

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
- `outputs/content_gap_alerts.csv`: repeated `track_only` content request clusters that cross the content-gap threshold.
- `outputs/daily_report.md`: deterministic aggregate content operations report.
- `outputs/exercise_embeddings.json`: local embedding cache for approved exercise records when embedding matching is used.
- `outputs/ml_evaluation.md`: illustrative local ML classifier evaluation report.
- `outputs/pipeline_evaluation.md`: held-out end-to-end evaluation report.
- `outputs/pipeline_evaluation_details.csv`: per-message expected vs actual pipeline outcome details.

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
- Content-gap clustering is an MVP signal for repeated demand, not a fully validated semantic clustering product.
- Optional LLM classification depends on OpenAI API availability.
- Embedding matching depends on OpenAI API availability.
- No clinical validation.
- No production authentication, storage, monitoring, or reviewer UI.

## Next Steps

- Add draft generation for confirmed content gaps.
- Add scheduled batch runs through n8n with a daily/weekly content ops digest.
- Add reviewer-facing output states and audit logs.
- Expand synthetic test cases and report checks.
