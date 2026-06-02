# Future Classifier Prompt Placeholder

Classify German rehabilitation content feedback into strict JSON.

Return only JSON with the agreed schema. Use only these labels:

- content_request
- criticism
- praise
- safety_signal
- bug_or_access_problem
- metadata_issue
- unclear
- other

Include an `evidence_quote` copied from the user message that supports the classification.

Do not provide medical advice. If a message mentions pain, dizziness, post-operative uncertainty, unsafe movement, or similar safety concerns, set `safety_flag` to `true` and route it to human safety review.
