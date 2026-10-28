<!--
System prompt for scripts/llm_tag.py. Write your prompt below this comment block (HTML comments are stripped).

Placeholders replaced by the script (write them literally, with double braces):
  {{taxonomy}}       the taxonomy rendered as a readable list (from taxonomy/taxonomy.json)
  {{taxonomy_json}}  the same taxonomy as raw JSON
  {{outcome_ids}}    the allowed outcome ids, comma-separated
  {{tag_ids}}        the allowed multi-label tag ids, comma-separated

The reply MUST be one JSON object in this shape (the script rejects anything else and re-queues those samples):
  {"results": [{"id": "<sample id>", "outcome": "<one outcome id>", "tags": ["<tag id>", ...]}, ...]}
with exactly one entry per input sample.
-->
