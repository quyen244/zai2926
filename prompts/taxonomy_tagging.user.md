<!--
User prompt for scripts/llm_tag.py, sent once per batch. Write your prompt below this comment block.

Placeholders (double braces):
  {{n}}        number of samples in this batch
  {{samples}}  one JSON object per line:
               {"id", "text" (final user turn), "installed_apps" ([app ids]),
                "history" ([{"role": "user"|"assistant", "text"}], last turns only, truncated)}
  The taxonomy placeholders of the system prompt also work here.
-->
