<!--
User prompt for scripts/plan_data.py, sent once per sample. Write your prompt below this comment block
(HTML comments are stripped).

Placeholders replaced by the script (write them literally, with double braces):
  {{current_time}}  ISO time with timezone of the request, e.g. 2031-04-16T20:10:00+07:00
  {{user_profile}}  JSON of the user profile (name, home/work address, city)
  {{history}}       earlier turns of the conversation, readable (user text, assistant calls/results/replies),
                    or "(không có)"
  {{request}}       the final user utterance (reference transcript, spoken Vietnamese)
  {{tools}}         schemas (name, description, parameters) of the tools used in the gold trajectory only,
                    or "(không có)" when n_calls is 0
  {{trajectory}}    the gold calls in order, one per line: "<step> <tool> <arguments JSON> -> <result JSON>",
                    or "(không có)"
  {{dependencies}}  machine-computed data dependencies, one per line, e.g.
                    "call 5 sms.send.contact_id <- call 4 contacts.search result", or "(không có)"
  {{n_calls}}       number of gold calls (can be 0)

The reply MUST be one JSON object in this shape (see plan_sft.system.md for the full rules):
  {"plan": {"goal": "...", "steps": [{"id": "s1", "subtask": "...", "tool": "app.tool" | null,
            "call_index": 1 | null, "depends_on": ["s?", ...], "expected_output": "..."}]},
   "final_response": "..."}
-->
Write the plan and final reply for this sample, following the system rules.

CURRENT TIME: {{current_time}}

USER PROFILE: {{user_profile}}

HISTORY:
{{history}}

REQUEST: "{{request}}"

TOOLS:
{{tools}}

TRAJECTORY ({{n_calls}} calls, fixed, do not change):
{{trajectory}}

DEPENDENCIES (exact, every one must appear in depends_on):
{{dependencies}}

N_CALLS: {{n_calls}}

Checklist before answering: exactly {{n_calls}} steps with a tool, call_index 1..{{n_calls}} in order; no value that only appears in a result inside "plan"; Vietnamese, short; JSON only.
