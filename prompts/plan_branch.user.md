<!--
User prompt for scripts/plan_branch.py, one per train sample. Write your prompt below this comment block (HTML comments
are stripped). Placeholders replaced by the script (write them literally, with double braces):
  {{current_time}}     ISO time with timezone of the request
  {{history}}          earlier turns, readable ("user: ...", "assistant gọi tool(args) -> result", "assistant: ..."),
                       or "(không có)"
  {{request}}          the final user utterance (reference transcript)
  {{installed_tools}}  every installed tool, one line each: "name [read|write]: description" (sys.ask_missing_info included)
  {{trajectory}}       the gold calls in order, one line each: "<k> <tool> <arguments JSON> -> <result JSON, cut at 300 chars>",
                       or "(không có)"
  {{n_calls}}          number of gold calls
-->
CURRENT TIME: {{current_time}}

HISTORY:
{{history}}

REQUEST: "{{request}}"

INSTALLED TOOLS:
{{installed_tools}}

TRAJECTORY ({{n_calls}} calls, fixed and correct):
{{trajectory}}

Is this request conditional, and which call checks the condition? Answer with the JSON object only.
