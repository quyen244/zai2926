<!--
Second-pass user prompt for scripts/plan_branch.py --pass2, one per sample. Same placeholders as plan_branch.user.md,
except {{trajectory}} marks each call: "<k> [read|write] <tool> <arguments JSON> -> <result JSON, cut at 300 chars>".
-->
CURRENT TIME: {{current_time}}

HISTORY:
{{history}}

REQUEST: "{{request}}"

INSTALLED TOOLS:
{{installed_tools}}

TRAJECTORY ({{n_calls}} calls, fixed and correct):
{{trajectory}}

Apply the opposite-result test. Is this request conditional, and which [read] call checks the condition? Answer with the JSON object only.
