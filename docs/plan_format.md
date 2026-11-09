# Plan format for the Plan + Trajectory SFT (experiment A)

Status: approved design, 2026-11-07 (plan with arguments + replan at decision points).
All examples below are invented; none is taken from the competition data.

## Why this shape

Published tool-calling planners write the planned calls with their arguments and refer to an earlier step's output
by a placeholder: TinyAgent (EMNLP 2024 demo, fine-tuned 1.1B/7B models, `$1`), LLMCompiler (ICML 2024, a DAG of calls),
ReWOO (2023, `#E1`).
None of them writes branches inside a plan; when what to do next depends on a result, the plan is revised after the
observation (Pre-Act 2025 refines the plan after every step; ReWOO names the fixed plan as its weakness).
Plan-and-Act (ICML 2025) builds its planner data the way we do: by annotating ground-truth trajectories with plans.

## What the model writes

Experiment A writes a plan in the content of an assistant message, right before that message's tool call (or before
the final reply).
Experiment B writes no plan; every other token of the training sequence is identical.

```
plan  := "<plan>" step " -> " ... " -> " ("end" | "replan") "</plan>"
         | "<plan>end</plan>"
step  := tool "(" param=value ", " ... ")"
value := a JSON literal known before any tool runs (the user's words, history, context, a date resolved from "mai")
       | $k   the value is copied from the RESULT of call k (1-based over the whole trajectory)
```

`end`: nothing is left to do after the listed steps.
`replan`: what comes next depends on the result of the last listed step; the assistant writes a new plan after reading it.
Dates and times are always literal, also when they equal a result value (they are computed, e.g. "one hour later").

## Rules (built by code from the gold trajectory, scripts/plan_data.py `plan_segments`)

**R1, plain chain:** one plan, before the first call, the whole gold chain, `-> end`.

```
<plan>contacts.search(query="Khoa") -> sms.send(contact_id=$1, content="mai họp 9 giờ") -> end</plan>
```

**R2, lookup then ask back** (the lookup found several matches, the gold calls `sys.ask_missing_info`):
the plan before the lookup is the happy path, as if the lookup had found exactly one match (the tool is the one named
in `sys.ask_missing_info.tool_name`, the missing params come from the lookup); the ask message carries a new plan.

```
before the lookup:  <plan>contacts.search(query="Khoa") -> sms.send(contact_id=$1) -> end</plan>
after 2 matches:    <plan>sys.ask_missing_info(tool_name="sms.send", params=["contact_id"]) -> end</plan>
```

So the first plan is the same whether the name turns out unique or ambiguous: it cannot reveal the lookup's result.
R2 applies only when the call right before the ask is a read (a lookup); an ask after a write (turn the TV on, then
ask the volume) does not depend on that result, so it stays a plain R1 chain that ends with the ask.

**R3, conditional request** (the user makes an action depend on a fact a tool must read):
the plan stops after the call that checks the condition, `-> replan`; the message right after that result carries the
plan of the rest of the gold trajectory, or `<plan>end</plan>` when the condition was false and nothing else is done.
Which call checks the condition is labelled by Qwen3-8B-AWQ (prompt `prompts/plan_branch.*.md`, job
`kaggle/plan_check/`), validated by code (the call must exist and be a read tool).

Filter on the 8B labels (decided 2026-11-08, the first pass called 745 of 1545 samples conditional, about half wrong):
- a sample whose gold calls no [read] tool is never R3, whatever the label says (nothing to wait on; such a condition
  is answered by the history or by the user's own words);
- tier 1: a request with no condition word (thì, nếu, hễ, lỡ, trường hợp, miễn là, còn không, không thì, kẻo) is not
  R3; the dropped samples that have a [read] call are hand-labelled (`scripts/label_web.py`) to measure what tier 1
  throws away;
- the rest goes to a second, stricter 8B pass (`prompts/plan_branch2.*`).

```
before the check:   <plan>weather.get_forecast(location="Huế", date="2031-03-02") -> replan</plan>
after "no rain":    <plan>end</plan>
after "rain":       <plan>reminders.create(content="mang dù", time="2031-03-02T18:00:00+07:00") -> end</plan>
```

**R4, no tool needed:** `<plan>end</plan>` before the final reply.

Every plan holds only what is known when it is written.

## Known limits

- `$k` is found by value matching (the argument value appears in call k's result and not in the user's words, the
  context or an earlier call's arguments). An aggregate ("the highest of five days") is written as a copy from the one
  call that holds the value, not as a max over all of them.
- R3 depends on the 8B label; a request it misses keeps a plain R1 plan (which then lists the branch the gold took).
- Only the first condition of a request triggers a replan.
- The final reply after the last call is a fixed short sentence per outcome (the score does not read it).

## How plans are scored at test time (scripts/ablation_metrics.py)

- Validity: every plan ends with `end` or `replan`, every tool is an offered tool.
- First-plan correctness: tool chain and ending equal the gold-derived first plan.
- Plan-to-execution consistency: the executed calls follow the written plans.
- Dependency accuracy: F1 of the `$k` references against the gold-derived plan (TinyAgent-style).
- Replan at the right check: on conditional test samples, the first plan ends with `replan` at the gold check call.
- Unnecessary planning: the gold calls nothing but a plan names a tool.
