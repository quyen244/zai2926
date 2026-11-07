# Hard-case tool-calling benchmark (hard_v1)

Built 2026-11-07 by `scripts/build_hard_bench.py`, scored by `scripts/eval_bench.py`.
The items themselves are per-sample data and live in `work/bench/hard_v1.jsonl` (git-ignored); this report holds aggregates only.

## What it is

529 items, every text unique, none equal or too close (token Jaccard >= 0.8) to a train transcript.
364 items are new requests written on top of a real train sample's environment (`env_id`): the contacts, events, beneficiaries, devices and forecasts they mention are read through ToolEnv.
165 items are real train samples selected by the same properties, with the official gold.
Synthetic gold follows the labelling conventions measured on train (lookup before action, `get_balance` + `list_beneficiaries` before `banking.transfer`, lookup then `sys.ask_missing_info` on an ambiguous name, `[]` for an app that is not installed or a request the user takes back, only the arguments the user said).
Every synthetic gold was replayed on a fresh ToolEnv: each call returned ok and no write was a no-op; the gold hash comes from that replay.
Texts are final-turn text only (no audio), so the benchmark measures the LLM stage, not ASR.

## Items per error group

| Group | What it tests | Synthetic | Train | Total |
|---|---|---|---|---|
| args | right tool, arguments easy to get wrong | 59 | 25 | 84 |
| under | several calls needed | 18 | 25 | 43 |
| over | only a read is needed | 10 | 25 | 35 |
| wrong_tool | a near-neighbour tool is installed too | 42 | 0 | 42 |
| no_tool | gold is `[]` | 91 | 40 | 131 |
| lookup | the reads the model skips | 90 | 25 | 115 |
| ambiguous | ask for missing or ambiguous info, conditional plans | 54 | 25 | 79 |
| **all** | | **364** | **165** | **529** |

Balance: 398 items need a call and 131 need none; 291 need a write and 238 need no write at all (no-tool plus read-only plus "no rain, do nothing" conditionals).

Synthetic subtypes (items):
contact_sms 18, contact_call 18, contact_ambiguous 18, calendar_move 18, calendar_cancel 18, bank_transfer 18, args_alarm 16, args_phone_digits 18, args_topup 18, args_weather_location 7, wrong_reminder 18, wrong_weather_current 6, wrong_ewallet 18, readonly 10, notool_not_installed 21, notool_cancel 35, notool_negation 21, notool_chitchat 14, conditional 18, multi_write 18, ask_missing 18.
The small ones (weather location, current weather, read-only, chit-chat) ran out of distinct phrasings before their quota; they were left small rather than filled with near-duplicates.

## Error labels the scorer assigns

`decision` (called when gold is `[]`, or no call when one is needed), `under` (a gold tool missing), `over` (a tool the gold does not have), `lookup_skip` (gold uses `contacts.search` / `calendar.list_events` / `banking.list_beneficiaries`, prediction does not), `wrong_args` (right tool, no exact match), `hash` (final state differs).

## Verification

| Check | Result |
|---|---|
| Gold replayed through `lifemate.agent.run_sample` | 100.0, every item full score, no error label |
| A model that never calls a tool | 24.8 (exactly the no-tool share), so negatives and positives are separated |
| Last Kaggle Qwen3-4B-AWQ run, on the 59 real items it had answered | 6.2, consistent with its 5.33 on the first 100 train samples |

Not done yet: running Qwen3-4B-AWQ on the 529 items (needs a Kaggle job).
