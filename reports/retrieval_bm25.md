# BM25 tool retrieval: offline recall on train

Train samples: 1545; with at least one non-sys gold tool: 1374; without (gold empty or only sys.ask_missing_info): 171.
Samples whose gold tool is not among today's candidates at all: 0.
Before (all tools, today's prompt): sample recall 100% by construction, tools per prompt mean 33.5, median 32 (incl. sys.ask_missing_info).

## Tool level: sample recall@k (all gold tools in top-k)

| config | @3 | @5 | @8 | @10 | @15 | @20 | call recall@10 |
|---|---|---|---|---|---|---|---|
| gold | desc | acc | 30.8 | 44.7 | 58.5 | 65.8 | 77.5 | 84.4 | 80.0 |
| gold | desc | noacc | 28.7 | 41.3 | 55.0 | 62.4 | 73.9 | 81.6 | 76.5 |
| gold | params | acc | 31.2 | 45.9 | 60.2 | 66.3 | 76.5 | 83.2 | 80.0 |
| gold | params | noacc | 28.7 | 41.3 | 55.6 | 63.0 | 73.4 | 81.2 | 76.4 |
| gold+hist | desc | acc | 27.3 | 40.4 | 54.3 | 61.7 | 74.2 | 82.2 | 74.1 |
| gold+hist | desc | noacc | 23.7 | 36.1 | 51.1 | 58.2 | 71.5 | 80.1 | 71.4 |
| gold+hist | params | acc | 26.2 | 40.6 | 54.7 | 61.5 | 72.3 | 80.3 | 73.4 |
| gold+hist | params | noacc | 24.2 | 35.8 | 50.7 | 56.8 | 69.7 | 77.9 | 70.0 |
| asr | desc | acc | 23.9 | 36.4 | 49.8 | 57.3 | 69.2 | 79.3 | 73.5 |
| asr | desc | noacc | 22.1 | 33.5 | 45.9 | 54.1 | 66.6 | 76.5 | 70.5 |
| asr | params | acc | 23.7 | 37.0 | 50.6 | 56.8 | 69.4 | 78.3 | 73.2 |
| asr | params | noacc | 22.4 | 34.6 | 47.9 | 54.3 | 67.5 | 76.5 | 70.7 |
| asr+hist | desc | acc | 21.0 | 34.0 | 46.2 | 54.4 | 67.1 | 78.9 | 69.4 |
| asr+hist | desc | noacc | 18.9 | 30.9 | 43.9 | 51.5 | 64.7 | 76.1 | 66.6 |
| asr+hist | params | acc | 20.3 | 33.7 | 47.0 | 54.0 | 65.8 | 76.6 | 68.5 |
| asr+hist | params | noacc | 19.9 | 30.1 | 44.4 | 52.0 | 64.8 | 74.9 | 65.9 |

## App level: keep top-k apps (all their tools) + contacts; sample recall % / mean tools kept / median tools kept

| config | max@1 | max@2 | max@3 | max@4 | sum@1 | sum@2 | sum@3 | sum@4 |
|---|---|---|---|---|---|---|---|---|
| gold | desc | acc | 39.2 / 6.2 / 6 | 62.9 / 9.6 / 10 | 75.8 / 13.2 / 13 | 82.2 / 16.6 / 16 | 40.9 / 6.7 / 7 | 67.0 / 10.3 / 10 | 79.8 / 13.7 / 14 | 85.9 / 17.0 / 17 |
| gold | desc | noacc | 36.9 / 6.2 / 6 | 58.4 / 9.6 / 10 | 71.9 / 13.1 / 13 | 79.8 / 16.5 / 16 | 37.1 / 6.8 / 7 | 58.6 / 10.6 / 11 | 73.0 / 14.1 / 14 | 81.3 / 17.4 / 17 |
| gold | params | acc | 40.5 / 6.3 / 6 | 64.8 / 9.8 / 10 | 76.4 / 13.3 / 13 | 84.3 / 16.8 / 17 | 40.8 / 6.8 / 7 | 67.1 / 10.5 / 11 | 78.8 / 14.0 / 14 | 85.2 / 17.4 / 17 |
| gold | params | noacc | 37.7 / 6.2 / 6 | 59.2 / 9.7 / 10 | 71.6 / 13.3 / 13 | 79.8 / 16.7 / 17 | 37.4 / 6.9 / 7 | 57.6 / 10.7 / 11 | 72.2 / 14.4 / 15 | 81.7 / 17.8 / 18 |
| gold+hist | desc | acc | 34.1 / 6.2 / 6 | 54.4 / 9.6 / 10 | 69.1 / 13.1 / 13 | 76.6 / 16.6 / 16 | 34.4 / 6.9 / 7 | 57.4 / 10.5 / 11 | 71.8 / 14.0 / 14 | 81.4 / 17.4 / 17 |
| gold+hist | desc | noacc | 30.9 / 6.1 / 6 | 50.6 / 9.5 / 9 | 64.4 / 12.9 / 13 | 74.2 / 16.3 / 16 | 31.4 / 7.0 / 7 | 51.3 / 10.9 / 11 | 65.1 / 14.4 / 14 | 75.9 / 17.7 / 18 |
| gold+hist | params | acc | 33.7 / 6.2 / 6 | 55.0 / 9.8 / 10 | 69.7 / 13.3 / 13 | 77.5 / 16.8 / 17 | 34.1 / 7.0 / 7 | 56.3 / 10.8 / 11 | 70.9 / 14.4 / 15 | 79.5 / 17.8 / 18 |
| gold+hist | params | noacc | 31.7 / 6.2 / 6 | 50.7 / 9.7 / 10 | 63.5 / 13.2 / 13 | 72.1 / 16.7 / 17 | 32.1 / 7.1 / 7 | 51.2 / 11.0 / 11 | 64.3 / 14.7 / 15 | 74.5 / 18.1 / 18 |
| asr | desc | acc | 32.4 / 6.2 / 6 | 51.6 / 9.7 / 10 | 64.3 / 13.2 / 13 | 72.1 / 16.6 / 16 | 33.8 / 6.7 / 7 | 55.9 / 10.3 / 10 | 68.3 / 13.7 / 14 | 75.0 / 16.9 / 17 |
| asr | desc | noacc | 30.6 / 6.2 / 6 | 47.8 / 9.7 / 10 | 60.0 / 13.2 / 13 | 68.7 / 16.5 / 16 | 30.1 / 6.9 / 7 | 48.7 / 10.7 / 11 | 63.1 / 14.1 / 14 | 71.1 / 17.4 / 17 |
| asr | params | acc | 33.3 / 6.3 / 6 | 51.9 / 9.8 / 10 | 65.4 / 13.4 / 13 | 73.6 / 16.8 / 17 | 33.8 / 6.8 / 7 | 55.1 / 10.5 / 11 | 68.1 / 14.1 / 14 | 75.9 / 17.4 / 17 |
| asr | params | noacc | 32.3 / 6.3 / 6 | 50.2 / 9.8 / 10 | 62.0 / 13.3 / 13 | 70.6 / 16.8 / 17 | 31.0 / 7.0 / 7 | 48.0 / 10.9 / 11 | 62.2 / 14.4 / 14 | 71.9 / 17.8 / 18 |
| asr+hist | desc | acc | 27.1 / 6.2 / 6 | 46.7 / 9.7 / 10 | 59.8 / 13.1 / 13 | 67.9 / 16.5 / 16 | 29.2 / 6.9 / 7 | 48.4 / 10.6 / 11 | 62.2 / 14.0 / 14 | 71.3 / 17.3 / 17 |
| asr+hist | desc | noacc | 26.2 / 6.1 / 6 | 43.5 / 9.5 / 9 | 56.0 / 13.0 / 13 | 64.9 / 16.4 / 16 | 26.9 / 7.0 / 7 | 44.0 / 10.9 / 11 | 56.6 / 14.4 / 14 | 67.2 / 17.7 / 18 |
| asr+hist | params | acc | 28.1 / 6.3 / 6 | 48.4 / 9.8 / 10 | 60.5 / 13.4 / 13 | 68.3 / 16.8 / 17 | 29.3 / 7.0 / 7 | 47.5 / 10.8 / 11 | 60.9 / 14.4 / 14 | 71.4 / 17.8 / 18 |
| asr+hist | params | noacc | 27.4 / 6.2 / 6 | 44.8 / 9.7 / 10 | 55.9 / 13.3 / 13 | 66.3 / 16.7 / 17 | 27.2 / 7.2 / 7 | 43.1 / 11.1 / 11 | 56.0 / 14.7 / 15 | 66.4 / 18.1 / 18 |

## Verdict

Best app-level config on ASR text: asr | params | acc, sum@4: sample recall 75.9%, tools kept mean 17.4, median 17 (today median 32).
Same config on the reference transcript: sample recall 85.2%.
No app-level config reaches the 99% sample-recall bar on ASR text; the best one is shown.
Verdict: BM25 app-level retrieval is NOT safe: it drops a needed tool in 24.1% of samples, and each such sample scores 0.
Tool-level top-k is weaker: best on ASR text is asr | desc | acc @20 with 79.3% sample recall; top-3 and top-5 miss far more.
Misses of the best config: 331 of 1374 samples. Most often missing gold tools: `weather.get_forecast` 49, `bill_payment.list_bills` 46, `bill_payment.pay` 36, `reminders.create` 30, `calendar.list_events` 22, `sms.send` 19, `chat_ott.list_chats` 16, `weather.get_current` 16.
Per-sample misses (ids and queries) are printed to the screen only: the repo is public.
