# Các script và cách dùng

Chạy từ thư mục gốc repo, bằng Python trong `.venv`:

```powershell
.venv\Scripts\python.exe -X utf8 scripts\<tên>.py --help
```

`-X utf8` để PowerShell in đúng tiếng Việt.
Mọi script đều có hướng dẫn chi tiết ở đầu file (docstring) và `--help`.
Script nào in dữ liệu của từng mẫu (id, câu, nhãn) thì chỉ in ra màn hình hoặc ghi vào `work/` (git-ignored); `reports/` chỉ có số liệu gộp vì repo public.

## 1. Xem một mẫu train

| Script | Dùng để | Ví dụ |
|---|---|---|
| `prompt_stats.py --id` | App đã cài và toàn bộ tool trong prompt của mẫu, gom theo app, kèm số token | `prompt_stats.py --id train_0018 --tokens` |
| `prompt_mock.py` | Prompt đầy đủ (SYSTEM / TOOLS / MESSAGES) đúng như gửi cho LLM | `prompt_mock.py --split train --id train_0018` |
| `trace_sample.py` | Đi qua cả pipeline từng mẫu (một hoặc nhiều id, mỗi id một file): normaliser, prompt, từng lượt LLM, parse, chạy ToolEnv, chấm điểm | `trace_sample.py --id train_0018 train_0095 --text gold --llm cache` |
| `show_tool.py` | Schema của một tool hay một app: mô tả, tham số, ràng buộc | `show_tool.py phone.call contacts.search`, `show_tool.py --list` |

`trace_sample.py --llm cache` phát lại câu trả lời đã cache của lượt Kaggle, không gọi model (mặc định file cache mới nhất, chọn model bằng `--cache`); file ra là `work/traces/<id>.<text>.cache-<model>.txt`; id không có trong cache bị bỏ qua; `--brief` cho bản rút gọn.

## 2. Khám phá dữ liệu train

| Script | Dùng để | Ví dụ |
|---|---|---|
| `request_groups.py` | Số app / nhóm app mỗi yêu cầu cần, độ dài, điểm theo số app -> `reports/request_groups.md` | `request_groups.py --show 10` |
| `show_requests.py` | Top k mẫu ngắn / dài của một nhóm hay một app, lọc có / không lịch sử | `show_requests.py --group "di chuyen" --top 5 --history yes --show-history` |
| `show_gt.py` | Xem ground truth của một hay nhiều mẫu train: câu cuối và các gold tool call, mỗi call một dòng (`--history`, `--asr`, `--results`) | `show_gt.py train_0921 --asr` |
| `prompt_stats.py` | Thống kê gộp: số tool, số app, ký tự, token mỗi prompt | `prompt_stats.py --split train --tokens` |
| `retrieval_recall.py` | Thử BM25 lọc bớt tool có làm mất tool cần dùng không -> `reports/retrieval_bm25.md` | `retrieval_recall.py --misses 20` |
| `retrieval_dense.py` | BGE-M3 vs BM25 vs RRF ở mức tool và app, cùng context tên + mô tả -> `reports/retrieval_dense.md` (vector cache ở `work/retrieval/`) | `retrieval_dense.py --misses 10` |
| `retrieval_sample.py` | Từng mẫu: xếp hạng các app đã cài bằng BM25, BGE-M3, RRF; top-4/top-5 có giữ đủ app gold không (chỉ in màn hình) | `retrieval_sample.py train_0018 --text gold`, `retrieval_sample.py --all` |
| `stage0_harness.py` | Dựng gold train (replay state hash), splits, kiểm validator -> `reports/stage0.md` | `stage0_harness.py` |
| `stage1_labels.py` | Giải phẫu nhãn, schema, environment, lịch sử -> `reports/stage1_labels.md` | `stage1_labels.py` |
| `stage1_taxonomy.py` | Gắn taxonomy tình huống cho từng mẫu -> `reports/stage1_taxonomy.md` | `stage1_taxonomy.py` |
| `llm_tag.py` | Gắn taxonomy bằng LLM rẻ (chỉ train, chỉ lúc phát triển) | `llm_tag.py --split train --dry-run` |

`show_requests.py`: mặc định một mẫu thuộc nhóm khi mọi app trong gold đều trong nhóm đó; `--any` chỉ cần chạm nhóm; `--app maps` lọc theo một app; `--list` xem nhóm.

## 3. Đánh giá một lượt chạy (đọc cache, không chạy model)

Sau mỗi thí nghiệm: ghép cache rồi chạy `error_eda.py` (rule trong `.claude/CLAUDE.md`).

| Script | Dùng để | Ví dụ |
|---|---|---|
| `error_eda.py` | Bảng % mẫu theo nhóm lỗi (1a thừa / 1b thiếu / 1c sai giá trị tham số, gọi thiếu, gọi thừa, ...) -> `reports/error_eda.md` | `error_eda.py` |
| `error_ids.py` | Id các mẫu thuộc nhóm lỗi (mặc định 1a, 1b, 2, 3) | `error_ids.py --variant gold --group missing_args` |
| `prompt_by_error.py` | Độ dài prompt (token, ký tự), số tool, số app, số lượt history theo nhóm: đúng chuỗi tool nhưng thừa/thiếu tham số, chỉ sai giá trị, đúng hết, sai chuỗi | `prompt_by_error.py --model Qwen3-4B-AWQ --variant gold` |
| `compare_sample.py` | Một mẫu: gold và predicted đặt cạnh nhau, kèm lí do sai | `compare_sample.py train_0018 --text asr` |
| `analyze_run.py` | Điểm mỗi variant, mất điểm ở đâu, mẫu được 1 / 0 điểm | `analyze_run.py --variant gold --show 10` |
| `tool_accuracy.py` | Độ chính xác gọi tool ở 3 mức (đúng tên / khớp hẳn / cho thừa tham số), theo từng tool | `tool_accuracy.py --variant gold --top 25` |

Định nghĩa nhóm lỗi nằm ở `lifemate/eda.py`.
Chọn lượt: `--text gold` là B0 (transcript gold), `--text asr` là B1 (ASR PhoWhisper).

## 4. Benchmark hard case

| Script | Dùng để | Ví dụ |
|---|---|---|
| `build_hard_bench.py` | Dựng 529 mẫu khó vào `work/bench/hard_v1.jsonl`, replay kiểm gold | `build_hard_bench.py` |
| `eval_bench.py` | Chấm dự đoán trên benchmark theo nhóm lỗi -> `reports/hard_bench.md` | `eval_bench.py --pred gold` |

## 5. Chạy model

| Script / thư mục | Dùng để | Ví dụ |
|---|---|---|
| `kaggle/baseline/`, `kaggle/vikhr/` | Job Kaggle T4 x2: chia shard, cache theo mẫu, chấm B0 / B1 | xem `.claude/CLAUDE.md`, phần Kaggle |
| `kaggle_pack.py` | Đóng gói code + data + cache thành dataset private `lifemate-data` | `kaggle_pack.py` |
| `kaggle_merge_cache.py` | Ghép cache tải từ Kaggle về `work/kaggle/cache/` | `kaggle_merge_cache.py work/kaggle/vikhr/cache` |
| `run_baseline.py` | Chạy baseline qua một API OpenAI-compatible, ghi submission.jsonl | `run_baseline.py --split train --text gold --limit 20` |
| `notebooks/ollama_qwen35_2b.ipynb` | Thử tool calling với Qwen3.5 2B local qua Ollama, thinking tắt | mở trong Jupyter / VS Code |

Không tự chạy job nào khi chưa vẽ luồng và được duyệt.

## 6. ASR

| Script | Dùng để | Ví dụ |
|---|---|---|
| `asr_run.py` | PhoWhisper-large chạy mọi clip -> `work/asr/phowhisper/<split>.jsonl` | `asr_run.py --splits public_test --limit 20` |
| `asr_bench.py` | So nhiều model ASR trên một tập con train | `asr_bench.py --models large-v3 ... --n 400` |
| `asr_report.py` | Chấm WER / CER, recall tham số, ảo giác -> `reports/stage1_asr.md` | `asr_report.py` |
| `asr_hallucination_probe.py` | Ảo giác của Whisper có liên quan tín hiệu âm thanh không | `asr_hallucination_probe.py --model large-v3` |
| `stage1_listening.py` | Trang nghe thử cục bộ `work/listening/index.html` | `stage1_listening.py` |
| `hf_upload.py` | Upload thư mục lên repo Hugging Face private | `hf_upload.py <dir> --repo <user>/<name> --dry-run` |
