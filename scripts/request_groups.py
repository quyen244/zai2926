"""How many apps / app groups one train request needs, and how long it is.

  python scripts/request_groups.py          # writes reports/request_groups.md (aggregates only)
  python scripts/request_groups.py --show 10  # also prints the 10 longest multi-group requests (screen only, never written)

App groups are our own reading aid (GROUPS below), not an official taxonomy. sys.* calls are not counted as an app.
The last table scores the cached Kaggle runs (work/kaggle/cache) by number of apps in the gold, when the cache exists.
"""
from __future__ import annotations

import argparse
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lifemate.data import PROJECT, load_catalog, load_samples  # noqa: E402

GROUPS = {
    "Nhà thông minh": "smart_home home_security door_lock robot_vacuum appliances home_energy home_automation household_chores",
    "Thời gian, ghi chú": "alarm reminders calendar notes tasks",
    "Liên lạc": "contacts phone sms chat_ott email video_meeting",
    "Tài chính": "banking ewallet bill_payment loyalty subscriptions",
    "Thông tin": "weather gold_price fuel_price lottery exchange_rate news calculator",
    "Di chuyển": "maps ride_hailing flights bike_rental",
    "Mua sắm": "food_delivery shopping_list package_tracking",
    "Giải trí": "music tv podcast_radio",
    "Sức khỏe": "medication fitness",
    "Học": "learning library",
}
GROUP_OF = {a: g for g, apps in GROUPS.items() for a in apps.split()}


def apps_of(calls: list[dict]) -> set[str]:
    return {c["name"].split(".")[0] for c in calls if not c["name"].startswith("sys.")}


def dist_table(title: str, counts: Counter, n: int) -> list[str]:
    out = [f"| {title} | Mẫu | Tỉ lệ |", "|---:|---:|---:|"]
    out += [f"| {k} | {v} | {100 * v / n:.1f}% |" for k, v in sorted(counts.items())]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", type=int, default=0)
    a = ap.parse_args()

    catalog = load_catalog("train")["apps"]
    missing = [x["id"] for x in catalog if x["id"] != "sys" and x["id"] not in GROUP_OF]
    if missing:
        raise SystemExit(f"apps without a group: {missing}")
    S = load_samples("train")
    n = len(S)
    rows = []
    for s in S:
        g = s["label"]["tool_calls"]
        apps = apps_of(g)
        rows.append({"id": s["id"], "text": s["audio_text"], "calls": len(g), "apps": len(apps),
                     "groups": sorted({GROUP_OF[x] for x in apps}), "words": len(s["audio_text"].split()),
                     "installed": len(s["context"]["installed_apps"]), "names": [c["name"] for c in g]})
    words = sorted(r["words"] for r in rows)
    p90 = words[int(0.9 * n)]
    inst = [r["installed"] for r in rows]
    by_groups = Counter(len(r["groups"]) for r in rows)
    combos = Counter(" + ".join(r["groups"]) for r in rows if len(r["groups"]) == 2)

    rep = ["# Yêu cầu nhiều app, nhiều nhóm trong train", "",
           f"Sinh bởi `scripts/request_groups.py` từ {n} mẫu train. Chỉ số liệu gộp, không có id hay câu của mẫu (repo public).",
           "Không tính `sys.*` (ask_missing_info) là một app.", "",
           "## Catalog", "",
           f"Catalog train có {len(catalog)} app, {sum(len(x['tools']) for x in catalog)} tool (45 app thường + `sys` luôn bật).",
           f"Mỗi mẫu chỉ cài một phần: trung bình {st.mean(inst):.1f} app, trung vị {st.median(inst):g}, ít nhất {min(inst)}, nhiều nhất {max(inst)}.",
           "Prompt chỉ đưa tool của app đã cài.", "",
           "Nhóm app tự đặt để đọc cho dễ (không phải phân loại chính thức):", "",
           "| Nhóm | App (số tool) |", "|---|---|"]
    ntools = {x["id"]: len(x["tools"]) for x in catalog}
    rep += [f"| {g} | " + ", ".join(f"{x} ({ntools[x]})" for x in apps.split()) + " |" for g, apps in GROUPS.items()]
    rep += ["", "## Một yêu cầu cần bao nhiêu app", ""] + dist_table("Số app trong gold", Counter(r["apps"] for r in rows), n)
    rep += ["", f"Từ 2 app trở lên: {sum(r['apps'] >= 2 for r in rows)} mẫu ({100 * sum(r['apps'] >= 2 for r in rows) / n:.1f}%).",
            f"Gold rỗng: {sum(r['calls'] == 0 for r in rows)} mẫu; 0 app nhưng có call: {sum(r['apps'] == 0 and r['calls'] > 0 for r in rows)} mẫu (chỉ `sys.ask_missing_info`)."]
    rep += ["", "## Một yêu cầu trải qua bao nhiêu nhóm", ""] + dist_table("Số nhóm trong gold", by_groups, n)
    g3 = [r for r in rows if len(r["groups"]) >= 3]
    rep += ["", f"Từ 3 nhóm trở lên: {len(g3)} mẫu ({100 * len(g3) / n:.1f}%), trong đó {sum(r['words'] >= p90 for r in g3)} mẫu cũng dài (>= {p90} từ, nhóm 10% dài nhất).",
            "", "Cặp nhóm hay đi cùng nhau nhất (mẫu đúng 2 nhóm):", "", "| Cặp nhóm | Mẫu |", "|---|---:|"]
    rep += [f"| {k} | {v} |" for k, v in combos.most_common(8)]
    rep += ["", "## Độ dài và số call", "",
            f"Số từ của câu: trung vị {words[n // 2]}, p90 {p90}, dài nhất {words[-1]}.", ""]
    rep += dist_table("Số call trong gold", Counter(r["calls"] for r in rows), n)
    mean_words = defaultdict(list)
    for r in rows:
        mean_words[len(r["groups"])].append(r["words"])
    rep += ["", "| Số nhóm | Số từ trung bình | Số call trung bình |", "|---:|---:|---:|"]
    rep += [f"| {k} | {st.mean(v):.1f} | {st.mean(r['calls'] for r in rows if len(r['groups']) == k):.1f} |" for k, v in sorted(mean_words.items())]

    cache = PROJECT / "work" / "kaggle" / "cache"
    if any(cache.glob("*.jsonl")):
        from lifemate.eda import load_runs
        runs = load_runs(cache)
        rep += ["", "## Điểm các lượt chạy đã cache, theo số app trong gold", "",
                "| Lượt | " + " | ".join(["0 app", "1 app", "2 app", "3+ app"]) + " |", "|---|---:|---:|---:|---:|"]
        for (v, f), res in runs.items():
            b = defaultdict(list)
            for r in res:
                b[min(3, len(apps_of(r["gold"])))].append(r["score"])
            rep.append(f"| {v} | " + " | ".join(
                f"{100 * st.mean(b[k]):.1f} (n={len(b[k])})" if b[k] else "-" for k in range(4)) + " |")
        rep += ["", "Ô = điểm trung bình x100 trên các mẫu của lượt đó có số app như cột."]
    rep += ["", "## Đọc", "",
            "Gần một nửa train cần từ 2 app trở lên; các yêu cầu này thường nối kết quả bước trước sang bước sau (tỉ giá ra số tiền, quãng đường ra số cây số, nhiệt độ quyết định bật quạt).",
            "Có các khuôn câu điều kiện lặp lại nhiều lần với tên hoặc hoá đơn khác nhau: \"nếu ví đủ thì trả bằng ví, không thì qua ngân hàng\" và \"mai mưa thì huỷ lịch, không thì đặt xe\".",
            "\"Thời gian, ghi chú\" hay là bước chốt cuối câu (nhắc, tạo việc, ghi chú): có mặt trong "
            f"{sum('Thời gian, ghi chú' in k for k, _ in combos.most_common(8))} trên 8 cặp nhóm phổ biến nhất ở bảng trên và trong "
            f"{sum(v for k, v in combos.items() if 'Thời gian, ghi chú' in k)} trên {sum(combos.values())} mẫu đúng 2 nhóm."]
    out = PROJECT / "reports" / "request_groups.md"
    out.write_text("\n".join(rep) + "\n", encoding="utf-8")
    print("\n".join(rep))
    print(f"\n-> {out}")
    if a.show:
        print("\n(chỉ in ra màn hình) yêu cầu dài nhất trong số mẫu >= 3 nhóm:")
        for r in sorted(g3, key=lambda r: (-len(r["groups"]), -r["words"]))[: a.show]:
            print(f"  {r['id']} {len(r['groups'])} nhóm {r['calls']} call {r['words']} từ | {r['text']} | {r['names']}")


if __name__ == "__main__":
    main()
