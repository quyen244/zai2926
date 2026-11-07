"""Print the shortest and/or longest train requests of an app group (screen only: per-sample data never goes to the repo).

  python scripts/show_requests.py --group "di chuyen" --top 5            # 5 shortest + 5 longest, gold apps all in Di chuyển
  python scripts/show_requests.py --group "Mua sắm" --top 10 --order long
  python scripts/show_requests.py --group "di chuyen" --any               # requests that touch the group (other groups allowed)
  python scripts/show_requests.py --app maps --top 5                      # one app instead of a group
  python scripts/show_requests.py --group "suc khoe" --history no         # only requests without earlier turns
  python scripts/show_requests.py --group "suc khoe" --history yes --show-history   # with history, print the turns too
  python scripts/show_requests.py --list                                  # groups and their apps

Groups are the ones in scripts/request_groups.py. Without --any a request belongs to a group when every app its gold
calls (sys.* not counted) is in that group, e.g. Di chuyển = only maps / ride_hailing / flights / bike_rental.
Length = number of words of the reference transcript.
"""
from __future__ import annotations

import argparse
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lifemate.data import load_samples  # noqa: E402
from request_groups import GROUPS, apps_of  # noqa: E402


def norm(s: str) -> str:
    s = unicodedata.normalize("NFD", s.lower().replace("đ", "d"))
    return "".join(c for c in s if c.isalnum())


def find_group(q: str) -> str:
    hit = [g for g in GROUPS if norm(g) == norm(q)] or [g for g in GROUPS if norm(g).startswith(norm(q))]
    if len(hit) != 1:
        raise SystemExit(f"nhóm '{q}' {'không có' if not hit else 'mơ hồ: ' + ', '.join(hit)}; có: {', '.join(GROUPS)}")
    return hit[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", help="tên nhóm, không cần dấu (di chuyen, mua sam, ...)")
    ap.add_argument("--app", help="một app (maps, ...) thay cho nhóm")
    ap.add_argument("--top", type=int, default=5, help="số mẫu in ra mỗi phía")
    ap.add_argument("--order", choices=["short", "long", "both"], default="both")
    ap.add_argument("--any", action="store_true", help="chỉ cần chạm nhóm/app, được có app nhóm khác")
    ap.add_argument("--history", choices=["all", "yes", "no"], default="all", help="yes: có lượt hội thoại trước; no: câu đứng một mình")
    ap.add_argument("--show-history", action="store_true", help="in luôn các lượt trước (user, assistant, tool call)")
    ap.add_argument("--split", default="train")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    if a.list or not (a.group or a.app):
        for g, apps in GROUPS.items():
            print(f"{g:22s} {apps}")
        return
    if a.app:
        name, allowed = f"app {a.app}", {a.app}
    else:
        name = find_group(a.group)
        allowed = set(GROUPS[name].split())

    rows = []
    for s in load_samples(a.split):
        apps = apps_of(s["label"]["tool_calls"])
        if not apps or not (apps & allowed if a.any else apps <= allowed):
            continue
        if a.history != "all" and bool(s["history"]) != (a.history == "yes"):
            continue
        rows.append((len(s["audio_text"].split()), s))
    rows.sort(key=lambda r: (r[0], r[1]["id"]))
    hist = {"all": "", "yes": ", có lịch sử", "no": ", không lịch sử"}[a.history]
    print(f"{name} ({'chạm nhóm' if a.any else 'chỉ app trong nhóm'}{hist}): {len(rows)} mẫu {a.split}"
          f" ({sum(bool(s['history']) for _, s in rows)} có lịch sử)")
    if not rows:
        return
    print(f"số từ: ngắn nhất {rows[0][0]}, trung vị {rows[len(rows) // 2][0]}, dài nhất {rows[-1][0]}")

    def show(title: str, part) -> None:
        print(f"\n== {title}")
        for w, s in part:
            calls = s["label"]["tool_calls"]
            h = s["history"]
            print(f"- {s['id']}  {w} từ, {len(calls)} call, app {', '.join(sorted(apps_of(calls)))}, "
                  + (f"lịch sử {len(h)} lượt" if h else "không lịch sử"))
            if a.show_history:
                for t in h:
                    if t["role"] == "user":
                        print(f"    [user] {t.get('audio_text', '')}")
                    else:
                        for c in t.get("tool_calls") or []:
                            print(f"    [assistant gọi] {c['name']} {c['arguments']}")
                        print(f"    [assistant] {t.get('text', '')}")
            print(f"    {'[câu cuối] ' if a.show_history and h else ''}{s['audio_text']}")
            for c in calls:
                print(f"    -> {c['name']} {c['arguments']}")

    if a.order in ("short", "both"):
        show(f"{a.top} mẫu ngắn nhất", rows[: a.top])
    if a.order in ("long", "both"):
        show(f"{a.top} mẫu dài nhất", rows[::-1][: a.top])


if __name__ == "__main__":
    main()
