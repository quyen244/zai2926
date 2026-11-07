"""Show a tool's schema from the catalog, to read its description and parameters.

  python scripts/show_tool.py contacts.search             # one tool, readable view + the JSON the model receives
  python scripts/show_tool.py phone.call contacts.search  # several tools
  python scripts/show_tool.py contacts                    # every tool of an app
  python scripts/show_tool.py --list                      # every tool name, grouped by app
  python scripts/show_tool.py phone.call --raw            # the full catalog entry (fields the prompt does not send too)

The JSON block is exactly what lifemate.prompt.build_tools puts in "tools" (name, description, parameters).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lifemate.data import load_catalog  # noqa: E402


def readable(t: dict) -> list[str]:
    p = t.get("parameters") or {}
    req = set(p.get("required") or [])
    out = [f"{t['name']}", f"  mô tả: {t['description']}"]
    props = p.get("properties") or {}
    if not props:
        out.append("  tham số: (không có)")
    for k, v in props.items():
        bits = [v.get("type", "?"), "bắt buộc" if k in req else "tùy chọn"]
        if "enum" in v:
            bits.append("enum " + "|".join(map(str, v["enum"])))
        if "default" in v:
            bits.append(f"mặc định {v['default']}")
        out.append(f"  - {k} ({', '.join(bits)}): {v.get('description', '')}")
        extra = {x: y for x, y in v.items() if x not in ("type", "description", "enum", "default")}
        if extra:
            out.append(f"      khác: {json.dumps(extra, ensure_ascii=False)}")
    rest = {x: y for x, y in p.items() if x not in ("type", "properties", "required")}
    if rest:
        out.append(f"  ràng buộc: {json.dumps(rest, ensure_ascii=False)}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", help="tool name (contacts.search) or app id (contacts)")
    ap.add_argument("--split", default="train")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--raw", action="store_true", help="print the whole catalog entry instead of the prompt fields")
    a = ap.parse_args()

    apps = load_catalog(a.split)["apps"]
    if a.list or not a.names:
        for app in apps:
            print(f"{app['id']:14s} {len(app['tools']):2d}  {', '.join(t['name'] for t in app['tools'])}")
        return
    tools = {t["name"]: t for app in apps for t in app["tools"]}
    for n in a.names:
        hit = [tools[n]] if n in tools else [t for app in apps if app["id"] == n for t in app["tools"]]
        if not hit:
            close = [x for x in tools if n.lower() in x.lower()]
            print(f"{n}: không có tool/app này" + (f"; gần giống: {', '.join(close)}" if close else ""))
            continue
        for t in hit:
            print("=" * 80)
            print("\n".join(readable(t)))
            body = t if a.raw else {"name": t["name"], "description": t["description"], "parameters": t["parameters"]}
            print("-- JSON" + (" (catalog đầy đủ)" if a.raw else " (đúng như đưa vào prompt)"))
            print(json.dumps(body, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
