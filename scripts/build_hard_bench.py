"""Build the hard-case tool-calling benchmark: work/bench/hard_v1.jsonl (per-sample data, git-ignored).

  python scripts/build_hard_bench.py              # build, verify every gold by replay, print counts per error group
  python scripts/build_hard_bench.py --per-gen 20 # items per synthetic generator (default 20)

Two sources, same item format:
  synthetic  a NEW request written on top of a real train sample's environment (env_id): the data it talks about
             (contact names, events, beneficiaries, devices, forecasts) is read from that ToolEnv, the gold calls follow
             the labelling conventions measured on train (e.g. contacts.search(query=<name as said>) before sms.send,
             get_balance + list_beneficiaries before banking.transfer, lookup then sys.ask_missing_info on an ambiguous
             name, [] when the app is not installed or the user takes the request back), and every gold is replayed on
             a fresh ToolEnv: each call must return ok, a write must not be a no-op (done: false). gold_hash comes from
             that replay. History is dropped (its effects are already in the state), the final turn is text only.
  train      real train samples picked by the same properties (gold = the official label), capped per group, with the
             error the last Kaggle run made on them when that sample was run.
Groups (one primary group per item, `tags` lists every group it exercises):
  1 args        right tool, arguments easy to get wrong (spoken digits, amounts, self-correction, unrequested options)
  2 under       several calls needed (multi-write chains)
  3 over        only a read is needed; any write is over-calling
  4 wrong_tool  a near-neighbour tool is installed too (reminder vs alarm, ewallet vs bank, current vs forecast)
  5 no_tool     gold is []: app not installed, self-cancel, negation, chit-chat
  6 lookup      the read the model skips: contacts.search, calendar.list_events, banking.list_beneficiaries
  7 ambiguous   ask for missing/ambiguous info, conditional plans
Reads data only through lifemate.data (tool_env, load_samples); nothing under <split>_environment/ is opened directly.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import unicodedata
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lifemate.data import PROJECT, load_samples, tool_env  # noqa: E402
from lifemate.gold import build_gold  # noqa: E402

OUT = PROJECT / "work" / "bench" / "hard_v1.jsonl"
GROUPS = {1: "args", 2: "under", 3: "over", 4: "wrong_tool", 5: "no_tool", 6: "lookup", 7: "ambiguous"}
DIG = "không một hai ba bốn năm sáu bảy tám chín".split()
NUM = ["", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]


# ---------------------------------------------------------------- Vietnamese phrasing helpers
def amount_words(n: int) -> str:
    """Multiples of 100k as people say them: 'hai triệu tám', 'một triệu rưỡi', 'ba trăm nghìn'."""
    m, r = divmod(n // 100_000, 10)
    if not m:
        return f"{NUM[r]} trăm nghìn"
    if not r:
        return f"{NUM[m]} triệu"
    return f"{NUM[m]} triệu {'rưỡi' if r == 5 else NUM[r]}"


def time_words(h: int, mi: int, rng: random.Random) -> str:
    part = "sáng" if h < 11 else "trưa" if h < 13 else "chiều" if h < 18 else "tối"
    hh = h if h <= 12 else h - 12
    if mi == 30 and rng.random() < 0.6:
        core = f"{hh} rưỡi"
    elif mi == 0:
        core = f"{hh} giờ"
    else:
        core = f"{hh} giờ {mi}"
    return f"{core} {part}"


def iso(d, h: int, mi: int) -> str:
    return f"{d.isoformat()}T{h:02d}:{mi:02d}:00+07:00"


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", s.lower())).strip()


def call(name: str, **args) -> dict:
    return {"name": name, "arguments": args}


# ---------------------------------------------------------------- env access (through ToolEnv only)
class Base:
    """One train sample used as the world of a synthetic item: its context and lazily read app data."""

    def __init__(self, s: dict):
        self.s, self.id = s, s["id"]
        self.apps = set(s["context"]["installed_apps"])
        self.now = datetime.fromisoformat(s["context"]["current_time"])
        self.tomorrow = (self.now + timedelta(days=1)).date()
        self._cache: dict = {}

    def read(self, name: str, **args):
        key = (name, json.dumps(args, sort_keys=True))
        if key not in self._cache:
            self._cache[key] = tool_env("train", self.id).call(name, args)
        return self._cache[key]

    def has(self, *apps: str) -> bool:
        return all(a in self.apps for a in apps)

    def contacts(self) -> list[dict]:
        return self.read("contacts.search").get("contacts", []) if self.has("contacts") else []

    def unique_contacts(self) -> list[dict]:
        return [c for c in self.contacts() if len(self.read("contacts.search", query=c["name"]).get("contacts", [])) == 1]

    def ambiguous_names(self) -> list[tuple[str, int]]:
        out = []
        for c in self.contacts():
            q = c["name"].split()[-1]
            n = len(self.read("contacts.search", query=q).get("contacts", []))
            if n >= 2 and len(q) >= 2:
                out.append((q, n))
        return sorted(set(out))


def replay(env_id: str, gold: list[dict]) -> tuple[bool, str, list[dict]]:
    env = tool_env("train", env_id)
    results = []
    for c in gold:
        r = env.call(c["name"], c["arguments"])
        results.append(r)
        if not r.get("ok") or r.get("done") is False:
            return False, f"{c['name']} -> {json.dumps(r, ensure_ascii=False)[:160]}", results
    return True, env.get_state_hash(), results


# ---------------------------------------------------------------- synthetic generators: (base, rng) -> item fields or None
CONTENTS = ["mình về muộn nửa tiếng", "tối nay ăn cơm nhà", "mai họp lúc chín giờ", "nhớ mang ô", "em đến nơi rồi",
            "chiều nay anh đón con", "đang kẹt xe, chờ chút", "tối nay không về ăn cơm", "mai nhớ gửi tài liệu"]


def g_contact_sms(b: Base, rng):
    if not b.has("contacts", "sms") or not (cs := b.unique_contacts()):
        return None
    c, content = rng.choice(cs), rng.choice(CONTENTS)
    text = rng.choice([f"Nhắn cho {c['name']} là {content} nhé.", f"Gửi tin nhắn cho {c['name']}: {content}.",
                       f"Ờ nhắn {c['name']} giúp anh là {content}."])
    return dict(group=6, tags=[6, 2], text=text, gold=[call("contacts.search", query=c["name"]), call("sms.send", contact_id=c["id"], content=content)],
                why="phải tra contacts.search lấy contact_id trước khi gửi")


def g_contact_call(b: Base, rng):
    if not b.has("contacts", "phone") or not (cs := b.unique_contacts()):
        return None
    c = rng.choice(cs)
    text = rng.choice([f"Gọi cho {c['name']} giúp tôi.", f"Alo, gọi {c['name']} cái nhé.", f"Em gọi điện cho {c['name']} đi."])
    return dict(group=6, tags=[6], text=text, gold=[call("contacts.search", query=c["name"]), call("phone.call", contact_id=c["id"])],
                why="gọi theo tên cần contact_id từ contacts.search")


def g_contact_ambiguous(b: Base, rng):
    if not b.has("contacts", "phone", "sms") or not (names := b.ambiguous_names()):
        return None
    q, n = rng.choice(names)
    if rng.random() < 0.5:
        return dict(group=7, tags=[7, 6], text=rng.choice([f"Gọi cho {q} nhé.", f"Gọi {q} giúp mình với."]),
                    gold=[call("contacts.search", query=q), call("sys.ask_missing_info", tool_name="phone.call", params=["contact_id"])],
                    why=f"'{q}' khớp {n} liên hệ: tra rồi hỏi lại, không tự chọn")
    content = rng.choice(CONTENTS)
    return dict(group=7, tags=[7, 6], text=f"Nhắn {q} là {content}.",
                gold=[call("contacts.search", query=q), call("sys.ask_missing_info", tool_name="sms.send", params=["contact_id"])],
                why=f"'{q}' khớp {n} liên hệ: tra rồi hỏi lại")


def _events(b: Base) -> list[dict]:
    if not b.has("calendar"):
        return []
    evs = b.read("calendar.list_events").get("events", [])
    titles = Counter(e["title"] for e in evs)
    return [e for e in evs if titles[e["title"]] == 1 and datetime.fromisoformat(e["start"]) > b.now]


def g_calendar_move(b: Base, rng):
    if not (evs := _events(b)):
        return None
    e = rng.choice(evs)
    st, en = datetime.fromisoformat(e["start"]), datetime.fromisoformat(e["end"])
    k = rng.choice([1, 2])
    ns, ne = st + timedelta(hours=k), en + timedelta(hours=k)
    if ns.date() != st.date():
        return None
    day = f"ngày {st.day}/{st.month}"
    text = rng.choice([f"Dời lịch {e['title']} {day} muộn {k} tiếng nhé.",
                       f"Lùi {e['title']} {day} sang {time_words(ns.hour, ns.minute, rng)} giúp anh."])
    return dict(group=6, tags=[6], text=text,
                gold=[call("calendar.list_events"), call("calendar.update_event", event_id=e["id"], start=ns.isoformat(), end=ne.isoformat())],
                why="cần calendar.list_events lấy event_id; giữ nguyên thời lượng")


def g_calendar_cancel(b: Base, rng):
    if not (evs := _events(b)):
        return None
    e = rng.choice(evs)
    st = datetime.fromisoformat(e["start"])
    text = rng.choice([f"Huỷ lịch {e['title']} ngày {st.day}/{st.month} giúp mình.", f"Bỏ cái lịch {e['title']} hôm {st.day}/{st.month} đi."])
    return dict(group=6, tags=[6], text=text, gold=[call("calendar.list_events"), call("calendar.cancel_event", event_id=e["id"])],
                why="cần calendar.list_events lấy event_id")


def _payment_account(b: Base):
    if not b.has("banking"):
        return None
    accs = b.read("banking.get_balance").get("accounts", [])
    return next((a for a in accs if "thanh toán" in a["name"].lower()), None)


def g_bank_transfer(b: Base, rng):
    acc = _payment_account(b)
    if not acc:
        return None
    bens = b.read("banking.list_beneficiaries").get("beneficiaries", [])
    names = Counter(x["name"] for x in bens)
    bens = [x for x in bens if names[x["name"]] == 1]
    top = min(5_000_000, acc["balance"]) // 100_000
    if not bens or top < 2:
        return None
    ben, amt = rng.choice(bens), rng.randint(2, top) * 100_000
    text = rng.choice([f"Chuyển cho {ben['name']} {amount_words(amt)} nhé.", f"Chuyển khoản {amount_words(amt)} cho {ben['name']} giúp anh."])
    return dict(group=6, tags=[6, 1], text=text,
                gold=[call("banking.get_balance"), call("banking.list_beneficiaries"),
                      call("banking.transfer", from_account_id=acc["id"], beneficiary_id=ben["id"], amount=amt)],
                why="cần list_beneficiaries lấy beneficiary_id, get_balance lấy tài khoản; số tiền đọc bằng chữ")


def g_args_alarm(b: Base, rng):
    if not b.has("alarm"):
        return None
    h, mi = rng.choice([(5, 30), (6, 0), (6, 15), (6, 45), (7, 0), (5, 45)])
    if rng.random() < 0.5:
        text, why = f"Đặt báo thức {time_words(h, mi, rng)} mai nhé.", "chỉ truyền time; không tự thêm label/repeat"
    else:
        h0 = h - 1
        text = f"Đặt báo thức {time_words(h0, mi, rng)} mai, à không, {time_words(h, mi, rng)} nhé."
        why = "người dùng tự sửa giờ: lấy giờ sau"
    return dict(group=1, tags=[1], text=text, gold=[call("alarm.create", time=iso(b.tomorrow, h, mi))], why=why)


def _phone(rng) -> str:
    return "09" + "".join(str(rng.randint(0, 9)) for _ in range(9))


def g_args_phone_digits(b: Base, rng):
    if not b.has("phone"):
        return None
    p = _phone(rng)
    spoken = " ".join(DIG[int(d)] for d in p)
    return dict(group=1, tags=[1], text=rng.choice([f"Gọi số {spoken} giúp tôi.", f"Bấm gọi {spoken} nha."]),
                gold=[call("phone.call", phone_number=p)], why="số đọc từng chữ, giữ số 0 đầu, không tra danh bạ")


def g_args_topup(b: Base, rng):
    if not b.has("ewallet"):
        return None
    bal = b.read("ewallet.get_balance").get("balance", 0)
    choices = [a for a in (50_000, 100_000, 200_000) if a <= bal]
    if not choices:
        return None
    p, amt = _phone(rng), rng.choice(choices)
    spoken = " ".join(DIG[int(d)] for d in p)
    say = {50_000: "năm mươi nghìn", 100_000: "một trăm nghìn", 200_000: "hai trăm nghìn"}[amt]
    return dict(group=1, tags=[1], text=f"Nạp {say} tiền điện thoại cho số {spoken} nhé.",
                gold=[call("ewallet.topup_mobile", phone=p, amount=amt)], why="số tiền bằng chữ + số điện thoại đọc từng chữ")


CITIES = ["Hà Nội", "Đà Nẵng", "TP.HCM", "Huế", "Cần Thơ", "Nha Trang", "Đà Lạt", "Hải Phòng"]


def g_args_weather_location(b: Base, rng):
    if not b.has("weather"):
        return None
    home = b.s["context"]["user_profile"].get("city", "")
    city = rng.choice([c for c in CITIES if c != home])
    g = call("weather.get_forecast", date=b.tomorrow.isoformat(), location=city)
    if not b.read("weather.get_forecast", **g["arguments"]).get("forecast"):
        return None
    return dict(group=1, tags=[1, 3], text=f"Mai ở {city} có mưa không em?", gold=[g], why="thành phố khác nơi đang ở: phải truyền location")


def g_wrong_reminder(b: Base, rng):
    if not b.has("reminders", "alarm"):
        return None
    content = rng.choice(["uống thuốc", "gọi điện cho mẹ", "mua sữa cho con", "nộp báo cáo", "đóng tiền nhà"])
    h, mi = rng.choice([(20, 0), (21, 30), (19, 0), (18, 30)])
    when = rng.choice(["tối nay", "tối mai"])
    d = b.now.date() if when == "tối nay" else b.tomorrow
    if when == "tối nay" and datetime(d.year, d.month, d.day, h, mi, tzinfo=b.now.tzinfo) <= b.now:
        return None
    return dict(group=4, tags=[4, 1], text=f"Nhắc tôi {content} lúc {time_words(h, mi, rng).replace(' tối', '')} {when} nhé.",
                gold=[call("reminders.create", content=content, time=iso(d, h, mi))], why="'nhắc' là reminders.create, không phải alarm.create")


def g_wrong_weather_current(b: Base, rng):
    if not b.has("weather"):
        return None
    return dict(group=4, tags=[4, 3], text=rng.choice(["Bây giờ ngoài trời thế nào em?", "Giờ đang nóng hay mát vậy?", "Ngoài kia đang mưa à?",
                                                       "Nhiệt độ bây giờ bao nhiêu độ?", "Lúc này ngoài đường có nắng không?", "Hiện tại trời đang ra sao?"]),
                gold=[call("weather.get_current")], why="hỏi thời tiết lúc này: get_current, không phải get_forecast")


def g_wrong_ewallet(b: Base, rng):
    if not b.has("ewallet", "contacts", "banking") or not (cs := b.unique_contacts()):
        return None
    bal = b.read("ewallet.get_balance").get("balance", 0)
    amt = min(bal // 100_000, 5) * 100_000
    if amt < 100_000:
        return None
    c = rng.choice(cs)
    return dict(group=4, tags=[4, 6], text=rng.choice([f"Chuyển qua ví cho {c['name']} {amount_words(amt)} nhé.", f"Gửi {c['name']} {amount_words(amt)} bằng ví điện tử."]),
                gold=[call("contacts.search", query=c["name"]), call("ewallet.transfer", contact_id=c["id"], amount=amt)],
                why="nói rõ 'ví': ewallet.transfer, không phải banking.transfer")


def g_readonly(b: Base, rng):
    opts = []
    if b.has("weather"):
        for t in ("Mai có mưa không vậy?", "Ngày mai thời tiết thế nào em?", "Xem giúp mai trời có mưa không, chỉ xem thôi nhé.",
                  "Mai đi chơi được không, có mưa không?"):
            opts.append((t, [call("weather.get_forecast", date=b.tomorrow.isoformat())], "chỉ hỏi dự báo, không làm gì thêm"))
    if b.has("banking"):
        for t in ("Xem giúp số dư ngân hàng còn bao nhiêu.", "Tài khoản ngân hàng của anh còn bao nhiêu tiền?", "Kiểm tra số dư tài khoản giúp chị."):
            opts.append((t, [call("banking.get_balance")], "chỉ đọc số dư, không chuyển tiền"))
    if b.has("smart_home"):
        for t in ("Kiểm tra xem đèn trong nhà đang bật cái nào.", "Trong nhà đang mở những thiết bị gì vậy?", "Xem giúp điều hoà với đèn đang thế nào."):
            opts.append((t, [call("smart_home.get_status")], "chỉ xem trạng thái, không bật/tắt"))
    if not opts:
        return None
    text, gold, why = rng.choice(opts)
    return dict(group=3, tags=[3], text=text, gold=gold, why=why)


NOT_INSTALLED = [("food_delivery", "Đặt giúp anh hai suất cơm tấm về nhà."), ("ride_hailing", "Gọi xe ra sân bay giúp em."),
                 ("flights", "Tìm vé máy bay đi Hà Nội thứ bảy này."), ("music", "Mở bài hát gì đó nhẹ nhàng đi."),
                 ("tv", "Chuyển sang kênh VTV3 giúp anh."), ("bike_rental", "Thuê giúp mình một chiếc xe đạp gần đây."),
                 ("lottery", "Kết quả xổ số hôm qua thế nào?"), ("fuel_price", "Giá xăng hôm nay bao nhiêu?"),
                 ("food_delivery", "Order cho chị ly trà sữa trân châu nhé."), ("ride_hailing", "Đặt cho anh chiếc xe bốn chỗ về nhà."),
                 ("flights", "Có chuyến bay nào vào Sài Gòn tối mai không?"), ("music", "Phát nhạc Sơn Tùng cho anh nghe."),
                 ("tv", "Bật ti vi lên xem thời sự."), ("lottery", "Dò giúp anh vé số đài Hà Nội kỳ này."),
                 ("fuel_price", "Xăng RON 95 tuần này tăng hay giảm?"), ("podcast_radio", "Mở radio VOV giao thông đi em."),
                 ("library", "Gia hạn cuốn sách anh mượn ở thư viện."), ("package_tracking", "Đơn hàng của chị tới đâu rồi?"),
                 (None, "Đặt phòng khách sạn ở Đà Lạt cuối tuần này giúp anh."), (None, "Mua giúp hai vé xem phim tối nay."),
                 (None, "Đặt bàn nhà hàng sáu người tối thứ bảy.")]


def g_notool_not_installed(b: Base, rng):
    opts = [t for app, t in NOT_INSTALLED if app is None or app not in b.apps]  # None: no such app exists at all
    if not opts:
        return None
    return dict(group=5, tags=[5], text=rng.choice(opts), gold=[], why="app không được cài: không gọi tool nào")


def _lights(b: Base) -> list[dict]:
    if not b.has("smart_home"):
        return []
    return [d for d in b.read("smart_home.get_status").get("devices", []) if d.get("type") == "light"]


def g_notool_cancel(b: Base, rng):
    opts = []
    if ls := _lights(b):
        d = rng.choice(ls)["name"].lower()
        opts += [f"Tắt {d} đi, à thôi khỏi, không cần đâu.", f"Bật {d} lên, mà thôi, đang sáng rồi."]
    if b.has("contacts", "sms") and (cs := b.unique_contacts()):
        opts.append(f"Nhắn cho {rng.choice(cs)['name']} là {rng.choice(CONTENTS)}, mà thôi để anh tự nhắn.")
    if b.has("alarm"):
        opts.append(f"Đặt báo thức {rng.choice([5, 6, 7])} giờ sáng mai, à mà thôi, mai được nghỉ.")
    if b.has("banking") and (bens := b.read("banking.list_beneficiaries").get("beneficiaries", [])):
        opts.append(f"Chuyển cho {rng.choice(bens)['name']} một triệu, à khoan, để anh hỏi lại đã.")
    if not opts:
        return None
    return dict(group=5, tags=[5], text=rng.choice(opts), gold=[], why="người dùng tự huỷ trong cùng lượt nói")


def g_notool_negation(b: Base, rng):
    if not (ls := _lights(b)):
        return None
    d = rng.choice(ls)["name"].lower()
    text = rng.choice([f"Đừng tắt {d} nhé, cứ để đó.", f"Khoan, {d} cứ để vậy, đừng đụng vào.", f"Không cần bật {d} đâu em.",
                       f"Đừng động vào {d}, để anh tự chỉnh."])
    return dict(group=5, tags=[5], text=text, gold=[], why="câu phủ định: không làm gì")


def g_notool_chitchat(b: Base, rng):
    return dict(group=5, tags=[5], text=rng.choice(["Cảm ơn em nhé.", "Em làm tốt lắm.", "Hôm nay mệt quá à.", "Em tên là gì thế?", "Chúc em ngủ ngon.",
                                                    "Ok, vậy được rồi.", "Thôi không có gì đâu.", "Em có biết hát không?", "Kể chuyện cười nghe đi.",
                                                    "Ừ để anh suy nghĩ đã.", "Hay quá, cảm ơn nha.", "Em ơi, anh buồn quá.", "Chào buổi sáng em.",
                                                    "Thôi được rồi, không cần nữa."]),
                gold=[], why="trò chuyện, không cần hành động")


def g_conditional(b: Base, rng):
    if not b.has("weather", "calendar"):
        return None
    evs = [e for e in _events(b) if datetime.fromisoformat(e["start"]).date() == b.tomorrow]
    if not evs:
        return None
    e = rng.choice(evs)
    fc = b.read("weather.get_forecast", date=b.tomorrow.isoformat()).get("forecast", [])
    if not fc:
        return None
    rain = bool(fc[0].get("rain"))
    if not rain and rng.random() < 0.5:  # most bases are dry: keep both outcomes represented
        return None
    gold = [call("weather.get_forecast", date=b.tomorrow.isoformat())]
    if rain:
        gold += [call("calendar.list_events"), call("calendar.cancel_event", event_id=e["id"])]
    return dict(group=7, tags=[7, 6, 2] if rain else [7, 3], text=f"Nếu mai mưa thì huỷ {e['title']} nhé, không mưa thì thôi.", gold=gold,
                why="mai " + ("có mưa: xem dự báo rồi huỷ" if rain else "không mưa: chỉ xem dự báo, không huỷ"))


def g_multi_write(b: Base, rng):
    if not b.has("alarm", "contacts", "sms") or not (cs := b.unique_contacts()):
        return None
    c, content = rng.choice(cs), rng.choice(CONTENTS)
    h, mi = rng.choice([(6, 0), (6, 30), (5, 45)])
    return dict(group=2, tags=[2, 6, 7], text=f"Đặt báo thức {time_words(h, mi, rng)} mai rồi nhắn {c['name']} là {content} nhé.",
                gold=[call("alarm.create", time=iso(b.tomorrow, h, mi)), call("contacts.search", query=c["name"]),
                      call("sms.send", contact_id=c["id"], content=content)], why="hai việc: báo thức + tra danh bạ + nhắn tin")


def g_ask_missing(b: Base, rng):
    opts = []
    if b.has("reminders"):
        opts.append((f"Nhắc tôi {rng.choice(['uống thuốc', 'nộp báo cáo', 'gọi cho mẹ'])} nhé.",
                     [call("sys.ask_missing_info", tool_name="reminders.create", params=["time"])], "thiếu giờ nhắc: hỏi lại"))
    if b.has("ewallet", "contacts") and (cs := b.unique_contacts()):
        n = rng.choice(cs)["name"]
        opts.append((f"Chuyển tiền qua ví cho {n} giúp anh.", [call("contacts.search", query=n),
                     call("sys.ask_missing_info", tool_name="ewallet.transfer", params=["amount"])], "thiếu số tiền: tra người nhận rồi hỏi"))
    if not opts:
        return None
    text, gold, why = rng.choice(opts)
    return dict(group=7, tags=[7], text=text, gold=gold, why=why)


GENERATORS = [g_contact_sms, g_contact_call, g_contact_ambiguous, g_calendar_move, g_calendar_cancel, g_bank_transfer,
              g_args_alarm, g_args_phone_digits, g_args_topup, g_args_weather_location, g_wrong_reminder, g_wrong_weather_current,
              g_wrong_ewallet, g_readonly, g_notool_not_installed, g_notool_cancel, g_notool_negation, g_notool_chitchat,
              g_conditional, g_multi_write, g_ask_missing]


# ---------------------------------------------------------------- real train items
READ = re.compile(r"\.(get|list|search|read|recent|guide|check|convert)|^sys\.")


def real_groups(s: dict) -> list[int]:
    calls = s["label"]["tool_calls"]
    names = [c["name"] for c in calls]
    g = []
    if not calls:
        g.append(5)
    if any(n in names for n in ("contacts.search", "calendar.list_events", "banking.list_beneficiaries")):
        g.append(6)
    if "sys.ask_missing_info" in names or "nếu" in s["audio_text"].lower():
        g.append(7)
    if len(calls) >= 4:
        g.append(2)
    if calls and all(READ.search(n) for n in names):
        g.append(3)
    if any(k in c["arguments"] for c in calls for k in ("phone_number", "phone", "amount")):
        g.append(1)
    return g


def last_run_errors() -> dict[str, str]:
    """id -> short error label from the cached B0 run (empty when the sample was not run)."""
    out = {}
    for f in (PROJECT / "work" / "kaggle" / "cache").glob("*.gold.*.jsonl"):
        for l in f.read_text(encoding="utf-8").splitlines():
            r = json.loads(l)
            out[r["id"]] = r
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-gen", type=int, default=18, help="items per positive generator")
    ap.add_argument("--per-neg-gen", type=int, default=35, help="items per no-tool generator (balances positives and negatives)")
    ap.add_argument("--real-per-group", type=int, default=25)
    ap.add_argument("--real-no-tool", type=int, default=40, help="real gold-[] train samples")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    train = load_samples("train")
    train_texts = {norm(s["audio_text"]) for s in train}
    train_tok = [set(t.split()) for t in train_texts]
    bases = [Base(s) for s in train]
    items, seen, rejected = [], set(), Counter()
    for gen in GENERATORS:
        quota = a.per_neg_gen if gen.__name__.startswith("g_notool") else a.per_gen
        rng = random.Random(f"{a.seed}-{gen.__name__}")
        order = bases[:]
        rng.shuffle(order)
        k = 0
        for b in order:
            if k >= quota:
                break
            it = gen(b, rng)
            if it is None:
                continue
            t = norm(it["text"])
            tt = set(t.split())
            if t in seen or t in train_texts or any(len(tt & x) / len(tt | x) >= 0.8 for x in train_tok):
                rejected[f"{gen.__name__}: duplicate / too close to train"] += 1
                continue
            ok, h, results = replay(b.id, it["gold"])
            if not ok:
                rejected[f"{gen.__name__}: gold replay failed"] += 1
                continue
            seen.add(t)  # texts are unique across the whole benchmark
            k += 1
            items.append({"id": f"hard_{gen.__name__[2:]}_{k:03d}", "env_id": b.id, "source": "synthetic", "subtype": gen.__name__[2:],
                          "group": GROUPS[it["group"]], "tags": [GROUPS[x] for x in it["tags"]], "why": it["why"],
                          "audio_text": it["text"], "context": b.s["context"], "history": [],
                          "label": {"tool_calls": it["gold"], "tool_results": results}, "gold_hash": h})

    gold = build_gold("train")
    runs = last_run_errors()
    rng = random.Random(a.seed)
    per = Counter()
    pool = sorted(train, key=lambda s: (s["id"] not in runs, rng.random()))  # samples the last run saw come first
    for s in pool:
        gs = [g for g in real_groups(s) if per[g] < (a.real_no_tool if g == 5 else a.real_per_group)]
        if not gs:
            continue
        per[gs[0]] += 1
        r = runs.get(s["id"])
        items.append({"id": s["id"], "env_id": s["id"], "source": "train", "subtype": "train", "group": GROUPS[gs[0]],
                      "tags": [GROUPS[x] for x in real_groups(s)], "why": "mẫu train thật",
                      "audio_text": s["audio_text"], "context": s["context"], "history": s["history"],
                      "label": {"tool_calls": s["label"]["tool_calls"]}, "gold_hash": gold[s["id"]]["state_hash"],
                      "last_run": None if r is None else {"tool_calls": r["tool_calls"], "state_hash": r["state_hash"]}})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    print(f"-> {OUT}: {len(items)} items")
    tab = Counter((it["group"], it["source"]) for it in items)
    print(f"{'group':12s} {'synthetic':>9s} {'train':>6s} {'total':>6s} {'gold=[]':>8s}")
    for g in GROUPS.values():
        n_s, n_t = tab[(g, "synthetic")], tab[(g, "train")]
        empty = sum(1 for it in items if it["group"] == g and not it["label"]["tool_calls"])
        print(f"{g:12s} {n_s:9d} {n_t:6d} {n_s + n_t:6d} {empty:8d}")
    pos = sum(1 for it in items if it["label"]["tool_calls"])
    writes = sum(1 for it in items if any(not READ.search(c["name"]) for c in it["label"]["tool_calls"]))
    print(f"positive (needs a call) {pos} | negative (gold []) {len(items) - pos} | needs a write {writes} | no write needed {len(items) - writes}")
    print(f"unique texts {len({norm(it['audio_text']) for it in items})} of {len(items)}")
    print("by subtype: " + ", ".join(f"{k} {v}" for k, v in Counter(it["subtype"] for it in items if it["source"] == "synthetic").items()))
    if rejected:
        print("rejected: " + "; ".join(f"{k} {v}" for k, v in rejected.items()))


if __name__ == "__main__":
    main()
