"""Stage 2.2: Vietnamese spoken-form normaliser for ASR / reference transcripts.

  normalize_text(text)        spoken numbers -> digits, money -> plain integer VND, clock times -> HH:MM
  resolve_dates(text, now)    relative / explicit dates found in the text -> [{"phrase", "date": "YYYY-MM-DD"}]

Deliberately conservative: a lone number word ("một cái", "không đủ", "sáng mốt") stays as it is unless a unit or
magnitude word makes the reading unambiguous. Capitalised number words ("anh Mười") are names and are never converted.
Clock times without a part of day keep the hour as spoken ("ba giờ" -> 03:00); the caller decides what that means.
"""
from __future__ import annotations

import re
from datetime import date, datetime, timedelta

DIGITS = {"không": 0, "một": 1, "mốt": 1, "hai": 2, "ba": 3, "bốn": 4, "tư": 4, "năm": 5, "lăm": 5, "sáu": 6,
          "bảy": 7, "bẩy": 7, "tám": 8, "chín": 9}
MAGNITUDES = {"trăm": 100, "nghìn": 1000, "ngàn": 1000, "triệu": 10**6, "tỷ": 10**9, "tỉ": 10**9}
TENS = {"mươi", "chục"}
SKIP = {"linh", "lẻ"}  # "một trăm linh năm"
NUMBER_WORDS = set(DIGITS) | set(MAGNITUDES) | TENS | SKIP | {"mười", "rưỡi"}
# a run of number words is converted when it has a magnitude, or is followed / preceded by one of these
NEXT_UNITS = {"giờ", "phút", "giây", "rưỡi", "đồng", "độ", "phần", "ký", "kg", "lít", "km", "mét", "bước", "ngày", "tuần", "tháng",
              "tiếng", "tuổi", "điểm", "sao", "%"}
PREV_UNITS = {"tháng", "mùng", "mồng", "ngày", "hôm", "số", "lúc"}
PERIODS = ("sáng", "trưa", "chiều", "tối", "đêm", "khuya")
TOKEN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def _run_value(words: list[str]) -> int | None:
    """Value of a run of Vietnamese number words, or None if it does not parse."""
    total = sec = 0
    pending: int | None = None
    last_mag = big = 0
    for w in words:
        if w in DIGITS:
            pending = DIGITS[w] if pending is None else pending * 10 + DIGITS[w]  # "bốn lăm" = 45
        elif w == "mười":
            sec += 10
        elif w in TENS:
            sec += (pending if pending is not None else 1) * 10
            pending = None
            last_mag = 10
        elif w == "trăm":
            sec += (pending if pending is not None else 1) * 100
            pending = None
            last_mag = 100
        elif w in MAGNITUDES:
            part = sec + (pending or 0)
            total += (part if part else 1) * MAGNITUDES[w]
            sec, pending, last_mag = 0, None, MAGNITUDES[w]
            big = MAGNITUDES[w] if MAGNITUDES[w] >= 10**6 else 0
        elif w == "rưỡi":
            if not last_mag:
                return None
            total += last_mag // 2 if sec == 0 and pending is None else 0
            if sec or pending is not None:  # "hai trăm rưỡi nghìn" is not a thing; treat as not parseable
                return None
        elif w in SKIP:
            continue
    rest = sec + (pending or 0)
    if big and rest:  # colloquial tail after millions: "hai triệu tám" = 2.8M, "hai triệu hai trăm" = 2.2M
        rest = rest * (big // 10) if rest < 10 else rest * (big // 1000) if rest < 1000 and rest % 100 == 0 else rest
    return total + rest


def _words_to_digits(text: str) -> str:
    toks = [(m.group(), m.start(), m.end()) for m in TOKEN.finditer(text)]
    out, i, last = [], 0, 0
    while i < len(toks):
        j = i
        while j < len(toks) and toks[j][0].lower() in NUMBER_WORDS and (
                toks[j][0].islower() or (j == 0) or toks[j - 1][0] in ".!?…"):  # capitalised mid-sentence = a name
            j += 1
        run = [t[0].lower() for t in toks[i:j]]
        while run and run[0] in ("rưỡi", "mốt"):  # neither can start a number
            run, i = run[1:], i + 1
        if run and run[-1] in DIGITS and len(run) > 2 and run[-2] in ("nghìn", "ngàn", "triệu", "tỷ", "tỉ") and \
                j < len(toks) and toks[j][0].lower() in NEXT_UNITS:
            run, j = run[:-1], j - 1  # "sáu mươi lăm ngàn | một ký": a new quantity starts
        if run and run[-1] == "rưỡi" and not any(r in MAGNITUDES or r in TENS for r in run[:-1]):
            run, j = run[:-1], j - 1  # half-hour "rưỡi" is left for the time pass
        if run:
            nxt = toks[j][0].lower() if j < len(toks) else ""
            prv = toks[i - 1][0].lower() if i else ""
            has_mag = any(r in MAGNITUDES or r in TENS or r == "mười" for r in run)
            ok = has_mag or (nxt in NEXT_UNITS and run != ["mốt"]) or (prv in PREV_UNITS and run not in (["mốt"], ["không"]))
            has_digit = any(r in DIGITS or r == "mười" for r in run)
            v = _run_value(run) if ok and has_digit and run != ["không"] else None
            if v is not None:
                out += [text[last:toks[i][1]], str(v)]
                last = toks[j - 1][2]
                i = j
                continue
        i = max(j, i + 1)
    out.append(text[last:])
    return "".join(out)


_MONEY = re.compile(r"(?<![\w.])(\d+(?:[.,]\d+)?)\s*(nghìn|ngàn|triệu|tỷ|tỉ|k)\b(\s*rưỡi\b)?", re.I)
_MULT = {"nghìn": 10**3, "ngàn": 10**3, "k": 10**3, "triệu": 10**6, "tỷ": 10**9, "tỉ": 10**9}


def _money(m: re.Match) -> str:
    n = float(m.group(1).replace(",", "."))
    n += 0.5 if m.group(3) else 0
    return str(int(round(n * _MULT[m.group(2).lower()])))


_TIME = re.compile(
    r"(?<![\d:])(\d{1,2})\s*(?:giờ|h|:)(?:\s*(?:(rưỡi)|kém\s+(\d{1,2})|(\d{1,2})(?:\s*phút)?))?"
    r"(?:\s+(sáng|trưa|chiều|tối|đêm|khuya))?(?!\d)", re.I)
_HALF = re.compile(r"(?<![\d:])(\d{1,2})\s+rưỡi(?:\s+(sáng|trưa|chiều|tối|đêm|khuya))?", re.I)


def _hour24(h: int, period: str | None) -> int:
    p = (period or "").lower()
    if p in ("chiều", "tối") and h < 12:
        h += 12
    elif p == "trưa" and h == 1:
        h = 13
    elif p in ("đêm", "khuya"):
        h = 0 if h == 12 else h + 12 if h >= 6 and h < 12 else h
    return h


def _time(m: re.Match) -> str:
    h, half, kem, mins, period = int(m.group(1)), m.group(2), m.group(3), m.group(4), m.group(5)
    minute = 30 if half else 0
    if kem:
        h, minute = h - 1, 60 - int(kem)
    elif mins:
        minute = int(mins)
    if h > 24 or minute > 59 or h < 0:
        return m.group(0)
    hhmm = f"{_hour24(h, period) % 24:02d}:{minute:02d}"
    if period and re.match(r"\s+(nay|mai|qua|mốt)\b", m.string[m.end():]):
        return f"{hhmm} {period}"  # "8 giờ tối nay": the resolver still needs "tối nay"
    return hhmm


def _half(m: re.Match) -> str:
    h = int(m.group(1))
    if h > 24:
        return m.group(0)
    hhmm = f"{_hour24(h, m.group(2)) % 24:02d}:30"
    return f"{hhmm} {m.group(2)}" if m.group(2) and re.match(r"\s+(nay|mai|qua|mốt)\b", m.string[m.end():]) else hhmm


def normalize_text(text: str) -> str:
    text = _words_to_digits(text)
    text = _MONEY.sub(_money, text)
    text = _TIME.sub(_time, text)
    return _HALF.sub(_half, text)


WEEKDAYS = {"hai": 0, "2": 0, "ba": 1, "3": 1, "tư": 2, "4": 2, "năm": 3, "5": 3, "sáu": 4, "6": 4, "bảy": 5, "7": 5}
_WD = re.compile(r"\b(?:thứ\s+(hai|ba|tư|năm|sáu|bảy|[2-7])|(chủ\s+nhật))(?:\s+(tuần\s+(?:này|sau|tới|trước|rồi)))?", re.I)
_REL = [
    (re.compile(r"\b(?:hôm|bữa)\s+nay\b|\b(?:sáng|trưa|chiều|tối|đêm)\s+nay\b", re.I), 0),
    (re.compile(r"\b(?:hôm|bữa)\s+qua\b|\b(?:sáng|trưa|chiều|tối|đêm)\s+qua\b", re.I), -1),
    (re.compile(r"\b(?:hôm|bữa)\s+kia\b", re.I), -2),
    (re.compile(r"\bngày\s+(?:mốt|kia)\b|\b(?:sáng|trưa|chiều|tối)\s+mốt\b|\bmốt\b", re.I), 2),
    (re.compile(r"\bngày\s+mai\b|\b(?:sáng|trưa|chiều|tối)\s+mai\b|(?<!\w)mai(?!\w)|(?:^|(?<=[.!?…]\s))Mai(?=\s+[a-zà-ỹ0-9])"), 1),  # case-sensitive: other "Mai" is a name
    (re.compile(r"\btuần\s+(?:sau|tới)\b", re.I), 7),
]
_DMY = re.compile(r"\b(?:ngày|hôm|mùng|mồng)?\s*(\d{1,2})\s*(?:/|\s+tháng\s+)(\d{1,2})(?:\s*/\s*(\d{4}))?\b", re.I)
_DAY = re.compile(r"\b(ngày|hôm|mùng|mồng)\s+(\d{1,2})\b(?!\s*(?:/|tháng|giờ|phút|h\b|:))", re.I)


def _day_in_month(y: int, mth: int, d: int) -> date | None:
    try:
        return date(y, mth, d)
    except ValueError:
        return None


def resolve_dates(text: str, now: str | datetime) -> list[dict]:
    """Dates mentioned in `text`, resolved against `now` (an ISO string or datetime). Run normalize_text first so that
    spoken day numbers are digits. Ordered by position; overlapping matches keep the first."""
    if isinstance(now, str):
        now = datetime.fromisoformat(now)
    today = now.date()
    found: list[tuple[int, int, str, date]] = []

    def add(m: re.Match, d: date | None) -> None:
        if d is not None:
            found.append((m.start(), m.end(), m.group(0).strip(), d))

    for pat, off in _REL:
        for m in pat.finditer(text):
            add(m, today + timedelta(days=off))
    for m in _WD.finditer(text):
        wd = 6 if m.group(2) else WEEKDAYS[m.group(1).lower()]
        rel = (m.group(3) or "").split()[-1].lower() if m.group(3) else ""
        monday = today - timedelta(days=today.weekday())
        if rel == "này":
            d = monday + timedelta(days=wd)
        elif rel in ("sau", "tới"):
            d = monday + timedelta(days=7 + wd)
        elif rel in ("trước", "rồi"):
            d = monday + timedelta(days=wd - 7)
        else:  # next occurrence strictly after today
            d = today + timedelta(days=(wd - today.weekday() - 1) % 7 + 1)
        add(m, d)
    for m in _DMY.finditer(text):
        d_, mth = int(m.group(1)), int(m.group(2))
        year = int(m.group(3)) if m.group(3) else today.year
        d = _day_in_month(year, mth, d_)
        if d and not m.group(3) and d < today - timedelta(days=183):
            d = _day_in_month(year + 1, mth, d_)
        add(m, d)
    for m in _DAY.finditer(text):
        d_ = int(m.group(2))
        d = _day_in_month(today.year, today.month, d_)
        if d is None or (d < today and m.group(1).lower() != "hôm"):  # "hôm 17" may be in the past, "ngày 17" is ahead
            nm = (today.replace(day=1) + timedelta(days=32)).replace(day=1)
            d = _day_in_month(nm.year, nm.month, d_) or d
        add(m, d)
    found.sort()
    out, end = [], -1
    for s, e, phrase, d in found:
        if s >= end:
            out.append({"phrase": phrase, "date": d.isoformat()})
            end = e
    return out
