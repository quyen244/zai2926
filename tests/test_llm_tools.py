import importlib.util
import time
from pathlib import Path

from lifemate.llm import RateLimiter, _api_message, _api_name, _real_message, _real_name

spec = importlib.util.spec_from_file_location("llm_tag", Path(__file__).resolve().parent.parent / "scripts" / "llm_tag.py")
llm_tag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(llm_tag)

OUT, TAGS = {"action", "ask_missing"}, {"conditional", "self_repair"}


def test_check_accepts_valid_and_requeues_the_rest():
    reply = {"results": [
        {"id": "a", "outcome": "action", "tags": ["conditional"]},
        {"id": "b", "outcome": "nope", "tags": []},           # unknown outcome
        {"id": "c", "outcome": "action", "tags": ["made_up"]},  # unknown tag
        {"id": "zzz", "outcome": "action", "tags": []},       # not in the batch: ignored
    ]}
    good, retry = llm_tag.check(reply, ["a", "b", "c", "d"], OUT, TAGS)
    assert list(good) == ["a"]
    assert retry == ["b", "c", "d"]


def test_check_rejects_duplicates():
    reply = {"results": [{"id": "a", "outcome": "action", "tags": []}, {"id": "a", "outcome": "ask_missing", "tags": []}]}
    good, retry = llm_tag.check(reply, ["a"], OUT, TAGS)
    assert good == {} and retry == ["a"]


def test_check_survives_garbage():
    assert llm_tag.check(["not a dict"], ["a"], OUT, TAGS) == ({}, ["a"])


def test_taxonomy_ids_unique_and_split_by_cardinality():
    tax, outcomes, tags, text = llm_tag.load_taxonomy()
    ids = [t["id"] for g in tax["groups"] for t in g["tags"]]
    assert len(ids) == len(set(ids))
    assert outcomes and tags and not outcomes & tags


def test_rate_limiter_spaces_requests():
    rl = RateLimiter(rpm=600, tpm=1e9)  # 10 requests/s, bucket starts full
    rl.req = 0
    t = time.monotonic()
    for _ in range(3):
        rl.acquire(1)
    assert time.monotonic() - t >= 0.25


def test_rate_limiter_token_budget():
    rl = RateLimiter(rpm=1e6, tpm=60_000)  # 1000 tokens/s
    rl.tok = 0
    t = time.monotonic()
    rl.acquire(300)
    assert time.monotonic() - t >= 0.25


def test_api_name_roundtrip():
    assert _api_name("alarm.list") == "alarm__list"
    assert _real_name(_api_name("ride_hailing.cancel")) == "ride_hailing.cancel"


def test_api_message_sanitises_tool_call_names_and_leaves_others_alone():
    plain = {"role": "user", "content": "hi"}
    assert _api_message(plain) == plain
    msg = {"role": "assistant", "content": None, "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "alarm.create", "arguments": "{}"}}]}
    api = _api_message(msg)
    assert api["tool_calls"][0]["function"]["name"] == "alarm__create"
    assert _real_message(api) == msg
