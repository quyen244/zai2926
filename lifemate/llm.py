"""Minimal OpenAI chat client for offline analysis jobs: .env config, client-side rate limiting, retries, usage log.

Config (D:/zaloai2026/.env, git-ignored):
  OPENAI_API_KEY   required
  LLM_MODEL        default gpt-6-luna (keep to the luna tier or below: cheap analysis model)
  LLM_RPM, LLM_TPM client-side budget, default 1000 requests and 400k tokens per minute (20% of the account limits
                   5000 RPM / 2M TPM read from the API headers on 2026-10-28)
Standard library only.
"""
from __future__ import annotations

import json
import os
import random
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from .data import PROJECT

URL = "https://api.openai.com/v1/chat/completions"
RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}


def _api_name(name: str) -> str:
    """Competition tool names ("app.tool") -> OpenAI function-name pattern (no dot allowed)."""
    return name.replace(".", "__")


def _real_name(name: str) -> str:
    return name.replace("__", ".")


def _api_message(m: dict) -> dict:
    if not m.get("tool_calls"):
        return m
    return {**m, "tool_calls": [{**c, "function": {**c["function"], "name": _api_name(c["function"]["name"])}} for c in m["tool_calls"]]}


def _real_message(m: dict) -> dict:
    if not m.get("tool_calls"):
        return m
    return {**m, "tool_calls": [{**c, "function": {**c["function"], "name": _real_name(c["function"]["name"])}} for c in m["tool_calls"]]}


def load_env(path: Path = PROJECT / ".env") -> None:
    """Put KEY=VALUE lines of .env into os.environ (existing variables win)."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


class RateLimiter:
    """Two token buckets (requests/min and tokens/min), refilled continuously. acquire() blocks until both allow."""

    def __init__(self, rpm: float, tpm: float):
        self.rpm, self.tpm = rpm, tpm
        self.req, self.tok = rpm, tpm
        self.t = time.monotonic()
        self.lock = threading.Lock()
        self.paused_until = 0.0

    def _refill(self) -> None:
        now = time.monotonic()
        dt = now - self.t
        self.t = now
        self.req = min(self.rpm, self.req + dt * self.rpm / 60)
        self.tok = min(self.tpm, self.tok + dt * self.tpm / 60)

    def acquire(self, tokens: int) -> None:
        tokens = min(tokens, self.tpm)
        while True:
            with self.lock:
                self._refill()
                wait = max(0.0, self.paused_until - time.monotonic())
                if not wait and self.req >= 1 and self.tok >= tokens:
                    self.req -= 1
                    self.tok -= tokens
                    return
                if not wait:
                    wait = max((1 - self.req) * 60 / self.rpm, (tokens - self.tok) * 60 / self.tpm, 0.05)
            time.sleep(wait)

    def pause(self, seconds: float) -> None:
        """Server said slow down: stop every worker for `seconds`."""
        with self.lock:
            self.paused_until = max(self.paused_until, time.monotonic() + seconds)


@dataclass
class Usage:
    requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    retries: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, u: dict) -> None:
        with self.lock:
            self.requests += 1
            self.prompt_tokens += u.get("prompt_tokens", 0)
            self.completion_tokens += u.get("completion_tokens", 0)


class LLM:
    def __init__(self, model: str | None = None, rpm: float | None = None, tpm: float | None = None, timeout: float = 180):
        load_env()
        self.key = os.environ.get("OPENAI_API_KEY")
        if not self.key:
            raise RuntimeError("OPENAI_API_KEY missing: put it in .env at the project root")
        self.model = model or os.environ.get("LLM_MODEL", "gpt-6-luna")
        self.limiter = RateLimiter(rpm or float(os.environ.get("LLM_RPM", 1000)), tpm or float(os.environ.get("LLM_TPM", 400_000)))
        self.timeout = timeout
        self.usage = Usage()

    def chat_json(self, system: str, user: str, max_tokens: int = 4000, max_retries: int = 6) -> dict:
        """One chat completion in JSON mode; returns the parsed JSON object. Retries rate-limit and server errors with
        exponential backoff + jitter (honouring Retry-After), and a reply that is not valid JSON once."""
        est = (len(system) + len(user)) // 3 + max_tokens  # rough token estimate for the budget (Vietnamese ~3 chars/token)
        body = {"model": self.model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "response_format": {"type": "json_object"}, "max_completion_tokens": max_tokens}
        data = json.dumps(body).encode("utf-8")
        bad_json = 0
        for attempt in range(max_retries + 1):
            self.limiter.acquire(est)
            req = urllib.request.Request(URL, data=data, headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    d = json.load(r)
                self.usage.add(d.get("usage", {}))
                text = d["choices"][0]["message"]["content"]
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    bad_json += 1
                    if bad_json > 1:
                        raise
                    continue
            except urllib.error.HTTPError as e:
                if e.code not in RETRY_STATUS or attempt == max_retries:
                    raise RuntimeError(f"OpenAI HTTP {e.code}: {e.read()[:300]!r}") from None
                delay = float(e.headers.get("retry-after") or 0) or min(60, 2 ** attempt + random.random())
                self.limiter.pause(delay)
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt == max_retries:
                    raise RuntimeError(f"OpenAI request failed: {e}") from None
                time.sleep(min(60, 2 ** attempt + random.random()))
            with self.usage.lock:
                self.usage.retries += 1
        raise RuntimeError("unreachable")

    def chat_tools(self, system: str, messages: list[dict], tools: list[dict], max_tokens: int = 1000,
                    temperature: float = 0, max_retries: int = 6) -> dict:
        """One chat completion with function-calling tools (native tool_calls, not JSON mode). Returns the assistant
        message dict straight from the API ({"role", "content", "tool_calls"}): tool_calls[i]["function"]["arguments"]
        stays a JSON string, same wire shape lifemate.prompt builds, so the caller appends it to `messages` as is.
        Tool names use the competition's "app.tool" form everywhere in `tools`/`messages`/the returned message;
        OpenAI's function name pattern rejects the dot, so it is swapped for "__" only on the wire, both ways,
        invisibly to the caller (lifemate.agent never sees the sanitised form).
        temperature=0 for reproducible agent runs. Retries rate-limit and server errors like chat_json."""
        api_tools = [{**t, "function": {**t["function"], "name": _api_name(t["function"]["name"])}} for t in tools]
        api_messages = [_api_message(m) for m in messages]
        full_messages = [{"role": "system", "content": system}] + api_messages
        body = {"model": self.model, "messages": full_messages, "tools": api_tools, "tool_choice": "auto",
                "temperature": temperature, "max_completion_tokens": max_tokens, "reasoning_effort": "none"}
        data = json.dumps(body).encode("utf-8")
        est = len(data) // 3 + max_tokens
        for attempt in range(max_retries + 1):
            self.limiter.acquire(est)
            req = urllib.request.Request(URL, data=data, headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    d = json.load(r)
                self.usage.add(d.get("usage", {}))
                return _real_message(d["choices"][0]["message"])
            except urllib.error.HTTPError as e:
                if e.code not in RETRY_STATUS or attempt == max_retries:
                    raise RuntimeError(f"OpenAI HTTP {e.code}: {e.read()[:300]!r}") from None
                delay = float(e.headers.get("retry-after") or 0) or min(60, 2 ** attempt + random.random())
                self.limiter.pause(delay)
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt == max_retries:
                    raise RuntimeError(f"OpenAI request failed: {e}") from None
                time.sleep(min(60, 2 ** attempt + random.random()))
            with self.usage.lock:
                self.usage.retries += 1
        raise RuntimeError("unreachable")
