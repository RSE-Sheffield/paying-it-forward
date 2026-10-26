"""Model providers. Each classifies one paper per call, with no state carried
between papers: every request is a fresh system prompt plus that paper alone.

Settings come from the reviewer file. Unknown settings are an error rather
than being silently ignored, so a typo can't produce a run that looks like it
used a setting it didn't.
"""
from __future__ import annotations

import base64
import json
import os
import time

import requests

from . import paths
from .prompting import UserMessage
from .registry import Reviewer
from .taxonomy import CATEGORIES, RESPONSE_SCHEMA

DEFAULT_OLLAMA_HOST = "http://localhost:11434"


class ProviderError(Exception):
    pass


def load_secrets() -> dict:
    """KEY=VALUE lines from .secrets; environment variables take precedence."""
    secrets = {}
    if paths.SECRETS.exists():
        for line in paths.SECRETS.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            secrets[key.strip()] = value.strip().strip('"').strip("'")
    for key in ("ANTHROPIC_API_KEY", "OLLAMA_HOST"):
        if os.environ.get(key):
            secrets[key] = os.environ[key]
    return secrets


def _check_settings(reviewer: Reviewer, allowed: set[str], hint: str = "") -> dict:
    unknown = set(reviewer.settings) - allowed
    if unknown:
        raise ProviderError(
            f"reviewer '{reviewer.id}': {reviewer.provider} doesn't support setting(s) "
            f"{', '.join(sorted(unknown))}. Supported: {', '.join(sorted(allowed))}.{hint}")
    return reviewer.settings


def make(reviewer: Reviewer):
    if reviewer.provider == "ollama":
        return Ollama(reviewer)
    if reviewer.provider == "anthropic":
        return Anthropic(reviewer)
    raise ProviderError(f"reviewer '{reviewer.id}': unknown provider '{reviewer.provider}'")


class Ollama:
    """A local model through Ollama's /api/chat with a JSON-schema response format.

    Full text that wouldn't fit in num_ctx is truncated to fit, and the
    judgement records that it was.
    """
    SETTINGS = {"num_ctx", "timeout_s", "temperature", "seed", "top_p", "top_k"}
    CHARS_PER_TOKEN = 4  # conservative estimate for English prose
    OUTPUT_TOKEN_BUDGET = 1000
    CTX_SAFETY_MARGIN = 256

    def __init__(self, reviewer: Reviewer):
        self.reviewer = reviewer
        self.settings = _check_settings(reviewer, self.SETTINGS)
        self.host = load_secrets().get("OLLAMA_HOST", DEFAULT_OLLAMA_HOST)
        self.num_ctx = self.settings.get("num_ctx", 262144)
        self.timeout = self.settings.get("timeout_s", 600)

    def describe(self) -> dict:
        """Where the model ran and exactly which weights, for run.json."""
        info = {"host": self.host, "digest": None}
        try:
            tags = requests.get(f"{self.host}/api/tags", timeout=10).json()
            info["digest"] = next((m["digest"] for m in tags.get("models", [])
                                   if m.get("name") == self.reviewer.model), None)
        except requests.RequestException:
            pass
        return info

    def _estimate(self, text: str) -> int:
        return len(text) // self.CHARS_PER_TOKEN

    def _fit(self, system: str, message: UserMessage) -> tuple[str, bool]:
        """The user message, with full text truncated if the request wouldn't fit num_ctx."""
        user, truncated = message.render()
        reserve = self.OUTPUT_TOKEN_BUDGET + self.CTX_SAFETY_MARGIN
        if self._estimate(system) + self._estimate(user) + reserve <= self.num_ctx or not message.text:
            return user, truncated
        budget_tokens = self.num_ctx - self._estimate(system) - reserve
        return message.render(max_chars=max(0, budget_tokens * self.CHARS_PER_TOKEN))[0], True

    def classify(self, system: str, message: UserMessage) -> dict:
        if message.pdf:
            raise ProviderError("Ollama can't read PDFs; use an input with extracted text")
        user, truncated = self._fit(system, message)
        options = {"num_ctx": self.num_ctx}
        options.update({k: v for k, v in self.settings.items() if k in ("temperature", "seed", "top_p", "top_k")})
        payload = {
            "model": self.reviewer.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "format": RESPONSE_SCHEMA,
            "stream": False,
            "options": options,
        }
        t0 = time.time()
        resp = requests.post(f"{self.host}/api/chat", json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        elapsed = time.time() - t0

        content = data.get("message", {}).get("content", "")
        try:
            parsed = json.loads(content)
            category, reasoning = parsed.get("category"), parsed.get("reasoning", "")
        except (json.JSONDecodeError, AttributeError):
            category, reasoning = None, ""
        valid = category in CATEGORIES
        return {
            "category": category if valid else None,
            "reasoning": reasoning if valid else "",
            "truncated": truncated,
            "prompt_tokens": data.get("prompt_eval_count"),
            "output_tokens": data.get("eval_count"),
            "duration_s": round(elapsed, 2),
            **({} if valid else {"error": "no valid category in response", "raw_response": content}),
        }


class Anthropic:
    """Claude through the Messages API. The answer comes back as a call to a
    classify_paper tool whose input schema is the response schema.

    tool_choice "tool" (the default) forces that call. Newer models (Sonnet
    5.5, Opus 5.5, Fable 5.1) reject forced tool use, so give their reviewers
    `tool_choice: auto`, which offers the tool with strict schema checking.
    Sampling parameters (temperature/top_p/top_k) are rejected by the API on
    every model since 2026-10-21; vary `effort` instead.
    """
    SETTINGS = {"max_tokens", "timeout_s", "effort", "thinking", "tool_choice"}

    def __init__(self, reviewer: Reviewer):
        import anthropic

        self.reviewer = reviewer
        self.settings = _check_settings(
            reviewer, self.SETTINGS,
            " The Messages API no longer accepts temperature/top_p/top_k; vary effort instead."
            if {"temperature", "top_p", "top_k"} & set(reviewer.settings) else "")
        key = load_secrets().get("ANTHROPIC_API_KEY")
        # With no key in .secrets or the environment, the SDK resolves credentials itself.
        self.client = anthropic.Anthropic(api_key=key) if key else anthropic.Anthropic()
        self.tool_choice = self.settings.get("tool_choice", "tool")
        if self.tool_choice not in ("tool", "auto"):
            raise ProviderError(f"reviewer '{reviewer.id}': tool_choice must be 'tool' or 'auto'")

    def describe(self) -> dict:
        return {}

    def classify(self, system: str, message: UserMessage) -> dict:
        user, truncated = message.render()
        content: list | str = user
        if message.pdf:
            data = base64.standard_b64encode(message.pdf.read_bytes()).decode()
            content = [
                {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": data}},
                {"type": "text", "text": user},
            ]
        tool = {
            "name": "classify_paper",
            "description": "Record the S*D* classification for this paper.",
            "input_schema": RESPONSE_SCHEMA,
        }
        if self.tool_choice == "auto":
            tool = {**tool, "strict": True,
                    "input_schema": {**RESPONSE_SCHEMA, "additionalProperties": False}}
        kwargs = {}
        if "effort" in self.settings:
            kwargs["output_config"] = {"effort": self.settings["effort"]}
        if "thinking" in self.settings:
            kwargs["thinking"] = self.settings["thinking"]

        t0 = time.time()
        resp = self.client.messages.create(
            model=self.reviewer.model,
            max_tokens=self.settings.get("max_tokens", 600),
            system=system,
            messages=[{"role": "user", "content": content}],
            tools=[tool],
            tool_choice={"type": "tool", "name": "classify_paper"} if self.tool_choice == "tool" else {"type": "auto"},
            timeout=self.settings.get("timeout_s", 600),
            **kwargs,
        )
        elapsed = time.time() - t0

        category, reasoning, error, raw = None, "", None, None
        tool_use = next((b for b in resp.content if b.type == "tool_use"), None)
        if resp.stop_reason == "refusal":
            error = f"refusal ({getattr(resp.stop_details, 'category', None)})"
        elif tool_use is None:
            error = f"no classify_paper call (stop_reason {resp.stop_reason})"
            raw = "".join(getattr(b, "text", "") for b in resp.content)
        elif tool_use.input.get("category") in CATEGORIES:
            category = tool_use.input["category"]
            reasoning = tool_use.input.get("reasoning", "")
        else:
            error, raw = "no valid category in response", json.dumps(tool_use.input)
        return {
            "category": category,
            "reasoning": reasoning,
            "truncated": truncated,
            "prompt_tokens": resp.usage.input_tokens,
            "output_tokens": resp.usage.output_tokens,
            "duration_s": round(elapsed, 2),
            **({"error": error} if error else {}),
            **({"raw_response": raw} if raw else {}),
        }
