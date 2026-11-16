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
import shutil
import subprocess
import tempfile
import time
from collections import Counter
from pathlib import Path

import requests

from . import paths
from .prompting import UserMessage
from .registry import Input, Reviewer
from .taxonomy import CATEGORIES, RESPONSE_SCHEMA, label

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
    if reviewer.provider == "ollama-systemone":
        return SystemOne(reviewer)
    if reviewer.provider == "claude-code":
        return ClaudeCode(reviewer)
    raise ProviderError(f"reviewer '{reviewer.id}': unknown provider '{reviewer.provider}'")


def input_problem(reviewer: Reviewer, input: Input) -> str | None:
    """Why a model reviewer can't be given this input, or None if it can.
    Checked before a run starts, so a grid skips the pairing instead of
    failing every paper."""
    if reviewer.provider == "ollama-systemone" and input.body != "none":
        return (f"'{reviewer.id}' is a System One model, which reads metadata and the abstract only, "
                f"not input '{input.id}' (body: {input.body}); use --input abstract-only")
    if input.as_file and reviewer.provider != "claude-code":
        return (f"'{reviewer.id}' is sent the paper in its message, but input '{input.id}' puts it in a "
                f"file for an agent to open; give it an inline input such as fulltext")
    return None


def _ollama_describe(host: str, model: str) -> dict:
    """Where an Ollama model ran and exactly which weights, for run.json."""
    info = {"host": host, "digest": None}
    try:
        tags = requests.get(f"{host}/api/tags", timeout=10).json()
        info["digest"] = next((m["digest"] for m in tags.get("models", [])
                               if m.get("name") == model), None)
    except requests.RequestException:
        pass
    return info


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
        return _ollama_describe(self.host, self.reviewer.model)

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


def _claude_env(ollama: tuple[str, str] | None = None) -> dict:
    """The environment for a `claude` child process, without the variables a
    parent Claude Code session sets (CLAUDECODE, CLAUDE_CODE_*, CLAUDE_EFFORT,
    ...). Started from inside Claude Code, the child would otherwise inherit
    that session's effort and a link back to it. The two that say how to log
    in are kept.

    With `ollama` = (host, model), every request goes to Ollama's
    Anthropic-compatible API instead, background ones included, and no
    Anthropic credentials are passed on."""
    keep = {"CLAUDE_CODE_OAUTH_TOKEN", "CLAUDE_CONFIG_DIR"}
    env = {k: v for k, v in os.environ.items() if k in keep or not k.startswith("CLAUDE")}
    if ollama:
        host, model = ollama
        env = {k: v for k, v in env.items() if not k.startswith("ANTHROPIC_") and k != "CLAUDE_CODE_OAUTH_TOKEN"}
        env.update(ANTHROPIC_BASE_URL=host, ANTHROPIC_AUTH_TOKEN="ollama",
                   ANTHROPIC_DEFAULT_HAIKU_MODEL=model, ANTHROPIC_SMALL_FAST_MODEL=model)
    return env


def _ollama_num_ctx(host: str, model: str) -> int | None:
    """The context length an Ollama model is configured with (its Modelfile's num_ctx)."""
    try:
        params = requests.post(f"{host}/api/show", json={"model": model}, timeout=10).json().get("parameters", "")
    except (requests.RequestException, ValueError):
        return None
    for line in params.splitlines():
        name, _, value = line.strip().partition(" ")
        if name == "num_ctx" and value.strip().isdigit():
            return int(value.strip())
    return None


class ClaudeCode:
    """Claude Code run headless (`claude -p`) as an agent: one fresh session
    per paper, which can use tools over several turns before answering.

    Each paper gets a new temporary folder holding only that paper (paper.txt
    or paper.pdf, when the input names {file}), and the session runs there with

        --restricted          file tools confined to that folder, nothing that
                              runs code, and user/project settings ignored
        --strict-mcp-config   no MCP servers (so no Google Drive or Sheets)
        --tools               only the tools the reviewer lists

    so earlier judgements, review sheets and ~/.claude are out of reach. The
    session's init event is checked against that: any other tool, a missing
    one, or an MCP server fails the paper instead of giving a judgement.
    Reads it attempts outside the folder are refused and recorded in the
    judgement's `permission_denials`.

    The system prompt is appended to Claude Code's own, which tells the model
    how to use its tools, and the answer comes back through --json-schema.
    Billing is the logged-in account, or the API if ANTHROPIC_API_KEY is set
    in the environment (the key in .secrets is not passed on).

    `backend: ollama` runs a local model in the same harness: Claude Code
    sends its requests to Ollama's Anthropic-compatible API (OLLAMA_HOST).
    That API can't set num_ctx per request, so the model should be a variant
    with its context set in a Modelfile (see factors/reviewers/modelfiles/);
    run.json records the num_ctx it was configured with. effort and
    max_budget_usd have no effect there, so they are refused, and cost_usd is
    left empty because Claude Code doesn't know the model's price.

    The session's events are returned as `transcript`, which the runner
    stores beside the judgements. Token counts are totals over every turn,
    cache reads included.
    """
    SETTINGS = {"tools", "effort", "timeout_s", "max_budget_usd", "backend"}
    BACKENDS = ("anthropic", "ollama")
    # Read-only file tools, which --restricted keeps inside the paper's folder,
    # and the web tools. A tool that runs code (Bash) could read the whole disk.
    ALLOWED_TOOLS = ("Read", "Grep", "Glob", "WebSearch", "WebFetch")
    DEFAULT_TOOLS = ("Read", "Grep", "Glob")
    INTERNAL_TOOLS = {"StructuredOutput"}  # how --json-schema returns the answer
    EFFORTS = ("low", "medium", "high", "xhigh", "max")

    def __init__(self, reviewer: Reviewer):
        self.reviewer = reviewer
        self.settings = _check_settings(reviewer, self.SETTINGS)
        self.tools = list(self.settings.get("tools", self.DEFAULT_TOOLS))
        bad = [t for t in self.tools if t not in self.ALLOWED_TOOLS]
        if bad:
            raise ProviderError(
                f"reviewer '{reviewer.id}': tool(s) {', '.join(bad)} not allowed. Allowed: "
                f"{', '.join(self.ALLOWED_TOOLS)}. Anything that runs code could read outside the paper's folder.")
        effort = self.settings.get("effort")
        if effort is not None and effort not in self.EFFORTS:
            raise ProviderError(f"reviewer '{reviewer.id}': effort must be one of {', '.join(self.EFFORTS)}")
        self.backend = self.settings.get("backend", "anthropic")
        if self.backend not in self.BACKENDS:
            raise ProviderError(f"reviewer '{reviewer.id}': backend must be one of {', '.join(self.BACKENDS)}")
        self.timeout = self.settings.get("timeout_s", 900)
        self.host = None
        if self.backend == "ollama":
            ignored = sorted({"effort", "max_budget_usd"} & set(self.settings))
            if ignored:
                raise ProviderError(f"reviewer '{reviewer.id}': {', '.join(ignored)} has no effect "
                                    f"on a model served by Ollama")
            self.host = load_secrets().get("OLLAMA_HOST", DEFAULT_OLLAMA_HOST)
        self.env = _claude_env((self.host, reviewer.model) if self.host else None)

    def describe(self) -> dict:
        try:
            version = subprocess.run(["claude", "--version"], capture_output=True, text=True,
                                     env=self.env, timeout=60).stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            version = None
        if self.backend == "ollama":
            return {"claude_code_version": version, "backend": "ollama",
                    **_ollama_describe(self.host, self.reviewer.model),
                    "num_ctx": _ollama_num_ctx(self.host, self.reviewer.model)}
        return {"claude_code_version": version,
                "auth": "ANTHROPIC_API_KEY" if self.env.get("ANTHROPIC_API_KEY") else "claude login"}

    def _command(self, system_file: Path) -> list[str]:
        tools = ",".join(self.tools)
        cmd = ["claude", "-p", "--restricted", "--strict-mcp-config",
               "--tools", tools, "--allowedTools", tools,
               "--model", self.reviewer.model,
               "--append-system-prompt-file", str(system_file),
               "--json-schema", json.dumps(RESPONSE_SCHEMA),
               "--output-format", "stream-json", "--verbose",
               "--no-session-persistence"]
        if "effort" in self.settings:
            cmd += ["--effort", self.settings["effort"]]
        if "max_budget_usd" in self.settings:
            cmd += ["--max-budget-usd", str(self.settings["max_budget_usd"])]
        return cmd

    def _check_isolation(self, init: dict) -> None:
        tools = set(init.get("tools") or []) - self.INTERNAL_TOOLS
        extra, missing = tools - set(self.tools), set(self.tools) - tools
        servers = [s.get("name", s) if isinstance(s, dict) else s for s in init.get("mcp_servers") or []]
        dirs = init.get("additional_directories") or []  # file tools could reach these too
        problems = ([f"unexpected tool(s) {', '.join(sorted(extra))}"] if extra else []) \
            + ([f"missing tool(s) {', '.join(sorted(missing))}"] if missing else []) \
            + ([f"MCP server(s) {', '.join(map(str, servers))}"] if servers else []) \
            + ([f"additional directories {', '.join(map(str, dirs))}"] if dirs else [])
        if problems:
            raise ProviderError("session not isolated as configured (" + "; ".join(problems)
                                + "); no judgement recorded")

    def classify(self, system: str, message: UserMessage) -> dict:
        if message.pdf and not message.file_name:
            raise ProviderError("Claude Code can't be sent a PDF in its message; use an input that gives it as {file}")
        user, truncated = message.render()
        with tempfile.TemporaryDirectory(prefix="pif-claude-code-") as tmp:
            # The session runs in tmp/paper; the system prompt sits beside it, out of the tools' reach.
            folder = Path(tmp) / "paper"
            folder.mkdir()
            system_file = Path(tmp) / "system.md"
            system_file.write_text(system)
            if message.file_name:
                if message.pdf:
                    shutil.copyfile(message.pdf, folder / message.file_name)  # a copy, never a link back
                else:
                    (folder / message.file_name).write_text(message.text or "")
            t0 = time.time()
            try:
                proc = subprocess.run(self._command(system_file), cwd=folder, input=user, env=self.env,
                                      capture_output=True, text=True, timeout=self.timeout)
            except subprocess.TimeoutExpired:
                raise ProviderError(f"no answer within {self.timeout}s") from None
            except FileNotFoundError:
                raise ProviderError("the `claude` command isn't on PATH") from None
            elapsed = time.time() - t0

        events = []
        for line in proc.stdout.splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        init = next((e for e in events if e.get("type") == "system" and e.get("subtype") == "init"), None)
        result = next((e for e in reversed(events) if e.get("type") == "result"), None)
        stderr = proc.stderr.strip()[:500]
        if init is None:
            raise ProviderError(f"claude exited {proc.returncode} before starting a session: {stderr}")
        self._check_isolation(init)
        if result is None:
            raise ProviderError(f"claude exited {proc.returncode} with no result: {stderr}")
        subtype = result.get("subtype") or ""
        if result.get("is_error") and not subtype.startswith("error_max"):
            # An API, auth or usage-limit error: not an answer, so the paper is retried on rerun.
            raise ProviderError(f"{subtype or 'error'}: {str(result.get('result') or stderr)[:500]}")

        answer = result.get("structured_output") or {}
        category, reasoning, error, raw = None, "", None, None
        if subtype.startswith("error_max"):  # a limit the reviewer sets (budget, turns) was reached
            error = subtype
        elif answer.get("category") in CATEGORIES:
            category, reasoning = answer["category"], answer.get("reasoning", "")
        else:
            error = "no valid category in response"
            raw = json.dumps(answer) if answer else str(result.get("result") or "")

        usage = result.get("usage") or {}
        tool_calls = Counter(
            b.get("name") for e in events if e.get("type") == "assistant"
            for b in (e.get("message") or {}).get("content") or []
            if isinstance(b, dict) and b.get("type") == "tool_use" and b.get("name") not in self.INTERNAL_TOOLS)
        return {
            "category": category,
            "reasoning": reasoning,
            "truncated": truncated,
            "prompt_tokens": sum(usage.get(k) or 0 for k in
                                 ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")),
            "output_tokens": usage.get("output_tokens"),
            "duration_s": round(elapsed, 2),
            "num_turns": result.get("num_turns"),
            "cost_usd": result.get("total_cost_usd") if self.backend == "anthropic" else None,
            "tool_calls": dict(tool_calls),
            "permission_denials": [{"tool": d.get("tool_name"), "input": d.get("tool_input")}
                                   for d in result.get("permission_denials") or []],
            "models": sorted(result.get("modelUsage") or {}),
            "claude_code_version": init.get("claude_code_version"),
            # Without the token counters streamed while the model thinks: thousands per paper
            # from a thinking model, and nothing that isn't in the assistant messages.
            "transcript": [e for e in events
                           if not (e.get("type") == "system" and e.get("subtype") == "thinking_tokens")],
            **({"error": error} if error else {}),
            **({"raw_response": raw} if raw else {}),
        }


class SystemOne:
    """A System One (decision) model through Ollama's /v1/systemone, e.g. nimble.

    It doesn't write text. It is asked one choice question whose options are
    the nine categories, and returns a probability for each: the category is
    the most probable (ties broken in taxonomy order), the probabilities and
    the model's confidence (how concentrated they are, not the chance it is
    right) are kept, and `reasoning` lists the top three so the judgement
    reads like any other.

    The system prompt is the question's instructions and the user message its
    state. The context is fixed at 8k tokens, too small for full text, so it
    is only given inputs without a body (see input_problem). The endpoint
    ignores sampling options, so none are accepted.
    """
    SETTINGS = {"timeout_s", "keep_alive"}

    def __init__(self, reviewer: Reviewer):
        self.reviewer = reviewer
        self.settings = _check_settings(reviewer, self.SETTINGS)
        self.host = load_secrets().get("OLLAMA_HOST", DEFAULT_OLLAMA_HOST)
        self.timeout = self.settings.get("timeout_s", 600)

    def describe(self) -> dict:
        return _ollama_describe(self.host, self.reviewer.model)

    def classify(self, system: str, message: UserMessage) -> dict:
        if message.text is not None or message.pdf:
            raise ProviderError("a System One model reads metadata and the abstract only")
        state, truncated = message.render()
        payload = {
            "model": self.reviewer.model,
            "state": state,
            "questions": {"category": {
                "type": "choice",
                "instructions": system,
                "criteria": {c: label(c) for c in CATEGORIES},
            }},
        }
        if "keep_alive" in self.settings:
            payload["keep_alive"] = self.settings["keep_alive"]
        t0 = time.time()
        resp = requests.post(f"{self.host}/v1/systemone", json=payload, timeout=self.timeout)
        if resp.status_code >= 400:
            # Ollama's message says why, e.g. a request longer than the context
            raise ProviderError(f"HTTP {resp.status_code}: {resp.text.strip()[:500]}")
        data = resp.json()
        elapsed = time.time() - t0

        answer = (data.get("answers") or {}).get("category") or {}
        probs = answer.get("probabilities") or {}
        valid = set(probs) == set(CATEGORIES)
        category, reasoning = None, ""
        if valid:
            ranked = sorted(CATEGORIES, key=lambda c: -probs[c])  # stable: ties stay in taxonomy order
            category = ranked[0]
            reasoning = " · ".join(f"{c} {probs[c]:.3f}" for c in ranked[:3])
            if answer.get("confidence") is not None:
                reasoning += f" (confidence {answer['confidence']:.3f})"
        usage = data.get("usage") or {}
        return {
            "category": category,
            "reasoning": reasoning,
            **({"probabilities": {c: probs[c] for c in CATEGORIES},
                "confidence": answer.get("confidence")} if valid else {}),
            "truncated": truncated,
            "prompt_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "duration_s": round(elapsed, 2),
            **({} if valid else {"error": "no probability for each category in response",
                                 "raw_response": resp.text}),
        }
