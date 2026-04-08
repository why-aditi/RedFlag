"""
RedFlag Inference Script
========================

MANDATORY root-level inference script for OpenEnv submission.

Environment Variables:
    API_BASE_URL   LLM endpoint (OpenAI-compatible). Validators inject LiteLLM proxy URL — use as-is.
    API_KEY        Key for API_BASE_URL. Validators inject the proxy key — do not substitute HF_TOKEN.
    MODEL_NAME     The model identifier (default: zai-org/GLM-5)
    HF_TOKEN       Optional; used only if API_KEY is unset (local HF Router).
    IMAGE_NAME     Docker image name for the environment

STDOUT FORMAT:
    [START] task=<task_name> env=redflag_env model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>
"""

import base64
import json
import os
import socket
import ssl
import struct
import sys
import textwrap
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import urllib.error
import urllib.request

try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment]


class _CompatMessage:
    def __init__(self, content: str):
        self.content = content


class _CompatChoice:
    def __init__(self, content: str):
        self.message = _CompatMessage(content)


class _CompatCompletion:
    def __init__(self, content: str):
        self.choices = [_CompatChoice(content)]


class _StdlibChatCompletions:
    def __init__(self, base_url: str, api_key: str):
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key

    def create(
        self,
        *,
        model: str,
        messages: list,
        temperature: float,
        max_tokens: int,
        stream: bool = False,
        **_kwargs: Any,
    ) -> _CompatCompletion:
        if stream:
            raise ValueError("Streaming not supported in stdlib fallback client.")

        url = f"{self._base_url}/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        req = urllib.request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as e:
            try:
                body = e.read().decode("utf-8", errors="replace")
            except Exception:
                body = "<unreadable response body>"
            raise RuntimeError(f"HTTP {e.code} from LLM endpoint: {body}") from e
        except Exception as e:
            raise RuntimeError(f"Failed to call LLM endpoint: {e}") from e

        data = json.loads(raw.decode("utf-8", errors="replace"))

        content = (
            (data.get("choices") or [{}])[0].get("message") or {}
        ).get("content", "")
        return _CompatCompletion((content or "").strip())


class _StdlibChat:
    def __init__(self, base_url: str, api_key: str):
        self.completions = _StdlibChatCompletions(base_url, api_key)


class _StdlibOpenAI:
    """Minimal OpenAI-compatible client (chat.completions.create only)."""

    def __init__(self, *, base_url: str, api_key: str):
        self.chat = _StdlibChat(base_url, api_key)


# ── OpenEnv-free WebSocket env client (stdlib only) ───────────────────────
# Validators often run inference.py with only the stdlib. The repo's client.py
# and models.py import openenv/pydantic; this block avoids those imports.


def _convert_http_to_ws_url(url: str) -> str:
    ws_url = url.rstrip("/")
    if ws_url.startswith("http://"):
        ws_url = "ws://" + ws_url[7:]
    elif ws_url.startswith("https://"):
        ws_url = "wss://" + ws_url[8:]
    elif not ws_url.startswith(("ws://", "wss://")):
        ws_url = "ws://" + ws_url
    return ws_url


class _StdlibWebSocket:
    """Minimal RFC6455 client (text + close + ping) for OpenEnv /ws."""

    def __init__(
        self,
        http_base_url: str,
        connect_timeout_s: float = 10.0,
        message_timeout_s: float = 60.0,
    ) -> None:
        self._http_base_url = http_base_url
        self._connect_timeout_s = connect_timeout_s
        self._message_timeout_s = message_timeout_s
        self._sock: Optional[socket.socket] = None
        self._pending = bytearray()

    def _read_exact(self, n: int) -> bytes:
        assert self._sock is not None
        while len(self._pending) < n:
            chunk = self._sock.recv(65536)
            if not chunk:
                raise ConnectionError("WebSocket connection closed unexpectedly")
            self._pending.extend(chunk)
        out = bytes(self._pending[:n])
        del self._pending[:n]
        return out

    def _recv_frame(self) -> tuple[bool, int, bytes]:
        b0, b1 = struct.unpack("!BB", self._read_exact(2))
        fin = bool((b0 >> 7) & 1)
        opcode = b0 & 0x0F
        masked = (b1 >> 7) & 1
        length = b1 & 0x7F
        if length == 126:
            length = struct.unpack("!H", self._read_exact(2))[0]
        elif length == 127:
            length = struct.unpack("!Q", self._read_exact(8))[0]
        if masked:
            mask = self._read_exact(4)
            payload = self._read_exact(length)
            payload = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        else:
            payload = self._read_exact(length)
        return fin, opcode, payload

    def _send_frame(self, opcode: int, data: bytes) -> None:
        assert self._sock is not None
        header = bytearray()
        b0 = 0x80 | (opcode & 0x0F)
        length = len(data)
        mask_bit = 0x80
        if length < 126:
            header.append(b0)
            header.append(mask_bit | length)
        elif length < 65536:
            header.append(b0)
            header.append(mask_bit | 126)
            header.extend(struct.pack("!H", length))
        else:
            header.append(b0)
            header.append(mask_bit | 127)
            header.extend(struct.pack("!Q", length))
        mask = os.urandom(4)
        header.extend(mask)
        masked = bytes(b ^ mask[i % 4] for i, b in enumerate(data))
        self._sock.sendall(bytes(header) + masked)

    def connect(self) -> None:
        ws_url = _convert_http_to_ws_url(self._http_base_url).rstrip("/") + "/ws"
        pseudo = ws_url.replace("ws://", "http://", 1).replace("wss://", "https://", 1)
        pr = urlparse(pseudo)
        host = pr.hostname or "localhost"
        path = pr.path or "/"
        port = pr.port
        use_tls = pr.scheme == "https"
        if port is None:
            port = 443 if use_tls else 80

        raw_sock = socket.create_connection((host, port), timeout=self._connect_timeout_s)
        if use_tls:
            ctx = ssl.create_default_context()
            raw_sock = ctx.wrap_socket(raw_sock, server_hostname=host)

        key = base64.b64encode(os.urandom(16)).decode("ascii")
        req = (
            f"GET {path} HTTP/1.1\r\n"
            f"Host: {host}:{port}\r\n"
            f"Upgrade: websocket\r\n"
            f"Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            f"Sec-WebSocket-Version: 13\r\n"
            f"\r\n"
        ).encode("ascii")
        raw_sock.sendall(req)

        self._pending.clear()
        self._sock = raw_sock
        self._sock.settimeout(self._message_timeout_s)
        while b"\r\n\r\n" not in self._pending:
            chunk = self._sock.recv(65536)
            if not chunk:
                raise ConnectionError("WebSocket handshake failed (EOF)")
            self._pending.extend(chunk)
            if len(self._pending) > 65536:
                raise ConnectionError("WebSocket handshake failed (headers too large)")
        idx = self._pending.index(b"\r\n\r\n")
        status_line = self._pending[: self._pending.index(b"\r\n")].decode(
            "latin-1", errors="replace"
        )
        if " 101 " not in status_line:
            raise ConnectionError(f"WebSocket handshake failed: {status_line!r}")
        del self._pending[: idx + 4]

    def recv_text(self) -> str:
        assert self._sock is not None
        out = bytearray()
        expect_first = True
        while True:
            fin, opcode, payload = self._recv_frame()
            if opcode == 8:
                raise ConnectionError("WebSocket closed by server")
            if opcode == 9:
                self._send_frame(0x0A, payload)
                continue
            if opcode == 0x0A:
                continue
            if expect_first:
                if opcode not in (1, 2):
                    raise RuntimeError(f"Expected data frame, got opcode {opcode}")
                out.extend(payload)
                if fin:
                    return out.decode("utf-8")
                expect_first = False
            else:
                if opcode != 0:
                    raise RuntimeError(f"Expected continuation frame, got {opcode}")
                out.extend(payload)
                if fin:
                    return out.decode("utf-8")

    def send_json(self, obj: Dict[str, Any]) -> None:
        self._send_frame(0x1, json.dumps(obj, ensure_ascii=False).encode("utf-8"))

    def recv_json(self) -> Dict[str, Any]:
        return json.loads(self.recv_text())

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._send_frame(0x8, b"")
            except Exception:
                pass
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None


class _SimpleObservation:
    def __init__(self, data: Dict[str, Any]) -> None:
        self._data = data

    def model_dump(self) -> Dict[str, Any]:
        return dict(self._data)


class _SimpleStepResult:
    def __init__(self, payload: Dict[str, Any]) -> None:
        obs_inner = payload.get("observation") or {}
        merged: Dict[str, Any] = dict(obs_inner) if isinstance(obs_inner, dict) else {}
        merged["reward"] = payload.get("reward")
        merged["done"] = payload.get("done", False)
        self.observation = _SimpleObservation(merged)
        self.reward = payload.get("reward")
        self.done = bool(payload.get("done", False))


def _parse_ws_message(msg: Dict[str, Any]) -> _SimpleStepResult:
    if msg.get("type") == "error":
        err = msg.get("data") or {}
        raise RuntimeError(
            f"Server error: {err.get('message', 'Unknown error')} "
            f"(code: {err.get('code', 'UNKNOWN')})"
        )
    if msg.get("type") != "observation":
        raise RuntimeError(f"Unexpected WebSocket message: {msg!r}")
    return _SimpleStepResult(msg.get("data") or {})


def _step_payload_from_action(action: Any) -> Dict[str, Any]:
    if hasattr(action, "model_dump"):
        raw: Dict[str, Any] = action.model_dump(exclude_none=True)
    else:
        raw = {
            "action_type": getattr(action, "action_type", "request_changes"),
        }
        if getattr(action, "comment", None) is not None:
            c = action.comment
            raw["comment"] = c.model_dump() if hasattr(c, "model_dump") else c
        if getattr(action, "context_request", None) is not None:
            cr = action.context_request
            raw["context_request"] = (
                cr.model_dump() if hasattr(cr, "model_dump") else cr
            )
        if getattr(action, "reasoning", None) is not None:
            raw["reasoning"] = action.reasoning

    payload: Dict[str, Any] = {"action_type": raw["action_type"]}
    if raw.get("comment") is not None:
        payload["comment"] = raw["comment"]
    if raw.get("context_request") is not None:
        payload["context_request"] = raw["context_request"]
    if raw.get("reasoning") is not None:
        payload["reasoning"] = raw["reasoning"]
    return payload


class _SimpleAction:
    """Stand-in when models.RedflagAction (pydantic) is unavailable."""

    def __init__(self, d: Dict[str, Any]) -> None:
        self.action_type = d.get("action_type", "request_changes")
        self.comment = d.get("comment")
        self.context_request = d.get("context_request")
        self.reasoning = d.get("reasoning")


class _FallbackSyncRedflagEnv:
    def __init__(self, parent: "_FallbackRedflagEnv") -> None:
        self._parent = parent
        self._ws: Optional[_StdlibWebSocket] = None

    def __enter__(self) -> "_FallbackSyncRedflagEnv":
        self._ws = _StdlibWebSocket(
            self._parent._base_url,
            connect_timeout_s=self._parent._connect_timeout_s,
            message_timeout_s=self._parent._message_timeout_s,
        )
        self._ws.connect()
        return self

    def __exit__(self, *args: Any) -> None:
        if self._ws is not None:
            try:
                self._ws.send_json({"type": "close"})
            except Exception:
                pass
            self._ws.close()
            self._ws = None

    def reset(self, **kwargs: Any) -> _SimpleStepResult:
        assert self._ws is not None
        self._ws.send_json({"type": "reset", "data": kwargs})
        return _parse_ws_message(self._ws.recv_json())

    def step(self, action: Any) -> _SimpleStepResult:
        assert self._ws is not None
        self._ws.send_json({"type": "step", "data": _step_payload_from_action(action)})
        return _parse_ws_message(self._ws.recv_json())


class _FallbackRedflagEnv:
    """Drop-in for client.RedflagEnv when openenv is not installed."""

    def __init__(
        self,
        base_url: str,
        connect_timeout_s: float = 10.0,
        message_timeout_s: float = 60.0,
        **_kwargs: Any,
    ) -> None:
        self._base_url = base_url
        self._connect_timeout_s = connect_timeout_s
        self._message_timeout_s = message_timeout_s

    def sync(self) -> _FallbackSyncRedflagEnv:
        return _FallbackSyncRedflagEnv(self)


_ENV_ACTION_CLASSES: Optional[tuple[Any, Any]] = None


def _load_redflag_env_and_action() -> tuple[Any, Any]:
    """Prefer repo client/models; fall back to stdlib WebSocket if imports fail."""
    global _ENV_ACTION_CLASSES
    if _ENV_ACTION_CLASSES is not None:
        return _ENV_ACTION_CLASSES
    try:
        from client import RedflagEnv as _RE  # type: ignore
        from models import RedflagAction as _RA  # type: ignore

        _ENV_ACTION_CLASSES = (_RE, _RA)
    except ImportError:
        _ENV_ACTION_CLASSES = (_FallbackRedflagEnv, None)
    return _ENV_ACTION_CLASSES


# ── Configuration ────────────────────────────────────────────────────────

def load_env():
    if os.path.exists(".env"):
        with open(".env", "r") as f:
            for line in f:
                if "=" in line and not line.startswith("#"):
                    k, v = line.strip().split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())
load_env()

IMAGE_NAME = os.getenv("IMAGE_NAME")
# Validators inject API_KEY + API_BASE_URL for LiteLLM; those must take precedence over HF_TOKEN
# so all chat completions go through the proxy (HF_TOKEN alone hits the public HF router).
API_KEY = os.environ.get("API_KEY") or os.environ.get("HF_TOKEN")
API_BASE_URL = os.environ.get("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "zai-org/GLM-5"
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")

if not API_KEY:
    print("\n[ERROR] Missing API key for the LLM endpoint.", file=sys.stderr)
    print("Set API_KEY (and API_BASE_URL) from the environment, or HF_TOKEN for local HF Router use.\n", file=sys.stderr)
    exit(1)

from pathlib import Path

BENCHMARK = "redflag_env"
TASKS_DIR = Path(__file__).parent / "tasks"
TASKS = [
    d.name for d in TASKS_DIR.iterdir()
    if d.is_dir() and (d / "golden.json").exists()
]
MAX_STEPS = 12
TEMPERATURE = 0.3
MAX_TOKENS = 2048
SUCCESS_SCORE_THRESHOLD = 0.6

SYSTEM_PROMPT = textwrap.dedent("""
You are a senior software engineer doing code review.
You will receive a code diff (pull request). Your job is to identify bugs,
security issues, and logic errors.

For each issue you find, respond with EXACTLY one JSON object per message.

### Required Structure:
{
  "reasoning": "<step-by-step logic>",
  "action_type": "comment",
  "comment": {
    "file": "<filename>",
    "line": <line_number>,
    "issue_type": "bug|security|logic",
    "severity": "critical|major|minor",
    "body": "<explanation>",
    "suggestion": "<fix>"
  }
}

### Rules:
- Be precise. Only comment on lines that have real issues.
- Keep the 'reasoning' and 'body' fields CONCISE (under 50 words each).
- When finished, use {"action_type": "request_changes"}.
""").strip()


import re

def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Robustly extract the first JSON object from text."""
    # Try literal match first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find the outermost { } block
    start = text.find('{')
    end = text.rfind('}')
    
    if start != -1 and end != -1 and end > start:
        json_candidates = []
        # Try best effort from start to end
        json_candidates.append(text[start:end+1])
        
        for candidate in json_candidates:
            try:
                # Basic cleanup: remove common markdown markers if they were inside found block
                cleaned = candidate.strip()
                return json.loads(cleaned)
            except json.JSONDecodeError:
                continue

    return None


CRITIC_SYSTEM_PROMPT = textwrap.dedent("""
You are an expert Review Supervisor.
Your job is to double-check a junior reviewer's proposed comment.

You will be given the original PR diff and Context, followed by the junior's Proposed Action.

Your tasks:
1. Verify the file and line number cited truly exist and have the problem indicated.
2. Ensure the vulnerability is a genuine flaw, not hallucinated or over-exaggerated.

Respond with EXACTLY ONE JSON object:
{
  "approved": true|false,
  "feedback": "<If approved, leave empty. If false, strictly explain why so the reviewer can correct it>"
}
""").strip()


# ── Logging ──────────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int, action: str, reward: float, done: bool, error: Optional[str]
) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={done_val} error={error_val}",
        flush=True,
    )


def log_end(
    success: bool, steps: int, rewards: List[float]
) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"rewards={rewards_str}",
        flush=True,
    )


# ── LLM Interface ───────────────────────────────────────────────────────

def call_llm_with_fallback(client: Any, messages: list, temperature: float, max_tokens: int) -> str:
    """Invokes LLM and automatically falls back to secondary models if Pro tier quota is depleted."""
    models = [
        MODEL_NAME,
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "XiaomiMiMo/MiMo-V2-Flash",
        "Qwen/Qwen2.5-7B-Instruct"
    ]
    for i, m in enumerate(models):
        try:
            completion = client.chat.completions.create(
                model=m,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
            )
            return (completion.choices[0].message.content or "").strip()
        except Exception as exc:
            err_msg = str(exc).lower()
            if "402" in err_msg or "429" in err_msg or "depleted" in err_msg:
                print(f"[DEBUG] Model {m} hit quota. Falling back to alternative...", file=sys.stderr, flush=True)
                continue
            # If it's a different error but we have more models, also try to fallback
            if i < len(models) - 1:
                print(f"[DEBUG] Model {m} failed: {exc}. Trying next...", file=sys.stderr, flush=True)
                continue
            raise exc
    raise Exception("All fallback models failed due to quota limits or errors.")

def build_user_prompt(
    observation: Dict[str, Any], last_reward: float = 0.0, last_error: Optional[str] = None, critic_feedback: Optional[str] = None
) -> str:
    """Build the user prompt from the environment observation."""
    diff_text = ""
    for hunk in observation.get("diff_hunks", []):
        if isinstance(hunk, dict):
            diff_text += f"\n--- {hunk.get('file', '')} ---\n{hunk.get('content', '')}\n"
        else:
            diff_text += f"\n--- {hunk.file} ---\n{hunk.content}\n"

    context_text = ""
    context_snippets = observation.get("context_snippets", {})
    if context_snippets:
        context_text = "\n\nAdditional context:\n"
        for key, snippet in context_snippets.items():
            context_text += f"\n--- {key} ---\n{snippet}\n"

    comments_text = ""
    comments = observation.get("comments_so_far", [])
    if comments:
        comments_text = "\n\nComments already posted:\n"
        for c in comments:
            if isinstance(c, dict):
                comments_text += f"- {c.get('file', '')}:{c.get('line', '')} [{c.get('issue_type', '')}] {c.get('body', '')}\n"
            else:
                comments_text += f"- {c.file}:{c.line} [{c.issue_type}] {c.body}\n"

    pr_title = observation.get("pr_title", "")
    pr_desc = observation.get("pr_description", "")
    step = observation.get("step", 0)
    max_steps = observation.get("max_steps", 8)

    feedback_text = ""
    if critic_feedback:
        feedback_text += f"\n\n[CRITIC SUPERVISOR REJECTED LAST PROPOSAL]\n{critic_feedback}\nPlease try again and fix the issue.\n"
    elif step > 0:
        feedback_text += f"\n\n[ENVIRONMENT FEEDBACK]\nYour last action gave a reward of {last_reward:.2f}.\n"
        if last_error:
            feedback_text += f"Error from last action: {last_error}\n"
        feedback_text += "(If reward <= 0, your comment was a false positive, a duplicate, or invalid. Adjust your strategy!)\n"

    return textwrap.dedent(f"""
PR Title: {pr_title}
PR Description: {pr_desc}
Step: {step}/{max_steps}

Code Diff:
{diff_text}
{context_text}
{comments_text}{feedback_text}

Identify the next issue in this diff, request context, or if you've found all issues, respond with {{"action_type": "request_changes"}}.
""").strip()


def get_model_action(
    client: Any,
    observation: Dict[str, Any],
    history: List[Dict],
    last_reward: float = 0.0,
    last_error: Optional[str] = None,
    critic_feedback: Optional[str] = None,
) -> Dict[str, Any]:
    """Call the LLM and parse its response into an action dict."""
    user_prompt = build_user_prompt(observation, last_reward, last_error, critic_feedback)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Add conversation history (last 4 exchanges)
    for h in history[-4:]:
        messages.append({"role": "user", "content": h["prompt"]})
        messages.append({"role": "assistant", "content": h["response"]})

    messages.append({"role": "user", "content": user_prompt})

    try:
        text = call_llm_with_fallback(client, messages, TEMPERATURE, MAX_TOKENS)

        action = extract_json(text)
        if action:
            # Healing: If model flattened the comment into top level
            if "file" in action and "line" in action and "action_type" not in action:
                action = {
                    "reasoning": action.get("reasoning", ""),
                    "action_type": "comment",
                    "comment": {
                        "file": action.get("file"),
                        "line": action.get("line"),
                        "issue_type": action.get("issue_type", "bug"),
                        "severity": action.get("severity", "major"),
                        "body": action.get("body", action.get("reasoning", "Issue found.")),
                        "suggestion": action.get("suggestion")
                    }
                }
            return action, text, user_prompt
        
        raise json.JSONDecodeError("No valid JSON found", text, 0)

    except json.JSONDecodeError:
        # If LLM returns non-JSON, try to end the episode
        print(f"[DEBUG] Failed to parse LLM response as JSON: {text[:200]}", file=sys.stderr, flush=True)
        return {"action_type": "request_changes"}, text, user_prompt
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", file=sys.stderr, flush=True)
        return {"action_type": "request_changes"}, str(exc), user_prompt


def get_critic_approval(client: Any, user_prompt: str, proposed_action: str) -> tuple[bool, str]:
    """Ask the critic model to verify the proposed action."""
    content = user_prompt + "\n\n--- JUNIOR PROPOSED ACTION ---\n" + proposed_action
    try:
        text = call_llm_with_fallback(
            client,
            [
                {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
                {"role": "user", "content": content}
            ],
            0.1,
            500
        )
        data = extract_json(text)
        if data:
            return data.get("approved", False), data.get("feedback", "")
        
        return True, ""  # Default to allow if critic returns non-JSON
    except Exception as e:
        print(f"[DEBUG CRITIC] Failed to parse critic: {e}", file=sys.stderr, flush=True)
        return True, ""  # Default to allow if critic crashes


# ── Environment Interaction ───────────────────────────────────────────────

def run_task(client: Any, base_url: str, task_id: str) -> Dict[str, Any]:
    """Run a single task episode and return results."""
    RedflagEnv, RedflagAction = _load_redflag_env_and_action()
    rewards: List[float] = []
    steps_taken = 0
    history: List[Dict] = []
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        with RedflagEnv(base_url=base_url).sync() as env:
            # Reset environment
            reset_result = env.reset(task_id=task_id)
            observation = reset_result.observation.model_dump()
            done = reset_result.done
            reward = 0.0
            error = None

            for step in range(1, MAX_STEPS + 1):
                if done:
                    break

                critic_feedback = None
                max_critic_retries = 2
                
                for attempt in range(max_critic_retries):
                    # Get action from LLM
                    action_dict, raw_response, prompt = get_model_action(
                        client, observation, history, reward, error, critic_feedback
                    )
                    
                    if action_dict.get("action_type") != "comment":
                        break # Only verify comments
                        
                    is_approved, critique = get_critic_approval(client, prompt, raw_response)
                    if is_approved:
                        break
                        
                    print(f"[CRITIC REJECTED] {critique}", file=sys.stderr, flush=True)
                    critic_feedback = critique

                history.append({"prompt": prompt, "response": raw_response})

                # Convert dict to action model (pydantic) or stdlib stand-in
                try:
                    if RedflagAction is not None:
                        action_obj = RedflagAction.model_validate(action_dict)
                    else:
                        action_obj = _SimpleAction(action_dict)
                except Exception as e:
                    print(f"[DEBUG] Validation error: {e}", file=sys.stderr)
                    if RedflagAction is not None:
                        action_obj = RedflagAction(action_type="request_changes")
                    else:
                        action_obj = _SimpleAction({"action_type": "request_changes"})
                    action_dict = {"action_type": "request_changes"}

                # Send action to environment
                step_result = env.step(action_obj)

                observation = step_result.observation.model_dump()
                reward = step_result.reward or 0.0
                done = step_result.done
                error = observation.get("last_action_error")

                rewards.append(reward)
                steps_taken = step

                # Format action string for logging
                action_str = action_dict.get("action_type", "unknown")
                if action_dict.get("comment"):
                    c = action_dict["comment"]
                    action_str = f"comment({c.get('file', '')}:{c.get('line', '')})"

                log_step(
                    step=step,
                    action=action_str,
                    reward=reward,
                    done=done,
                    error=error,
                )

                if done:
                    break
        
        # Compute final success
        max_possible = 1.7
        score = sum(rewards) / max_possible if max_possible > 0 else 0.0
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"Task {task_id} error: {e}", file=sys.stderr, flush=True)
    finally:
        log_end(success=success, steps=steps_taken, rewards=rewards)

    return {
        "task_id": task_id,
        "success": success,
        "steps": steps_taken,
        "rewards": rewards,
    }


def main() -> None:
    """Run inference on all tasks."""
    if OpenAI is not None:
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    else:
        client = _StdlibOpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    base_url = ENV_BASE_URL

    print(f"Using model: {MODEL_NAME}", file=sys.stderr, flush=True)
    print(f"Using LLM base_url: {API_BASE_URL}", file=sys.stderr, flush=True)
    print(f"Using env: {base_url}", file=sys.stderr, flush=True)

    for task_id in TASKS:
        run_task(client, base_url, task_id)


if __name__ == "__main__":
    main()

