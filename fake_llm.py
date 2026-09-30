#!/usr/bin/env python3
"""Ollama / Qwen 互換のテスト用フェイク LLM サーバ。

実 LLM を呼ばずに、Ollama と OpenAI 互換の両 API を話す決定論的なサーバを立てる。
標準ライブラリのみで動くので、CI でも `python fake_llm.py` だけで起動できる。

    # 単体起動（既定は Ollama と同じ 11434 番）
    python fake_llm.py
    python fake_llm.py --port 8080 --model qwen2.5-coder:7b --reply "OK"
    python fake_llm.py --script fake_llm_script.example.json

    # テストから使う
    from fake_llm import FakeLLMServer
    with FakeLLMServer(script={"queue": [{"content": "hi"}]}) as srv:
        os.environ["OPENAI_BASE_URL"] = srv.openai_base_url   # .../v1
        os.environ["OLLAMA_HOST"] = srv.base_url
        ...
        assert srv.requests[-1]["body"]["messages"][-1]["content"] == "..."

応答の決まり方（先に当たったものを使う）:
    1. script の "queue" に残りがあれば先頭から消費する
    2. script の "rules" の正規表現が直近の user メッセージに当たればそれを返す
    3. どちらも無ければ、入力から決定論的に組み立てたエコー応答を返す

対応エンドポイント:
    Ollama         GET  /                     ヘルスチェック
                   GET  /api/version
                   GET  /api/tags             モデル一覧
                   GET  /api/ps
                   POST /api/show
                   POST /api/chat             stream: NDJSON / 非 stream: JSON
                   POST /api/generate
                   POST /api/embed, /api/embeddings
    OpenAI 互換    GET  /v1/models
                   POST /v1/chat/completions  stream: SSE / 非 stream: JSON
                   POST /v1/completions
                   POST /v1/embeddings
    検査用         GET  /_fake/state, /_fake/requests
                   POST /_fake/script         実行中に応答を差し替える
                   POST /_fake/reset          キューと記録を初期状態に戻す
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Iterator, Sequence

__all__ = ["Reply", "FakeEngine", "FakeLLMServer", "serve", "main"]

DEFAULT_PORT = 11434  # Ollama の既定ポート
DEFAULT_MODEL = "qwen3:8b"
DEFAULT_MODELS = ("qwen3:8b", "qwen2.5-coder:7b", "qwen3-embedding:0.6b")
DEFAULT_EMBEDDING_DIM = 1024


# --------------------------------------------------------------------------- #
# 応答の定義
# --------------------------------------------------------------------------- #


@dataclass
class Reply:
    """1 リクエストぶんのフェイク応答。

    script の JSON では、この dataclass のフィールド名をそのままキーに使う。
        {"content": "hello", "thinking": "...", "delay_ms": 50}
        {"tool_calls": [{"name": "get_weather", "arguments": {"city": "Tokyo"}}]}
        {"status": 429, "error": "rate limit exceeded"}
    """

    content: str = ""
    thinking: str = ""  # qwen3 の思考ブロック相当（Ollama: message.thinking）
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str = ""  # 空なら tool_calls の有無から決める
    delay_ms: int = 0  # 応答前に待つ（タイムアウト試験用）
    chunk_size: int = 8  # ストリーミング 1 チャンクの文字数
    status: int = 200  # 200 以外ならエラー応答を返す
    error: str = ""

    @classmethod
    def from_obj(cls, obj: Any) -> "Reply":
        """文字列または dict を Reply にする。未知のキーは弾く。"""
        if isinstance(obj, str):
            return cls(content=obj)
        if isinstance(obj, Reply):
            return obj
        if not isinstance(obj, dict):
            raise ValueError(f"応答は str か dict で指定してください: {obj!r}")

        known = {f for f in cls.__dataclass_fields__}
        unknown = set(obj) - known
        if unknown:
            raise ValueError(
                f"未知のキー {sorted(unknown)} （使えるのは {sorted(known)}）"
            )
        kwargs = dict(obj)
        calls = [_normalize_tool_call(c) for c in kwargs.pop("tool_calls", [])]
        return cls(tool_calls=calls, **kwargs)

    @property
    def effective_finish_reason(self) -> str:
        if self.finish_reason:
            return self.finish_reason
        return "tool_calls" if self.tool_calls else "stop"

    @property
    def is_error(self) -> bool:
        return self.status != 200


def _normalize_tool_call(call: dict[str, Any]) -> dict[str, Any]:
    """tool_call 指定を {"name", "arguments"} に正規化する。

    OpenAI 形式（function.arguments が JSON 文字列）と素の形式の両方を受ける。
    """
    if "function" in call:
        fn = call["function"]
        name, args = fn.get("name", ""), fn.get("arguments", {})
    else:
        name, args = call.get("name", ""), call.get("arguments", {})
    if isinstance(args, str):
        args = json.loads(args) if args.strip() else {}
    if not name:
        raise ValueError(f"tool_call に name がありません: {call!r}")
    return {"id": call.get("id") or "", "name": name, "arguments": args}


# --------------------------------------------------------------------------- #
# 応答エンジン
# --------------------------------------------------------------------------- #


class FakeEngine:
    """リクエストを受けて決定論的に Reply を決め、内容を記録する。

    HTTP を介さず単体でも使えるので、サーバを立てたくないテストではこれを直接呼ぶ。
    """

    def __init__(
        self,
        script: dict[str, Any] | None = None,
        *,
        default_model: str = DEFAULT_MODEL,
        models: Sequence[str] = DEFAULT_MODELS,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        latency_ms: int = 0,
    ) -> None:
        self.default_model = default_model
        self.models = list(dict.fromkeys([default_model, *models]))
        self.embedding_dim = embedding_dim
        self.latency_ms = latency_ms
        self.requests: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._initial_script = _validate_script(script or {})
        self.load_script(self._initial_script)

    # -- script -------------------------------------------------------------

    def load_script(self, script: dict[str, Any]) -> None:
        """応答定義を入れ替える。キューは先頭から積み直される。"""
        script = _validate_script(script)
        with self._lock:
            self._queue = [Reply.from_obj(o) for o in script.get("queue", [])]
            self._rules = [
                (
                    re.compile(r["match"], _regex_flags(r.get("flags", ""))),
                    Reply.from_obj(r["response"]),
                )
                for r in script.get("rules", [])
            ]
            default = script.get("default")
            self._default = Reply.from_obj(default) if default is not None else None

    def reset(self) -> None:
        """キューと記録を初期状態に戻す。テストの teardown 用。"""
        with self._lock:
            self.requests.clear()
        self.load_script(self._initial_script)

    @property
    def pending(self) -> int:
        with self._lock:
            return len(self._queue)

    # -- 記録 ---------------------------------------------------------------

    def record(self, path: str, body: Any) -> None:
        with self._lock:
            self.requests.append(
                {"seq": len(self.requests), "path": path, "body": body}
            )

    # -- 応答決定 -----------------------------------------------------------

    def next_reply(self, messages: Sequence[dict[str, Any]]) -> Reply:
        """会話履歴から次の応答を決める。queue → rules → 既定エコーの順。"""
        with self._lock:
            if self._queue:
                reply = self._queue.pop(0)
            else:
                reply = self._match_rules(messages) or self._default or _echo(
                    messages, self.default_model
                )

        wait_ms = reply.delay_ms or self.latency_ms
        if wait_ms > 0:
            time.sleep(wait_ms / 1000.0)
        return reply

    def _match_rules(self, messages: Sequence[dict[str, Any]]) -> Reply | None:
        text = last_user_text(messages)
        for pattern, reply in self._rules:
            if pattern.search(text):
                return reply
        return None

    # -- 埋め込み -----------------------------------------------------------

    def embed(self, text: str) -> list[float]:
        """入力から決定論的に作る単位ベクトル。同じ文字列なら常に同じ値。"""
        dim = self.embedding_dim
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        # 必要な長さになるまでハッシュを鎖状に伸ばす
        raw = bytearray(digest)
        while len(raw) < dim * 2:
            digest = hashlib.sha256(digest).digest()
            raw.extend(digest)
        vec = [
            int.from_bytes(raw[i * 2 : i * 2 + 2], "big") / 32767.5 - 1.0
            for i in range(dim)
        ]
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


def _regex_flags(spec: str) -> int:
    flags = 0
    for ch in spec:
        if ch == "i":
            flags |= re.IGNORECASE
        elif ch == "m":
            flags |= re.MULTILINE
        elif ch == "s":
            flags |= re.DOTALL
        else:
            raise ValueError(f"未知の正規表現フラグ: {ch!r} （使えるのは i, m, s）")
    return flags


def _validate_script(script: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(script, dict):
        raise ValueError("script はオブジェクトで指定してください")
    unknown = set(script) - {"queue", "rules", "default"}
    if unknown:
        raise ValueError(
            f"script に未知のキー {sorted(unknown)} （使えるのは queue, rules, default）"
        )
    for rule in script.get("rules", []):
        if "match" not in rule or "response" not in rule:
            raise ValueError(f"rule には match と response が必要です: {rule!r}")
    return script


def message_text(message: dict[str, Any]) -> str:
    """content が文字列でも OpenAI のパート配列でもテキストを取り出す。"""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            p.get("text", "")
            for p in content
            if isinstance(p, dict) and p.get("type") in (None, "text", "input_text")
        ]
        return "".join(parts)
    return "" if content is None else str(content)


def last_user_text(messages: Sequence[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return message_text(message)
    return message_text(messages[-1]) if messages else ""


def _echo(messages: Sequence[dict[str, Any]], model: str) -> Reply:
    """script が何も指定していないときの既定応答。

    入力のダイジェストを含めるので、テスト側で「どの入力への応答か」を照合できる。
    """
    text = last_user_text(messages)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
    head = text if len(text) <= 200 else text[:197] + "..."
    return Reply(
        content=(
            f"[fake-llm:{model}] messages={len(messages)} digest={digest}\n"
            f"echo: {head}"
        )
    )


def count_tokens(text: str) -> int:
    """おおまかで決定論的なトークン数。usage の値を assert できれば十分。"""
    return len(re.findall(r"\w+|[^\w\s]", text)) or (1 if text else 0)


def chunk_text(text: str, size: int) -> Iterator[str]:
    size = max(1, size)
    for i in range(0, len(text), size):
        yield text[i : i + size]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


# --------------------------------------------------------------------------- #
# HTTP ハンドラ
# --------------------------------------------------------------------------- #


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "fake-llm/1.0"

    # server 側で差し込む
    engine: FakeEngine
    verbose: bool

    # -- ルーティング -------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler の規約
        path = self._path()
        routes = {
            "/": lambda: self._send_text("Ollama is running"),
            "/api/version": lambda: self._send_json({"version": "0.5.0-fake"}),
            "/api/tags": self._ollama_tags,
            "/api/ps": self._ollama_ps,
            "/v1/models": self._openai_models,
            "/_fake/state": self._fake_state,
            "/_fake/requests": self._fake_requests,
        }
        handler = routes.get(path)
        if handler is None:
            self._send_error(404, f"unknown path: {path}")
            return
        handler()

    def do_POST(self) -> None:  # noqa: N802
        path = self._path()
        routes = {
            "/api/chat": self._ollama_chat,
            "/api/generate": self._ollama_generate,
            "/api/show": self._ollama_show,
            "/api/embed": self._ollama_embed,
            "/api/embeddings": self._ollama_embeddings_legacy,
            "/v1/chat/completions": self._openai_chat,
            "/v1/completions": self._openai_completions,
            "/v1/embeddings": self._openai_embeddings,
            "/_fake/script": self._fake_set_script,
            "/_fake/reset": self._fake_reset,
        }
        handler = routes.get(path)
        if handler is None:
            self._send_error(404, f"unknown path: {path}")
            return

        try:
            body = self._read_json()
        except ValueError as exc:
            self._send_error(400, f"invalid JSON body: {exc}")
            return

        if not path.startswith("/_fake/"):
            self.engine.record(path, body)
        try:
            handler(body)
        except ValueError as exc:
            self._send_error(400, str(exc))

    def do_DELETE(self) -> None:  # noqa: N802
        if self._path() == "/_fake/requests":
            self.engine.reset()
            self._send_json({"ok": True})
        else:
            self._send_error(404, f"unknown path: {self._path()}")

    def _path(self) -> str:
        path = self.path.split("?", 1)[0]
        return path.rstrip("/") or "/"

    # -- Ollama -------------------------------------------------------------

    def _ollama_tags(self) -> None:
        self._send_json({"models": [self._model_entry(m) for m in self.engine.models]})

    def _ollama_ps(self) -> None:
        entry = self._model_entry(self.engine.default_model)
        entry["expires_at"] = "2099-01-01T00:00:00Z"
        entry["size_vram"] = entry["size"]
        self._send_json({"models": [entry]})

    def _model_entry(self, name: str) -> dict[str, Any]:
        family = name.split(":", 1)[0].split("-", 1)[0]
        return {
            "name": name,
            "model": name,
            "modified_at": "2026-01-01T00:00:00Z",
            "size": 4_000_000_000,
            "digest": hashlib.sha256(name.encode()).hexdigest(),
            "details": {
                "parent_model": "",
                "format": "gguf",
                "family": family,
                "families": [family],
                "parameter_size": "8.2B",
                "quantization_level": "Q4_K_M",
            },
        }

    def _ollama_show(self, body: dict[str, Any]) -> None:
        name = body.get("model") or body.get("name") or self.engine.default_model
        entry = self._model_entry(name)
        self._send_json(
            {
                "license": "fake",
                "modelfile": f"FROM {name}",
                "parameters": "stop \"<|im_end|>\"",
                "template": "{{ .Prompt }}",
                "details": entry["details"],
                "model_info": {"general.architecture": entry["details"]["family"]},
                "capabilities": ["completion", "tools", "thinking"],
            }
        )

    def _ollama_chat(self, body: dict[str, Any]) -> None:
        messages = body.get("messages") or []
        model = body.get("model") or self.engine.default_model
        reply = self.engine.next_reply(messages)
        if reply.is_error:
            self._send_error(reply.status, reply.error or "fake error")
            return

        prompt_tokens = sum(count_tokens(message_text(m)) for m in messages)
        stream = body.get("stream", True)  # Ollama の既定は stream=True

        if not stream:
            message: dict[str, Any] = {"role": "assistant", "content": reply.content}
            if reply.thinking:
                message["thinking"] = reply.thinking
            if reply.tool_calls:
                message["tool_calls"] = [
                    {"function": {"name": c["name"], "arguments": c["arguments"]}}
                    for c in reply.tool_calls
                ]
            self._send_json(
                {
                    "model": model,
                    "created_at": _now_iso(),
                    "message": message,
                    "done": True,
                    "done_reason": reply.effective_finish_reason,
                    **_ollama_stats(prompt_tokens, count_tokens(reply.content)),
                }
            )
            return

        self._begin_stream("application/x-ndjson")
        base = {"model": model, "created_at": _now_iso()}
        if reply.thinking:
            for piece in chunk_text(reply.thinking, reply.chunk_size):
                self._write_chunk(
                    json.dumps(
                        {
                            **base,
                            "message": {
                                "role": "assistant",
                                "content": "",
                                "thinking": piece,
                            },
                            "done": False,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        for piece in chunk_text(reply.content, reply.chunk_size):
            self._write_chunk(
                json.dumps(
                    {
                        **base,
                        "message": {"role": "assistant", "content": piece},
                        "done": False,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        final_message: dict[str, Any] = {"role": "assistant", "content": ""}
        if reply.tool_calls:
            final_message["tool_calls"] = [
                {"function": {"name": c["name"], "arguments": c["arguments"]}}
                for c in reply.tool_calls
            ]
        self._write_chunk(
            json.dumps(
                {
                    **base,
                    "message": final_message,
                    "done": True,
                    "done_reason": reply.effective_finish_reason,
                    **_ollama_stats(prompt_tokens, count_tokens(reply.content)),
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        self._end_stream()

    def _ollama_generate(self, body: dict[str, Any]) -> None:
        prompt = body.get("prompt", "")
        model = body.get("model") or self.engine.default_model
        messages = [{"role": "user", "content": prompt}]
        if body.get("system"):
            messages.insert(0, {"role": "system", "content": body["system"]})
        reply = self.engine.next_reply(messages)
        if reply.is_error:
            self._send_error(reply.status, reply.error or "fake error")
            return

        stats = _ollama_stats(count_tokens(prompt), count_tokens(reply.content))
        base = {"model": model, "created_at": _now_iso()}
        if not body.get("stream", True):
            self._send_json(
                {
                    **base,
                    "response": reply.content,
                    "done": True,
                    "done_reason": reply.effective_finish_reason,
                    "context": [],
                    **stats,
                }
            )
            return

        self._begin_stream("application/x-ndjson")
        for piece in chunk_text(reply.content, reply.chunk_size):
            self._write_chunk(
                json.dumps({**base, "response": piece, "done": False}, ensure_ascii=False)
                + "\n"
            )
        self._write_chunk(
            json.dumps(
                {
                    **base,
                    "response": "",
                    "done": True,
                    "done_reason": reply.effective_finish_reason,
                    "context": [],
                    **stats,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        self._end_stream()

    def _ollama_embed(self, body: dict[str, Any]) -> None:
        model = body.get("model") or self.engine.default_model
        raw = body.get("input", "")
        inputs = raw if isinstance(raw, list) else [raw]
        vectors = [self.engine.embed(str(t)) for t in inputs]
        self._send_json(
            {
                "model": model,
                "embeddings": vectors,
                "total_duration": 1_000_000,
                "load_duration": 100_000,
                "prompt_eval_count": sum(count_tokens(str(t)) for t in inputs),
            }
        )

    def _ollama_embeddings_legacy(self, body: dict[str, Any]) -> None:
        text = str(body.get("prompt", ""))
        self._send_json({"embedding": self.engine.embed(text)})

    # -- OpenAI 互換 --------------------------------------------------------

    def _openai_models(self) -> None:
        self._send_json(
            {
                "object": "list",
                "data": [
                    {
                        "id": name,
                        "object": "model",
                        "created": 1767225600,
                        "owned_by": "fake-llm",
                    }
                    for name in self.engine.models
                ],
            }
        )

    def _openai_chat(self, body: dict[str, Any]) -> None:
        messages = body.get("messages") or []
        model = body.get("model") or self.engine.default_model
        reply = self.engine.next_reply(messages)
        if reply.is_error:
            self._send_openai_error(reply.status, reply.error or "fake error")
            return

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())
        usage = _openai_usage(
            sum(count_tokens(message_text(m)) for m in messages),
            count_tokens(reply.content),
        )
        calls = [
            {
                "id": c["id"] or f"call_{uuid.uuid4().hex[:20]}",
                "type": "function",
                "function": {
                    "name": c["name"],
                    "arguments": json.dumps(c["arguments"], ensure_ascii=False),
                },
            }
            for c in reply.tool_calls
        ]

        if not body.get("stream", False):  # OpenAI の既定は stream=False
            message: dict[str, Any] = {"role": "assistant", "content": reply.content}
            if reply.thinking:
                message["reasoning_content"] = reply.thinking
            if calls:
                message["tool_calls"] = calls
            self._send_json(
                {
                    "id": completion_id,
                    "object": "chat.completion",
                    "created": created,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "message": message,
                            "logprobs": None,
                            "finish_reason": reply.effective_finish_reason,
                        }
                    ],
                    "usage": usage,
                }
            )
            return

        self._begin_stream("text/event-stream")

        def emit(delta: dict[str, Any], finish: str | None = None) -> None:
            self._write_chunk(
                "data: "
                + json.dumps(
                    {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [
                            {"index": 0, "delta": delta, "finish_reason": finish}
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n\n"
            )

        emit({"role": "assistant", "content": ""})
        for piece in chunk_text(reply.thinking, reply.chunk_size):
            emit({"reasoning_content": piece})
        for piece in chunk_text(reply.content, reply.chunk_size):
            emit({"content": piece})
        for index, call in enumerate(calls):
            emit({"tool_calls": [{"index": index, **call}]})
        emit({}, reply.effective_finish_reason)
        if body.get("stream_options", {}).get("include_usage"):
            self._write_chunk(
                "data: "
                + json.dumps(
                    {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [],
                        "usage": usage,
                    }
                )
                + "\n\n"
            )
        self._write_chunk("data: [DONE]\n\n")
        self._end_stream()

    def _openai_completions(self, body: dict[str, Any]) -> None:
        prompt = body.get("prompt", "")
        if isinstance(prompt, list):
            prompt = "".join(str(p) for p in prompt)
        model = body.get("model") or self.engine.default_model
        reply = self.engine.next_reply([{"role": "user", "content": prompt}])
        if reply.is_error:
            self._send_openai_error(reply.status, reply.error or "fake error")
            return
        self._send_json(
            {
                "id": f"cmpl-{uuid.uuid4().hex[:24]}",
                "object": "text_completion",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "text": reply.content,
                        "logprobs": None,
                        "finish_reason": reply.effective_finish_reason,
                    }
                ],
                "usage": _openai_usage(
                    count_tokens(prompt), count_tokens(reply.content)
                ),
            }
        )

    def _openai_embeddings(self, body: dict[str, Any]) -> None:
        raw = body.get("input", "")
        inputs = raw if isinstance(raw, list) else [raw]
        model = body.get("model") or self.engine.default_model
        prompt_tokens = sum(count_tokens(str(t)) for t in inputs)
        self._send_json(
            {
                "object": "list",
                "data": [
                    {
                        "object": "embedding",
                        "index": i,
                        "embedding": self.engine.embed(str(text)),
                    }
                    for i, text in enumerate(inputs)
                ],
                "model": model,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "total_tokens": prompt_tokens,
                },
            }
        )

    # -- 検査用 -------------------------------------------------------------

    def _fake_state(self) -> None:
        self._send_json(
            {
                "default_model": self.engine.default_model,
                "models": self.engine.models,
                "pending_queue": self.engine.pending,
                "request_count": len(self.engine.requests),
            }
        )

    def _fake_requests(self) -> None:
        self._send_json({"requests": self.engine.requests})

    def _fake_set_script(self, body: dict[str, Any]) -> None:
        self.engine.load_script(body)
        self._send_json({"ok": True, "pending_queue": self.engine.pending})

    def _fake_reset(self, body: dict[str, Any]) -> None:
        self.engine.reset()
        self._send_json({"ok": True})

    # -- 送信ヘルパ ---------------------------------------------------------

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        if not raw:
            return {}
        try:
            body = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(str(exc)) from exc
        if not isinstance(body, dict):
            raise ValueError("body must be a JSON object")
        return body

    def _send_bytes(self, status: int, content_type: str, payload: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_json(self, obj: Any, status: int = 200) -> None:
        self._send_bytes(
            status,
            "application/json",
            json.dumps(obj, ensure_ascii=False).encode("utf-8"),
        )

    def _send_text(self, text: str, status: int = 200) -> None:
        self._send_bytes(status, "text/plain; charset=utf-8", text.encode("utf-8"))

    def _send_error(self, status: int, message: str) -> None:
        self._send_json({"error": message}, status)

    def _send_openai_error(self, status: int, message: str) -> None:
        kind = {400: "invalid_request_error", 429: "rate_limit_error"}.get(
            status, "api_error"
        )
        self._send_json(
            {"error": {"message": message, "type": kind, "code": status}}, status
        )

    def _begin_stream(self, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

    def _write_chunk(self, text: str) -> None:
        payload = text.encode("utf-8")
        self.wfile.write(f"{len(payload):X}\r\n".encode("ascii"))
        self.wfile.write(payload + b"\r\n")
        self.wfile.flush()

    def _end_stream(self) -> None:
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    def log_message(self, fmt: str, *args: Any) -> None:
        if self.verbose:
            super().log_message(fmt, *args)


def _ollama_stats(prompt_tokens: int, eval_tokens: int) -> dict[str, int]:
    return {
        "total_duration": 1_000_000,
        "load_duration": 100_000,
        "prompt_eval_count": prompt_tokens,
        "prompt_eval_duration": 200_000,
        "eval_count": eval_tokens,
        "eval_duration": 700_000,
    }


def _openai_usage(prompt_tokens: int, completion_tokens: int) -> dict[str, int]:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


# --------------------------------------------------------------------------- #
# サーバ
# --------------------------------------------------------------------------- #


class FakeLLMServer:
    """バックグラウンドスレッドでフェイクサーバを動かすコンテキストマネージャ。

    port=0 なら空きポートを OS に選ばせるので、テストの並列実行でも衝突しない。

        with FakeLLMServer(script={"queue": ["hello"]}) as srv:
            print(srv.base_url, srv.openai_base_url)
    """

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        script: dict[str, Any] | None = None,
        default_model: str = DEFAULT_MODEL,
        models: Sequence[str] = DEFAULT_MODELS,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        latency_ms: int = 0,
        verbose: bool = False,
    ) -> None:
        self.engine = FakeEngine(
            script,
            default_model=default_model,
            models=models,
            embedding_dim=embedding_dim,
            latency_ms=latency_ms,
        )
        handler = type(
            "_BoundHandler", (_Handler,), {"engine": self.engine, "verbose": verbose}
        )
        self._httpd = ThreadingHTTPServer((host, port), handler)
        self._httpd.daemon_threads = True
        self._thread: threading.Thread | None = None

    @property
    def host(self) -> str:
        return self._httpd.server_address[0]

    @property
    def port(self) -> int:
        return self._httpd.server_address[1]

    @property
    def base_url(self) -> str:
        """Ollama 互換のベース URL（OLLAMA_HOST に渡す）。"""
        return f"http://{self.host}:{self.port}"

    @property
    def openai_base_url(self) -> str:
        """OpenAI 互換のベース URL（OPENAI_BASE_URL に渡す）。"""
        return f"{self.base_url}/v1"

    @property
    def requests(self) -> list[dict[str, Any]]:
        return self.engine.requests

    def start(self) -> "FakeLLMServer":
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._httpd.serve_forever, name="fake-llm", daemon=True
            )
            self._thread.start()
        return self

    def stop(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def __enter__(self) -> "FakeLLMServer":
        return self.start()

    def __exit__(self, *_exc: object) -> None:
        self.stop()


def serve(**kwargs: Any) -> None:
    """フォアグラウンドで動かす。Ctrl-C で止まる。"""
    server = FakeLLMServer(**kwargs)
    # CI で「起動したか」を待ち受けられるよう即 flush する
    print(f"fake-llm listening on {server.base_url}", flush=True)
    print(f"  Ollama API   : {server.base_url}/api/chat", flush=True)
    print(f"  OpenAI API   : {server.openai_base_url}/chat/completions", flush=True)
    print(f"  default model: {server.engine.default_model}", flush=True)
    server.start()
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        server.stop()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ollama / Qwen 互換のテスト用フェイク LLM サーバ",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="既定モデル名")
    parser.add_argument(
        "--models", nargs="*", default=list(DEFAULT_MODELS), help="一覧に出すモデル名"
    )
    parser.add_argument("--script", help="応答定義 JSON のパス")
    parser.add_argument("--reply", help="常に返す固定応答（--script より弱い）")
    parser.add_argument(
        "--latency-ms", type=int, default=0, help="全応答に足す待ち時間"
    )
    parser.add_argument(
        "--embedding-dim", type=int, default=DEFAULT_EMBEDDING_DIM
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="アクセスログを出す")
    args = parser.parse_args(argv)

    script: dict[str, Any] = {}
    if args.script:
        with open(args.script, encoding="utf-8") as fh:
            script = json.load(fh)
    elif args.reply:
        script = {"default": {"content": args.reply}}

    serve(
        host=args.host,
        port=args.port,
        script=script,
        default_model=args.model,
        models=args.models,
        embedding_dim=args.embedding_dim,
        latency_ms=args.latency_ms,
        verbose=args.verbose,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
