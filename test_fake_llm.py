#!/usr/bin/env python3
"""fake_llm.py の自己テスト。

    python test_fake_llm.py          # 単体で実行
    python -m unittest test_fake_llm
    pytest test_fake_llm.py          # pytest でも収集できる

標準ライブラリのみ。fake_llm を実際に起動して HTTP 越しに叩く。
"""

from __future__ import annotations

import json
import unittest
import urllib.error
import urllib.request
from typing import Any

from fake_llm import FakeEngine, FakeLLMServer, Reply, count_tokens

SERVER: FakeLLMServer


def setUpModule() -> None:  # noqa: N802 - unittest の規約
    global SERVER
    SERVER = FakeLLMServer(port=0).start()


def tearDownModule() -> None:  # noqa: N802
    SERVER.stop()


# --------------------------------------------------------------------------- #
# HTTP ヘルパ
# --------------------------------------------------------------------------- #


def post(path: str, body: dict[str, Any]) -> Any:
    req = urllib.request.Request(
        SERVER.base_url + path,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as res:
        return json.loads(res.read())


def post_stream(path: str, body: dict[str, Any]) -> str:
    req = urllib.request.Request(
        SERVER.base_url + path,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as res:
        return res.read().decode("utf-8")


def get(path: str) -> Any:
    with urllib.request.urlopen(SERVER.base_url + path, timeout=10) as res:
        payload = res.read()
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return payload.decode("utf-8")


def set_script(script: dict[str, Any]) -> None:
    post("/_fake/script", script)


class FakeLLMTestCase(unittest.TestCase):
    """各テストの前に script と記録をまっさらに戻す。"""

    def setUp(self) -> None:
        post("/_fake/reset", {})


# --------------------------------------------------------------------------- #
# エンジン単体（HTTP なし）
# --------------------------------------------------------------------------- #


class TestEngine(unittest.TestCase):
    def test_queue_is_consumed_in_order(self) -> None:
        engine = FakeEngine({"queue": ["one", "two"]})
        messages = [{"role": "user", "content": "x"}]
        self.assertEqual(engine.next_reply(messages).content, "one")
        self.assertEqual(engine.next_reply(messages).content, "two")
        # 枯れたら既定のエコーに落ちる
        self.assertIn("[fake-llm:", engine.next_reply(messages).content)

    def test_rules_match_last_user_message(self) -> None:
        engine = FakeEngine(
            {
                "rules": [
                    {"match": "天気", "response": "晴れ"},
                    {"match": "hello", "flags": "i", "response": "hi"},
                ],
                "default": "fallback",
            }
        )
        self.assertEqual(
            engine.next_reply([{"role": "user", "content": "明日の天気は?"}]).content,
            "晴れ",
        )
        self.assertEqual(
            engine.next_reply([{"role": "user", "content": "HELLO"}]).content, "hi"
        )
        self.assertEqual(
            engine.next_reply([{"role": "user", "content": "その他"}]).content,
            "fallback",
        )

    def test_queue_wins_over_rules(self) -> None:
        engine = FakeEngine(
            {"queue": ["queued"], "rules": [{"match": "x", "response": "ruled"}]}
        )
        messages = [{"role": "user", "content": "x"}]
        self.assertEqual(engine.next_reply(messages).content, "queued")
        self.assertEqual(engine.next_reply(messages).content, "ruled")

    def test_echo_is_deterministic(self) -> None:
        engine = FakeEngine()
        messages = [{"role": "user", "content": "同じ入力"}]
        first = engine.next_reply(messages).content
        self.assertEqual(first, FakeEngine().next_reply(messages).content)
        self.assertNotEqual(
            first, engine.next_reply([{"role": "user", "content": "違う入力"}]).content
        )

    def test_multimodal_content_parts(self) -> None:
        engine = FakeEngine({"rules": [{"match": "画像の説明", "response": "ok"}]})
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "画像の説明"},
                    {"type": "image_url", "image_url": {"url": "data:..."}},
                ],
            }
        ]
        self.assertEqual(engine.next_reply(messages).content, "ok")

    def test_embeddings_are_deterministic_unit_vectors(self) -> None:
        engine = FakeEngine(embedding_dim=64)
        vec = engine.embed("hello")
        self.assertEqual(len(vec), 64)
        self.assertAlmostEqual(sum(v * v for v in vec), 1.0, places=6)
        self.assertEqual(vec, FakeEngine(embedding_dim=64).embed("hello"))
        self.assertNotEqual(vec, engine.embed("world"))

    def test_reply_from_obj_rejects_unknown_keys(self) -> None:
        with self.assertRaises(ValueError):
            Reply.from_obj({"contnet": "typo"})

    def test_invalid_script_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            FakeEngine({"rulez": []})
        with self.assertRaises(ValueError):
            FakeEngine({"rules": [{"match": "x"}]})

    def test_tool_call_accepts_openai_shape(self) -> None:
        reply = Reply.from_obj(
            {
                "tool_calls": [
                    {
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Tokyo"}',
                        }
                    }
                ]
            }
        )
        self.assertEqual(reply.tool_calls[0]["name"], "get_weather")
        self.assertEqual(reply.tool_calls[0]["arguments"], {"city": "Tokyo"})
        self.assertEqual(reply.effective_finish_reason, "tool_calls")

    def test_count_tokens(self) -> None:
        self.assertEqual(count_tokens(""), 0)
        self.assertEqual(count_tokens("hello world"), 2)
        self.assertEqual(count_tokens("hi!"), 2)


# --------------------------------------------------------------------------- #
# Ollama API
# --------------------------------------------------------------------------- #


class TestOllamaAPI(FakeLLMTestCase):
    def test_health_and_version(self) -> None:
        self.assertIn("Ollama", get("/"))
        self.assertIn("version", get("/api/version"))

    def test_tags_lists_qwen_models(self) -> None:
        models = [m["name"] for m in get("/api/tags")["models"]]
        self.assertIn("qwen3:8b", models)
        self.assertTrue(all(m["details"]["format"] == "gguf" for m in get("/api/tags")["models"]))

    def test_show(self) -> None:
        shown = post("/api/show", {"model": "qwen3:8b"})
        self.assertIn("tools", shown["capabilities"])

    def test_chat_non_stream(self) -> None:
        set_script({"queue": ["こんにちは"]})
        res = post(
            "/api/chat",
            {
                "model": "qwen3:8b",
                "messages": [{"role": "user", "content": "hi there"}],
                "stream": False,
            },
        )
        self.assertEqual(res["message"]["content"], "こんにちは")
        self.assertEqual(res["message"]["role"], "assistant")
        self.assertTrue(res["done"])
        self.assertEqual(res["done_reason"], "stop")
        self.assertEqual(res["prompt_eval_count"], 2)

    def test_chat_streams_ndjson_and_finishes_with_done(self) -> None:
        set_script({"queue": [{"content": "abcdefghij", "chunk_size": 3}]})
        raw = post_stream(
            "/api/chat", {"messages": [{"role": "user", "content": "go"}]}
        )
        lines = [json.loads(line) for line in raw.splitlines() if line.strip()]
        self.assertEqual(len(lines), 5)  # 4 チャンク + 終端
        self.assertEqual(
            "".join(line["message"]["content"] for line in lines), "abcdefghij"
        )
        self.assertFalse(any(line["done"] for line in lines[:-1]))
        self.assertTrue(lines[-1]["done"])

    def test_chat_thinking_block(self) -> None:
        set_script({"queue": [{"content": "答え", "thinking": "考え中"}]})
        res = post(
            "/api/chat",
            {"messages": [{"role": "user", "content": "q"}], "stream": False},
        )
        self.assertEqual(res["message"]["thinking"], "考え中")

    def test_chat_tool_calls(self) -> None:
        set_script(
            {
                "queue": [
                    {
                        "tool_calls": [
                            {"name": "get_weather", "arguments": {"city": "Tokyo"}}
                        ]
                    }
                ]
            }
        )
        res = post(
            "/api/chat",
            {"messages": [{"role": "user", "content": "天気"}], "stream": False},
        )
        call = res["message"]["tool_calls"][0]["function"]
        self.assertEqual(call["name"], "get_weather")
        self.assertEqual(call["arguments"], {"city": "Tokyo"})  # Ollama は dict のまま
        self.assertEqual(res["done_reason"], "tool_calls")

    def test_generate(self) -> None:
        set_script({"default": "生成結果"})
        res = post(
            "/api/generate", {"model": "qwen3:8b", "prompt": "書いて", "stream": False}
        )
        self.assertEqual(res["response"], "生成結果")

    def test_embed_and_legacy_embeddings(self) -> None:
        res = post("/api/embed", {"input": ["a", "b"]})
        self.assertEqual(len(res["embeddings"]), 2)
        self.assertEqual(len(res["embeddings"][0]), 1024)

        legacy = post("/api/embeddings", {"prompt": "a"})
        self.assertEqual(legacy["embedding"], res["embeddings"][0])


# --------------------------------------------------------------------------- #
# OpenAI 互換 API
# --------------------------------------------------------------------------- #


class TestOpenAICompatAPI(FakeLLMTestCase):
    def test_models(self) -> None:
        res = get("/v1/models")
        self.assertEqual(res["object"], "list")
        self.assertIn("qwen3:8b", [m["id"] for m in res["data"]])

    def test_chat_completion(self) -> None:
        set_script({"queue": ["hello world"]})
        res = post(
            "/v1/chat/completions",
            {
                "model": "qwen3:8b",
                "messages": [
                    {"role": "system", "content": "you are a test"},
                    {"role": "user", "content": "hi"},
                ],
            },
        )
        choice = res["choices"][0]
        self.assertEqual(res["object"], "chat.completion")
        self.assertTrue(res["id"].startswith("chatcmpl-"))
        self.assertEqual(choice["message"]["content"], "hello world")
        self.assertEqual(choice["finish_reason"], "stop")
        self.assertEqual(res["usage"]["completion_tokens"], 2)
        self.assertEqual(res["usage"]["prompt_tokens"], 5)
        self.assertEqual(
            res["usage"]["total_tokens"],
            res["usage"]["prompt_tokens"] + res["usage"]["completion_tokens"],
        )

    def test_chat_completion_stream_sse(self) -> None:
        set_script({"queue": [{"content": "abcdef", "chunk_size": 2}]})
        raw = post_stream(
            "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "go"}], "stream": True},
        )
        events = [
            line[len("data: ") :]
            for line in raw.splitlines()
            if line.startswith("data: ")
        ]
        self.assertEqual(events[-1], "[DONE]")
        chunks = [json.loads(e) for e in events[:-1]]
        self.assertTrue(all(c["object"] == "chat.completion.chunk" for c in chunks))
        text = "".join(
            c["choices"][0]["delta"].get("content", "") for c in chunks
        )
        self.assertEqual(text, "abcdef")
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "stop")

    def test_stream_options_include_usage(self) -> None:
        set_script({"queue": ["ok"]})
        raw = post_stream(
            "/v1/chat/completions",
            {
                "messages": [{"role": "user", "content": "go"}],
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        )
        events = [
            json.loads(line[len("data: ") :])
            for line in raw.splitlines()
            if line.startswith("data: ") and not line.endswith("[DONE]")
        ]
        self.assertIn("usage", events[-1])
        self.assertEqual(events[-1]["choices"], [])

    def test_tool_calls_are_json_encoded_strings(self) -> None:
        set_script(
            {
                "queue": [
                    {
                        "tool_calls": [
                            {"name": "get_weather", "arguments": {"city": "Tokyo"}}
                        ]
                    }
                ]
            }
        )
        res = post(
            "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "天気"}]},
        )
        choice = res["choices"][0]
        call = choice["message"]["tool_calls"][0]
        self.assertEqual(call["type"], "function")
        self.assertTrue(call["id"].startswith("call_"))
        self.assertEqual(json.loads(call["function"]["arguments"]), {"city": "Tokyo"})
        self.assertEqual(choice["finish_reason"], "tool_calls")

    def test_reasoning_content(self) -> None:
        set_script({"queue": [{"content": "答え", "thinking": "考え中"}]})
        res = post("/v1/chat/completions", {"messages": [{"role": "user", "content": "q"}]})
        self.assertEqual(res["choices"][0]["message"]["reasoning_content"], "考え中")

    def test_completions(self) -> None:
        set_script({"default": "text out"})
        res = post("/v1/completions", {"prompt": "in"})
        self.assertEqual(res["choices"][0]["text"], "text out")

    def test_embeddings(self) -> None:
        res = post("/v1/embeddings", {"input": ["a", "b"], "model": "qwen3-embedding:0.6b"})
        self.assertEqual([d["index"] for d in res["data"]], [0, 1])
        self.assertEqual(len(res["data"][0]["embedding"]), 1024)


# --------------------------------------------------------------------------- #
# エラー系と検査用エンドポイント
# --------------------------------------------------------------------------- #


class TestErrorsAndIntrospection(FakeLLMTestCase):
    def test_scripted_error_status(self) -> None:
        set_script({"queue": [{"status": 429, "error": "rate limit exceeded"}]})
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            post("/v1/chat/completions", {"messages": [{"role": "user", "content": "x"}]})
        self.assertEqual(ctx.exception.code, 429)
        body = json.loads(ctx.exception.read())
        self.assertEqual(body["error"]["type"], "rate_limit_error")
        self.assertEqual(body["error"]["message"], "rate limit exceeded")

    def test_ollama_scripted_error(self) -> None:
        set_script({"queue": [{"status": 500, "error": "boom"}]})
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            post("/api/chat", {"messages": [{"role": "user", "content": "x"}], "stream": False})
        self.assertEqual(ctx.exception.code, 500)

    def test_unknown_path_is_404(self) -> None:
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            get("/api/nope")
        self.assertEqual(ctx.exception.code, 404)

    def test_invalid_json_is_400(self) -> None:
        req = urllib.request.Request(
            SERVER.base_url + "/v1/chat/completions",
            data=b"{not json",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(req, timeout=10)
        self.assertEqual(ctx.exception.code, 400)

    def test_bad_script_is_400(self) -> None:
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            post("/_fake/script", {"rules": [{"match": "x"}]})
        self.assertEqual(ctx.exception.code, 400)

    def test_requests_are_recorded(self) -> None:
        post("/v1/chat/completions", {"messages": [{"role": "user", "content": "一件目"}]})
        post("/api/chat", {"messages": [{"role": "user", "content": "二件目"}], "stream": False})
        recorded = get("/_fake/requests")["requests"]
        self.assertEqual([r["path"] for r in recorded], ["/v1/chat/completions", "/api/chat"])
        self.assertEqual(recorded[0]["body"]["messages"][0]["content"], "一件目")
        self.assertEqual([r["seq"] for r in recorded], [0, 1])

        # 検査用エンドポイント自身は記録しない
        self.assertEqual(get("/_fake/state")["request_count"], 2)

    def test_reset_clears_queue_and_records(self) -> None:
        set_script({"queue": ["a", "b"]})
        self.assertEqual(get("/_fake/state")["pending_queue"], 2)
        post("/v1/chat/completions", {"messages": [{"role": "user", "content": "x"}]})
        self.assertEqual(get("/_fake/state")["pending_queue"], 1)

        post("/_fake/reset", {})
        state = get("/_fake/state")
        self.assertEqual(state["request_count"], 0)
        self.assertEqual(state["pending_queue"], 0)  # 起動時の script は空だった


class TestServerLifecycle(unittest.TestCase):
    def test_context_manager_picks_free_port(self) -> None:
        with FakeLLMServer(port=0, default_model="qwen2.5-coder:7b") as srv:
            self.assertGreater(srv.port, 0)
            self.assertTrue(srv.openai_base_url.endswith("/v1"))
            with urllib.request.urlopen(srv.base_url + "/api/version", timeout=10) as res:
                self.assertEqual(res.status, 200)
            self.assertEqual(srv.engine.models[0], "qwen2.5-coder:7b")


if __name__ == "__main__":
    unittest.main(verbosity=2)
