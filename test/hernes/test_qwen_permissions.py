#!/usr/bin/env python3
"""`.qwen/settings.json` の permissions が実際に効くかを検証するテスト。

静的検査（スキーマ）と E2E（qwen CLI を実際に起動）の 2 段構え。
E2E はリポジトリ直下の `fake_llm.py` をモデルの代わりに立てるので、
本物の LLM も API キーも要らず、tool call を狙いどおりに発火できる。

    仕組み:
      qwen CLI ──OpenAI 互換──> fake_llm.py
                                  └─ read_file / grep_search / run_shell_command の
                                     tool call を台本どおりに返す
      その tool call を qwen が実行したか、permissions.deny で拒否したかを
      fake_llm 側に返ってくる tool 結果メッセージから判定する。

    実行:
      python3 test/hernes/test_qwen_permissions.py       # 全部
      python3 -m unittest discover -s test/hernes        # 同じ
      pytest test/hernes                                 # pytest でも収集できる

      QWEN_SKIP_E2E=1 ...        E2E を飛ばし静的検査だけ走らせる
      QWEN_CLI=/path/to/qwen ... npx ではなくこのバイナリを使う（速い）
      QWEN_CODE_VERSION=0.24.7   npx で取るバージョン

    注意:
      E2E は 1 ケースごとに qwen を起動するので数分かかる。
      また qwen は `~/.qwen/` に状態を書き、`.qwen/settings.json` に
      `"$version"` を追記する（設定マイグレーションの印）。
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from fake_llm import FakeLLMServer  # noqa: E402  PROJECT_ROOT を通してから import

SETTINGS_PATH = PROJECT_ROOT / ".qwen" / "settings.json"
QWEN_CODE_VERSION = os.environ.get("QWEN_CODE_VERSION", "0.24.7")

# deny されたとき qwen が tool 結果に入れる文言（0.24.7 で確認）
DENY_MARKER = "denied by permission rules"

# permissions のルールに書けるツール表示名（qwen 0.24.7 の TOOL_NAME_ALIASES より）。
# 別名は canonical 名に解決されるので、ここには代表的な表示名だけを持つ。
KNOWN_RULE_TOOLS = {
    "Agent", "Bash", "Edit", "EnterPlanMode", "Exec", "ExitPlanMode", "FindFiles",
    "Glob", "Grep", "Lsp", "ListFiles", "Monitor", "NotebookEdit", "Read",
    "ReadFile", "ReadFolder", "ReadMcpResource", "SaveMemory", "SearchFiles",
    "Shell", "Skill", "TodoList", "TodoWrite", "WebFetch", "WebSearch", "Write",
    "WriteFile", "ZoomImage",
}

# 読み取り拒否を試すためのおとり。テスト中だけ置いて必ず消す。
ENV_FIXTURE = PROJECT_ROOT / ".env.permissions-e2e"
ENV_FIXTURE_SECRET = "SECRET_TOKEN=should-not-be-readable"

# deny 対象外の対照ファイル（リポジトリに元からあるもの）
ALLOWED_FIXTURE = PROJECT_ROOT / "fake_llm_script.example.json"


# --------------------------------------------------------------------------- #
# 静的検査
# --------------------------------------------------------------------------- #


class TestSettingsSchema(unittest.TestCase):
    """qwen を起動せずに settings.json の形だけ見る。CI で常に走らせられる。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.settings: dict[str, Any] = json.loads(
            SETTINGS_PATH.read_text(encoding="utf-8")
        )

    def test_settings_file_is_valid_json(self) -> None:
        self.assertIsInstance(self.settings, dict)

    def test_permissions_key_exists(self) -> None:
        self.assertIn(
            "permissions",
            self.settings,
            "qwen 0.24.7 の設定キーは permissions（allow/ask/deny）",
        )

    def test_only_known_permission_sections(self) -> None:
        sections = set(self.settings["permissions"])
        self.assertTrue(
            sections <= {"allow", "ask", "deny"},
            f"未知のセクション: {sorted(sections - {'allow', 'ask', 'deny'})}",
        )

    def test_sections_are_string_lists(self) -> None:
        for name, rules in self.settings["permissions"].items():
            with self.subTest(section=name):
                self.assertIsInstance(rules, list)
                self.assertTrue(all(isinstance(r, str) for r in rules))

    def test_rules_are_well_formed(self) -> None:
        """`Tool` か `Tool(specifier)` の形で、括弧が閉じていること。

        qwen は括弧が閉じていないルールを警告だけ出して黙って無視するので、
        設定ミスがここで気づけないと「効いているつもり」になる。
        """
        for section, rules in self.settings["permissions"].items():
            for rule in rules:
                with self.subTest(rule=rule):
                    self.assertEqual(rule, rule.strip(), "前後の空白は不要")
                    open_paren = rule.find("(")
                    if open_paren == -1:
                        tool = rule
                    else:
                        self.assertTrue(
                            rule.endswith(")"),
                            f"括弧が閉じていないルールは無視される（{section}: {rule}）",
                        )
                        tool = rule[:open_paren].strip()
                    self.assertIn(
                        tool,
                        KNOWN_RULE_TOOLS,
                        f"未知のツール名 {tool!r} はどのツールにも一致しない",
                    )

    def test_env_files_are_denied(self) -> None:
        self.assertIn("Read(.env*)", self.settings["permissions"].get("deny", []))


# --------------------------------------------------------------------------- #
# E2E
# --------------------------------------------------------------------------- #


def _qwen_command() -> list[str] | None:
    """qwen CLI の起動コマンド。見つからなければ None。"""
    override = os.environ.get("QWEN_CLI")
    if override:
        return shlex.split(override)
    if shutil.which("qwen"):
        return ["qwen"]
    if shutil.which("npx"):
        return ["npx", "--yes", f"@qwen-code/qwen-code@{QWEN_CODE_VERSION}"]
    return None


def _content_text(content: Any) -> str:
    """tool メッセージの content を文字列に潰す（文字列 / パート配列の両対応）。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            p.get("text", "") for p in content if isinstance(p, dict)
        )
    return "" if content is None else str(content)


@dataclass
class QwenRun:
    """qwen を 1 回走らせた結果。"""

    returncode: int
    stdout: str
    stderr: str
    tool_results: list[str]  # fake LLM に返ってきた tool 結果（重複排除・順序保持）

    @property
    def tool_output(self) -> str:
        return "\n".join(self.tool_results)

    @property
    def was_denied(self) -> bool:
        return DENY_MARKER in self.tool_output

    def debug(self) -> str:
        return (
            f"\n--- returncode: {self.returncode}"
            f"\n--- tool 結果:\n{self.tool_output or '(なし)'}"
            f"\n--- stdout:\n{self.stdout[-2000:]}"
            f"\n--- stderr:\n{self.stderr[-2000:]}"
        )


class TestQwenPermissionsE2E(unittest.TestCase):
    """fake LLM に tool call を吐かせ、deny が本当に止めるかを見る。"""

    server: FakeLLMServer
    cli: list[str]

    @classmethod
    def setUpClass(cls) -> None:
        if os.environ.get("QWEN_SKIP_E2E"):
            raise unittest.SkipTest("QWEN_SKIP_E2E が設定されている")
        cli = _qwen_command()
        if cli is None:
            raise unittest.SkipTest(
                "qwen CLI も npx も見つからない（QWEN_CLI で指定できる）"
            )
        if ENV_FIXTURE.exists():
            raise unittest.SkipTest(
                f"{ENV_FIXTURE.name} が既に存在する（上書きを避けて中止）"
            )
        if not ALLOWED_FIXTURE.exists():
            raise unittest.SkipTest(f"対照ファイルがない: {ALLOWED_FIXTURE.name}")

        cls.cli = cli
        ENV_FIXTURE.write_text(ENV_FIXTURE_SECRET + "\n", encoding="utf-8")
        cls.server = FakeLLMServer(port=0).start()

    @classmethod
    def tearDownClass(cls) -> None:
        if getattr(cls, "server", None) is not None:
            cls.server.stop()
        ENV_FIXTURE.unlink(missing_ok=True)

    # -- ヘルパ -------------------------------------------------------------

    def _run_tool_call(self, tool: str, arguments: dict[str, Any]) -> QwenRun:
        """fake LLM に `tool` の呼び出しを 1 回返させ、qwen を走らせる。"""
        self.server.engine.reset()
        self.server.engine.load_script(
            {
                "queue": [
                    {"tool_calls": [{"name": tool, "arguments": arguments}]},
                    "終了します。",
                ],
                "default": "終了します。",
            }
        )

        proc = subprocess.run(
            [
                *self.cli,
                "--auth-type", "openai",
                "--openai-api-key", "dummy",  # fake LLM は検証しない
                "--openai-base-url", self.server.openai_base_url,
                "-m", "qwen3:8b",
                # deny 以外の確認を全部通し、止まるのは deny だけという状況を作る
                "--yolo",
                "--max-wall-time", "90",
                "--max-tool-calls", "3",
                "-p", "指示どおりに実行してください",
            ],
            cwd=PROJECT_ROOT,  # .qwen/settings.json を読ませるため
            capture_output=True,
            text=True,
            timeout=300,
            env={**os.environ, "QWEN_CODE_SUPPRESS_YOLO_WARNING": "1"},
        )

        seen: list[str] = []
        for request in self.server.requests:
            for message in request["body"].get("messages", []):
                if message.get("role") != "tool":
                    continue
                text = _content_text(message.get("content"))
                if text and text not in seen:
                    seen.append(text)

        return QwenRun(proc.returncode, proc.stdout, proc.stderr, seen)

    def assertToolCallReached(self, run: QwenRun) -> None:
        """tool 結果が 1 件も無いなら、そもそも tool call が発火していない。

        これを確認しないと「拒否された」と「呼ばれなかった」を区別できず、
        deny が効いていなくてもテストが通ってしまう。
        """
        self.assertTrue(
            run.tool_results,
            "tool 結果が返っていない（tool call が発火していない）" + run.debug(),
        )

    def assertDenied(self, run: QwenRun) -> None:
        self.assertToolCallReached(run)
        self.assertTrue(run.was_denied, "deny されていない" + run.debug())
        self.assertIn("Read(.env*)", run.tool_output, run.debug())

    def assertAllowed(self, run: QwenRun) -> None:
        self.assertToolCallReached(run)
        self.assertFalse(run.was_denied, "deny されてしまった" + run.debug())

    # -- deny が効くこと ----------------------------------------------------

    def test_read_file_on_env_is_denied(self) -> None:
        """`Read(.env*)` が read_file を止める（ルールの本筋）。"""
        run = self._run_tool_call("read_file", {"file_path": str(ENV_FIXTURE)})
        self.assertDenied(run)
        self.assertNotIn(
            ENV_FIXTURE_SECRET, run.tool_output, "中身が漏れている" + run.debug()
        )

    def test_grep_search_on_env_is_denied(self) -> None:
        """`Read` ルールは read_file だけでなく検索系にも及ぶ。

        qwen は Read ルールを read_file / grep_search / glob / list_directory /
        zoom_image の読み取り一族に当てるので、grep 経由の読み出しも塞がれる。
        """
        run = self._run_tool_call(
            "grep_search", {"pattern": "SECRET_TOKEN", "path": str(ENV_FIXTURE)}
        )
        self.assertDenied(run)
        self.assertNotIn("should-not-be-readable", run.tool_output, run.debug())

    def test_shell_read_of_env_is_denied(self) -> None:
        """シェル経由の迂回も塞がれる（qwen がコマンドの読み取り先を解析する）。"""
        run = self._run_tool_call(
            "run_shell_command",
            {"command": f"cat {ENV_FIXTURE.name}", "description": "read env file"},
        )
        self.assertDenied(run)
        self.assertNotIn(ENV_FIXTURE_SECRET, run.tool_output, run.debug())

    # -- 効きすぎていないこと（対照） ---------------------------------------

    def test_read_file_on_other_file_is_allowed(self) -> None:
        """deny 対象外のファイルは普通に読める。"""
        run = self._run_tool_call("read_file", {"file_path": str(ALLOWED_FIXTURE)})
        self.assertAllowed(run)
        self.assertIn("queue", run.tool_output, run.debug())

    def test_harmless_shell_command_is_allowed(self) -> None:
        """シェルが丸ごと塞がれているわけではない。"""
        run = self._run_tool_call(
            "run_shell_command",
            {"command": "echo permissions-control-case", "description": "control"},
        )
        self.assertAllowed(run)
        self.assertIn("permissions-control-case", run.tool_output, run.debug())


if __name__ == "__main__":
    unittest.main(verbosity=2)
