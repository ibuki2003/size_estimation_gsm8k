import asyncio
import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import benchmark as b


class FakeClient:
    def __init__(self, fail_on=None):
        self.chat = SimpleNamespace(completions=self)
        self.fail_on = fail_on
        self.calls = []

    async def create(self, **kwargs):
        question = kwargs["messages"][1]["content"]
        self.calls.append(question)
        await asyncio.sleep(0)
        if question == self.fail_on:
            raise RuntimeError("simulated API failure")
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=question),
                    finish_reason="stop",
                )
            ]
        )


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = self.root / "config.toml"
        self.config.write_text(
            'system_prompt = "Only output a number."\n'
            "[providers.test]\n"
            'base_url = "https://example.invalid/v1"\n'
            'token_env = "OPENAI_API_KEY"\n',
            encoding="utf-8",
        )
        self.args = b.parse_args(
            [
                "--config",
                str(self.config),
                "--provider",
                "test",
                "--model",
                "test",
                "--checkpoint",
                str(self.root / "checkpoint.json"),
                "--csv",
                str(self.root / "results.csv"),
                "--concurrency",
                "2",
            ]
        )
        self.questions = [{"question": str(i), "answer": str(i)} for i in range(3)]

    def state(self):
        return b.load_state(
            self.args.checkpoint, b.configuration(self.args, self.questions)
        )

    def test_numeric_grading(self):
        for value in ["42", " 42\n", "42.00", "+42"]:
            with self.subTest(value=value):
                self.assertTrue(b.is_correct(value, "42"))
        for value in [
            "",
            "Answer: 42",
            "42 dollars",
            "40+2=42",
            "**42**",
            "4.2e1",
            "4,200",
            "NaN",
            "Infinity",
            "４２",
            "41",
        ]:
            with self.subTest(value=value):
                self.assertFalse(b.is_correct(value, "42"))
        self.assertTrue(b.is_correct("-.5", "-0.5"))

    def test_partial_run_and_resume(self):
        with patch.object(b, "TOTAL", 3):
            state = self.state()
            with self.assertRaisesRegex(RuntimeError, "simulated API failure"):
                asyncio.run(
                    b.solve(FakeClient(fail_on="1"), self.args, self.questions, state)
                )
            saved = json.loads(self.args.checkpoint.read_text())
            self.assertIn("0", saved["results"])
            self.assertNotIn("1", saved["results"])
            self.assertFalse(self.args.csv.exists())
            resumed = self.state()
            client = FakeClient()
            asyncio.run(b.solve(client, self.args, self.questions, resumed))
            self.assertNotIn("0", client.calls)
            self.assertEqual(len(resumed["results"]), 3)
            resumed["timestamp"] = "2026-01-01T00:00:00+00:00"
            b.record_csv(self.args.csv, resumed)
            b.record_csv(self.args.csv, resumed)
            with self.args.csv.open(newline="") as stream:
                reader = csv.DictReader(stream)
                self.assertEqual(reader.fieldnames, b.COLUMNS)
                rows = list(reader)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["correct"], "3")
            self.assertEqual(rows[0]["parameters"], "")

    def test_incomplete_results_never_touch_csv(self):
        self.args.csv.write_text("existing data\n")
        with self.assertRaises(ValueError):
            b.record_csv(self.args.csv, self.state())
        self.assertEqual(self.args.csv.read_text(), "existing data\n")

    def test_changed_conditions_rejected(self):
        self.state()
        self.args.model = "different-model"
        with self.assertRaisesRegex(ValueError, "実行条件"):
            self.state()

    def test_parameter_strings_are_preserved_in_csv(self):
        parameters = " 32B, estimated\nunknown "
        active = "N/A (dense?)"
        self.args = b.parse_args(
            [
                "--config",
                str(self.config),
                "--provider",
                "test",
                "--model",
                "test",
                "--parameters",
                parameters,
                "--active-parameters",
                active,
                "--checkpoint",
                str(self.args.checkpoint),
                "--csv",
                str(self.args.csv),
            ]
        )
        with patch.object(b, "TOTAL", 3):
            state = self.state()
            asyncio.run(b.solve(FakeClient(), self.args, self.questions, state))
            state["timestamp"] = "test"
            b.record_csv(self.args.csv, state)
        with self.args.csv.open(newline="") as stream:
            row = next(csv.DictReader(stream))
        self.assertEqual(row["parameters"], parameters)
        self.assertEqual(row["active_parameters"], active)

    def test_provider_settings_and_overrides(self):
        self.config.write_text(
            'system_prompt = "Custom prompt"\n'
            "[providers.test]\n"
            'base_url = "https://example.invalid/v1"\n'
            'token_env = "CUSTOM_TOKEN"\n'
            "max_tokens = 12\n"
            "temperature = 0.5\n"
            "timeout = 30\n"
            "[providers.test.extra_body.structured_outputs]\n"
            'regex = "-?[0-9]+"\n',
            encoding="utf-8",
        )
        argv = ["--config", str(self.config), "--provider", "test", "--model", "test"]
        args = b.parse_args(argv)
        self.assertEqual(args.system_prompt, "Custom prompt")
        self.assertEqual(args.token_env, "CUSTOM_TOKEN")
        self.assertEqual(args.max_tokens, 12)
        self.assertEqual(args.temperature, 0.5)
        self.assertEqual(args.timeout, 30)
        self.assertEqual(args.extra_body, {"structured_outputs": {"regex": "-?[0-9]+"}})
        args = b.parse_args(argv + ["--max-tokens", "16", "--extra-body", "{}"])
        self.assertEqual(args.max_tokens, 16)
        self.assertEqual(args.extra_body, {})

    def test_changed_prompt_rejected_on_resume(self):
        self.state()
        self.args.system_prompt = "Different prompt"
        with self.assertRaisesRegex(ValueError, "実行条件"):
            self.state()

    def test_model_and_provider_are_required(self):
        import contextlib
        import io

        for arguments in (["--model", "test"], ["--provider", "test"]):
            with (
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                b.parse_args(["--config", str(self.config)] + arguments)

    def test_truncation_is_incorrect(self):
        class TruncatedClient(FakeClient):
            async def create(self, **kwargs):
                response = await super().create(**kwargs)
                response.choices[0].finish_reason = "length"
                return response

        with patch.object(b, "TOTAL", 3):
            state = self.state()
            asyncio.run(b.solve(TruncatedClient(), self.args, self.questions, state))
            self.assertFalse(any(item["correct"] for item in state["results"].values()))

    def test_other_csv_rows_preserved(self):
        with patch.object(b, "TOTAL", 3):
            state = self.state()
            asyncio.run(b.solve(FakeClient(), self.args, self.questions, state))
            state["timestamp"] = "first"
            b.record_csv(self.args.csv, state)
            state["timestamp"] = "second"
            b.record_csv(self.args.csv, state)
            with self.args.csv.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row["timestamp"] for row in rows], ["first", "second"])

    def test_dataset_extracts_only_final_answer(self):
        dataset = [
            {"question": "question", "answer": "secret chain of thought\n#### 1,234"}
        ]
        import sys

        fake = SimpleNamespace(load_dataset=lambda *args, **kwargs: dataset)
        with patch.dict(sys.modules, {"datasets": fake}), patch.object(b, "TOTAL", 1):
            self.assertEqual(
                b.load_questions(), [{"question": "question", "answer": "1234"}]
            )
        with patch.dict(sys.modules, {"datasets": fake}):
            with self.assertRaisesRegex(ValueError, "1319"):
                b.load_questions()

    def test_worker_cancellation_preserves_completed_answers(self):
        class SlowClient(FakeClient):
            async def create(self, **kwargs):
                if kwargs["messages"][1]["content"] != "0":
                    await asyncio.sleep(60)
                return await super().create(**kwargs)

        async def interrupt():
            task = asyncio.create_task(
                b.solve(SlowClient(), self.args, self.questions, self.state())
            )
            while not json.loads(self.args.checkpoint.read_text())["results"]:
                await asyncio.sleep(0)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

        with patch.object(b, "TOTAL", 3):
            asyncio.run(interrupt())
        saved = json.loads(self.args.checkpoint.read_text())
        self.assertEqual(list(saved["results"]), ["0"])
        self.assertFalse(self.args.csv.exists())

    @unittest.skipUnless(importlib.util.find_spec("openai"), "OpenAI SDK未導入")
    def test_sdk_request_and_complete_run(self):
        import httpx
        from openai import AsyncOpenAI

        requests = []

        def respond(request):
            payload = json.loads(request.content)
            requests.append(payload)
            self.assertEqual(request.headers["authorization"], "Bearer test-token")
            self.assertEqual(payload["messages"][0]["content"], self.args.system_prompt)
            self.assertNotIn("answer", payload)
            return httpx.Response(
                200,
                json={
                    "id": "test",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "test",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": payload["messages"][1]["content"],
                            },
                            "finish_reason": "stop",
                        }
                    ],
                },
            )

        def make_client(**kwargs):
            return AsyncOpenAI(
                **kwargs,
                http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
            )

        with (
            patch.object(b, "TOTAL", 3),
            patch.object(b, "load_questions", return_value=self.questions),
            patch.dict("os.environ", {"OPENAI_API_KEY": "test-token"}),
            patch("openai.AsyncOpenAI", side_effect=make_client),
        ):
            asyncio.run(b.run(self.args))
            asyncio.run(b.run(self.args))
        self.assertEqual(len(requests), 3)
        with self.args.csv.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["correct"], "3")
        self.assertTrue(rows[0]["timestamp"].endswith("+00:00"))


if __name__ == "__main__":
    unittest.main()
