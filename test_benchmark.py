import asyncio
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import benchmark as b


class BenchmarkTests(unittest.TestCase):
    def test_numeric_answers_only(self):
        self.assertTrue(b.is_correct(" 42.0\n", "42"))
        for response in ("Answer: 42", "42 dollars", "40+2=42", "4.2e1", ""):
            self.assertFalse(b.is_correct(response, "42"))

    def test_effort_request_formats(self):
        for provider, expected in (
            ("openai", {"reasoning_effort": "none"}),
            ("openrouter", {"reasoning": {"effort": "none"}}),
        ):
            args = SimpleNamespace(
                provider=provider,
                effort="none",
                answer_format="plain",
                extra_body={
                    "reasoning_effort": "high",
                    "reasoning": {"effort": "high"},
                },
            )
            self.assertEqual(b.request_body(args), expected)
        args.effort = ""
        self.assertEqual(b.request_body(args), args.extra_body)

    def test_csv_trailing_columns_can_be_omitted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            header = ",".join(b.COLUMNS) + "\n"
            path.write_text(
                header + "model,32B,,author,host\nother\n", encoding="utf-8"
            )
            rows = b.read_csv(path)
            self.assertEqual(rows[0]["active_parameters"], "")
            self.assertEqual(rows[0]["provider"], "host")
            self.assertEqual(rows[0]["effort"], "")
            self.assertEqual(rows[0]["correct"], "")
            self.assertEqual(rows[0]["timestamp"], "")
            self.assertEqual(
                rows[1],
                {
                    key: (
                        "other" if key == "model"
                        else "plain" if key == "answer_format"
                        else ""
                    )
                    for key in b.COLUMNS
                },
            )
            path.write_text(
                header + ",".join(["extra"] * (len(b.COLUMNS) + 1)) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                b.read_csv(path)

    def test_interrupted_run_resumes_and_records_only_when_complete(self):
        response = SimpleNamespace(
            model_dump=lambda: {"choices": [{"message": {"content": "42"}}]},
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="42"), finish_reason="stop"
                )
            ]
        )
        with tempfile.TemporaryDirectory() as directory, patch.object(b, "TOTAL", 2):
            path = Path(directory)
            args = SimpleNamespace(
                model="model",
                provider="openai",
                system_prompt="Number only",
                temperature=0,
                max_tokens=64,
                extra_body={},
                effort="",
                answer_format="plain",
                concurrency=1,
                checkpoint=path / "checkpoint.json",
            )
            config = dict(
                model="model",
                parameters="32B",
                active_parameters=None,
                model_author="author",
                provider="host",
                effort="",
                answer_format="plain",
            )
            questions = [{"question": "question", "answer": "42"}] * 2
            create = AsyncMock(side_effect=[response, RuntimeError("API error")])
            client = SimpleNamespace(
                chat=SimpleNamespace(completions=SimpleNamespace(create=create))
            )
            state = b.load_state(args.checkpoint, config)
            with self.assertRaises(RuntimeError):
                asyncio.run(b.solve(client, args, questions, state))
            self.assertFalse(args.checkpoint.exists())
            b.atomic_json(args.checkpoint, state)
            state = b.load_state(args.checkpoint, config)
            self.assertEqual(state["completed"], [0])
            self.assertEqual(state["correct"], 1)
            csv_path = path / "results.csv"
            with self.assertRaises(ValueError):
                b.record_csv(csv_path, state)
            self.assertFalse(csv_path.exists())

            create.side_effect = [response]
            asyncio.run(b.solve(client, args, questions, state))
            state["timestamp"] = "completed"
            b.record_csv(csv_path, state)
            b.record_csv(csv_path, state)
            rows = b.read_csv(csv_path)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["correct"], "2")
            self.assertEqual(create.call_args.kwargs["max_completion_tokens"], 64)
            self.assertNotIn("max_tokens", create.call_args.kwargs)

    def test_json_answers_are_strict_integer_objects(self):
        self.assertTrue(b.is_correct('{"answer": 10}', "10", "json"))
        self.assertTrue(b.is_correct('{"answer": -10}', "-10", "json"))
        for content in (
            "10",
            '{"answer": "10"}',
            '{"answer": 10.0}',
            '{"answer": true}',
            '{"answer": 10, "reason": "x"}',
            '{"answer": 11}',
            "not json",
            "null",
            "[]",
            "{}",
        ):
            with self.subTest(content=content):
                self.assertFalse(b.is_correct(content, "10", "json"))

    def test_json_request_and_batch_column(self):
        import batch

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            path.write_text(
                "model,parameters,active_parameters,provider,correct,timestamp\n"
                "model,32B,,openai,,\n",
                encoding="utf-8",
            )
            rows = b.read_csv(path)
            self.assertEqual(rows[0]["answer_format"], "plain")
            rows[0]["answer_format"] = "json"
            b.write_csv(path, rows)
            jobs = batch.plan(batch.parse_args([str(path)]))
            self.assertEqual(jobs[0].answer_format, "json")
            self.assertIn('"answer"', jobs[0].system_prompt)
            body = b.request_body(jobs[0])
            schema = body["response_format"]["json_schema"]
            self.assertTrue(schema["strict"])
            self.assertEqual(
                schema["schema"]["properties"], {"answer": {"type": "integer"}}
            )
            self.assertEqual(schema["schema"]["required"], ["answer"])
            self.assertFalse(schema["schema"]["additionalProperties"])
            plain = b.parse_args(["--model", "model", "--provider", "openai"])
            self.assertNotIn("response_format", b.request_body(plain))
            self.assertNotEqual(
                b.checkpoint_path(plain, b.configuration(plain, [])),
                b.checkpoint_path(plain, b.configuration(jobs[0], [])),
            )

    def test_checkpoint_tracks_full_configuration(self):
        args = SimpleNamespace(checkpoint=None, csv=Path("results.csv"))
        config = {"model": "a", "extra_body": {"x": 1}, "system_prompt": "number"}
        original = b.checkpoint_path(args, config)
        self.assertEqual(
            original, b.checkpoint_path(args, dict(reversed(config.items())))
        )
        for key, value in (
            ("model", "b"),
            ("extra_body", {"x": 2}),
            ("system_prompt", "other"),
        ):
            self.assertNotEqual(original, b.checkpoint_path(args, {**config, key: value}))
        args.checkpoint_prefix = "row-"
        self.assertNotEqual(original, b.checkpoint_path(args, config))
        args.checkpoint = Path("custom.json")
        self.assertEqual(b.checkpoint_path(args, config), args.checkpoint)

    def test_reasoning_detection(self):
        clean = {
            "choices": [{"message": {"content": "42", "reasoning": None}}],
            "usage": {"completion_tokens_details": {"reasoning_tokens": 0}},
        }
        b.check_reasoning(SimpleNamespace(model_dump=lambda: clean))
        for key, value in (
            ("reasoning", "thinking"),
            ("reasoning_content", "thinking"),
            ("reasoning_details", [{"type": "reasoning.text", "text": "thinking"}]),
            ("content", "<think>thinking</think>42"),
        ):
            body = copy.deepcopy(clean)
            body["choices"][0]["message"][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "reasoning"):
                b.check_reasoning(SimpleNamespace(model_dump=lambda: body))
        body = copy.deepcopy(clean)
        body["usage"]["completion_tokens_details"]["reasoning_tokens"] = 36
        with self.assertRaisesRegex(ValueError, "reasoning_tokens=36"):
            b.check_reasoning(SimpleNamespace(model_dump=lambda: body))

    def test_reasoning_stops_before_scoring_or_next_request(self):
        response = SimpleNamespace(
            model_dump=lambda: {
                "choices": [{"message": {"content": "42", "reasoning": "thinking"}}]
            }
        )
        create = AsyncMock(return_value=response)
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        args = SimpleNamespace(
            model="model",
            provider="openai",
            system_prompt="number",
            temperature=0,
            max_tokens=64,
            extra_body={},
            effort="",
            answer_format="plain",
            concurrency=1,
        )
        state = {"completed": [], "correct": 0}
        with patch.object(b, "TOTAL", 2), self.assertRaisesRegex(ValueError, "reasoning"):
            asyncio.run(
                b.solve(client, args, [{"question": "q", "answer": "42"}] * 2, state)
            )
        self.assertEqual(state, {"completed": [], "correct": 0})
        self.assertEqual(create.await_count, 1)

    def test_error_drains_inflight_requests_without_starting_new_ones(self):
        async def scenario(reasoning):
            started = 0
            all_started = asyncio.Event()
            release = asyncio.Event()
            cancelled = []
            first_error = RuntimeError("first API error")
            response = SimpleNamespace(
                model_dump=lambda: {"choices": [{"message": {"content": "42"}}]},
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="42"), finish_reason="stop"
                    )
                ],
            )

            async def create(**kwargs):
                nonlocal started
                index = started
                started += 1
                if started == 3:
                    all_started.set()
                await all_started.wait()
                if index == 0:
                    if reasoning:
                        return SimpleNamespace(
                            model_dump=lambda: {
                                "choices": [{"message": {"reasoning": "thinking"}}]
                            }
                        )
                    raise first_error
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    cancelled.append(index)
                    raise
                if index == 2:
                    raise RuntimeError("later API error")
                return response

            args = SimpleNamespace(
                model="model",
                provider="openai",
                system_prompt="number",
                temperature=0,
                max_tokens=64,
                extra_body={},
                effort="",
                answer_format="plain",
                concurrency=3,
            )
            client = SimpleNamespace(
                chat=SimpleNamespace(completions=SimpleNamespace(create=create))
            )
            state = {"completed": [], "correct": 0}
            task = asyncio.create_task(
                b.solve(client, args, [{"question": "q", "answer": "42"}] * 5, state)
            )
            try:
                await all_started.wait()
                # Let the first worker fail while the other requests remain in flight.
                await asyncio.sleep(0)
                await asyncio.sleep(0)
                self.assertFalse(task.done())
                self.assertEqual(cancelled, [])
                self.assertEqual(started, 3)
            finally:
                release.set()
                with self.assertRaisesRegex(
                    ValueError if reasoning else RuntimeError,
                    "reasoning" if reasoning else "first API error",
                ) as caught:
                    await task
            if not reasoning:
                self.assertIs(caught.exception, first_error)
            self.assertEqual(state, {"completed": [1], "correct": 1})
            self.assertEqual(started, 3)
            self.assertEqual(cancelled, [])

        for reasoning in (False, True):
            with self.subTest(reasoning=reasoning), patch.object(b, "TOTAL", 5):
                asyncio.run(asyncio.wait_for(scenario(reasoning), timeout=5))


if __name__ == "__main__":
    unittest.main()
