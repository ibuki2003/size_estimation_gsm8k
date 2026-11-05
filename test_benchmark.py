import asyncio
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
            self.assertEqual(rows[0]["correct"], "")
            self.assertEqual(rows[0]["timestamp"], "")
            self.assertEqual(
                rows[1], {key: "other" if key == "model" else "" for key in b.COLUMNS}
            )
            path.write_text(
                header + "model,32B,,author,host,,,extra\n", encoding="utf-8"
            )
            with self.assertRaises(ValueError):
                b.read_csv(path)

    def test_interrupted_run_resumes_and_records_only_when_complete(self):
        response = SimpleNamespace(
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
                system_prompt="Number only",
                temperature=0,
                max_tokens=64,
                extra_body={},
                concurrency=1,
                checkpoint=path / "checkpoint.json",
            )
            config = dict(
                model="model",
                parameters="32B",
                active_parameters=None,
                model_author="author",
                provider="host",
            )
            questions = [{"question": "question", "answer": "42"}] * 2
            create = AsyncMock(side_effect=[response, RuntimeError("API error")])
            client = SimpleNamespace(
                chat=SimpleNamespace(completions=SimpleNamespace(create=create))
            )
            state = b.load_state(args.checkpoint, config)
            with self.assertRaises(RuntimeError):
                asyncio.run(b.solve(client, args, questions, state))
            state = b.load_state(args.checkpoint, config)
            self.assertEqual(len(state["results"]), 1)
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


if __name__ == "__main__":
    unittest.main()
