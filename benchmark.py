# /// script
# requires-python = ">=3.11"
# dependencies = ["openai>=1.55,<3", "datasets>=3,<5"]
# ///
import argparse
import asyncio
import csv
import hashlib
import json
import os
import re
import sys
import tomllib
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path


TOTAL = 1319
COLUMNS = [
    "model",
    "parameters",
    "active_parameters",
    "correct",
    "timestamp",
    "provider",
]
NUMBER = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)")


def is_correct(response: str, answer: str) -> bool:
    value = response.strip()
    return bool(NUMBER.fullmatch(value)) and Decimal(value) == Decimal(answer)


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def load_questions() -> list[dict]:
    from datasets import load_dataset

    dataset = load_dataset("openai/gsm8k", "main", split="test")
    questions = []
    for item in dataset:
        _, separator, answer = item["answer"].rpartition("####")
        answer = answer.strip().replace(",", "")
        if not separator or not NUMBER.fullmatch(answer):
            raise ValueError("GSM8Kの正解値を解析できません")
        questions.append({"question": item["question"], "answer": answer})
    if len(questions) != TOTAL:
        raise ValueError(f"GSM8K testは{TOTAL}件必要です: {len(questions)}件")
    return questions


def configuration(args: argparse.Namespace, questions: list[dict]) -> dict:
    digest = hashlib.sha256(
        json.dumps(questions, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()
    return {
        "model": args.model,
        "base_url": args.base_url.rstrip("/"),
        "parameters": args.parameters,
        "active_parameters": args.active_parameters,
        "provider": args.provider,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "extra_body": args.extra_body,
        "system_prompt": args.system_prompt,
        "dataset_sha256": digest,
    }


def load_state(path: Path, config: dict) -> dict:
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
        if state["version"] != 1 or state["config"] != config:
            raise ValueError(
                "再開ファイルと実行条件が異なります。別の--checkpointを指定してください"
            )
        results = state["results"]
        if not isinstance(results, dict) or any(
            not key.isdigit() or str(int(key)) != key or not 0 <= int(key) < TOTAL
            for key in results
        ):
            raise ValueError("再開ファイルの問題番号が不正です")
        return state
    state = {"version": 1, "config": config, "results": {}, "timestamp": None}
    atomic_json(path, state)
    return state


async def solve(client, args, questions: list[dict], state: dict) -> None:
    pending = iter(i for i in range(TOTAL) if str(i) not in state["results"])

    async def worker():
        for index in pending:
            response = await client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": args.system_prompt},
                    {"role": "user", "content": questions[index]["question"]},
                ],
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                extra_body=args.extra_body,
            )
            choice = response.choices[0]
            content = choice.message.content
            # A truncated response must not accidentally count as the right answer.
            correct = (
                choice.finish_reason == "stop"
                and content is not None
                and is_correct(content, questions[index]["answer"])
            )
            state["results"][str(index)] = {
                "response": content,
                "finish_reason": choice.finish_reason,
                "correct": correct,
            }
            # No await between updating the state and persisting it.
            atomic_json(args.checkpoint, state)
            print(
                f"\r完了: {len(state['results'])}/{TOTAL}",
                end="",
                file=sys.stderr,
                flush=True,
            )

    tasks = [asyncio.create_task(worker()) for _ in range(args.concurrency)]
    try:
        await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def record_csv(path: Path, state: dict) -> None:
    if len(state["results"]) != TOTAL:
        raise ValueError("全件が完了するまでCSVには記録できません")
    config = state["config"]
    row = {
        "model": config["model"],
        "parameters": config["parameters"],
        "active_parameters": config["active_parameters"],
        "correct": sum(result["correct"] for result in state["results"].values()),
        "timestamp": state["timestamp"],
        "provider": config["provider"],
    }
    normalized = {
        key: "" if value is None else str(value) for key, value in row.items()
    }
    rows = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != COLUMNS:
                raise ValueError("既存CSVの列が期待する形式と異なります")
            rows = list(reader)
        # Recover safely if interrupted after CSV replacement but before exit.
        if normalized in rows:
            return
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow(row)
    temporary.replace(path)


async def run(args) -> None:
    questions = load_questions()
    state = load_state(args.checkpoint, configuration(args, questions))
    if len(state["results"]) < TOTAL:
        from openai import AsyncOpenAI

        token = os.environ.get(args.token_env)
        if not token:
            raise ValueError(f"環境変数{args.token_env}にAPIトークンを設定してください")
        async with AsyncOpenAI(
            api_key=token,
            base_url=args.base_url,
            timeout=args.timeout,
            max_retries=0,
        ) as client:
            await solve(client, args, questions, state)
    if not state["timestamp"]:
        state["timestamp"] = datetime.now(timezone.utc).isoformat()
        atomic_json(args.checkpoint, state)
    record_csv(args.csv, state)
    correct = sum(item["correct"] for item in state["results"].values())
    print(
        f"\n正解: {correct}/{TOTAL} ({correct / TOTAL:.2%}) → {args.csv}",
        file=sys.stderr,
    )


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("1以上の整数を指定してください")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="数値回答限定のGSM8K testベンチマーク")
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--provider", required=True, help="config内のprovider名")
    parser.add_argument("--base-url", help="configの接続先を上書き")
    parser.add_argument("--token-env", help="configのトークン環境変数名を上書き")
    parser.add_argument("--parameters", help="総パラメータ数の表記（文字列のまま記録）")
    parser.add_argument(
        "--active-parameters", help="有効パラメータ数の表記（文字列のまま記録）"
    )
    parser.add_argument("--concurrency", type=positive_int, default=4)
    parser.add_argument("--max-tokens", type=positive_int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--timeout", type=float)
    parser.add_argument(
        "--extra-body", type=json.loads, help="configの追加JSON全体を上書き"
    )
    parser.add_argument("--csv", type=Path, default=Path("results.csv"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoint.json"))
    args = parser.parse_args(argv)
    try:
        with args.config.open("rb") as stream:
            config = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as error:
        parser.error(f"configを読み込めません: {error}")
    providers = config.get("providers", {})
    if args.provider not in providers or not isinstance(providers[args.provider], dict):
        parser.error(f"configにprovider {args.provider!r} がありません")
    provider = providers[args.provider]
    args.system_prompt = config.get("system_prompt")
    if not isinstance(args.system_prompt, str) or not args.system_prompt.strip():
        parser.error("configに空でないsystem_promptが必要です")
    for name, default in (
        ("base_url", None),
        ("token_env", None),
        ("max_tokens", 64),
        ("temperature", 0),
        ("timeout", 120),
        ("extra_body", {}),
    ):
        if getattr(args, name) is None:
            setattr(args, name, provider.get(name, default))
    for name in ("base_url", "token_env"):
        value = getattr(args, name)
        if not isinstance(value, str) or not value.strip():
            parser.error(f"providerに空でない{name}が必要です")
    if type(args.max_tokens) is not int or args.max_tokens < 1:
        parser.error("max_tokensには1以上の整数を指定してください")
    for name in ("temperature", "timeout"):
        if type(getattr(args, name)) not in (int, float):
            parser.error(f"{name}には数値を指定してください")
    if not isinstance(args.extra_body, dict):
        parser.error("extra_bodyにはオブジェクトを指定してください")
    reserved = {"model", "messages", "temperature", "max_tokens", "stream", "n"}
    if reserved & args.extra_body.keys():
        parser.error("--extra-bodyで標準リクエスト項目は上書きできません")
    if args.timeout <= 0:
        parser.error("--timeoutは正の数にしてください")
    if args.csv.resolve() == args.checkpoint.resolve():
        parser.error("--csvと--checkpointには別のパスを指定してください")
    return args


def main() -> int:
    args = parse_args()
    try:
        asyncio.run(run(args))
    except (Exception, KeyboardInterrupt) as error:
        print(
            f"\n中断: {error or 'KeyboardInterrupt'}\n"
            f"CSVへの記録は完了していません。同じ引数で再実行してください。\n"
            f"再開ファイル: {args.checkpoint}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
