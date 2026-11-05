# /// script
# requires-python = ">=3.11"
# dependencies = ["openai>=1.55,<3", "datasets>=3,<5", "python-dotenv>=1,<2", "httpx>=0.27,<1"]
# ///
import argparse
import asyncio
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import benchmark


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="CSVの未完了行を順番にベンチマーク")
    parser.add_argument("csv", type=Path)
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--request-log", type=Path, help="指定時のみJSONLログを保存")
    parser.add_argument("--concurrency", type=benchmark.positive_int, default=4)
    return parser.parse_args(argv)


def plan(args) -> list:
    rows = benchmark.read_csv(args.csv)
    directory = args.checkpoint_dir or args.csv.with_name(
        args.csv.name + ".checkpoints"
    )
    occurrences = Counter()
    jobs = []
    for index, row in enumerate(rows):
        identity = [
            row[key] for key in benchmark.INPUT_COLUMNS if key != "effort" or row[key]
        ]
        key = json.dumps(identity, ensure_ascii=False)
        occurrence = occurrences[key]
        occurrences[key] += 1
        if not row["model"].strip() or not row["provider"].strip():
            raise ValueError(f"CSVの{index + 2}行目: modelとproviderは必須です")
        if bool(row["correct"]) != bool(row["timestamp"]):
            raise ValueError(
                f"CSVの{index + 2}行目: correctとtimestampは両方空欄にしてください"
            )
        if row["correct"]:
            continue
        digest = hashlib.sha256(
            json.dumps(
                [str(args.csv.resolve()), identity, occurrence], ensure_ascii=False
            ).encode()
        ).hexdigest()
        job = benchmark.parse_args(
            [
                "--config",
                str(args.config),
                "--env-file",
                str(args.env_file),
                "--model=" + row["model"],
                "--provider=" + row["provider"],
                "--model-author=" + row["model_author"],
                "--effort=" + row["effort"],
                "--parameters=" + row["parameters"],
                "--active-parameters=" + row["active_parameters"],
                "--csv",
                str(args.csv),
                "--checkpoint",
                str(directory / f"{digest}.json"),
                "--concurrency",
                str(args.concurrency),
            ]
        )
        job.request_log = args.request_log
        job.csv_row = index
        jobs.append(job)
    return jobs


async def run(args) -> None:
    jobs = plan(args)
    for job in jobs:
        job.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        print(
            f"CSV {job.csv_row + 2}行目: {job.provider} / {job.model}"
            f"\n再開ファイル: {job.checkpoint}",
            file=sys.stderr,
        )
        await benchmark.run(job)
    print(f"未完了行 {len(jobs)}件の処理が完了しました。", file=sys.stderr)


def main() -> int:
    args = parse_args()
    try:
        asyncio.run(run(args))
    except (Exception, KeyboardInterrupt) as error:
        print(
            f"\n中断: {error or 'KeyboardInterrupt'}\n"
            "未完了行は結果を空欄のまま残しています。同じコマンドで再開してください。",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
