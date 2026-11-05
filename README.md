# 数値回答限定 GSM8K ベンチマーク

GSM8Kを算数の問題集として使い、途中式・説明なしの回答精度を測ります。
通常のCoTありGSM8Kスコアとは別の指標です。
プロンプトと出力制約で推論の言語化を抑えますが、モデル内部の推論禁止は保証しません。

## 実行

Python 3.11以上と[uv](https://docs.astral.sh/uv/)が必要です。
初回は依存パッケージとHugging Faceの `openai/gsm8k`（main/test）を取得します。
問題文と最終正解値だけを使い、データセットの解説はモデルに渡しません。

```sh
export OPENAI_API_KEY='...'
uv run benchmark.py \
  --model your-model \
  --provider openai \
  --model-author 'model-creator' \
  --parameters '32B' \
  --active-parameters '3B' \
  --concurrency 4 \
  --csv results.csv \
  --checkpoint your-model.json
```

`--provider`と`--model`は毎回必須です。既定のprovider・modelはありません。
`parameters`・`active_parameters`は省略可能で、その場合はCSVが空欄になります。
指定した文字列は数値変換・単位変換・空白除去をせず記録します。
`model_author`はモデル作成者の記録用メタデータです（省略可能）。
接続先・APIキーを選ぶのはAPI提供者を表す`provider`で、作成者は接続に影響しません。

## CSVから一括実行

同じCSVに実験予定と結果をまとめられます。まず結果の2列を空欄にします。

```csv
model,parameters,active_parameters,model_author,provider,correct,timestamp
your-model,32B,3B,model-creator,openai
another-model,unknown,,another-creator,vllm
```

```sh
uv run batch.py results.csv --concurrency 4
```

未完了行を上から順に実行し、1319問が揃った行だけ結果を埋めます。
完了済み行はスキップします。`correct`と`timestamp`の片方だけが空欄の行はエラーです。
末尾の空欄はカンマごと省略できます。途中の空欄には区切りのカンマが必要です。
従来の6列CSV・旧列順も読み込め、次の結果保存時に新しい列順になります。

中断したら同じコマンドで再開してください。
再開ファイルは既定で`results.csv.checkpoints/`に保存します。
同一条件の行を複数書いても、それぞれ独立した実験になります。
再開時は同一条件の行どうしの順序を維持し、CSVや実験条件は実行中に変更しないでください。
`--checkpoint-dir`で再開ファイルの保存先を変更できます。

## 設定

カレントディレクトリの`config.toml`にsystem promptとproviderごとの設定を置きます。
別のファイルを使う場合は`--config path/to/config.toml`を指定します。

```toml
system_prompt = "Output only the final answer as a number. No explanations."

[providers.local]
base_url = "http://localhost:8000/v1"
token_env = "LOCAL_API_KEY"
max_tokens = 64
temperature = 0
timeout = 120

# 対応するvLLMで、整数だけの出力に制約する例
[providers.local.extra_body.structured_outputs]
regex = '-?[0-9]+'
```

この例は`--provider local --model your-model`で選択します。
トークンの値は設定ファイルには書かず、`token_env`で指定した環境変数に設定してください。

カレントディレクトリの`.env`も自動で読み込みます。
`.env.example`を参考に、providerごとのキーを記入してください。
既存の環境変数が優先されます。別のファイルは`--env-file path/to/.env`で指定できます。
`.env`・`.env.*`はGit管理対象外です（`.env.example`を除く）。

API固有の推論抑制・出力制約は`extra_body`に指定できます。
**対応する設定・値は提供元に確認してください。**
OpenAIのStructured Outputsはトップレベルの整数を許可しないため、
数値だけの出力を強制する方法としては利用できません。

接続先や生成設定は`--base-url`・`--token-env`・`--max-tokens`・`--temperature`・
`--timeout`・`--extra-body`でも上書きできます。
`--extra-body`は設定ファイルの`extra_body`全体を置き換えます。
出力上限・温度の指定に非対応のモデルでは利用できません。

## 採点・再開

前後の空白を除いて、符号付き整数または小数だけを正解候補にします。
正解値と数値比較するため、`42` と `42.0` は同じです。
単位・説明・Markdown・桁区切り・指数表記は許可せず、出力上限による打ち切りも不正解です。

CSVの列は `model,parameters,active_parameters,model_author,provider,correct,timestamp` です。
`correct` は1319問中の正解数、`timestamp` は全件完了時のUTC日時です。

APIエラーやCtrl+Cで停止した場合は、**同じ引数で再実行**してください。
回答済みの問題は再送信しません。APIエラーは不正解として消費せず、処理を中断します。
CSVは全件が揃ったときだけ更新します。中断時や実験条件を変える際は再開ファイルを残してください。
新しい実験には別の `--checkpoint` を指定します。

再開ファイルには問題ごとの応答と採点結果が含まれます。APIトークンは保存しません。
同じCSV・再開ファイルを使う複数プロセスの同時実行には対応していません。

## ファイル

- `benchmark.py`: CLI・API呼び出し・採点・記録
- `batch.py`: CSVの未完了行を一括実行
- `config.toml`: system prompt・providerごとの接続先とAPI設定
- `.env.example`: APIキー設定のひな形
- `test_benchmark.py`: 採点・CSV・中断と再開の最小限のテスト

テスト: `python -m unittest -v`
