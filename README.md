# Open LLM Leaderboard Local
Huggingface の [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) と同様の検証をローカルで実施するスクリプト

## 目次

- [環境](#環境)
- [動かし方](#動かし方)
- [オプションなどの詳細](#オプションなどの詳細)
  - [モデル出力は wandb にアップロードしない方法](#モデル出力は-wandb-にアップロードしない方法)
  - [PEFT モデルを動かす方法](#peft-モデルを動かす方法)
- [たくさんのモデルを自動で動かす例](#たくさんのモデルを自動で動かす例)
- [タスクを追加検証したい場合](#タスクを追加検証したい場合)
- [その他の想定されるエラー](#その他の想定されるエラー)
  - [gsm8k 実施時に ValueError が出る](#gsm8k-実施時に-valueerror-が出る)

## 環境
- Python 3.9 以上
- ライブラリは lm-evaluation-harness + wandb を使用します
  ```bash
    git clone --recuresive https://github.com/llm-jp/open-llm-leaderboard-local.git
    cd lm-evaluation-harness  # lm-evaluation-harness の commit-id が b281b09 であることを確認
    pip install -e ".[sentencepiece]"  # 必要に応じて extras_require を指定
    pip install wandb
  ```

## 動かし方
下記の通りにして `run_open_llm_leaderboard.sh` を実行することで検証から wandb への結果のアップロードまで自動で行われます。

1. [前述](#環境)のライブラリ設定を行う
2. root に置いてある `run_open_llm_leaderboard.sh` の `HF_HOME` と `WANDB_ENTITY` と `WANDB_PROJECT` を記入
3. `wandb login` を実行
4. `cp ./run_open_llm_leaderboard.sh ./save_wandb.py ./lm-evaluation-harness/  # スクリプトを lm-evaluation-harness に配置する`
5. `cd lm-evaluation-harness`
6. `bash run_open_llm_leaderboard.sh -w {検証したいモデル名} {バッチサイズ} {出力ディレクトリ}` を実行すると出力ディレクトリに結果が保存され、その中身が wandb にアップロードされる

なお、2番で設定する変数は下記を意味しています。

- `HF_HOME`: Huggingface の各種データ、トークン情報などが保存されるパス
- `WANDB_ENTITY`: wandb のエンティティ名
- `WANDB_PROJECT`: wandb のプロジェクト名

wandb のエンティティ、プロジェクトに関しては[公式ドキュメント](https://docs.wandb.ai/)をご確認ください。

> [!CAUTION]
> 2023年11月時点で `run_open_llm_leaderboard.sh` は一つのモデル評価におよそ一日かかります。

## オプションなどの詳細
オプションを設定することで以下のようなことが可能です。
- lm-evaluation-harness 実行時のモデル出力は wandb にアップロードしない
- PEFT モデルを動かす

### モデル出力は wandb にアップロードしない方法
lm-evaluation-harness はオプションを指定すると、スコア以外にも実際のモデルの出力（logit）を出力、保存することができます。

`run_open_llm_leaderboard.sh` では `w` オプションの有無で実際のモデル出力を wandb にアップロードするか指定しています。
（[動かし方](#動かし方)の実行例ではモデル出力を wandb に保存する設定となっています。）

モデル出力を保存したくない場合は下記のように実行してください。

```bash
bash run_open_llm_leaderboard.sh {検証したいモデル名} {バッチサイズ} {出力ディレクトリ}
```

### PEFT モデルを動かす方法
`run_open_llm_leaderboard.sh` では `l` オプションの有無で PEFT モデルの推論を実施します。

具体的には `l` オプションは引数として `{ベースモデル名}` をとり、これまでの `{検証したいモデル名}` のところに PEFT モデル名を渡す形になります。

```bash
bash run_open_llm_leaderboard.sh -w -l {ベースモデル名} {検証したい PEFT モデル名} {バッチサイズ} {出力ディレクトリ}
```

## たくさんのモデルを自動で動かす例
下記のようなシェルスクリプトを使用すると自動で様々なモデルが検証されます。

```bash
#! /bin/bash

original_models=(
"llm-jp/llm-jp-1.3b-v1.0"
"llm-jp/llm-jp-13b-v1.0"
"llm-jp/llm-jp-13b-instruct-full-jaster-v1.0"
"llm-jp/llm-jp-13b-instruct-full-jaster-dolly-oasst-v1.0"
"llm-jp/llm-jp-13b-instruct-full-dolly-oasst-v1.0"
"elyza/ELYZA-japanese-Llama-2-7b"
"elyza/ELYZA-japanese-Llama-2-7b-instruct"
"elyza/ELYZA-japanese-Llama-2-7b-fast"
"elyza/ELYZA-japanese-Llama-2-7b-fast-instruct"
"matsuo-lab/weblab-10b"
"matsuo-lab/weblab-10b-instruction-sft"
"pfnet/plamo-13b"
"pfnet/plamo-13b-instruct"
"pfnet/plamo-13b-instruct-nc"
"rinna/youri-7b"
"rinna/youri-7b-chat"
"rinna/youri-7b-instruction"
"stabilityai/japanese-stablelm-base-beta-7b"
"stabilityai/japanese-stablelm-instruct-beta-7b"
"stabilityai/japanese-stablelm-base-ja_vocab-beta-7b"
"stabilityai/japanese-stablelm-instruct-ja_vocab-beta-7b"
"stabilityai/japanese-stablelm-base-gamma-7b"
"stabilityai/japanese-stablelm-instruct-gamma-7b"
"cyberagent/calm2-7b"
"cyberagent/calm2-7b-chat"
"meta-llama/Llama-2-7b-hf"
"meta-llama/Llama-2-7b-chat-hf"
"tiiuae/falcon-rw-1b"
)

peft_models=(
"llm-jp/llm-jp-13b-v1.0 llm-jp/llm-jp-13b-instruct-lora-jaster-v1.0"
"llm-jp/llm-jp-13b-v1.0 llm-jp/llm-jp-13b-instruct-lora-jaster-dolly-oasst-v1.0"
"llm-jp/llm-jp-13b-v1.0 llm-jp/llm-jp-13b-instruct-lora-dolly-oasst-v1.0"
)

for model_name in "${original_models[@]}"
do
    echo ${model_name}

    result_path=results/${model_name}
    mkdir -p ${result_path}
    bash run_open_llm_leaderboard.sh -w ${model_name} 2 ${result_path}
done

for peft_model_name in "${peft_models[@]}"
do
    base_peft=(${peft_model_name})
    echo "pretrained model ${base_peft[0]}"
    echo "peft_model ${base_peft[1]}"

    result_path=results/${base_peft[1]}
    mkdir -p ${result_path}
    bash run_open_llm_leaderboard.sh -w -l ${base_peft[0]} ${base_peft[1]} 2 ${result_path}
done
```

## タスクを追加検証したい場合
既に wandb に登録済みのモデルに対して、新たにタスクを追加したい場合は下記のスクリプトを実行してください。

```bash
cp ./run_open_llm_leaderboard_add_task.sh ./additional_save_wandb.py ./lm-evaluation-harness/
cd lm-evaluation-harness
bash run_open_llm_leaderboard_add_task.sh -w {検証したいモデル名} {バッチサイズ} {出力ディレクトリ}
```

パラメータなどは `run_open_llm_leaderboard.sh` と同様です。
（環境変数の設定も同様に実施してください）

（wandb へ結果を保存するスクリプトを `save_wandb.py` から `additional_save_wandb.py` へ変更しています。）

## その他の想定されるエラー
日本語モデルを評価する際、次のようなエラーが出る場合があります。

- gsm8k 実施時に ValueError が出る

### gsm8k 実施時に ValueError が出る

gsm8k 実施時に以下のようなエラーが出る場合があります。

```
(primary_until,) = self.tok_encode(until[0])
ValueError: too many values to unpack (expected 1)
```

このエラーは lm-evaluation-harness の `hf-causal` を使用している際に発生することがあります。
https://github.com/EleutherAI/lm-evaluation-harness/issues/628

このエラーが発生した場合、 `run_open_llm_leaderboard_add_task.sh` 中の `HARNESS_MODEL_TYPE=hf-causal-experimental` と設定して再度動かしてください。

（デフォルトは Huggingface の Open LLM Leaderboard に従って `hf-causal` にしています。）
