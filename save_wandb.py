"""lm-evaluation-harness の結果を wandb にアップロードするスニペット

アップロードする項目は次の通り:
- テストデータに対する各指標
- lm-evaluation-harness の json ファイル一覧
- lm-evaluation-harness の commit-id
- lm-evaluation-harness の実行時間
- lm-evaluation-harness の引数（評価対象のモデル、バッチサイズ）
- lm-evaluation-hanress の write_out 出力ファイル
"""
import argparse
import json
import glob
import os
import math
import warnings
from dataclasses import dataclass
from typing import Dict, Optional

import wandb


WANDB_TABLE_NAME = "open-llm-leaderboard"
TASK_METRIC_MAPPING = {  # https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard
    "arc-challenge": "acc_norm",
    "hellaswag": "acc_norm",
    "truthfulqa-mc": "mc2",
    "mmlu": "acc",
    "winogrande": "acc",
    "gsm8k": "acc",
    "drop": "f1",
}


@dataclass
class SingleTaskResult:
    results: Dict[str, Dict[str, float]]
    result_json_file: str
    write_out_dir: Optional[str]
    config: Dict


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, required=True, help="lm-evaluation-harness の評価結果出力ディレクトリ")
    parser.add_argument("--is_write_out", action="store_true", help="lm-evaluation-harness の write_out を保存するかどうか")
    parser.add_argument("--target_model", type=str, required=True, help="評価対象のモデル")
    parser.add_argument("--batch_size", type=int, required=True, help="評価時のバッチサイズ")
    parser.add_argument("--commit_id", type=str, required=True, help="lm-evaluation-harness の commit-id")
    parser.add_argument("--elapsed_time", type=int, required=True, help="lm-evaluation-harness の実行時間 [sec]")
    parser.add_argument("--wandb_entity_name", required=True, help="WandB の Entity 名")
    parser.add_argument("--wandb_project_name", required=True, help="WandB の Project 名")
    return parser.parse_args()


def load_results(result_dir: str, is_write_out: bool, target_model: str) -> Dict[str, SingleTaskResult]:
    """lm-evaluation-harness の結果を読み込む

    Returns:
        Dict[str, SingleTaskResult]: タスク名をキーとした結果の辞書
    """
    result_files = glob.glob(f"{result_dir}/*.json")
    results: Dict[str, SingleTaskResult] = {}
    for result_file in result_files:
        with open(result_file) as f:
            result = json.load(f)

        task_name = os.path.basename(result_file).replace(".json", "")
        task_scores = result["results"]
        config = result["config"]
        write_out_dir = os.path.join("write_out", target_model, task_name) if is_write_out else None

        results[task_name] = SingleTaskResult(
            results=task_scores,
            result_json_file=result_file,
            write_out_dir=write_out_dir,
            config=config,
        )
    return results


def upload_wandb(
    data: Dict[str, SingleTaskResult],
    entity_name: str,
    project_name: str,
    target_model: str,
    commit_id: str,
    elapsed_time: int,
    batch_size: int,
) -> None:
    """wandb に Upload を試みる

    wandb に保存する内容は次の通り
    - テストデータに対する各指標（avg 含む）: Table として保存する
    - lm-evaluation-harness の json ファイル一覧: artifact として保存する
    - lm-evaluation-harness の commit-id: config
    - lm-evaluation-harness の実行時間: Table
    - lm-evaluation-harness の引数（評価対象のモデル、バッチサイズ）: config（モデル名は Table にも保存）
    - lm-evaluation-hanress の write_out 出力ファイル: artifact
    """
    def post_process_results(lm_evaluation_results: Dict[str, Dict[str, float]], task_name: str) -> float:
        """lm-evaluation-harness で出力された結果を平均を利用して単一の float 値にする"""
        scores = []  # 設定した評価指標のスコアが格納される
        for _, metric_score in lm_evaluation_results.items():
            target_metric = TASK_METRIC_MAPPING[task_name]
            score: float = metric_score[target_metric]
            if math.isnan(score):
                warnings.warn(f"Task: {task_name} において、指標 {target_metric} が NaN でした。")
            scores.append(score)
        return sum(scores) / len(scores)

    def extract_dir_info(data: Dict[str, SingleTaskResult], data_type: str) -> str:
        """lm-evaluation-harness の出力ディレクトリを抽出する"""
        if data_type == "result":
            return os.path.dirname(data[list(data.keys())[0]].result_json_file)
        elif data_type == "output":
            return os.path.dirname(data[list(data.keys())[0]].write_out_dir)
        else:
            raise ValueError(f"Invalid data_type: {data_type}")

    wandb_config = {
        "lm_evaluation_harness_commit_id": commit_id,
        "target_model": target_model,
        "batch_size": batch_size,
    }
    with wandb.init(
        entity=entity_name, project=project_name, name=target_model, config=wandb_config
    ) as run:
        columns = ["model_name", "Average"]  # model_name + avg_score + task_name + elapsed_time
        items = [target_model]  # model_name + avg_score (insert) + task_name + elapsed_time
        for task_name, result in data.items():
            # make table
            columns.append(task_name)
            result_for_table = post_process_results(result.results, task_name)
            items.append(result_for_table)

        scores_without_nan = [score for score in items[1:] if not math.isnan(score)]
        avg_score = sum(scores_without_nan) / len(scores_without_nan)
        items.insert(1, avg_score)
        # Add time column
        columns.append("Elapsed Time")
        items.append(elapsed_time)

        # Add 1D array and save table
        lm_evaluation_harness_table = wandb.Table(columns=columns, data=[items])
        run.log({WANDB_TABLE_NAME: lm_evaluation_harness_table})

        # save artifact
        result_json_dir = extract_dir_info(data, data_type="result")
        result_artifact_name = target_model.replace("/", ".") + ".result"
        result_artifact = wandb.Artifact(
            result_artifact_name, type="lm-evaluation-harness-result"
        )
        result_artifact.add_dir(result_json_dir)
        wandb.log_artifact(result_artifact)
        if result.write_out_dir is not None:
            result_output_dir = extract_dir_info(data, data_type="output")
            output_artifact_name = target_model.replace("/", ".") + ".output"
            output_artifact = wandb.Artifact(
                output_artifact_name, type="lm-evaluation-harness-output"
            )
            output_artifact.add_dir(result_output_dir)
            wandb.log_artifact(output_artifact)

    print("Finish Upload.")


def main():
    args = get_args()

    # lm-evaluation-harness の結果を読み込む
    data = load_results(args.result_dir, args.is_write_out, args.target_model)

    # wandb にアップロードする
    upload_wandb(
        data, args.wandb_entity_name, args.wandb_project_name, args.target_model, args.commit_id, args.elapsed_time, args.batch_size
    )


if __name__ == "__main__":
    main()
