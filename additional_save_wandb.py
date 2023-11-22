"""追加で実施した lm-evaluation-harness の結果を wandb に Upload する

注意事項:
- batch_size, commit_id は、lm-evaluation-harness の実行時のものを指定すること
- is_write_out もできれば lm-evaluation-harness の実行時のものを指定すること
- average は追加したタスクを反映させた結果が上書きされる
- artifact は追加で実施した lm-evaluation-harness の結果のみ Upload される（ただし、以前に実行した結果がローカルに残っている場合は、それも Upload される）
  - 古い結果は wandb の UI 上で version を選択して確認する
"""
import argparse
import json
import os
import math
import warnings
from typing import List, Dict

import wandb

from save_wandb import WANDB_TABLE_NAME, TASK_METRIC_MAPPING, SingleTaskResult


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, required=True, help="lm-evaluation-harness の評価結果出力ディレクトリ")
    parser.add_argument("--is_write_out", action="store_true", help="lm-evaluation-harness の write_out を保存するかどうか")
    parser.add_argument("--target_model", type=str, required=True, help="評価対象のモデル")
    parser.add_argument("--elapsed_time", type=int, required=True, help="lm-evaluation-harness の追加実行時間 [sec]")
    parser.add_argument("--wandb_entity_name", required=True, help="WandB の Entity 名")
    parser.add_argument("--wandb_project_name", required=True, help="WandB の Project 名")
    parser.add_argument("--tasks", nargs="+", required=True, help="追加評価対象のタスク名")
    return parser.parse_args()


def load_results(result_dir: str, is_write_out: bool, target_model: str, tasks: List[str]) -> Dict[str, SingleTaskResult]:
    """tasks に含まれるタスクの結果を読み込む"""
    result_files = [f"{result_dir}/{task}.json" for task in tasks]
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
    elapsed_time: int,
) -> None:
    """wandb に Upload を試みる"""
    def get_run(entity_name: str, project_name: str, target_model: str) -> wandb.apis.public.Run:
        api = wandb.Api()
        runs = api.runs(f"{entity_name}/{project_name}", filters={"config.target_model": target_model})
        assert len(runs) == 1, f"len(runs) must be 1, but {len(runs)}"
        run = list(runs)[0]
        return run

    def post_process_results(lm_evaluation_results: Dict[str, Dict[str, float]], task_name: str) -> float:
        """lm-evaluation-harness で出力された結果を平均して単一の値にする"""
        scores = []
        for _, metric_score in lm_evaluation_results.items():
            target_metric = TASK_METRIC_MAPPING[task_name]
            score: float = metric_score[target_metric]
            if math.isnan(score):
                warnings.warn(f"Task: {task_name} において、指標 {target_metric} が NaN でした。")
            scores.append(score)
        return sum(scores) / len(scores)

    def extract_dir_info(data: Dict[str, SingleTaskResult], data_type: str) -> str:
        """lm-evaulation-harness の出力ディレクトリを抽出する"""
        if data_type == "result":
            return os.path.dirname(data[list(data.keys())[0]].result_json_file)
        elif data_type == "output":
            return os.path.dirname(data[list(data.keys())[0]].write_out_dir)
        else:
            raise ValueError(f"Invalid data_type: {data_type}")

    def get_new_average(updated_table: wandb.Table) -> float:
        """average を再計算する

        除くカラム: model_name, Average, Elapsed Time と nan と None が含まれるカラム
            wandb は nan の値を None として扱うため、None が含まれるカラムも除く
        """
        exclude_columns = ["model_name", "Average", "Elapsed Time"]
        target_scores = updated_table.data[0]
        target_scores = [
            score
            for column, score in zip(updated_table.columns, target_scores)
            if column not in exclude_columns and score is not None and not math.isnan(score)
        ]
        return sum(target_scores) / len(target_scores)

    run = get_run(entity_name, project_name, target_model)
    run_id = run.id
    run_config = json.loads(run.json_config)  # {Key: {"value": value}, {"desc": desc}, ...}
    wandb_config = {key: value["value"] for key, value in run_config.items()}
    with wandb.init(id=run_id, project=project_name, entity=entity_name, config=wandb_config) as run:
        # get table
        table_name = WANDB_TABLE_NAME.replace("-", "")  # wandb.Table に渡すときに - が消えるので、wandb.Table から取得するときには - を消す
        artifact_name = f"{entity_name}/{project_name}/run-{run.id}-{table_name}:latest"
        api_artifact = wandb.Api().artifact(artifact_name)
        table = run.use_artifact(api_artifact).get(WANDB_TABLE_NAME)

        # update average, elapsed time and task
        updated_table = wandb.Table(columns=table.columns, data=table.data)
        column2index = {column: index for index, column in enumerate(updated_table.columns)}

        updated_table.data[0][column2index["Elapsed Time"]] += elapsed_time

        for task_name, task_result in data.items():
            updated_table.add_column(task_name, [post_process_results(task_result.results, task_name)])

        # average の再計算
        # model_name と古い Average と elapsed_time を除いて、average を計算する
        updated_table.data[0][column2index["Average"]] = get_new_average(updated_table)

        # update table
        run.log({WANDB_TABLE_NAME: updated_table})

        # save artifact
        result_json_dir = extract_dir_info(data, data_type="result")
        result_artifact_name = target_model.replace("/", ".") + ".result"
        result_artifact = wandb.Artifact(
            result_artifact_name, type="lm-evaluation-harness-result"
        )
        result_artifact.add_dir(result_json_dir)
        wandb.log_artifact(result_artifact)
        if task_result.write_out_dir is not None:
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

    # 追加で実施された lm-evaluation-harness の結果を読み込む
    data = load_results(args.result_dir, args.is_write_out, args.target_model, args.tasks)

    # wandb に Upload する
    upload_wandb(
        data, args.wandb_entity_name, args.wandb_project_name, args.target_model, args.elapsed_time
    )


if __name__ == "__main__":
    main()
