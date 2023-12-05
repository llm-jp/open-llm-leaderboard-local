"""wandb の average に対して、特定のカラムを除いた値を Upload する"""
import argparse
import json
import math
from typing import List

import wandb

from save_wandb import WANDB_TABLE_NAME


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_model", type=str, required=True, help="評価対象のモデル")
    parser.add_argument("--wandb_entity_name", required=True, help="WandB の Entity 名")
    parser.add_argument("--wandb_project_name", required=True, help="WandB の Project 名")
    parser.add_argument("--exclude_tasks", nargs="+", required=True, help="平均から取り除きたいタスク名")
    return parser.parse_args()


def update_average(
    entity_name: str,
    project_name: str,
    target_model: str,
    exclude_tasks: List[str],
) -> None:
    """wandb の Average の更新を行う"""
    def get_run(entity_name: str, project_name: str, target_model: str) -> wandb.apis.public.Run:
        api = wandb.Api()
        runs = api.runs(f"{entity_name}/{project_name}", filters={"config.target_model": target_model})
        assert len(runs) == 1, f"len(runs) must be 1, but {len(runs)}"
        run = list(runs)[0]
        return run

    def get_new_average(updated_table: wandb.Table, exclude_tasks: List[str]) -> float:
        """average を再計算する

        除くカラム: model_name, Average, Elapsed Time と nan と None が含まれるカラム
            wandb は nan の値を None として扱うため、None が含まれるカラムも除く
        """
        assert all([column in updated_table.columns for column in exclude_tasks]), f"exclude_tasks must be in {updated_table.columns}"
        exclude_columns = ["model_name", "Average", "Elapsed Time"]
        exclude_columns.extend(exclude_tasks)
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

        # average の再計算
        # model_name と古い Average と elapsed_time と exclude_tasks を除いて、average を計算する
        updated_table.data[0][column2index["Average"]] = get_new_average(updated_table, exclude_tasks)

        # update table
        run.log({WANDB_TABLE_NAME: updated_table})

    print("Finish Upload.")



def main():
    args = get_args()

    # wandb の結果を更新する
    update_average(
        args.wandb_entity_name, args.wandb_project_name, args.target_model, args.exclude_tasks
    )


if __name__ == "__main__":
    main()
