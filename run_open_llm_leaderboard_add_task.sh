#!/bin/bash
set -eu

export HF_HOME=  # Need setting

WANDB_ENTITY=  # Need setting
WANDB_PROJECT=  # Need setting

WRITE_OUT_HARNESS=""
WRITE_OUT_PATH=""
WRITE_OUT_WANDB=""
PEFT_BASE_MODEL=""
while getopts "wl:" OPT
do
    case $OPT in
        w)
            WRITE_OUT_HARNESS="--write_out --output_base_path"
            WRITE_OUT_PATH="./write_out"
            WRITE_OUT_WANDB="--is_write_out"
            ;;
        l)
            PEFT_BASE_MODEL="${OPTARG}";;
    esac
done
shift $((OPTIND - 1))

target_model=$1
batch_size=$2
result_dir=$3

# task-name number-of-shot task-list
n_shot_task=(
"arc-challenge 25 arc_challenge"
"hellaswag 10 hellaswag"
"truthfulqa-mc 0 truthfulqa_mc"
"mmlu 5 hendrycksTest-abstract_algebra,hendrycksTest-anatomy,hendrycksTest-astronomy,hendrycksTest-business_ethics,hendrycksTest-clinical_knowledge,hendrycksTest-college_biology,hendrycksTest-college_chemistry,hendrycksTest-college_computer_science,hendrycksTest-college_mathematics,hendrycksTest-college_medicine,hendrycksTest-college_physics,hendrycksTest-computer_security,hendrycksTest-conceptual_physics,hendrycksTest-econometrics,hendrycksTest-electrical_engineering,hendrycksTest-elementary_mathematics,hendrycksTest-formal_logic,hendrycksTest-global_facts,hendrycksTest-high_school_biology,hendrycksTest-high_school_chemistry,hendrycksTest-high_school_computer_science,hendrycksTest-high_school_european_history,hendrycksTest-high_school_geography,hendrycksTest-high_school_government_and_politics,hendrycksTest-high_school_macroeconomics,hendrycksTest-high_school_mathematics,hendrycksTest-high_school_microeconomics,hendrycksTest-high_school_physics,hendrycksTest-high_school_psychology,hendrycksTest-high_school_statistics,hendrycksTest-high_school_us_history,hendrycksTest-high_school_world_history,hendrycksTest-human_aging,hendrycksTest-human_sexuality,hendrycksTest-international_law,hendrycksTest-jurisprudence,hendrycksTest-logical_fallacies,hendrycksTest-machine_learning,hendrycksTest-management,hendrycksTest-marketing,hendrycksTest-medical_genetics,hendrycksTest-miscellaneous,hendrycksTest-moral_disputes,hendrycksTest-moral_scenarios,hendrycksTest-nutrition,hendrycksTest-philosophy,hendrycksTest-prehistory,hendrycksTest-professional_accounting,hendrycksTest-professional_law,hendrycksTest-professional_medicine,hendrycksTest-professional_psychology,hendrycksTest-public_relations,hendrycksTest-security_studies,hendrycksTest-sociology,hendrycksTest-us_foreign_policy,hendrycksTest-virology,hendrycksTest-world_religions"
"winogrande 5 winogrande"
"gsm8k 5 gsm8k"
)

echo "Target model: ${target_model}"
echo "Batch size: ${batch_size}"
echo "Result dir: ${result_dir}"

if [ "${PEFT_BASE_MODEL}" != "" ]; then
    echo "PEFT base model: ${PEFT_BASE_MODEL}"
    HARNESS_MODEL_TYPE="hf-causal-experimental"
    HARNESS_MODEL_ARGS="pretrained=${PEFT_BASE_MODEL},revision=main,peft=${target_model},trust_remote_code=True"
else
    # default
    HARNESS_MODEL_TYPE="hf-causal"
    HARNESS_MODEL_ARGS="pretrained=${target_model},revision=main,trust_remote_code=True"
fi

WRITE_OUT_PATH=${WRITE_OUT_PATH}/${target_model}
mkdir -p ${WRITE_OUT_PATH}
start_time=`date +%s`
done_task=""
for current_n_shot_task in "${n_shot_task[@]}"
do
        # task-name number-of-shot task-list
        current_task=(${current_n_shot_task})
        echo "---"
        echo "task name: ${current_task[0]}"
        echo "n-shot: ${current_task[1]}"
        echo "task list: ${current_task[2]}"

        if [ "${WRITE_OUT_HARNESS}" != "" ]; then
            WRITE_OUT_PATH=${WRITE_OUT_PATH}/${current_task[0]}
        fi

        output_path="${result_dir}/${current_task[0]}".json
        python main.py \
            --model ${HARNESS_MODEL_TYPE} \
            --model_args ${HARNESS_MODEL_ARGS} \
            --num_fewshot ${current_task[1]} \
            --tasks ${current_task[2]} \
            --batch_size ${batch_size} \
            --output_path ${output_path} \
            ${WRITE_OUT_HARNESS} ${WRITE_OUT_PATH}

        if [ "${WRITE_OUT_HARNESS}" != "" ]; then
            WRITE_OUT_PATH=`dirname ${WRITE_OUT_PATH}`
        fi
        done_task="${done_task} ${current_task[0]}"
done
end_time=`date +%s`
elapsed_time=$((end_time - start_time))

echo "Tasks: ${done_task}"
python additional_save_wandb.py \
  --result_dir ${result_dir}\
  --target_model ${target_model} \
  --elapsed_time ${elapsed_time} \
  --wandb_entity_name ${WANDB_ENTITY} \
  --wandb_project_name ${WANDB_PROJECT} \
  --tasks ${done_task} \
  ${WRITE_OUT_WANDB}

exit
