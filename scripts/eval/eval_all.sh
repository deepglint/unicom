#!/bin/bash -e

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <MODEL_DIR> <MAX_NUM_IMAGES>"
    exit 1
fi

model_dir=$1
max_num_images=$2
if [ ! -d "$model_dir" ]; then
    echo "Error: $model_dir does not exist"
    exit 1
fi

export PYTHONPATH=$(pwd)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_PORT=29501

benchmarks="[
'ai2d',
'mme',
'mmbench.cc.test',
'mmbench.cn.dev',
'mmbench.cn.test',
'mmbench.en.dev',
'mmbench.en.test',
'seed-bench',
'scienceqa.test',
'scienceqa.validation',
'hallusion',
'mmstar',
'nlvr2',
'mmmu.validation',
'mmmu.dev',
'cmmmu.val',
'cmmmu.dev',
'mantis',
'q-bench2.dev',
'mmlu.dev',
'mmlu.test',
'mmlu.val',
'cmmlu.dev',
'cmmlu.test'
]"

python llava/benchmark/eval.py \
    --model_dir=$model_dir \
    --max_num_images=$max_num_images \
    --benchmarks="$(echo $benchmarks | tr -d '\n')" \
    --bmk_root=/home/vlm/benchmarks \
    --batch_size=2 \
    --num_workers=2


generative_benchmarks="[
'ocrbench',
'chartqa',
'textvqa',
'docvqa',
'robovqa',
'openeqa'
]"

python llava/benchmark/eval_generative.py \
    --model_dir=$model_dir  \
    --benchmarks="$(echo $generative_benchmarks | tr -d '\n')" \
    --image_folder=/home/vlm/eval_images \
    --max_num_images=$max_num_images \
    --bmk_root=/home/vlm/benchmarks \
    --batch_size=1 \
    --max_new_tokens=64 \
    --num_workers=1

python scripts/eval/compute_scores.py $model_dir/eval
