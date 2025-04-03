#!/bin/bash -e
export PYTHONPATH=$(pwd)

export YOUR_API_KEY="YOUR_API_KEY"
export YOUR_ENDPOINT="YOUR_ENDPOINT"

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <MODEL_DIR>"
    exit 1
fi

model_dir=$1
bmk_root=/vlm/data/benchmarks
image_folder=/vlm/data/eval_data/eval_images

if [ ! -d "$model_dir" ]; then
    echo "Error: $model_dir does not exist"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_PORT=29501

generative_benchmarks="[
'robovqa',
'openeqa'
]"

python llava/benchmark/eval_robo.py \
    --model_dir=$model_dir  \
    --benchmarks="$(echo $generative_benchmarks | tr -d '\n')" \
    --image_folder=$image_folder \
    --bmk_root=$bmk_root \
    --batch_size=1 \
    --max_new_tokens=128 \
    --num_workers=1

python scripts/eval/compute_scores.py $model_dir/eval_robo
