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

generative_benchmarks="[
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