#!/bin/bash

# Set the Python path to the current directory
export PYTHONPATH=$(pwd)

# Define the model path, conversation template, port, and model name
MODEL_PATH="checkpoints/llavanext-DeepGlint-AI_mlcd-vit-bigG-patch14-448-Qwen_Qwen2.5-7B-Instruct-mlp2x_gelu-pretrain_blip558k-finetune_llavanext780k"  # Replace with your actual model path
CONV_TEMPLATE='qwen_1_5'
RUN_PORT=12444
MODEL_NAME='mlcd'

# Define the tasks to be evaluated
TASKS="mmbench,mme,mmmu,ocrbench,scienceqa,scienceqa_img,seedbench,gqa,pope,textvqa_val,ai2d,chartqa,docvqa_val,infovqa_val,realworldqa,mmstar"

# Set the GPUs to be used
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 

# Run the evaluation script with the specified parameters
python -m accelerate.commands.launch \
    --main_process_port=$RUN_PORT \
    --num_processes=8 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained=$MODEL_PATH,conv_template=$CONV_TEMPLATE \
    --tasks $TASKS \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix ${MODEL_NAME}_$(date +%Y%m%d) \
    --output_path ./eval_log/${MODEL_NAME}/
