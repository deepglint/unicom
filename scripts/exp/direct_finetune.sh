############### Finetune ################

export BASE_RUN_NAME=direct_finetune_Llava-Onevision-baseline-qwen2

export LLM_VERSION=/home/vlm/pretrain_model/Qwen2-7B-Instruct
export VISION_MODEL_VERSION=/home/vlm/pretrain_model/siglip-so400m-patch14-384

export MM_PROJECTOR=mm-projection_Qwen2-7B-Instruct_siglip-so400m-patch14-384
export PRETRAIN_MM_MLP_ADAPTER=/home/vlm/workspace/checkpoints/projectors/${MM_PROJECTOR}/mm_projector.bin

export DATA_PATH=/home/vlm/finetune_json/yaml/llava1008k_robovqa800k.yaml
export IMAGE_FOLDER=/home/vlm/train_images
export VIDEO_FOLDER=/home/vlm/train_videos
export OUTPUT_DIR=/home/vlm/workspace/checkpoints/${BASE_RUN_NAME}

export PROMPT_VERSION=qwen_2

export IMAGE_ASPECT_RATIO=anyres
export MM_TUNABLE_PARTS="mm_mlp_adapter,mm_language_model"
export IMAGE_GRID_PINPOINTS="[(384, 768), (768, 384), (768, 768), (1152, 384), (384, 1152)]"

export NUM_GPUS=8
export NNODES=4
export HOSTFILE=/home/vlm/workspace/hostfile/hostfile_group2_4

export DATA_WORKERS=4
export DEV_BATCHSIZE=4
export GRAD_ACC_STEPS=2

export LEARNING_RATE=2e-5
export VIT_LEARNING_RATE=2e-6
export MAX_SEQ_LEN=4096
export MAX_IMAGE_NUM=16
export ZERO_VERSION=3

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir "$OUTPUT_DIR"
fi

deepspeed --num_gpus $NUM_GPUS --num_nodes $NNODES --hostfile $HOSTFILE \
    llava/train/train_mem.py \
    --deepspeed scripts/zero${ZERO_VERSION}.json \
    --model_name_or_path $LLM_VERSION \
    --version $PROMPT_VERSION \
    --data_path $DATA_PATH \
    --image_folder $IMAGE_FOLDER \
    --video_folder $VIDEO_FOLDER \
    --pretrain_mm_mlp_adapter $PRETRAIN_MM_MLP_ADAPTER \
    --mm_tunable_parts $MM_TUNABLE_PARTS \
    --mm_vision_tower_lr $VIT_LEARNING_RATE \
    --vision_tower $VISION_MODEL_VERSION \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio $IMAGE_ASPECT_RATIO \
    --image_grid_pinpoints "$IMAGE_GRID_PINPOINTS" \
    --bf16 True \
    --run_name $BASE_RUN_NAME \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size $DEV_BATCHSIZE \
    --per_device_eval_batch_size $DEV_BATCHSIZE \
    --gradient_accumulation_steps $GRAD_ACC_STEPS \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 3000 \
    --save_total_limit 1 \
    --learning_rate $LEARNING_RATE \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length $MAX_SEQ_LEN \
    --gradient_checkpointing True \
    --dataloader_num_workers $DATA_WORKERS \
    --lazy_preprocess True \
    --report_to wandb \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --attn_implementation sdpa \
    --frames_upbound $MAX_IMAGE_NUM  \
    --max_num_images $MAX_IMAGE_NUM \
    2>&1 | tee $OUTPUT_DIR/train.log

# You can delete the sdpa attn_implementation if you want to use flash attn
