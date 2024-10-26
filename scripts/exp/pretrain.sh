############### Pretrain ################

export BASE_RUN_NAME=mm-projection_Qwen2-7B-Instruct_siglip-so400m-patch14-384

export LLM_VERSION=/home/vlm/pretrain_model/Qwen2-7B-Instruct
export VISION_MODEL_VERSION=/home/vlm/pretrain_model/siglip-so400m-patch14-384

export DATA_PATH=/home/vlm/pretrain_json/blip_laion_cc_sbu_558k.json
export IMAGE_FOLDER=/home/vlm/train_images/pretrain_laion
export OUTPUT_DIR=/home/vlm/workspace/checkpoints/projectors/${BASE_RUN_NAME}

export PROMPT_VERSION=plain

export MM_TUNABLE_PARTS="mm_mlp_adapter"

export NUM_GPUS=8
export NNODES=4
export HOSTFILE=/home/vlm/workspace/hostfile/hostfile_group2_4

export DATA_WORKERS=4
export DEV_BATCHSIZE=4
export GRAD_ACC_STEPS=8

export LEARNING_RATE=1e-3
export MAX_SEQ_LEN=4096
export ZERO_VERSION=2

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
    --vision_tower $VISION_MODEL_VERSION \
    --mm_tunable_parts $MM_TUNABLE_PARTS \
    --output_dir $OUTPUT_DIR \
    --per_device_eval_batch_size $DEV_BATCHSIZE \
    --per_device_train_batch_size $DEV_BATCHSIZE \
    --gradient_accumulation_steps $GRAD_ACC_STEPS \
    --model_max_length $MAX_SEQ_LEN \
    --dataloader_num_workers $DATA_WORKERS \
    --learning_rate $LEARNING_RATE \
    --mm_vision_select_layer -2 \
    --mm_projector_type mlp2x_gelu \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --num_train_epochs 1 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --save_steps 50000 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name $BASE_RUN_NAME \
    --attn_implementation sdpa \
    2>&1 | tee $OUTPUT_DIR/train.log

# You can delete the sdpa attn_implementation if you want to use flash attn