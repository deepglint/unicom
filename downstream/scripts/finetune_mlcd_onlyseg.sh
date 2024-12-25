export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
# export NCCL_DEBUG=INFO
export NUM_GPUS=8
export NNODES=1
export RANK=0
export ADDR="localhost"
export PORT="29500"
export PYTHONPATH=$(pwd)

LLM_VERSION="Qwen/Qwen2.5-7B-Instruct"
# for 7b model we recommend bs=1, accum=2, 16 nodes, 128 gpus, lr=1e-5, warmup=0.03
# for 72b model we recommend bs=1, accum=1, 32 nodes, 256 gpus, lr=1e-5, warmup=0.03
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="DeepGlint-AI/mlcd-vit-large-patch14-336"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

############### Pretrain ################

BASE_RUN_NAME="llavanext-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-ov-robo"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

############### Finetune ################

# Stage 2
PROMPT_VERSION="qwen_1_5"
RUN_NAME="llava-seg-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-1.8m_miniHD" 
PREV_STAGE_CHECKPOINT="DeepGlint-AI/MLCD-Embodied-7B" # replace it with your last checkpoint training from single image collection
DATA_ROOT="llava-next and other data root"
echo "PREV_STAGE_CHECKPOINT: ${PREV_STAGE_CHECKPOINT}"
echo "MID_RUN_NAME: ${RUN_NAME}"

ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node="${NUM_GPUS}" --nnodes="${NNODES}" --node_rank="${RANK}" --master_addr="${ADDR}" --master_port="${PORT}" \
    llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path $PREV_STAGE_CHECKPOINT \
    --version $PROMPT_VERSION \
    --sam_path ./checkpoints/sam_vit_h_4b8939.pth \
    --data_path ${DATA_ROOT}/seg.json \
    --image_folder ${DATA_ROOT} \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model,sam" \
    --mm_vision_tower_lr=2e-6 \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres \
    --image_grid_pinpoints  "[[336, 336],[336,672],[336,1008],[336,1344],[672,336],[672,672],[672,1008],[1008,336],[1008,672],[1344,336]]" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name $RUN_NAME \
    --output_dir ./checkpoints/seg/$RUN_NAME \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 6000 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --torch_compile_backend "inductor" \
    --lazy_preprocess True \
    --report_to none \
    --torch_compile True \
    --dataloader_drop_last True \
    --frames_upbound 32
    2>&1 | tee -a $RUN_NAME/train.log
exit 0;
# You can delete the sdpa attn_implementation if you want to use flash attn