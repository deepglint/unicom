export PYTHONPATH=$(pwd)
model_path="model path"
conv_template='qwen_1_5'
run_port=12444
model_name='mlcd'

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m accelerate.commands.launch \
    --main_process_port=$run_port \
    --num_processes=8 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained=$model_path,conv_template=$conv_template \
    --tasks mmbench,mme,mmmu,ocrbench,scienceqa,scienceqa_img,seedbench,gqa,pope,textvqa_val,ai2d,chartqa,docvqa_val,infovqa_val,realworldqa,mmstar \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix $model_name \
    --output_path ./eval_log/ 
