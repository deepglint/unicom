json_path=./eval
gpu_num=8
checkpoints_name=./checkpoints/
result_name=./eval/results

model_name=llava-seg-DeepGlint-AI_mlcd-vit-large-patch14-336-Qwen_Qwen2.5-7B-Instruct-1.8m
echo $model_name

./eval/script/eval_multiprocess.sh $checkpoints_name/$model_name $json_path/refcoco.json $result_name/$model_name/refcoco /vlm/kunwu/data/llava_train_img/glamm_data "" $gpu_num 0.2
python ./eval/eval/evaluate_refcoco.py --result-dir $result_name/$model_name/refcoco
