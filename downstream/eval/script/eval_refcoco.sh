json_path=./eval
gpu_num=8
result_name=./eval/results
train_image_path=/vlm/kunwu/data/llava_train_img/glamm_data

model_name=DeepGlint-AI/MLCD-Seg-7B
echo $model_name

./eval/script/eval_multiprocess.sh $model_name $json_path/refcoco.json $result_name/$model_name/refcoco $train_image_path "" $gpu_num 0.2
python ./eval/eval/evaluate_refcoco.py --result-dir $result_name/$model_name/refcoco