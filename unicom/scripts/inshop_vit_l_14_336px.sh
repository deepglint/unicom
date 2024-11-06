CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --nproc_per_node 8 --master_port 12345 \
    retrieval.py \
                --batch_size           16 \
                --dataset              inshop \
                --debug                0 \
                --epochs               128 \
                --lr                   1e-05 \
                --lr_pfc_weight        10.0 \
                --input_size           336 \
                --gradient_acc         1 \
                --model_name           ViT-L/14@336px \
                --margin_loss_m1       1.0 \
                --margin_loss_m2       0.25 \
                --margin_loss_m3       0.0 \
                --margin_loss_s        32.0 \
                --margin_loss_filter   0.0 \
                --num_workers          4 \
                --num_feat             512 \
                --optimizer            adamw \
                --output_dim           768 \
                --output               /tmp/tmp_for_training \
                --resume               NULL \
                --sample_rate          1.0 \
                --seed                 1024 \
                --transform            timm \
                --weight_decay         0 >> l14_336px_inshop.log
