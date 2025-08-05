#!/bin/bash
# muP hyperparameter transfer with MOE

for width in 128 256 512 1024
do
    for lr in 0.125 0.0625 0.03125 0.015625 0.0078125 0.00390625 0.001953125 0.0009765625 0.00048828125 0.000244140625 0.0001220703125 0.00006103515625
    do
        for num_exp in 4 8 16
        do
            for seed in 1 2 3
            do
                head_size=64
                n_heads=$((width / head_size))
                mup_base_width=256
                mup_width_multiplier=$(echo "scale=8; $width/$mup_base_width" | bc -l)
                num_act=2  # top-k experts
                out_dir="mup_examples/mutransfer_lr_shakespeare_char/mup_moe/out/width${width}_depth2_experts${num_exp}_active${num_act}_seed${seed}_lr${lr}"
                python train.py \
                    --out_dir=$out_dir \
                    --eval_interval=1 \
                    --log_interval=1 \
                    --eval_iters=1 \
                    --eval_only=False \
                    --skip_val_loss=True \
                    --always_save_checkpoint=False \
                    --never_save_checkpoint=True \
                    --init_from='scratch' \
                    --wandb_log=False \
                    --csv_log=True \
                    --dataset='shakespeare_char' \
                    --gradient_accumulation_steps=8 \
                    --batch_size=1 \
                    --block_size=1024 \
                    --n_layer=2 \
                    --n_head=$n_heads \
                    --n_embd=$width \
                    --dropout=0.0 \
                    --bias=False \
                    --init_std=0.02 \
                    --learning_rate=$lr \
                    --max_iters=122 \
                    --weight_decay=1e-1 \
                    --beta1=0.9 \
                    --beta2=0.95 \
                    --grad_clip=1.0 \
                    --decay_lr=False \
                    --mup_enabled=True \
                    --mup_width_multiplier=$mup_width_multiplier \
                    --mup_input_alpha=1.0 \
                    --mup_output_alpha=1.0 \
                    --num_exp=$num_exp \
                    --num_act=$num_act \
                    --moe_tau=1.0 \
                    --moe_bias_lr=1e-2 \
                    --seed=$seed \
                    --backend='nccl' \
                    --device='cuda' \
                    --dtype='float32' \
                    --compile=False
            done
        done
    done
done