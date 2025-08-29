#!/bin/bash
# muP hyperparameter transfer with MOE - Bash version
# Multi-GPU support: set NGPUS environment variable (default: 1)
# Usage: NGPUS=4 ./run.sh

NGPUS=${NGPUS:-1}
if [ $NGPUS -gt 1 ]; then
    echo "Running with $NGPUS GPUs using torchrun"
    LAUNCHER="torchrun --standalone --nproc_per_node=$NGPUS"
    # Note: gradient_accumulation_steps is automatically adjusted by train.py for DDP
else
    echo "Running on single GPU"
    LAUNCHER="python3"
fi

timestamp=$(python -c "from datetime import datetime; print(datetime.now().strftime('%Y%m%d_%H%M%S'))")


for width in 256 512 1024 2048
do
    for num_exp in 8 4 2
    do
        for lr in 0.125 0.0625 0.03125 0.015625 0.0078125 0.00390625 0.001953125 0.0009765625 0.00048828125 0.000244140625 0.0001220703125 0.00006103515625
        do
            for seed in 1 2 3
            do
                head_size=64
                n_heads=$((width / head_size))
                min_lr=$lr
                mup_base_width=256
                mup_width_multiplier=$(echo "scale=8; $width/$mup_base_width" | bc -l)
                num_act=$((num_exp/2))
                out_dir="run_data/mutransfer_lr_owt/out_${timestamp}/width${width}_depth${LAYERS}_experts${num_exp}_active${num_act}_seed${seed}_lr${lr}"
                $LAUNCHER train.py \
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
                    --dataset='openwebtext' \
                    --gradient_accumulation_steps=8 \
                    --batch_size=256 \
                    --block_size=1024 \
                    --n_layer=8 \
                    --n_head=$n_heads \
                    --n_embd=$width \
                    --dropout=0.0 \
                    --bias=False \
                    --init_std=0.02 \
                    --learning_rate=$lr \
                    --lr_decay_iters=1000 \
                    --min_lr=$min_lr \
                    --max_iters=1000 \
                    --weight_decay=0.0 \
                    --beta1=0.9 \
                    --beta2=0.95 \
                    --grad_clip=3.0 \
                    --decay_lr=False \
                    --mup_enabled=True \
                    --mup_width_multiplier=$mup_width_multiplier \
                    --mup_input_alpha=1.0 \
                    --mup_output_alpha=1.0 \
                    --num_exp=$num_exp \
                    --num_act=$num_act \
                    --moe_tau=1.0 \
                    --moe_bias_lr=1e-2 \
                    --moe_bias_momentum=0.5 \
                    --moe_bias_momentum_enabled=True \
                    --moe_load_balance_method='bias' \
                    --moe_aux_loss_weight=1.0 \
                    --seed=$seed \
                    --backend='nccl' \
                    --device='cuda' \
                    --dtype='float16' \
                    --compile=True
            done
        done
    done
done