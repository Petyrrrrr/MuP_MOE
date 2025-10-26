#!/bin/bash
# muP hyperparameter transfer with MOE - Multi-GPU DDP version

# Number of GPUs to use for DDP training
NUM_GPUS=4
export CUDA_VISIBLE_DEVICES=0,1,2,3
LAUNCHER="torchrun --standalone --nproc_per_node=$NUM_GPUS"

timestamp=$(python -c "from datetime import datetime; print(datetime.now().strftime('%Y%m%d_%H%M%S'))")

max_iters=5000
warmup_iters=300
router_lr_mult=0.5
init_std=0.02
moe_tau=0.02
n_layer=14

total_batch_size=480
gradient_accumulation_steps=12
batch_size=40

t_ema_inv=0.0
bias_update_interval=1
moe_bias_lr_mult=1.0
for width in 512
do
    for num_exp in 16
    do
        for lr in 0.007
        do
            for seed in 1
            do
                head_size=64
                n_heads=$((width / head_size))
                min_lr=$lr
                mup_base_width=256
                mup_width_multiplier=$(echo "scale=8; $width/$mup_base_width" | bc -l)
                num_act=$((num_exp/4))
                weight_decay=0.0
                moe_bias_lr=0.1
                out_dir="run_data/mutransfer_lr_cccc/out_${timestamp}/width${width}_depth${n_layer}_experts${num_exp}_active${num_act}_seed${seed}_lr${lr}"

                $LAUNCHER train.py \
                    --out_dir=$out_dir \
                    --eval_interval=1 \
                    --log_interval=1 \
                    --eval_iters=$((100 * gradient_accumulation_steps / NUM_GPUS)) \
                    --eval_only=False \
                    --skip_val_loss=True \
                    --always_save_checkpoint=False \
                    --never_save_checkpoint=True \
                    --init_from='scratch' \
                    --csv_log=True \
                    --warmup_iters=$warmup_iters \
                    --dataset='cccc' \
                    --gradient_accumulation_steps=$gradient_accumulation_steps \
                    --batch_size=$batch_size \
                    --block_size=1024 \
                    --n_layer=$n_layer \
                    --n_head=$n_heads \
                    --n_embd=$width \
                    --dropout=0.0 \
                    --bias=False \
                    --init_std=$init_std \
                    --learning_rate=$lr \
                    --lr_decay_iters=2000 \
                    --min_lr=$min_lr \
                    --max_iters=$max_iters \
                    --weight_decay=$weight_decay \
                    --beta1=0.9 \
                    --beta2=0.95 \
                    --grad_clip=3.0 \
                    --decay_lr=True \
                    --mup_enabled=True \
                    --mup_width_multiplier=$mup_width_multiplier \
                    --mup_input_alpha=1.0 \
                    --mup_output_alpha=1.0 \
                    --num_exp=$num_exp \
                    --num_act=$num_act \
                    --moe_tau=$moe_tau \
                    --moe_bias_lr=$moe_bias_lr \
                    --moe_bias_momentum=0.8 \
                    --router_lr_mult=$router_lr_mult \
                    --moe_bias_momentum_enabled=True \
                    --moe_load_balance_method='bias' \
                    --moe_aux_loss_weight=1.0 \
                    --seed=$seed \
                    --alpha=1.0 \
                    --backend='nccl' \
                    --device='cuda' \
                    --dtype='bfloat16' \
                    --compile=False \
                    --bias_update_interval=$bias_update_interval \
                    --wandb_log=False \
                    --wandb_project=cccc_lr_7e-3 \
                    --wandb_run_name=width${width}_e${num_exp}_a${num_act}_server_a \
                    >> /home/ubuntu/MuP_MOE/std_out/debugged_outlog_cccc_${timestamp}
            done
        done
    done
done