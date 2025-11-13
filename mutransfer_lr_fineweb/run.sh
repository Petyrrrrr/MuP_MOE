#!/bin/bash
# muP hyperparameter transfer with MOE - Multi-GPU DDP version

# Number of GPUs to use for DDP training
NUM_GPUS=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Enable deterministic behavior
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDNN_DETERMINISTIC=1
export CUDNN_BENCHMARK=0

LAUNCHER="torchrun --standalone --nproc_per_node=$NUM_GPUS"
timestamp=$(python -c "from datetime import datetime; print(datetime.now().strftime('%Y%m%d_%H%M%S'))")

max_iters=2000
warmup_iters=250
head_size=64
total_batch_size=480
gradient_accumulation_steps=8
batch_size=60
seed=1

init_std=0.02
moe_tau=1.0
t_ema_inv=1.0
lr=0.064
router_lr=0.005
attn_lr_mult=2.0
n_layer=8
for width in 256
do
    for num_exp in 16
    do
        for moe_tau in 1.0
        do
                moe_bias_lr=0.1
                expert_gamma=1.0
                ffn_alpha=1.0
                depth_alpha_exp=1.0
                n_heads=$((width / head_size))
                num_act=$((num_exp/4))

                mup_base_width=256
                completep_base_depth=8
                completep_depth_multiplier=$(echo "scale=8; $n_layer/$completep_base_depth" | bc -l)
                mup_width_multiplier=$(echo "scale=8; $width/$mup_base_width" | bc -l)
                router_lr_mult=$(echo "scale=8; $router_lr/$lr" | bc -l)
                weight_decay=$(echo "scale=8; $t_ema_inv/($max_iters*$lr)" | bc -l)
                
                out_dir="run_data/mutransfer_lr_fineweb/out_${timestamp}/width${width}_depth${n_layer}_experts${num_exp}_active${num_act}_seed${seed}_lr${lr}"
                $LAUNCHER train.py \
                    --out_dir=$out_dir \
                    --eval_iters=$((100 * gradient_accumulation_steps / NUM_GPUS)) \
                    --csv_log=True \
                    --warmup_iters=$warmup_iters \
                    --dataset='fineweb' \
                    --gradient_accumulation_steps=$gradient_accumulation_steps \
                    --batch_size=$batch_size \
                    --n_layer=$n_layer \
                    --n_head=$n_heads \
                    --n_embd=$width \
                    --dropout=0.0 \
                    --bias=False \
                    --init_std=$init_std \
                    --learning_rate=$lr \
                    --max_iters=$max_iters \
                    --weight_decay=$weight_decay \
                    --mup_enabled=True \
                    --mup_width_multiplier=$mup_width_multiplier \
                    --mup_input_alpha=1.0 \
                    --mup_output_alpha=1.0 \
                    --num_exp=$num_exp \
                    --num_act=$num_act \
                    --moe_tau=$moe_tau \
                    --moe_bias_lr=$moe_bias_lr \
                    --router_lr_mult=$router_lr_mult \
                    --seed=$seed \
                    --alpha=$ffn_alpha \
                    --dtype='bfloat16' \
                    --compile=False \
                    --attn_lr_mult=$attn_lr_mult \
                    --depth_multiplier=$completep_depth_multiplier \
                    --depth_alpha_exp=$depth_alpha_exp \
                    --expert_gamma=$expert_gamma \
                    --wandb_log=False \
                    --wandb_project=CompleteP_opt_moe_tau \
                    --wandb_run_name=width${width}_moe_tau${moe_tau} \
                    >> /home/ubuntu/MuP_MOE/std_out/debugged_outlog_fineweb_${timestamp}
            done
        done
    done
done