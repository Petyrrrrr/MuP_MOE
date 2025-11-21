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
t_ema_inv=1.0
total_batch_size=480
gradient_accumulation_steps=8
batch_size=60

n_layer=8

mup_base_width=256
completep_base_depth=8
seed=1

init_std=0.02
moe_tau=1.0
base_lr=0.005
router_lr=0.00125
router_init_mult=1.0
beta_moe=0.25
beta_attn=1.0

others_lr_mult=1.0
mlp_up_lr_mult=1.0
attn_qkv_lr_mult=0.0625
attn_lr_down_mult=0.0625
mlp_down_lr_mult=0.0625
t_ema_inv=1.0

num_exp=4

for width in 512
do
    for beta_moe in 0.125 0.25 0.5 1.0
    do
        for num_exp in 4 16
        do
                moe_bias_lr=0.1
                expert_gamma=1.0
                ffn_alpha=1.0
                depth_alpha_exp=1.0
                n_heads=$((width / head_size))
                num_act=$((num_exp/4))
                completep_depth_multiplier=$(echo "scale=8; $n_layer/$completep_base_depth" | bc -l)
                mup_width_multiplier=$(echo "scale=8; $width/$mup_base_width" | bc -l)
                router_lr_mult=$(echo "scale=8; $router_lr/$base_lr" | bc -l)
                weight_decay=$(echo "scale=8; $t_ema_inv/($max_iters*$base_lr)" | bc -l)
                
                out_dir="run_data/mutransfer_lr_fineweb/out_${timestamp}/width${width}_depth${n_layer}_experts${num_exp}_active${num_act}_seed${seed}_lr${base_lr}"
                $LAUNCHER train.py \
                    --out_dir=$out_dir \
                    --eval_iters=$((100 * gradient_accumulation_steps / NUM_GPUS)) \
                    --csv_log=True \
                    --warmup_iters=$warmup_iters \
                    --dataset='fineweb' \
                    --gradient_accumulation_steps=$gradient_accumulation_steps \
                    --batch_size=$batch_size \
                    --n_layer=$n_layer \
                    --router_init_mult=$router_init_mult \
                    --n_head=$n_heads \
                    --n_embd=$width \
                    --dropout=0.0 \
                    --bias=False \
                    --init_std=$init_std \
                    --router_lr=$router_lr \
                    --learning_rate=$base_lr \
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
                    --beta_moe=$beta_moe \
                    --beta_attn=$beta_attn \
                    --seed=$seed \
                    --alpha=$ffn_alpha \
                    --dtype='bfloat16' \
                    --compile=False \
                    --mlp_up_lr_mult=$mlp_up_lr_mult \
                    --attn_qkv_lr_mult=$attn_qkv_lr_mult \
                    --mlp_down_lr_mult=$mlp_down_lr_mult \
                    --attn_lr_down_mult=$attn_lr_down_mult \
                    --others_lr_mult=$others_lr_mult \
                    --depth_multiplier=$completep_depth_multiplier \
                    --depth_alpha_exp=$depth_alpha_exp \
                    --expert_gamma=$expert_gamma \
                    --wandb_log=True \
                    --wandb_project=CompleteP_sweep_beta_moe \
                    --wandb_run_name=width${width}_beta_moe${beta_moe} \
                    >> /home/ubuntu/MuP_MOE/std_out/debugged_outlog_fineweb_${timestamp}
        done
    done
done