#!/bin/bash
# Multi-GPU version of muP hyperparameter transfer with MOE
# Distributes different runs across available GPUs

LAUNCHER="python3"
NUM_GPUS=8
timestamp=$(python -c "from datetime import datetime; print(datetime.now().strftime('%Y%m%d_%H%M%S'))")

# Create log directory
LOG_DIR="/home/ubuntu/MuP_MOE/std_out/mutransfer_lr_owt/${timestamp}"
mkdir -p "$LOG_DIR"

# Track running processes
declare -a PIDS=()
declare -a GPU_JOBS=()

# Initialize GPU job counters
for ((i=0; i<$NUM_GPUS; i++)); do
    GPU_JOBS[$i]=0
done

# Function to get the GPU with fewest jobs
get_next_gpu() {
    local min_jobs=${GPU_JOBS[0]}
    local min_gpu=0
    
    for ((i=1; i<$NUM_GPUS; i++)); do
        if [ ${GPU_JOBS[$i]} -lt $min_jobs ]; then
            min_jobs=${GPU_JOBS[$i]}
            min_gpu=$i
        fi
    done
    
    echo $min_gpu
}

# Function to wait for a process to finish
wait_for_slot() {
    while true; do
        # Check each running process
        for i in "${!PIDS[@]}"; do
            pid=${PIDS[$i]}
            gpu=${GPU_ASSIGNMENTS[$i]}
            
            # Check if process is still running
            if ! kill -0 $pid 2>/dev/null; then
                # Process finished
                echo "Process $pid on GPU $gpu finished"
                unset PIDS[$i]
                unset GPU_ASSIGNMENTS[$i]
                ((GPU_JOBS[$gpu]--))
                return
            fi
        done
        
        # If we have free slots, return
        total_jobs=0
        for jobs in "${GPU_JOBS[@]}"; do
            ((total_jobs += jobs))
        done
        
        if [ $total_jobs -lt $NUM_GPUS ]; then
            return
        fi
        
        # Wait a bit before checking again
        sleep 2
    done
}

# Arrays to track GPU assignments
declare -a GPU_ASSIGNMENTS=()

# Counter for job IDs
job_id=0

# Main loop through all configurations
for width in 512
do
    for num_exp in 8 4 2
    do
        for lr in 0.0625 0.03125 0.015625 0.0078125 0.00390625 0.001953125 0.0009765625 0.00048828125 0.000244140625 0.0001220703125 0.00006103515625
        do
            for seed in 1
            do
                # Wait for an available GPU slot
                wait_for_slot
                
                # Get next available GPU
                gpu=$(get_next_gpu)
                
                # Calculate parameters
                head_size=64
                n_heads=$((width / head_size))
                min_lr=$lr
                mup_base_width=256
                mup_width_multiplier=$(echo "scale=8; $width/$mup_base_width" | bc -l)
                num_act=$((num_exp/2))
                
                # Set output directory
                out_dir="run_data/mutransfer_lr_owt/out_${timestamp}/width${width}_depth8_experts${num_exp}_active${num_act}_seed${seed}_lr${lr}"
                
                # Create log file names
                log_file="${LOG_DIR}/job_${job_id}_gpu${gpu}_w${width}_exp${num_exp}_lr${lr}_seed${seed}.log"
                err_file="${LOG_DIR}/job_${job_id}_gpu${gpu}_w${width}_exp${num_exp}_lr${lr}_seed${seed}.err"
                
                echo "Starting job $job_id on GPU $gpu: width=$width, num_exp=$num_exp, lr=$lr, seed=$seed"
                echo "Logs: $log_file"
                
                # Launch the training process on specific GPU
                CUDA_VISIBLE_DEVICES=$gpu $LAUNCHER train.py \
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
                    --warmup_iters=1000 \
                    --dataset='openwebtext' \
                    --gradient_accumulation_steps=16 \
                    --batch_size=16 \
                    --block_size=1024 \
                    --n_layer=8 \
                    --n_head=$n_heads \
                    --n_embd=$width \
                    --dropout=0.0 \
                    --bias=False \
                    --init_std=0.02 \
                    --learning_rate=$lr \
                    --lr_decay_iters=2000 \
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
                    --moe_bias_lr=$lr \
                    --moe_bias_momentum=0.9 \
                    --moe_bias_momentum_enabled=True \
                    --moe_load_balance_method='bias' \
                    --moe_aux_loss_weight=1.0 \
                    --seed=$seed \
                    --backend='nccl' \
                    --device='cuda' \
                    --dtype='float16' \
                    --compile=False \
                    > "$log_file" 2> "$err_file" &
                
                # Store PID and GPU assignment
                pid=$!
                PIDS+=($pid)
                GPU_ASSIGNMENTS+=($gpu)
                ((GPU_JOBS[$gpu]++))
                
                # Increment job counter
                ((job_id++))
                
                # Small delay to avoid race conditions
                sleep 0.5
            done
        done
    done
done

echo "All jobs launched. Waiting for completion..."

# Wait for all remaining processes to finish
for pid in "${PIDS[@]}"; do
    if [ ! -z "$pid" ]; then
        wait $pid
    fi
done

echo "All jobs completed!"
echo "Logs saved in: $LOG_DIR"