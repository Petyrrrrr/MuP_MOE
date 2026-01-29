# nanoGPT-mup-moe

To start (Mithril Ubuntu environment, fresh box), run 

startup_script.sh 

to set up venv and necessary packages (may or may not need to change GitHub/wandb config for that). Run source venv/bin/activate to activate venv before calling Python.

========DATA========

Running 

startup_cccc_data.sh 

downloads 85M training documents, which is slightly under 30B tokens. You will need to configure a large enough storage space because the downloaded documents will be ~200GB (the tokenized .bin takes up <70GB).

For running in the simplified branch, no data batch pre-splitting is implemented, so just make sure that train.bin and val.bin exist in ./data/cccc/. For dense branch, we implemented batch splitting for deterministic batches. Run

python split_cccc_batches.py --dataset-dir data/cccc --train-batches 10000 --val-batches 100 --overwrite --shuffle --shuffle-seed <SEED>

Each split corresponds to a unique batch size (by default, 480 * 1024 tokens), but this is configurable, and splitting batches should run quickly (< 1 minute).

========IMPLEMENTATION========

train.py -- takes in config HPs and sets up the run. get_batch and get_lr are there

trainer.py -- scripts that do the actual training loops, including implementation of micro-batches and moe expert bias update schedule.

model.py -- sets up the model. Specific places to pay attention to are the initialization (search # init all weights), forward pass of the MLP (search h_func and s_func), and learning rate (search lr_scale).

The current forward pass MOE structure is as follows:

F_{moe} = \frac{\sum q_i * s(r_i) * E_i}{\eps + \sum q_i * s(r_i)}, where r_i = nn.Linear(n_embd, n_exp) is the router output and the activation indicator is q_i = 1 (top_K of h(r_i) + bias).

Here, s_func is the expert weights, and h_func is used for load balancing. There are arguments to make them the same, and there are arguments to make them different. 

There are also arguments to make eps non-trivial (0-FLOP "Null expert"), although in my experience, large eps is not necessarily good. By default, s and h are both sigmoid (so biases don't need to overflow).


====Counting parameters====

Each transformer block contributes 4 * n_embd^2 parameters from the attention projections (c_attn + c_proj) regardless of head count because both matrices are n_embd × (3·n_embd) and n_embd × n_embd respectively. The feed-forward hidden width is hidden_size = int(alpha * n_embd) (model.py:140), so every MLP instance adds about 2 * alpha * n_embd^2 weights. In MoE mode, there are num_exp copies of this MLP per block, so the expert-side contribution is 2 * alpha * num_exp * n_embd^2 per block.

Putting those together gives a first-order total-parameter heuristic

P_total ≈ n_layer * (4 + 2 * alpha * num_exp) * n_embd^2 + n_embd * vocab_size + O(n_layer * n_embd * num_exp)

count_parameters.py splits parameters into “expert” vs “non-expert” by summing over the expert modules only , then defines “active” parameters as non_expert + (num_act / num_exp) * expert_params (count_parameters.py:102-108). Using the same leading-order logic, the activated expert weight per block is 2 * alpha * num_act * n_embd^2, so
    
P_active ≈ n_layer * (4 + 2 * alpha * num_act) * n_embd^2 + n_embd * (vocab_size + block_size) + O(n_layer * n_embd * num_exp)
