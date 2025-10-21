# nanoGPT-mup-moe

To start (on a fresh Ubuntu environment), run 

startup_script.sh 

to set up venv and necessary packages (you may or may not need to change GitHub/wandb config for that). Run source venv/bin/activate to activate venv before calling python.

========DATA========

Running 

startup_cccc_data.sh 

downloads 85M training documents, which is slightly under 30B tokens. You will need to configure a large enough storage space because the downloaded documents will be ~200GB (the tokenized .bin takes up <70GB).

For running in the simplified branch, no data batch pre-splitting is implemented, so just make sure that train.bin and val.bin exist in ./data/cccc/. For dense branch, we implemented batch splitting for deterministic batches. Run

python split_cccc_batches.py --dataset-dir data/cccc --train-batches 10000 --val-batches 100 --overwrite --shuffle --shuffle-seed <SEED>

Each split corresponds to a unique batch size (by default 480 * 1024 tokens), but that is configurable, and splitting batches should run pretty quickly (< 1min).

========RUNNING CCCC SCRIPTS========

The two main scripts I run that call train.py in customizable ways are

python mutransfer_lr_cccc/run-multi-gpu.py

which sets up a sweep over hyperparameters to run each experiment on one GPU (this is faster, I believe, if you have more jobs than GPUs). To train a large model quickly on multiple GPUs, run

bash mutransfer_lr_cccc/run.sh

In which the models are trained on all visible devices sequentially, and you have to configure the HP rules manually. By default, wandb logging is enabled when calling.

========IMPLEMENTATION========

train.py -- takes in config HPs and sets up the run. get_batch and get_lr are there
trainer.py -- scripts that do the actual training loops, including implementation of micro-batches and moe expert bias update schedule.
model.py -- sets up the model. Specific places to pay attention to are the initialization (search # init all weights), forward pass of the MLP (search h_func and s_func), and learning rate (search lr_scale).

The current forward pass MOE structure is as follows:

F_{moe} = \frac{\sum q_i * s(r_i) * E_i}{\eps + \sum q_i * s(r_i)}, where r_i = nn.Linear(n_embd, n_exp) is the router output and q_i = 1 (top_K of h(r_i) + bias).

Here, s_func is the expert weights, and h_func is used for load balancing. There are arguments to make them the same, and there are arguments to make them different. There are also arguments to make eps non-trivial (so one can think of there being an expert that always outputs zero), although in my experience, large eps is not necessarily good.

By default, h is sigmoid (so biases don't need to overflow) and s is softmax (which is equivalent to exp for this purpose).

========COMMENTS========

These claims are pretty non-rigorous, and I'm not sure how tested/statistically significant these are.

====Comments Oct.21st
(-1) HP transfer on hidden MLP size (no MOE): I tried to run some stuff on a dense model, varying only the hidden MLP dim, and it seems like I get good transfer following the recipe of Spectral Conditioning for Feature Learning, which is basically that you scale down 1/ffn_mult on the forward pass and keep everything (LR and init) the same. I did see a pretty significant boost in val loss by increasing ffn_mult up to 100.
(-1.a) The transfer parametrization was supported by The Hidden Width of Deep ResNets, but I think more rigorous tests may be required to make a conclusion.

(0) HP transfer: on a coarse scale, pretty much everything transfers, even stuff that doesn't make sense. I think this is because the attn and MLP are doing good transfer work, so you can be sloppy in the MOE and still get good transfer.
(0.a) Under the "good setup" (see below), I have observed that transfer on a finer scale seems to hold pretty well (on a sweep on the base LR range from 0.004 to 0.01).

(1) Load balancing: Based on my existing runs, I think the necessary and sufficient conditions for good load balancing are (under 1.a and 1.b I have yet seen any not-balanced run under a reasonable base LR):
 (1.a) normalizing expert weights (softmax/exp as well as sigmoid are all fine) i.e. F = \sum_{top k} p_i E_i / (\sum_{top k} p_i+eps), where top k is selected by sigmoid(logit) + bias and p is some activated logit.
 (comment 1.a.1) Numerical stability was pretty bad at F = \sum_{top k} p_i E_i / num_exp when num_exp is large, not just at the beginning but also randomly amidst training.  The same if you replace num_exp with sqrt(num_exp) (which isn't the most sensible thing to do to begin with). This was what I wrote to you on Sunday.
 (comment 1.a.2) In pretty much all runs before this weekend, I was trying Cerebras MoE's "null expert bias" setup, which was basically you do F = \sum_{top k} p_i E_i / (1 + \sum_{top k} p_i) where the extra 1 factor allows p_i to be small (i.e. you are less confident when experts are all bad). There is some literature supporting this setup, and it's basically equivalent to what Andrey mentioned as the zero-FLOP expert. Turns out that this trick hurts balancing stability at the edge, where for larger LR with (+eps) you get balancing but not with (+1) (see the nullexp wandb folder). [This claim is not super rigorously tested]
 (1.b) num_exp needs to be at least 12. Even for 8 experts, in a few runs I saw instability kick in (most of the time it's ok though). 4 experts are a total disaster (that's where the whole torch compile business comes in).
 (comment 1.b.1) Somehow, "being balanced" is a binary thing later during training; either all experts is being very close to uniform and staying there, or in some layer, experts are struggling and moving wildly.
 (comment 1.b.2) However, I'm not sure if I do see that bad load balancing in some of the layers necessarily hurts performance by a ton; the effect is not very consistent

(2) Loss: We are definitely doing better than dense by a margin.
(2.a) On 2.5B tokens with the simplest 5% warmup + cosine decay to zero, our 400M-activate-124M (124M is the og GPT2 param count) following lr found in (0) and balancing recipe in (1) matches (if not exceeds? it's hard to tell from their plot) the modded speedrun non-Muon tweak checkpoint, which largely outperforms GPT2 dense baseline (our final val loss is close to 3.2). However, obviously, the speedrun's objective was to cram an insane batch size very quickly without really bothering about model size, so it wasn't optimized on this front.
(2.b) Our 1.25B-activate-434M model, following the same recipe, matches the gpt2-large 1.5B dense model at the same 2.5B checkpoint (our final val loss is <3.4), which is actually pretty nice. In both of the models, I did no tuning except trying to find a reasonable (width, n_exp) combo to match parameter count, and they both balance perfectly.
(2.c) Compared to no router learning or no balancing effort: I'm not certain because it is fairly setup-dependent (which we change from here to there), and it seems like even the literature cannot fully agree on this.
