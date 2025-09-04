The script uses streaming so it won’t fetch the entire C4; it stops when it has tokenized enough examples to hit your targets (it clips the last example to fit exactly).

Default targets are 30B train / 10M val; adjust with --train-tokens and --val-tokens.

Disk: training will be ~60 GB (2 bytes/token) for 30B tokens; ensure you have room.

For full parity with your other scripts, meta.pkl contains vocab_size, tokenizer, eot_token_id, and token counts.

If you want light randomization, set --shuffle-buffer (e.g., --shuffle-buffer 100000), which shuffles the streaming iterator with a bounded buffer.