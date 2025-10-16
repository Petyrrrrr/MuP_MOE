import argparse, json, os, re, csv
from collections import defaultdict

from huggingface_hub import hf_hub_download
from safetensors.torch import safe_open
import torch
import numpy as np
from tqdm import tqdm

ROUTER_KEY_RE = re.compile(r"^model\.layers\.\d+\.mlp\.router\.(weight|bias)$")
LAYER_NUM_RE = re.compile(r"^model\.layers\.(\d+)\.")

def list_router_shards(model_id: str):
    """Read HF index, return (weight_map, router_keys, involved_shards)."""
    index_path = hf_hub_download(repo_id=model_id, filename="model.safetensors.index.json")
    with open(index_path, "r") as f:
        idx = json.load(f)
    wm = idx["weight_map"]
    router_keys = [k for k in wm if ROUTER_KEY_RE.search(k)]
    if not router_keys:
        raise RuntimeError(
            f"No router keys found in {model_id}. "
            "Expected keys like 'model.layers.N.mlp.router.weight'."
        )
    shards = sorted({wm[k] for k in router_keys})
    return wm, router_keys, shards

def download_shards(model_id: str, shards):
    """Download only shards that hold router weights/biases."""
    return {s: hf_hub_download(repo_id=model_id, filename=s) for s in shards}

def spectral_summary(W: torch.Tensor, topk=8):
    W = W.to(torch.float32)
    try:
        s = torch.linalg.svdvals(W)
    except Exception:
        _, s, _ = torch.linalg.svd(W, full_matrices=False)
    s_sorted, _ = torch.sort(s, descending=True)
    spec_norm = float(s_sorted[0])
    fro = float(torch.linalg.norm(W, ord='fro'))
    eff_rank = (fro**2) / (spec_norm**2)  # stable rank
    smin = float(s_sorted[-1]) if s_sorted.numel() else 0.0
    cond = float(spec_norm / max(smin, 1e-8))
    return dict(
        shape=tuple(W.shape),
        spectral_norm=spec_norm,
        fro_norm=fro,
        effective_rank=eff_rank,
        condition_number=cond,
        top_singular_values=s_sorted[:topk].cpu().numpy().tolist(),
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-oss-120b",
                    help="Model id: openai/gpt-oss-20b or openai/gpt-oss-120b (or a compatible fork)")
    ap.add_argument("--topk", type=int, default=8, help="How many top singular values to store")
    args = ap.parse_args()

    model_id = args.model
    os.makedirs("gpt_oss_router_120", exist_ok=True)

    print(f"[1/3] Reading index for {model_id} ...")
    weight_map, router_keys, shards = list_router_shards(model_id)
    print(f"Found {len(router_keys)} router keys across {len(shards)} shard file(s).")

    print(f"[2/3] Downloading only necessary shard(s) ...")
    shard_paths = download_shards(model_id, shards)

    routers = {}
    spectra = {}

    print(f"[3/3] Extracting router weights layer-by-layer ...")
    shard_to_weight_keys = defaultdict(list)
    for k in router_keys:
        if k.endswith(".weight"):
            shard_to_weight_keys[weight_map[k]].append(k)

    for shard_name in shards:
        with safe_open(shard_paths[shard_name], framework="pt", device="cpu") as f:
            shard_keys = set(f.keys())
            for k in shard_to_weight_keys[shard_name]:
                if k not in shard_keys:
                    continue
                L = int(LAYER_NUM_RE.search(k).group(1))
                W = f.get_tensor(k)  # router is BF16 per config; cast later
                routers[L] = W

    layers = sorted(routers.keys())
    if not layers:
        raise RuntimeError(
            "Router weights dictionary is empty after extraction. "
            "This usually means the index used different key names."
        )
    print(f"Collected router weights for layers: {layers[0]}..{layers[-1]} ({len(layers)} total)")


    torch.save({f"layer_{L}": routers[L].to(torch.float16).cpu() for L in layers},
               "gpt_oss_router_120/router_weights.pt")
    np.savez("gpt_oss_router_120/router_weights.npz",
             **{f"layer_{L}": routers[L].to(torch.float32).cpu().numpy() for L in layers})


    print("     Done. Files written to gpt_oss_router_120/:")
    print("   - router_weights.pt  (PyTorch state dict with FP16 weights)")
    print("   - router_weights.npz (NumPy arrays per layer, FP32)")

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
