"""Turn a wake/sleep LoRA checkpoint into a vLLM-servable full Qwen3.5-9B checkpoint.

Applies the LoRA deltas DIRECTLY onto the base snapshot tensors (W += alpha/r * B @ A),
because every indirect route fails on Qwen3.5:
  - standalone vLLM --lora-modules IndexErrors on the hybrid-attention stack;
  - peft-merging into the transformers TEXT model silently matches ZERO adapter keys —
    wake/sleep adapters are keyed `base_model.model.model.language_model.layers...`
    (the full multimodal wrapper), the text model has no `language_model.` segment,
    and peft only WARNS about the missing keys (the "merged" model equals base).

Adapter key -> base key: strip `base_model.model.` and `.lora_{A,B}.weight`, append
`.weight`; e.g. `base_model.model.model.language_model.layers.0.mlp.down_proj.lora_A.weight`
-> `model.language_model.layers.0.mlp.down_proj.weight` (1:1 with the snapshot index).

Usage:
    python merge_adapter_and_splice.py <adapter_dir> <out_dir> [--base-snapshot DIR]
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("adapter_dir")
    parser.add_argument("out_dir")
    parser.add_argument("--base", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--base-snapshot", default=None,
                        help="local HF snapshot dir of the base (auto-resolved if omitted)")
    args = parser.parse_args()

    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    adapter_dir = Path(args.adapter_dir)
    config = json.loads((adapter_dir / "adapter_config.json").read_text())
    scaling = config["lora_alpha"] / config["r"]
    print(f"lora r={config['r']} alpha={config['lora_alpha']} scaling={scaling}")

    lora_a: dict[str, torch.Tensor] = {}
    lora_b: dict[str, torch.Tensor] = {}
    with safe_open(str(adapter_dir / "adapter_model.safetensors"), framework="pt") as f:
        for key in f.keys():
            if key.endswith(".lora_A.weight"):
                lora_a[key[: -len(".lora_A.weight")]] = f.get_tensor(key)
            elif key.endswith(".lora_B.weight"):
                lora_b[key[: -len(".lora_B.weight")]] = f.get_tensor(key)
    assert set(lora_a) == set(lora_b), "unpaired lora_A/lora_B keys"
    print(f"adapter modules: {len(lora_a)}")

    def to_base_key(module: str) -> str:
        prefix = "base_model.model."
        assert module.startswith(prefix), module
        return module[len(prefix):] + ".weight"

    deltas = {
        to_base_key(module): (lora_b[module].float() @ lora_a[module].float()) * scaling
        for module in lora_a
    }

    snapshot_dir = args.base_snapshot
    if snapshot_dir is None:
        from huggingface_hub import snapshot_download

        snapshot_dir = snapshot_download(args.base)
    snapshot = Path(snapshot_dir)
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())
    weight_map: dict[str, str] = index["weight_map"]
    hits = sum(1 for k in deltas if k in weight_map)
    print(f"delta keys matching base: {hits}/{len(deltas)}")
    if hits != len(deltas):
        missing = [k for k in deltas if k not in weight_map][:5]
        raise SystemExit(f"unmatched delta keys, sample: {missing}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    shards: dict[str, list[str]] = {}
    for key, shard in weight_map.items():
        shards.setdefault(shard, []).append(key)
    applied = 0
    max_delta = 0.0
    for shard, keys in sorted(shards.items()):
        tensors: dict[str, torch.Tensor] = {}
        with safe_open(str(snapshot / shard), framework="pt") as f:
            for key in keys:
                tensor = f.get_tensor(key)
                if key in deltas:
                    delta = deltas[key]
                    max_delta = max(max_delta, float(delta.abs().max()))
                    tensor = (tensor.float() + delta).to(tensor.dtype)
                    applied += 1
                tensors[key] = tensor
        save_file(tensors, str(out / shard), metadata={"format": "pt"})
        print(f"wrote {shard}")
    for extra in snapshot.iterdir():
        if extra.suffix != ".safetensors" and extra.is_file():
            shutil.copy(extra, out / extra.name)
    print(f"applied {applied} deltas; max |delta| = {max_delta:.6f}")
    if applied != len(deltas) or max_delta == 0.0:
        raise SystemExit("merge ineffective — refusing to produce a base-identical checkpoint")
    print("done")


if __name__ == "__main__":
    main()
