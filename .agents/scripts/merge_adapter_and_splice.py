"""Turn a wake/sleep LoRA checkpoint into a vLLM-servable full Qwen3.5-9B checkpoint.

Standalone vLLM can neither dynamic-LoRA Qwen3.5's hybrid-attention stack (IndexError)
nor serve the text-only merged model (registry lacks Qwen3_5ForCausalLM), so checkpoint
eval goes: adapter -> merge into the TEXT model (transformers CausalLM view) -> splice
the text weights over a copy of the full base snapshot (see splice_sft_into_base.py).

Usage:
    python merge_adapter_and_splice.py <adapter_dir> <out_dir> [--base Qwen/Qwen3.5-9B]

Runs in the claas-verl venv (torch/peft/transformers). GPU not required (CPU merge).
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("adapter_dir")
    parser.add_argument("out_dir")
    parser.add_argument("--base", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--base-snapshot", default=None,
                        help="local HF snapshot dir of the base (auto-resolved if omitted)")
    args = parser.parse_args()

    import json

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    # verl saves target_modules as its regex string (^.*\.(q_proj|...)$); this peft
    # version iterates the string into characters. Rewrite to the explicit module list.
    config_path = Path(args.adapter_dir) / "adapter_config.json"
    adapter_config = json.loads(config_path.read_text())
    modules = adapter_config.get("target_modules")
    mangled = isinstance(modules, str) or (
        isinstance(modules, list) and modules and all(len(str(m)) <= 1 for m in modules)
    )
    if mangled:
        # Derive the true module set from the adapter weights themselves.
        from safetensors import safe_open

        names: set[str] = set()
        adapter_weights = str(Path(args.adapter_dir) / "adapter_model.safetensors")
        with safe_open(adapter_weights, framework="pt") as f:
            for key in f.keys():
                if ".lora_A" in key or ".lora_B" in key:
                    prefix = key.split(".lora_")[0]
                    names.add(prefix.rsplit(".", 1)[-1])
        if not names:
            raise SystemExit("no lora_A/lora_B keys found in adapter weights")
        adapter_config["target_modules"] = sorted(names)
        config_path.write_text(json.dumps(adapter_config, indent=2))
        print(f"rewrote mangled target_modules -> {adapter_config['target_modules']}")

    print(f"loading base text model {args.base} (cpu, bf16)...")
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16)
    print(f"applying adapter {args.adapter_dir}...")
    model = PeftModel.from_pretrained(model, args.adapter_dir)
    merged = model.merge_and_unload()

    snapshot_dir = args.base_snapshot
    if snapshot_dir is None:
        from huggingface_hub import snapshot_download

        snapshot_dir = snapshot_download(args.base)
    snapshot = Path(snapshot_dir)

    # Splice IN-PROCESS from the merged model's tensors (no /tmp text-merged copy:
    # the two-copy pipeline needs ~36G transient and filled the disk).
    import shutil

    from safetensors import safe_open
    from safetensors.torch import save_file

    state = {k: v for k, v in merged.state_dict().items()}
    del merged, model
    print(f"merged state tensors: {len(state)}")

    index = json.loads((snapshot / "model.safetensors.index.json").read_text())
    weight_map: dict[str, str] = index["weight_map"]
    hits = sum(1 for k in state if k in weight_map)
    print(f"merged keys matching base: {hits}/{len(state)}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    shards: dict[str, list[str]] = {}
    for key, shard in weight_map.items():
        shards.setdefault(shard, []).append(key)
    substituted = 0
    for shard, keys in sorted(shards.items()):
        tensors = {}
        with safe_open(str(snapshot / shard), framework="pt") as f:
            for key in keys:
                if key in state:
                    tensors[key] = state[key].to(f.get_tensor(key).dtype)
                    substituted += 1
                else:
                    tensors[key] = f.get_tensor(key)
        save_file(tensors, str(out / shard), metadata={"format": "pt"})
        print(f"wrote {shard}")
    for extra in snapshot.iterdir():
        if extra.suffix != ".safetensors" and extra.is_file():
            shutil.copy(extra, out / extra.name)
    print(f"substituted {substituted} tensors")
    if substituted != len(state):
        missing = [k for k in state if k not in weight_map][:5]
        print("WARNING: unplaced merged tensors, sample:", missing)
        raise SystemExit(1)
    print("done")


if __name__ == "__main__":
    main()
