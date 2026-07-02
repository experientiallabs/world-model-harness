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
import subprocess
import sys
import tempfile
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
        with safe_open(str(Path(args.adapter_dir) / "adapter_model.safetensors"), framework="pt") as f:
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

    with tempfile.TemporaryDirectory(prefix="wm_tau_merge_") as tmp:
        merged.save_pretrained(tmp)
        del merged, model

        snapshot = args.base_snapshot
        if snapshot is None:
            from huggingface_hub import snapshot_download

            snapshot = snapshot_download(args.base)
        print(f"splicing over base snapshot {snapshot} -> {args.out_dir}")
        splice = Path(__file__).with_name("splice_sft_into_base.py")
        subprocess.run(
            [sys.executable, str(splice), snapshot, tmp, args.out_dir], check=True
        )
    print("done")


if __name__ == "__main__":
    main()
