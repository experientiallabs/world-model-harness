"""Splice merged SFT text-LM weights back into a full Qwen3.5-9B checkpoint for vLLM.

transformers' AutoModelForCausalLM loads Qwen3.5-9B's TEXT model only (keys `model.*`,
config Qwen3_5TextConfig) — vLLM can't serve that (registry only has the full
Qwen3_5ForConditionalGeneration). This script rebuilds a servable checkpoint: hardlink
the base snapshot, then overwrite every language-model tensor with the SFT-merged value
under the base namespace (`model.language_model.*`), keeping vision/mtp/config intact.

Usage: python splice_sft_into_base.py <base_snapshot_dir> <merged_dir> <out_dir>
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def load_all(dir_path: Path) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for st in sorted(dir_path.glob("*.safetensors")):
        with safe_open(str(st), framework="pt") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
    return tensors


def main() -> None:
    base_dir, merged_dir, out_dir = (Path(p) for p in sys.argv[1:4])
    out_dir.mkdir(parents=True, exist_ok=True)

    merged = load_all(merged_dir)
    print(f"merged text tensors: {len(merged)}")

    # Map merged text keys (model.*, lm_head.*) -> base namespace.
    def to_base_key(key: str) -> str:
        if key.startswith("model."):
            return "model.language_model." + key[len("model.") :]
        return key  # lm_head.weight is top-level in both

    remapped = {to_base_key(k): v for k, v in merged.items()}

    index = json.loads((base_dir / "model.safetensors.index.json").read_text())
    weight_map: dict[str, str] = index["weight_map"]
    base_keys = set(weight_map)
    hits = sum(1 for k in remapped if k in base_keys)
    print(f"remapped keys matching base: {hits}/{len(remapped)}")
    missing = [k for k in remapped if k not in base_keys][:5]
    if missing:
        print("sample non-matching:", missing)

    # Rewrite each shard: keep base tensors, substitute remapped ones.
    shards: dict[str, list[str]] = {}
    for key, shard in weight_map.items():
        shards.setdefault(shard, []).append(key)
    substituted = 0
    for shard, keys in sorted(shards.items()):
        out_tensors: dict[str, torch.Tensor] = {}
        with safe_open(str(base_dir / shard), framework="pt") as f:
            for key in keys:
                if key in remapped:
                    out_tensors[key] = remapped[key].to(f.get_tensor(key).dtype)
                    substituted += 1
                else:
                    out_tensors[key] = f.get_tensor(key)
        save_file(out_tensors, str(out_dir / shard), metadata={"format": "pt"})
        print(f"wrote {shard} ({len(keys)} tensors)")
    print(f"substituted {substituted} tensors")

    for extra in base_dir.iterdir():
        if extra.suffix != ".safetensors" and extra.is_file():
            shutil.copy(extra, out_dir / extra.name)
    print(f"done -> {out_dir}")
    if substituted != len(remapped):
        print("WARNING: some merged tensors were not placed — investigate before serving")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
