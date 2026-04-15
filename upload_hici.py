#!/usr/bin/env python3
"""
Upload HiCI model adapters to HuggingFace.
Supports: qwen3 (Qwen3-8B HiCI 48K) and llama2 (Llama-2-7B HiCI 8K)

Usage:
    python upload_hici.py --model qwen3
    python upload_hici.py --model llama2
"""

import argparse
import os
import shutil
from huggingface_hub import HfApi, create_repo

# ====================================================
# Per-model configuration
# ====================================================
HF_USERNAME = "ZengXiangyu"

CONFIGS = {
    "qwen3": {
        "model_name":      "Qwen3-8b-HiCI-48k-1000steps",
        "checkpoint_path": "./checkpoints/Qwen3-8b-hici-48k/checkpoint-1000",
        "adapter_file":    "adapter_model.safetensors",
        "lora_files": [
            "adapter_model.safetensors",  # LoRA weights (safetensors format)
            "adapter_config.json",        # LoRA config (includes base_model path)
            "trainable_params.bin",       # embed + norm + global_memory + hierarchical_aggregator
        ],
        "tokenizer_files": [              # BPE tokenizer, no tokenizer.model
            "tokenizer.json",
            "tokenizer_config.json",
            "added_tokens.json",
            "merges.txt",
            "vocab.json",
            "special_tokens_map.json",
        ],
    },
    "llama2": {
        "model_name":      "Llama-2-7b-8k-hici-causal_gi-G4",
        "checkpoint_path": "./checkpoints/Llama-2-7b-8k-hici-causal_gi-G4/checkpoint-2000",
        "adapter_file":    "adapter_model.bin",
        "lora_files": [
            "adapter_model.bin",    # LoRA weights (.bin format for Llama-2)
            "adapter_config.json",  # LoRA config (includes base_model path)
            "trainable_params.bin", # embed + norm + global_memory + hierarchical_aggregator
        ],
        "tokenizer_files": [        # SentencePiece tokenizer
            "tokenizer.model",
            "tokenizer.json",
            "tokenizer_config.json",
            "added_tokens.json",
            "special_tokens_map.json",
        ],
    },
}

# ====================================================
# README templates (per model)
# ====================================================
HICI_ARCH_DESCRIPTION = """\
Three-stage hierarchy per transformer layer:
1. **Local Construction** — M learnable query slots attend to each segment via bottleneck cross-attention → local summary L_i
2. **Global Integration** — multi-view statistics (mean/max/min/std/ℓ2-norm) → shared compression → attention-based selection → gated expansion → G
3. **Top-down Broadcast** — per-segment attention with augmented KV=[G, L_i, segment tokens]; queries from segment tokens only\
"""

CITATION = """\
```bibtex
@article{zeng2026hici,
  title={HiCI: Hierarchical Construction-Integration for Long-Context Attention},
  author={Zeng, Xiangyu and Xu, Qi and Wang, Yunke and Xu, Chang},
  journal={arXiv preprint arXiv:2603.20843},
  year={2026}
}
```\
"""


def create_readme_qwen3(model_name):
    return f"""---
language:
- en
- zh
license: apache-2.0
tags:
- long-context
- context-extension
- hierarchical-attention
- segmented-attention
- qwen3
- peft
- lora
- hici
base_model: Qwen/Qwen3-8B
---

# {model_name}

## Model Description

This is a **LoRA adapter** for Qwen3-8B with **HiCI (Hierarchical Construction-Integration)** architecture,
trained for long-context understanding up to **48K tokens**.

Paper: [HiCI (arXiv 2603.20843)](https://arxiv.org/abs/2603.20843)

### HiCI Architecture

{HICI_ARCH_DESCRIPTION}

```
Input (48K tokens) → 8 segments × 6K
  Stage 1: 8 local slots per segment → L_i
  Stage 2: multi-view stats → K=4 global slots G
  Stage 3: Q=[chunk], KV=[G, L_i, chunk] → Flash Attention
```

## Trainable Components

```
adapter_model.safetensors  (27 MB)
└── LoRA Adapters (r=8, alpha=16): q_proj, k_proj, v_proj, o_proj

trainable_params.bin  (~4 GB)
├── global_memory.*            — Local Construction modules (36 layers)
├── hierarchical_aggregator.*  — Global Integration modules (36 layers)
├── self_attn.q_norm / k_norm  — QK-Norm weights (Qwen3-specific, 36 layers)
├── input_layernorm / post_attention_layernorm — LayerNorm weights (36 layers)
├── model.embed_tokens.weight  — Token embeddings
└── model.norm.weight          — Final LayerNorm
```

## Training Details

- **Base Model**: Qwen/Qwen3-8B
- **Context Length**: 49,152 tokens (48K)
- **Segments**: 8 × 6,144 tokens
- **Local Memory Slots (M)**: 8 per segment
- **Global Memory Slots (K)**: 4
- **Memory Heads**: 8, Bottleneck dim: 512
- **LoRA**: r=8, alpha=16, target: q/k/v/o_proj
- **Checkpoint**: step 500 / 1000
- **Batch**: per_device=1, grad_accum=8 (effective batch=8)
- **LR**: 2e-5 (LoRA), 2e-4 (memory modules), grad clip=0.3
- **Precision**: bf16
- **Hardware**: 8× H200 141GB, DeepSpeed Stage 2

## Usage

**Requires `qwen3_attn_hici.py` from this repo.**

```python
import torch
import transformers
from peft import PeftModel
import qwen3_attn_hici as hici_attn

# 1. Replace attention with HiCI BEFORE loading model
hici_attn.MIXED_GROUP_TRAINING = False
hici_attn.replace_qwen3_attn(use_flash_attn=True, use_full=False, use_hierarchical_forward=True)

# 2. Load base model
base_model = transformers.AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B", torch_dtype=torch.bfloat16, device_map="auto",
)

# 3. Register HiCI modules (must match training config)
hici_attn.register_hici_to_model(base_model, num_memory_slots=8, global_slots=4, num_heads=8, bottleneck_dim=512)

# 4. Load LoRA adapter + trainable_params
model = PeftModel.from_pretrained(base_model, "{HF_USERNAME}/{model_name}")

# 5. Tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained("{HF_USERNAME}/{model_name}")
```

## Citation

{CITATION}

## License

Apache 2.0 (follows Qwen3 license)
"""


def create_readme_llama2(model_name):
    return f"""---
language:
- en
- zh
license: llama2
tags:
- long-context
- context-extension
- hierarchical-attention
- segmented-attention
- llama-2
- peft
- lora
- hici
base_model: meta-llama/Llama-2-7b-hf
---

# {model_name}

## Model Description

This is a **LoRA adapter** for Llama-2-7B with **HiCI (Hierarchical Construction-Integration)** architecture,
trained for long-context understanding up to **8K tokens**.

Paper: [HiCI (arXiv 2603.20843)](https://arxiv.org/abs/2603.20843)

### HiCI Architecture

{HICI_ARCH_DESCRIPTION}

```
Input (8K tokens) → 4 segments × 2K
  Stage 1: 8 local slots per segment → L_i
  Stage 2: multi-view stats → K=4 global slots G
  Stage 3: Q=[chunk], KV=[G, L_i, chunk] → Flash Attention
```

## Trainable Components

```
adapter_model.bin  (27 MB)
└── LoRA Adapters (r=8, alpha=16): q_proj, k_proj, v_proj, o_proj

trainable_params.bin  (~2 GB)
├── global_memory.*            — Local Construction modules (32 layers)
├── hierarchical_aggregator.*  — Global Integration modules (32 layers)
├── input_layernorm / post_attention_layernorm — LayerNorm weights (32 layers)
├── model.embed_tokens.weight  — Token embeddings
└── model.norm.weight          — Final LayerNorm
```

## Training Details

- **Base Model**: meta-llama/Llama-2-7b-hf
- **Context Length**: 8,192 tokens (8K)
- **Segments**: 4 × 2,048 tokens
- **Local Memory Slots (M)**: 8 per segment
- **Global Memory Slots (K)**: 4
- **Memory Heads**: 8, Bottleneck dim: 512
- **LoRA**: r=8, alpha=16, target: q/k/v/o_proj
- **Checkpoint**: step 2000 / 2000
- **Batch**: per_device=1, grad_accum=8 (effective batch=8)
- **LR**: 2e-5 (LoRA), 2e-4 (memory modules), grad clip=0.3
- **Precision**: bf16
- **Hardware**: 8× H100 80GB, DeepSpeed Stage 2

## Usage

**Requires `llama_attn_hici.py` from this repo.**

```python
import torch
import transformers
from peft import PeftModel
import llama_attn_hici as hici_attn

# 1. Replace attention with HiCI BEFORE loading model
hici_attn.MIXED_GROUP_TRAINING = False
hici_attn.replace_llama_attn(use_flash_attn=True, use_full=False, use_hierarchical_forward=True)

# 2. Load base model
base_model = transformers.AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf", torch_dtype=torch.bfloat16, device_map="auto",
)

# 3. Register HiCI modules (must match training config)
hici_attn.register_hici_to_model(base_model, num_memory_slots=8, global_slots=4, num_heads=8, bottleneck_dim=512)

# 4. Load LoRA adapter + trainable_params
model = PeftModel.from_pretrained(base_model, "{HF_USERNAME}/{model_name}")

# 5. Tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained("{HF_USERNAME}/{model_name}")
```

## Citation

{CITATION}

## License

This model follows the [Llama 2 Community License](https://ai.meta.com/llama/license/).
"""


README_CREATORS = {
    "qwen3":  create_readme_qwen3,
    "llama2": create_readme_llama2,
}

# ====================================================
# Shared upload logic
# ====================================================

def check_files(cfg):
    checkpoint_path = cfg["checkpoint_path"]
    print("\nChecking files...")

    if not os.path.exists(checkpoint_path):
        print(f"  ERROR: checkpoint path not found: {checkpoint_path}")
        return False

    total_size = 0
    print("\nLoRA + HiCI weights:")
    for f in cfg["lora_files"]:
        path = os.path.join(checkpoint_path, f)
        if os.path.exists(path):
            size = os.path.getsize(path) / (1024 ** 2)
            total_size += size
            print(f"  OK  {f}: {size:.1f} MB")
        else:
            print(f"  MISSING  {f}")
            return False

    print("\nTokenizer files:")
    for f in cfg["tokenizer_files"]:
        path = os.path.join(checkpoint_path, f)
        if os.path.exists(path):
            size = os.path.getsize(path) / 1024
            print(f"  OK  {f}: {size:.1f} KB")
        else:
            print(f"  WARN  {f} not found")

    print(f"\nTotal: {total_size:.0f} MB (~{total_size/1024:.2f} GB)")
    return True


def prepare_upload_files(cfg, temp_dir):
    checkpoint_path = cfg["checkpoint_path"]
    os.makedirs(temp_dir, exist_ok=True)

    for f in cfg["lora_files"] + cfg["tokenizer_files"]:
        src = os.path.join(checkpoint_path, f)
        dst = os.path.join(temp_dir, f)
        if os.path.exists(src):
            print(f"  copy: {f}")
            shutil.copy2(src, dst)

    readme = README_CREATORS[cfg["_model_key"]](cfg["model_name"])
    with open(os.path.join(temp_dir, "README.md"), "w", encoding="utf-8") as fh:
        fh.write(readme)
    print("  write: README.md")

    with open(os.path.join(temp_dir, ".gitattributes"), "w") as fh:
        fh.write("*.bin filter=lfs diff=lfs merge=lfs -text\n")
        fh.write("*.safetensors filter=lfs diff=lfs merge=lfs -text\n")
    print("  write: .gitattributes")


def upload(temp_dir, repo_id):
    api = HfApi()
    print(f"\nCreating/verifying repo: {repo_id}")
    try:
        create_repo(repo_id, repo_type="model", exist_ok=True)
    except Exception as e:
        print(f"  ERROR creating repo: {e}")
        print("  Run first: huggingface-cli login")
        return False

    print(f"Uploading: {temp_dir}")
    try:
        api.upload_folder(folder_path=temp_dir, repo_id=repo_id, repo_type="model")
        print("Upload complete.")
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Upload HiCI model to HuggingFace")
    parser.add_argument("--model", required=True, choices=["qwen3", "llama2"],
                        help="Which model to upload: qwen3 or llama2")
    args = parser.parse_args()

    cfg = CONFIGS[args.model].copy()
    cfg["_model_key"] = args.model  # pass key for README dispatch

    print("=" * 50)
    print(f"Upload {cfg['model_name']} to HuggingFace")
    print("=" * 50)
    print(f"User:       {HF_USERNAME}")
    print(f"Model:      {cfg['model_name']}")
    print(f"Checkpoint: {cfg['checkpoint_path']}")
    print("=" * 50)

    if not check_files(cfg):
        print("\nAbort: file check failed.")
        return

    checkpoint_path = cfg["checkpoint_path"]
    adapter_path = os.path.join(checkpoint_path, cfg["adapter_file"])
    trainable_path = os.path.join(checkpoint_path, "trainable_params.bin")
    adapter_size = os.path.getsize(adapter_path) / 1024 / 1024 if os.path.exists(adapter_path) else 0
    trainable_size = os.path.getsize(trainable_path) / 1024 / 1024 / 1024 if os.path.exists(trainable_path) else 0

    print("\nFiles to upload:")
    print(f"  {cfg['adapter_file']:<35} ({adapter_size:.0f} MB, LoRA)")
    print(f"  trainable_params.bin               ({trainable_size:.2f} GB, HiCI modules + embed + norm)")
    print("  tokenizer files                    (~few MB)")
    print("  README.md")
    print(f"\n  Target: https://huggingface.co/{HF_USERNAME}/{cfg['model_name']}")

    confirm = input("\nConfirm upload? (yes/no): ")
    if confirm.lower() != "yes":
        print("Cancelled.")
        return

    temp_dir = f"/tmp/hf_upload_{cfg['model_name']}"
    print(f"\nPreparing temp dir: {temp_dir}")

    try:
        prepare_upload_files(cfg, temp_dir)
        repo_id = f"{HF_USERNAME}/{cfg['model_name']}"
        success = upload(temp_dir, repo_id)

        if success:
            print("\n" + "=" * 50)
            print("Upload successful!")
            print(f"  https://huggingface.co/{repo_id}")
            print("=" * 50)
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print("Temp dir cleaned up.")


if __name__ == "__main__":
    # Login first: huggingface-cli login
    main()
