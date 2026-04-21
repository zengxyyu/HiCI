#!/usr/bin/env python3
"""
Upload HiCI evaluation data to HuggingFace as a dataset.
Includes PG19 (tokenized for Llama-2/3 and Qwen3) and Proof-pile binaries.
"""

from huggingface_hub import HfApi, create_repo
import os

HF_USERNAME = "ZengXiangyu"
DATASET_NAME = "pg19-and-proof-pile"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

README = """\
---
license: other
tags:
- evaluation
- long-context
- long-context modeling
- pg19
- proof-pile
- hici
---

# HiCI Evaluation Data

Pre-tokenized binary evaluation splits used in the [HiCI](https://arxiv.org/abs/2603.20843) paper
(Hierarchical Construction-Integration for long-context LLMs).

## Contents

| Path | Description |
|------|-------------|
| `pg19_llama2/test.bin` | PG19 test set, Llama-2 tokenizer (uint16) |
| `pg19_llama2/validation.bin` | PG19 validation set, Llama-2 tokenizer (uint16) |
| `pg19_llama3/test.bin` | PG19 test set, Llama-3 tokenizer (uint32) |
| `pg19_llama3/validation.bin` | PG19 validation set, Llama-3 tokenizer (uint32) |
| `pg19_qwen3/test.bin` | PG19 test set, Qwen3 tokenizer (uint32) |
| `pg19_qwen3/validation.bin` | PG19 validation set, Qwen3 tokenizer (uint32) |
| `pg19_raw/test.txt` | PG19 test set, raw text |
| `pg19_raw/validation.txt` | PG19 validation set, raw text |
| `proof-pile_llama2/test_sampled_data.bin` | Proof-pile 128-doc sampled test set, Llama-2 tokenizer (uint16) |
| `proof-pile_llama3/test_sampled_data.bin` | Proof-pile 128-doc sampled test set, Llama-3 tokenizer (uint32) |
| `proof-pile_qwen3/test_sampled_data.bin` | Proof-pile 128-doc sampled test set, Qwen3 tokenizer (uint32) |

## Format

`.bin` files are memory-mapped token ID arrays, compatible with the evaluation scripts in the HiCI repo.
- Llama-2 tokenized files: `uint16` (vocab size 32,000)
- Llama-3 / Qwen3 tokenized files: `uint32` (vocab size > 65,535)

```python
import numpy as np
data = np.memmap("pg19_llama2/test.bin", dtype=np.uint16, mode="r")   # Llama-2
data = np.memmap("pg19_qwen3/test.bin",  dtype=np.uint32, mode="r")   # Qwen3 / Llama-3
```

## Usage

Download a single file:
```bash
huggingface-cli download ZengXiangyu/pg19-and-proof-pile proof-pile_llama2/test_sampled_data.bin --repo-type dataset
```

Or the full dataset:
```bash
huggingface-cli download ZengXiangyu/pg19-and-proof-pile --repo-type dataset --local-dir ./data
```

## Proof-pile Sampling

`proof-pile_llama2/test_sampled_data.bin` is identical to the file released by
[LongLoRA](https://github.com/dvlab-research/LongLoRA): 128 documents randomly sampled from the
proof-pile test split, each with at least 32,768 tokens, tokenized with the LLaMA-2 tokenizer.

`proof-pile_llama3` and `proof-pile_qwen3` contain the **same 128 documents** re-tokenized with
their respective tokenizers, enabling fair cross-model comparison.

## Source

- PG19: [deepmind/pg19](https://huggingface.co/datasets/deepmind/pg19)
- Proof-pile: [EleutherAI/proof-pile](https://huggingface.co/datasets/EleutherAI/proof-pile)
- Proof-pile LLaMA-2 tokenized (original): [LongLoRA](https://github.com/dvlab-research/LongLoRA)
"""


def main():
    repo_id = f"{HF_USERNAME}/{DATASET_NAME}"
    api = HfApi()

    print(f"Creating dataset repo: {repo_id}")
    create_repo(repo_id, repo_type="dataset", exist_ok=True)

    # Upload README
    api.upload_file(
        path_or_fileobj=README.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )
    print("Uploaded README.md")

    # Upload all data files
    subdirs = [
        "pg19_llama2",
        "pg19_llama3",
        "pg19_qwen3",
        "pg19_raw",
        "proof-pile_llama2",
        "proof-pile_llama3",
        "proof-pile_qwen3",
    ]

    for subdir in subdirs:
        subdir_path = os.path.join(DATA_DIR, subdir)
        if not os.path.isdir(subdir_path):
            print(f"Skipping {subdir} (not found)")
            continue
        for fname in os.listdir(subdir_path):
            local_path = os.path.join(subdir_path, fname)
            if not os.path.isfile(local_path):
                continue
            repo_path = f"{subdir}/{fname}"
            size_mb = os.path.getsize(local_path) / 1024 / 1024
            print(f"Uploading {repo_path} ({size_mb:.1f} MB)...")
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=repo_path,
                repo_id=repo_id,
                repo_type="dataset",
            )
            print(f"  Done: {repo_path}")

    print(f"\nAll done! https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    main()
