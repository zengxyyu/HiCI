"""
Prepare evaluation data (PG19 / proof-pile) for any model's tokenizer.

The existing .bin files were tokenized with LLaMA-2 tokenizer.
Other models (LLaMA-3, Qwen3) need re-tokenized versions.

Usage (login node, needs internet for --from_hf):

  # PG19 from local text (no internet)
  python prepare_eval_data.py --model_path ./models/Qwen3-8B \
      --text_file ChunkLlama/ppl/pg19_raw.txt --output_dir data/pg19_qwen3

  # PG19 from HuggingFace
  python prepare_eval_data.py --model_path ./models/Qwen3-8B \
      --from_hf --dataset pg19 --split test --output_dir data/pg19_qwen3

  # proof-pile from HuggingFace (login node)
  python prepare_eval_data.py --model_path ./models/Qwen3-8B \
      --from_hf --dataset proof-pile --split test --output_dir data/proof-pile_qwen3

  # LLaMA-3 versions
  python prepare_eval_data.py --model_path ./models/Meta-Llama-3-8B \
      --text_file ChunkLlama/ppl/pg19_raw.txt --output_dir data/pg19_llama3
  python prepare_eval_data.py --model_path ./models/Meta-Llama-3-8B \
      --from_hf --dataset proof-pile --split test --output_dir data/proof-pile_llama3
"""

import argparse
import os
import numpy as np
from transformers import AutoTokenizer


def tokenize_from_hf(tokenizer, dataset_name, split, sample_size=None):
    from datasets import load_dataset
    from tqdm import tqdm

    if dataset_name == "proof-pile":
        # proof-pile on HuggingFace: EleutherAI/proof-pile-2 or similar
        # Try multiple possible names
        for name in ["EleutherAI/proof-pile-2", "hoskinson-center/proof-pile", "proof-pile"]:
            try:
                print(f"Trying to load {name} ({split})...")
                dataset = load_dataset(name, split=split, trust_remote_code=True)
                break
            except Exception as e:
                print(f"  Failed: {e}")
                continue
        else:
            raise RuntimeError("Could not load proof-pile from any known source")
    else:
        print(f"Loading {dataset_name} ({split}) from HuggingFace...")
        dataset = load_dataset(dataset_name, split=split, trust_remote_code=True)

    print(f"Number of examples: {len(dataset)}")

    if sample_size and sample_size < len(dataset):
        import random
        random.seed(42)
        indices = random.sample(range(len(dataset)), sample_size)
        dataset = dataset.select(indices)
        print(f"Sampled {sample_size} examples")

    all_tokens = []
    for example in tqdm(dataset, desc="Tokenizing"):
        text = example.get("text", "")
        if text:
            tokens = tokenizer.encode(text, add_special_tokens=False)
            all_tokens.extend(tokens)

    return all_tokens


def tokenize_from_file(tokenizer, text_file, chunk_chars=1_000_000):
    print(f"Loading text from {text_file}...")
    file_size = os.path.getsize(text_file)
    print(f"File size: {file_size / 1024 / 1024:.1f} MB")

    all_tokens = []
    with open(text_file, "r", encoding="utf-8") as f:
        chunk_idx = 0
        while True:
            text = f.read(chunk_chars)
            if not text:
                break
            tokens = tokenizer.encode(text, add_special_tokens=False)
            all_tokens.extend(tokens)
            chunk_idx += 1
            print(f"  Chunk {chunk_idx}: {len(tokens)} tokens (total: {len(all_tokens)})")

    return all_tokens


def main():
    parser = argparse.ArgumentParser(description="Prepare eval data for any tokenizer")
    parser.add_argument("--model_path", type=str, required=True, help="Model path for tokenizer")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--text_file", type=str, default=None, help="Local text file (no internet needed)")
    parser.add_argument("--from_hf", action="store_true", help="Load from HuggingFace")
    parser.add_argument("--dataset", type=str, default="pg19", help="HF dataset name")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--sample_size", type=int, default=None, help="Sample N examples (for large datasets)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading tokenizer from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    print(f"Vocab size: {tokenizer.vocab_size}")

    if args.text_file:
        all_tokens = tokenize_from_file(tokenizer, args.text_file)
    elif args.from_hf:
        all_tokens = tokenize_from_hf(tokenizer, args.dataset, args.split, args.sample_size)
    else:
        print("ERROR: Must specify --text_file or --from_hf")
        return

    all_tokens = np.array(all_tokens, dtype=np.uint32)
    print(f"Total tokens: {len(all_tokens)}")
    print(f"Max token id: {all_tokens.max()}")
    print(f"Min token id: {all_tokens.min()}")

    output_path = os.path.join(args.output_dir, f"{args.split}.bin")
    all_tokens.tofile(output_path)
    print(f"Saved to {output_path} ({os.path.getsize(output_path) / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
