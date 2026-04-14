"""
Download PG19 test split raw text from HuggingFace and save locally.
Run on login node (needs internet).

Usage:
    python download_pg19.py
    python download_pg19.py --split validation
"""

import argparse
import os
from datasets import load_dataset
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output_dir", type=str, default="data/pg19_raw")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Downloading PG19 {args.split} split from HuggingFace...")
    dataset = load_dataset("emozilla/pg19", split=args.split)
    print(f"Number of books: {len(dataset)}")

    # Save as single concatenated text file
    output_path = os.path.join(args.output_dir, f"{args.split}.txt")
    total_chars = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for i, example in enumerate(tqdm(dataset, desc="Writing")):
            text = example["text"]
            f.write(text)
            f.write("\n")
            total_chars += len(text)

    print(f"Saved to {output_path}")
    print(f"Total chars: {total_chars:,}")
    print(f"File size: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")
    print(f"Number of books: {len(dataset)}")


if __name__ == "__main__":
    main()
