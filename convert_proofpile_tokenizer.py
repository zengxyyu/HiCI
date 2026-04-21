"""
Convert proof-pile .bin from LLaMA-2 tokenizer (uint16) to another tokenizer.

Source file download (LongLoRA, LLaMA-2 tokenized):
  test_sampled_data.bin: https://drive.google.com/file/d/1bUI5lPDvrqzY_XXJJ2sSuvZx0Y9AZClE/view?usp=share_link

dst_dtype is auto-detected from the target tokenizer's vocab size:
  vocab <= 65535  -> uint16  (e.g. LLaMA-2: 32000)
  vocab >  65535  -> uint32  (e.g. LLaMA-3: 128256, Qwen3: 151936)

Usage:
    # LLaMA-3
    python convert_proofpile_tokenizer.py \
        --input  data/proof-pile/test_sampled_data.bin \
        --src_model  ./models/Llama-2-7b-hf \
        --dst_model  ./models/Meta-Llama-3-8B \
        --output data/proof-pile_llama3/test_sampled_data.bin

    # Qwen3
    python convert_proofpile_tokenizer.py \
        --input  data/proof-pile/test_sampled_data.bin \
        --src_model  ./models/Llama-2-7b-hf \
        --dst_model  ./models/Qwen3-8B \
        --output data/proof-pile_qwen3/test_sampled_data.bin
"""

import argparse
import os
import numpy as np
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",     required=True, help="Source .bin (LLaMA-2 uint16)")
    parser.add_argument("--src_model", required=True, help="Source tokenizer path (LLaMA-2)")
    parser.add_argument("--dst_model", required=True, help="Target tokenizer path")
    parser.add_argument("--output",    required=True, help="Output .bin path")
    parser.add_argument("--src_dtype", default="uint16", choices=["uint16", "uint32"])
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    print(f"Loading source tokenizer : {args.src_model}")
    src_tok = AutoTokenizer.from_pretrained(args.src_model, trust_remote_code=True)

    print(f"Loading target tokenizer : {args.dst_model}")
    dst_tok = AutoTokenizer.from_pretrained(args.dst_model, trust_remote_code=True)

    # auto-detect output dtype from vocab size (add_special_tokens=False so max id < vocab_size)
    dst_dtype = "uint16" if dst_tok.vocab_size <= 65535 else "uint32"
    print(f"  vocab_size={dst_tok.vocab_size} -> dst_dtype={dst_dtype}")

    print(f"Reading {args.input} ...")
    src_ids = np.fromfile(args.input, dtype=np.dtype(args.src_dtype))
    total = len(src_ids)
    print(f"  {total:,} tokens ({os.path.getsize(args.input) / 1024**2:.1f} MB, {args.src_dtype})")

    # Decode all tokens at once to avoid SentencePiece boundary artifacts
    # (chunked decode loses leading ▁ spaces at chunk edges).
    print("Decoding to text...")
    text = src_tok.decode(src_ids.tolist(), skip_special_tokens=True)
    print(f"  Text length: {len(text):,} chars")

    print("Re-tokenizing with target tokenizer...")
    all_dst_ids = dst_tok.encode(text, add_special_tokens=False)

    out_arr = np.array(all_dst_ids, dtype=np.dtype(dst_dtype))
    out_arr.tofile(args.output)

    src_mb = os.path.getsize(args.input)  / 1024**2
    dst_mb = os.path.getsize(args.output) / 1024**2
    ratio  = len(out_arr) / total

    print(f"\nDone.")
    print(f"  Source : {total:,} tokens  ({src_mb:.1f} MB, {args.src_dtype})")
    print(f"  Target : {len(out_arr):,} tokens  ({dst_mb:.1f} MB, {dst_dtype})")
    print(f"  Token ratio (dst/src): {ratio:.3f}")
    print(f"  Saved  : {args.output}")


if __name__ == "__main__":
    main()
