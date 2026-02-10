#!/usr/bin/env python3
"""
Merge Stage3 + LoRA checkpoint into a standalone HuggingFace model.

This script handles the case where DeepSpeed Stage 3 + LoRA training produces
a checkpoint with:
- pytorch_model.bin containing both base weights and separate LoRA weights
- Empty adapter_model.bin

It merges LoRA weights into base weights and saves a clean HuggingFace model.
"""

import argparse
import os
import json
import shutil
import torch
from collections import OrderedDict
from tqdm import tqdm

def merge_lora_weights(checkpoint_path, save_path, base_model_path=None,
                       lora_alpha=16, lora_r=8, include_hici=False):
    """
    Merge LoRA weights from pytorch_model.bin and save as HuggingFace model.

    Args:
        checkpoint_path: Path to checkpoint directory containing pytorch_model.bin
        save_path: Path to save the merged model
        base_model_path: Path to base model (for copying config/tokenizer if needed)
        lora_alpha: LoRA alpha parameter (default: 16)
        lora_r: LoRA rank parameter (default: 8)
        include_hici: Whether to include HiCI local_constructor and global_integrator
    """

    pytorch_model_path = os.path.join(checkpoint_path, "pytorch_model.bin")

    if not os.path.exists(pytorch_model_path):
        raise FileNotFoundError(f"pytorch_model.bin not found at {pytorch_model_path}")

    print("=" * 60)
    print("Stage3 + LoRA Checkpoint Merger")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Save path: {save_path}")
    print(f"LoRA alpha: {lora_alpha}, rank: {lora_r}")
    print(f"Include HiCI: {include_hici}")
    print()

    # Load checkpoint
    print("Loading pytorch_model.bin (this may take a while for 15GB file)...")
    state_dict = torch.load(pytorch_model_path, map_location='cpu', weights_only=False)
    print(f"Loaded {len(state_dict)} keys")

    # Analyze keys
    all_keys = list(state_dict.keys())
    lora_a_keys = [k for k in all_keys if '.lora_A.' in k]
    lora_b_keys = [k for k in all_keys if '.lora_B.' in k]
    local_constructor_keys = [k for k in all_keys if 'local_constructor' in k and 'lora' not in k]
    hierarchical_keys = [k for k in all_keys if 'hierarchical' in k]

    print(f"\nKey analysis:")
    print(f"  Total keys: {len(all_keys)}")
    print(f"  LoRA A keys: {len(lora_a_keys)}")
    print(f"  LoRA B keys: {len(lora_b_keys)}")
    print(f"  LocalConstructor keys (non-LoRA): {len(local_constructor_keys)}")
    print(f"  Hierarchical keys: {len(hierarchical_keys)}")

    # Prefix to remove
    prefix = "base_model.model."

    # Organize weights
    base_weights = {}  # Original base weights
    lora_a_weights = {}  # LoRA A matrices
    lora_b_weights = {}  # LoRA B matrices
    other_weights = {}  # embed, norm, etc.
    hici_weights = {}  # local_constructor, hierarchical (if include_hici)

    print("\nOrganizing weights...")
    for key, value in tqdm(state_dict.items(), desc="Processing"):
        # Remove prefix
        if key.startswith(prefix):
            clean_key = key[len(prefix):]
        else:
            clean_key = key

        if '.lora_A.default.weight' in clean_key:
            # LoRA A: get the base key
            base_key = clean_key.replace('.lora_A.default.weight', '.weight')
            lora_a_weights[base_key] = value
        elif '.lora_B.default.weight' in clean_key:
            # LoRA B: get the base key
            base_key = clean_key.replace('.lora_B.default.weight', '.weight')
            lora_b_weights[base_key] = value
        elif 'local_constructor' in clean_key or 'hierarchical' in clean_key:
            if include_hici:
                hici_weights[clean_key] = value
        elif '.lora_' not in clean_key:
            # Regular weight (base model weight or other trainable)
            if any(x in clean_key for x in ['q_proj.weight', 'k_proj.weight', 'v_proj.weight', 'o_proj.weight']):
                base_weights[clean_key] = value
            else:
                other_weights[clean_key] = value

    print(f"\nOrganized:")
    print(f"  Base weights (to merge): {len(base_weights)}")
    print(f"  LoRA A weights: {len(lora_a_weights)}")
    print(f"  LoRA B weights: {len(lora_b_weights)}")
    print(f"  Other weights: {len(other_weights)}")
    if include_hici:
        print(f"  HiCI weights: {len(hici_weights)}")

    # Merge LoRA into base weights
    # Formula: W_merged = W_base + (lora_B @ lora_A) * scaling
    scaling = lora_alpha / lora_r
    print(f"\nMerging LoRA weights (scaling = {lora_alpha}/{lora_r} = {scaling})...")

    merged_weights = OrderedDict()
    merge_count = 0

    for base_key, base_value in tqdm(base_weights.items(), desc="Merging"):
        if base_key in lora_a_weights and base_key in lora_b_weights:
            lora_a = lora_a_weights[base_key]  # [r, in_features]
            lora_b = lora_b_weights[base_key]  # [out_features, r]

            # Compute delta: B @ A -> [out_features, in_features]
            delta = torch.matmul(lora_b.float(), lora_a.float()) * scaling

            # Merge: W_merged = W_base + delta
            merged = base_value.float() + delta
            merged_weights[base_key] = merged.to(base_value.dtype)
            merge_count += 1
        else:
            # No LoRA for this weight, keep original
            merged_weights[base_key] = base_value

    print(f"Merged {merge_count} weight matrices")

    # Add other weights
    for key, value in other_weights.items():
        merged_weights[key] = value

    # Add HiCI weights if requested
    if include_hici:
        for key, value in hici_weights.items():
            merged_weights[key] = value
        print(f"Added {len(hici_weights)} HiCI weights")

    print(f"\nTotal merged weights: {len(merged_weights)}")

    # Create save directory
    os.makedirs(save_path, exist_ok=True)

    # Save in HuggingFace format (sharded for large models)
    print(f"\nSaving to {save_path}...")

    # Calculate total size
    total_size = sum(v.numel() * v.element_size() for v in merged_weights.values())
    print(f"Total model size: {total_size / (1024**3):.2f} GB")

    # Shard if larger than 5GB
    max_shard_size = 5 * 1024 * 1024 * 1024  # 5GB

    if total_size > max_shard_size:
        # Save sharded
        print("Saving as sharded model...")
        current_shard = OrderedDict()
        current_size = 0
        shard_idx = 1
        weight_map = {}

        for key, value in tqdm(merged_weights.items(), desc="Saving shards"):
            tensor_size = value.numel() * value.element_size()

            if current_size + tensor_size > max_shard_size and current_shard:
                # Save current shard
                shard_file = f"pytorch_model-{shard_idx:05d}-of-XXXXX.bin"
                torch.save(current_shard, os.path.join(save_path, shard_file))
                shard_idx += 1
                current_shard = OrderedDict()
                current_size = 0

            current_shard[key] = value
            current_size += tensor_size
            weight_map[key] = f"pytorch_model-{shard_idx:05d}-of-XXXXX.bin"

        # Save last shard
        if current_shard:
            shard_file = f"pytorch_model-{shard_idx:05d}-of-XXXXX.bin"
            torch.save(current_shard, os.path.join(save_path, shard_file))

        # Rename shards with correct total count
        total_shards = shard_idx
        for i in range(1, total_shards + 1):
            old_name = f"pytorch_model-{i:05d}-of-XXXXX.bin"
            new_name = f"pytorch_model-{i:05d}-of-{total_shards:05d}.bin"
            os.rename(
                os.path.join(save_path, old_name),
                os.path.join(save_path, new_name)
            )
            # Update weight map
            for key in weight_map:
                if weight_map[key] == old_name:
                    weight_map[key] = new_name

        # Save index
        index = {
            "metadata": {"total_size": total_size},
            "weight_map": weight_map
        }
        with open(os.path.join(save_path, "pytorch_model.bin.index.json"), "w") as f:
            json.dump(index, f, indent=2)

        print(f"Saved {total_shards} shards")
    else:
        # Save as single file
        torch.save(merged_weights, os.path.join(save_path, "pytorch_model.bin"))
        print("Saved as single pytorch_model.bin")

    # Copy config and tokenizer files
    print("\nCopying config and tokenizer files...")

    # Try to copy from checkpoint first, then from base_model_path
    config_sources = [checkpoint_path]
    if base_model_path:
        config_sources.append(base_model_path)

    files_to_copy = [
        "config.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "generation_config.json"
    ]

    for filename in files_to_copy:
        copied = False
        for source in config_sources:
            src_path = os.path.join(source, filename)
            if os.path.exists(src_path):
                shutil.copy2(src_path, os.path.join(save_path, filename))
                print(f"  Copied {filename} from {source}")
                copied = True
                break
        if not copied and filename == "config.json":
            print(f"  WARNING: {filename} not found!")

    # Update config.json if needed
    config_path = os.path.join(save_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)

        # Remove PEFT-related fields if present
        peft_fields = ['peft_config', 'base_model_name_or_path']
        for field in peft_fields:
            if field in config:
                del config[field]

        # Fix vocab_size if embed_tokens has different size
        if 'model.embed_tokens.weight' in merged_weights:
            actual_vocab_size = merged_weights['model.embed_tokens.weight'].shape[0]
            if config.get('vocab_size') != actual_vocab_size:
                print(f"  Fixing vocab_size: {config.get('vocab_size')} -> {actual_vocab_size}")
                config['vocab_size'] = actual_vocab_size

        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

    print("\n" + "=" * 60)
    print("Merge complete!")
    print(f"Merged model saved to: {save_path}")
    print("=" * 60)

    return save_path

def main():
    parser = argparse.ArgumentParser(description="Merge Stage3 + LoRA checkpoint")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to checkpoint directory containing pytorch_model.bin")
    parser.add_argument("--save_path", type=str, required=True,
                        help="Path to save the merged model")
    parser.add_argument("--base_model_path", type=str, default=None,
                        help="Path to base model (for config/tokenizer)")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="LoRA alpha parameter")
    parser.add_argument("--lora_r", type=int, default=8,
                        help="LoRA rank parameter")
    parser.add_argument("--include_hici", action="store_true",
                        help="Include HiCI local_constructor and global_integrator weights")

    args = parser.parse_args()

    merge_lora_weights(
        checkpoint_path=args.checkpoint_path,
        save_path=args.save_path,
        base_model_path=args.base_model_path,
        lora_alpha=args.lora_alpha,
        lora_r=args.lora_r,
        include_hici=args.include_hici
    )

if __name__ == "__main__":
    main()
