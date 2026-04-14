# Merge LoRA weights with HiCI memory modules
#
# Key difference from original merge script:
# 1. Register HiCI memory modules BEFORE loading trainable_params.bin
# 2. This ensures local_constructor.* parameters are properly loaded

import os
import torch
import argparse
import transformers
from peft import PeftModel
from typing import Dict

# Import HiCI attention module
import llama_attn_hici as hici_attn

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


def parse_config():
    parser = argparse.ArgumentParser(
        description="Merge LoRA weights with HiCI memory modules"
    )
    parser.add_argument(
        "--base_model", type=str, required=True, help="Path to base LLaMA model"
    )
    parser.add_argument(
        "--peft_model", type=str, required=True, help="Path to PEFT/LoRA checkpoint"
    )
    parser.add_argument("--context_size", type=int, default=16384, help="Context size")
    parser.add_argument(
        "--save_path", type=str, required=True, help="Path to save merged model"
    )
    parser.add_argument("--cache_dir", type=str, default=None, help="Cache directory")

    # HiCI memory module parameters (should match training)
    # These can be inferred by inspecting trainable_params.bin
    parser.add_argument(
        "--num_local_slots", type=int, default=8, help="Number of memory slots"
    )
    parser.add_argument(
        "--global_slots",
        type=int,
        default=4,
        help="Number of global slots (inferred from global_queries shape)",
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=8,
        help="Number of attention heads (40 for 13B, 32 for 7B)",
    )
    parser.add_argument(
        "--compress_dim", type=int, default=512, help="Compression dimension"
    )
    parser.add_argument(
        "--shared_compress_dim",
        type=int,
        default=128,
        help="Shared compressor dimension",
    )
    # LocalConstructor bottleneck dimension
    parser.add_argument(
        "--bottleneck_dim",
        type=int,
        default=512,
        help="Bottleneck dimension for LocalConstructorFlash (default: 512)",
    )

    args = parser.parse_args()
    return args


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding."""
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def main(args):
    device = "cuda:0"
    torch.cuda.set_device(device)

    print("=" * 60)
    print("HiCI Model Merge Script")
    print("=" * 60)
    print(f"Base model: {args.base_model}")
    print(f"PEFT model: {args.peft_model}")
    print(f"Context size: {args.context_size}")
    print(f"Save path: {args.save_path}")
    print("=" * 60)

    # Step 1: Replace attention mechanism with HiCI version
    print("\n[1/6] Replacing attention mechanism...")
    hici_attn.MIXED_GROUP_TRAINING = False  # Disable for inference
    hici_attn.replace_llama_attn(
        use_flash_attn=True, use_full=False, use_hierarchical_forward=True
    )

    # Step 2: Load base model
    print("\n[2/6] Loading base model...")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.base_model,
        cache_dir=args.cache_dir,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # Step 3: Load tokenizer and resize embeddings
    print("\n[3/6] Loading tokenizer...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.base_model,
        cache_dir=args.cache_dir,
        model_max_length=args.context_size,
        padding_side="right",
        use_fast=False,
    )

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    # Step 4: Register HiCI memory modules (CRITICAL!)
    print("\n[4/6] Registering HiCI memory modules...")
    print(f"  num_local_slots: {args.num_local_slots}")
    print(f"  global_slots: {args.global_slots}")
    print(f"  num_heads: {args.num_heads}")
    print(f"  bottleneck_dim: {args.bottleneck_dim}")
    print(f"  compress_dim: {args.compress_dim}")
    print(f"  shared_compress_dim: {args.shared_compress_dim}")

    hici_attn.register_hici_to_model(
        model,
        num_local_slots=args.num_local_slots,
        global_slots=args.global_slots,
        num_heads=args.num_heads,
        use_global_integrator=True,
        use_local_constructor_flash=False,
        use_bottleneck=True,
        bottleneck_dim=args.bottleneck_dim,  # must match training config
        compress_dim=args.compress_dim,
        shared_compress_dim=args.shared_compress_dim,
    )

    # Step 5: Load trainable_params.bin (now HiCI modules exist!)
    print("\n[5/6] Loading trainable parameters...")
    trainable_params_path = os.path.join(args.peft_model, "trainable_params.bin")
    if os.path.isfile(trainable_params_path):
        trainable_params = torch.load(trainable_params_path, map_location="cpu")

        # Count HiCI parameters
        hici_keys = [
            k
            for k in trainable_params.keys()
            if "memory" in k.lower() or "global" in k.lower()
        ]
        print(f"  Found {len(trainable_params)} parameters in trainable_params.bin")
        print(f"  Including {len(hici_keys)} HiCI memory parameters")

        # Load with strict=False but check what was loaded
        missing, unexpected = model.load_state_dict(trainable_params, strict=False)

        # Verify HiCI parameters were loaded
        loaded_hici = len(hici_keys) - len(
            [k for k in missing if "memory" in k.lower() or "global" in k.lower()]
        )
        print(f"  Successfully loaded {loaded_hici} HiCI parameters")

        if missing:
            hici_missing = [
                k for k in missing if "memory" in k.lower() or "global" in k.lower()
            ]
            if hici_missing:
                print(f"  ⚠️ Warning: {len(hici_missing)} HiCI parameters not loaded!")
                for k in hici_missing[:5]:
                    print(f"    - {k}")
    else:
        print(f"  ⚠️ Warning: {trainable_params_path} not found!")

    # Step 6: Load and merge LoRA
    print("\n[6/6] Loading and merging LoRA weights...")
    model = PeftModel.from_pretrained(
        model,
        args.peft_model,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model = model.merge_and_unload()

    # Verify final model has HiCI modules
    state_dict = model.state_dict()
    final_hici_keys = [
        k for k in state_dict.keys() if "memory" in k.lower() or "global" in k.lower()
    ]
    print(f"\n✅ Final model has {len(final_hici_keys)} HiCI parameters")

    # Save
    print(f"\nSaving merged model to {args.save_path}...")
    model.save_pretrained(args.save_path)
    tokenizer.save_pretrained(args.save_path)

    print("\n" + "=" * 60)
    print("✅ Merge complete!")
    print("=" * 60)


if __name__ == "__main__":
    args = parse_config()
    main(args)
