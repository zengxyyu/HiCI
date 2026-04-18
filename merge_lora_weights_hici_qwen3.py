# Merge LoRA weights with HiCI modules (Qwen3 version)
#
# Adapted from merge_lora_weights_hici.py for Qwen3 models.
# Key difference: uses qwen3_attn_hici instead of llama_attn_hici.
#
# Usage:
#   python merge_lora_weights_hici_qwen3.py \
#       --base_model ./models/Qwen3-8B \
#       --peft_model ./checkpoints/Qwen3-8b-HiCI-48k-1000steps \
#       --save_path ./models/merged/Qwen3-8b-HiCI-48k-merged \
#       --num_local_slots 8 --global_slots 4 --num_heads 8 \
#       --bottleneck_dim 512 --context_size 49152

import os
import torch
import argparse
import transformers
from peft import PeftModel

import qwen3_attn_hici as hici_attn


def parse_config():
    parser = argparse.ArgumentParser(
        description="Merge LoRA weights with HiCI modules (Qwen3)"
    )
    parser.add_argument(
        "--base_model", type=str, required=True, help="Path to base Qwen3 model"
    )
    parser.add_argument(
        "--peft_model", type=str, required=True, help="Path to PEFT/LoRA checkpoint"
    )
    parser.add_argument(
        "--save_path", type=str, required=True, help="Path to save merged model"
    )
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument(
        "--context_size", type=int, default=-1,
        help="Context size used during training (for RoPE scaling). e.g. 49152 for 48k"
    )

    # HiCI module parameters (must match training!)
    parser.add_argument("--num_local_slots", type=int, default=8)
    parser.add_argument("--global_slots", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--compress_dim", type=int, default=512)
    parser.add_argument("--shared_compress_dim", type=int, default=128)
    parser.add_argument("--use_local_constructor_flash", action="store_true", default=False)
    parser.add_argument("--bottleneck_dim", type=int, default=512)

    args = parser.parse_args()
    return args


def main(args):
    device = "cuda:0"
    torch.cuda.set_device(device)

    print("=" * 60)
    print("HiCI Model Merge Script (Qwen3)")
    print("=" * 60)
    print(f"Base model: {args.base_model}")
    print(f"PEFT model: {args.peft_model}")
    print(f"Save path:  {args.save_path}")
    print("=" * 60)

    # Step 1: Replace attention with HiCI
    print("\n[1/6] Replacing Qwen3 attention with HiCI...")
    hici_attn.MIXED_GROUP_TRAINING = False
    hici_attn.replace_qwen3_attn(
        use_flash_attn=True, use_full=False, use_hierarchical_forward=True
    )

    # Step 2: Load base model
    print("\n[2/6] Loading base Qwen3 model...")
    config = transformers.AutoConfig.from_pretrained(args.base_model, cache_dir=args.cache_dir)
    if args.context_size > 0:
        orig_ctx_len = getattr(config, "max_position_embeddings", None)
        if orig_ctx_len and args.context_size > orig_ctx_len:
            scaling_factor = args.context_size / orig_ctx_len
            config.rope_scaling = {"type": "linear", "factor": scaling_factor}
            print(f"  Applied RoPE scaling: factor={scaling_factor} ({orig_ctx_len} -> {args.context_size})")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.base_model,
        config=config,
        cache_dir=args.cache_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Step 3: Load tokenizer
    print("\n[3/6] Loading tokenizer...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.base_model,
        cache_dir=args.cache_dir,
        padding_side="right",
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|endoftext|>"})
        model.resize_token_embeddings(len(tokenizer))

    # Step 4: Register HiCI modules (CRITICAL: before loading weights!)
    print("\n[4/6] Registering HiCI modules...")
    print(f"  num_local_slots: {args.num_local_slots}")
    print(f"  global_slots:     {args.global_slots}")
    print(f"  num_heads:        {args.num_heads}")
    print(f"  use_local_constructor_flash: {args.use_local_constructor_flash}")
    print(f"  bottleneck_dim:   {args.bottleneck_dim}")

    hici_attn.register_hici_to_qwen3_model(
        model,
        num_local_slots=args.num_local_slots,
        global_slots=args.global_slots,
        num_heads=args.num_heads,
        use_global_integrator=True,
        use_local_constructor_flash=args.use_local_constructor_flash,
        use_bottleneck=True,
        bottleneck_dim=args.bottleneck_dim,
        compress_dim=args.compress_dim,
        shared_compress_dim=args.shared_compress_dim,
    )

    # Step 5: Load trainable_params.bin
    print("\n[5/6] Loading trainable parameters...")
    trainable_params_path = os.path.join(args.peft_model, "trainable_params.bin")
    if os.path.isfile(trainable_params_path):
        trainable_params = torch.load(trainable_params_path, map_location="cpu")

        hici_keys = [
            k for k in trainable_params.keys()
            if "local_constructor" in k or "global_integrator" in k
        ]
        print(f"  Found {len(trainable_params)} params in trainable_params.bin")
        print(f"  Including {len(hici_keys)} HiCI params")

        missing, unexpected = model.load_state_dict(trainable_params, strict=False)

        loaded_hici = len(hici_keys) - len(
            [k for k in missing if "local_constructor" in k or "global_integrator" in k]
        )
        print(f"  Loaded {loaded_hici} HiCI params successfully")

        if missing:
            hici_missing = [
                k for k in missing
                if "local_constructor" in k or "global_integrator" in k
            ]
            if hici_missing:
                print(f"  WARNING: {len(hici_missing)} HiCI params NOT loaded:")
                for k in hici_missing[:5]:
                    print(f"    - {k}")
    else:
        print(f"  WARNING: {trainable_params_path} not found!")
        print("  HiCI modules will have random weights.")
        print("  Run get_trainable_weights.py first:")
        print(f"    python get_trainable_weights.py --checkpoint_path {args.peft_model} "
              f"--trainable_params embed,norm,local_constructor,global_integrator")

    # Step 6: Load and merge LoRA
    print("\n[6/6] Loading and merging LoRA weights...")
    model = PeftModel.from_pretrained(
        model,
        args.peft_model,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = model.merge_and_unload()

    # Verify
    state_dict = model.state_dict()
    final_hici_keys = [
        k for k in state_dict.keys()
        if "local_constructor" in k or "global_integrator" in k
    ]
    print(f"\nFinal model has {len(final_hici_keys)} HiCI params")

    # Save
    print(f"\nSaving merged model to {args.save_path}...")
    model.save_pretrained(args.save_path)
    tokenizer.save_pretrained(args.save_path)

    print("\n" + "=" * 60)
    print("Merge complete!")
    print("=" * 60)


if __name__ == "__main__":
    args = parse_config()
    main(args)
