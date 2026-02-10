# Modified passkey retrieval for HiCI (Hierarchical Cognitive-Inspired) models
# Uses grouped attention during evaluation to match training

import os
import math
import torch
import argparse
import random
import numpy as np
from numpy import random
from tqdm import tqdm
import transformers
from peft import PeftModel

# Import HiCI attention replacement instead of standard LongLoRA
import llama_attn_hici as hici_attn

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--base_model', type=str, default="/path/to/llama-7b-hf")
    parser.add_argument('--cache_dir', type=str, default="./cache")
    parser.add_argument('--context_size', type=int, default=-1, help='context size during fine-tuning')
    parser.add_argument('--flash_attn', type=bool, default=True, help='whether to use flash attention 2')
    parser.add_argument('--max_tokens', type=int, default=32000, help='maximum token length for evaluation')
    parser.add_argument('--interval', type=int, default=1000, help='interval for evaluation')
    parser.add_argument('--num_tests', type=int, default=10, help='number of repeat testing for each length')

    # HiCI specific parameters
    parser.add_argument('--segment_size', type=int, default=1024,
                        help='segment size for grouped attention (should match training)')
    parser.add_argument('--use_grouped_attn', action='store_true', default=True,
                        help='use grouped attention (matching training) instead of full attention')

    args = parser.parse_args()
    return args

def generate_prompt_landmark(n_garbage, seed):
    """Generates a text file and inserts an passkey at a random position."""
    rnd_state = random.get_state()
    random.seed(seed)
    n_garbage_prefix = random.randint(0, n_garbage)
    n_garbage_suffix = n_garbage - n_garbage_prefix

    task_description = "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there."
    garbage = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
    garbage_inf = " ".join([garbage] * 50000)
    assert len(garbage_inf) >= n_garbage, f"garbage_inf length {len(garbage_inf)} < n_garbage {n_garbage}"
    garbage_prefix = garbage_inf[:n_garbage_prefix]
    garbage_suffix = garbage_inf[:n_garbage_suffix]
    pass_key = random.randint(1, 50000)
    information_line = f"The pass key is {pass_key}. Remember it. {pass_key} is the pass key."
    final_question = "What is the pass key? The pass key is"
    lines = [
        task_description,
        garbage_prefix,
        information_line,
        garbage_suffix,
        final_question,
    ]
    random.set_state(rnd_state)
    return "\n".join(lines), str(pass_key)

def passkey_retrieval_test(model, tokenizer, device, use_cache=False, n_garbage=60000, seed=666):
    prompt, answer = generate_prompt_landmark(n_garbage, seed)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(device)
    len_token = input_ids.shape[-1]

    answer_ids = tokenizer(answer, return_tensors="pt").input_ids[:, 1:] # drop BOS
    generation_output = model.generate(
        input_ids=input_ids, max_new_tokens=answer_ids.shape[-1], num_beams=1, use_cache=use_cache
    )

    model_answer = generation_output[0, -answer_ids.shape[-1]:].cpu()

    is_correct = (model_answer == answer_ids[0]).all().item()
    return is_correct, len_token

def main(args):
    device = "cuda:0"
    torch.cuda.set_device(device)

    print("base model", args.base_model)
    print(f"HiCI Mode: use_grouped_attn={args.use_grouped_attn}, segment_size={args.segment_size}")

    if args.flash_attn:
        if args.use_grouped_attn:
            # Set group_size_ratio based on segment_size and context_size
            # For 100k context with 1024-token segments: 1024/102400 = 1/100
            group_size_ratio = args.segment_size / args.context_size
            hici_attn.group_size_ratio = group_size_ratio

            # Disable mixed group training for evaluation
            hici_attn.MIXED_GROUP_TRAINING = False

            print(f"Setting group_size_ratio = {args.segment_size}/{args.context_size} = {group_size_ratio}")
            print(f"Number of groups = {int(1/group_size_ratio)}")

            # Use hierarchical forward with grouped attention
            hici_attn.replace_llama_attn(use_flash_attn=True, use_full=False, use_hierarchical_forward=True)
        else:
            # Fallback: use full attention (original LongLoRA behavior)
            hici_attn.replace_llama_attn(use_flash_attn=True, use_full=True)

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        args.base_model,
        cache_dir=args.cache_dir,
    )

    context_size = args.context_size
    orig_ctx_len = getattr(config, "max_position_embeddings", None)

    LLAMA2_ORIG_CTX = 4096

    existing_rope_scaling = getattr(config, "rope_scaling", None)

    if existing_rope_scaling is not None:
        print(f"[RoPE] Using existing rope_scaling from config: {existing_rope_scaling}")
    elif context_size > LLAMA2_ORIG_CTX:
        scaling_factor = float(math.ceil(context_size / LLAMA2_ORIG_CTX))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
        config.max_position_embeddings = context_size
        print(f"[RoPE] Setting rope_scaling: type=linear, factor={scaling_factor}")
        print(f"[RoPE] Setting max_position_embeddings: {context_size}")
    else:
        print(f"[RoPE] No rope_scaling needed (context_size={context_size} <= {LLAMA2_ORIG_CTX})")

    print(f"[RoPE] Final config: max_position_embeddings={config.max_position_embeddings}, rope_scaling={config.rope_scaling}")

    # Load model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.base_model,
        config=config,
        cache_dir=args.cache_dir,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.resize_token_embeddings(32001)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.base_model,
        cache_dir=args.cache_dir,
        model_max_length=context_size,
        padding_side="right",
        use_fast=False,
    )

    # ========================================================================
    # ========================================================================
    if args.use_grouped_attn:
        import os
        import json

        print("\n" + "=" * 60)
        print(" HiCI Module Loading...")
        print("=" * 60)

        hici_params = {
            "num_local_slots": 7,
            "global_slots": 5,
            "num_heads": 8,
            "use_hierarchical": True,
            "use_flash_plus": False,
            "use_flash": False,
            "use_bottleneck": True,
            "bottleneck_dim": 512,
            "compress_dim": 512,
            "shared_compress_dim": 128,
        }

        print(f"  num_heads: {hici_params['num_heads']}")
        print(f"  num_local_slots: {hici_params['num_local_slots']}")
        print(f"  global_slots: {hici_params['global_slots']}")
        print(f"  bottleneck_dim: {hici_params['bottleneck_dim']}")

        hici_attn.register_hici_to_model(model, **hici_params)
        print(" HiCI Module Loaded")

        index_file = os.path.join(args.base_model, "pytorch_model.bin.index.json")
        single_file = os.path.join(args.base_model, "pytorch_model.bin")

        if os.path.exists(index_file):
            with open(index_file, 'r') as f:
                index = json.load(f)

            shard_files = set(index["weight_map"].values())
            print(f" {len(shard_files)} ...")

            hici_loaded = 0
            for shard_file in shard_files:
                shard_path = os.path.join(args.base_model, shard_file)
                shard_weights = torch.load(shard_path, map_location="cpu")

                hici_weights = {k: v for k, v in shard_weights.items()
                               if "local_constructor" in k or "hierarchical" in k}

                if hici_weights:
                    missing, unexpected = model.load_state_dict(hici_weights, strict=False)
                    hici_loaded += len(hici_weights) - len(missing)

                del shard_weights
                torch.cuda.empty_cache()

                print(f" {hici_loaded} HiCI ")

            print(" HiCI ...")
            for layer in model.model.layers:
                if hasattr(layer.self_attn, 'local_constructor'):
                    layer_device = layer.self_attn.q_proj.weight.device
                    layer.self_attn.local_constructor = layer.self_attn.local_constructor.to(layer_device)
                if hasattr(layer.self_attn, 'global_integrator'):
                    layer_device = layer.self_attn.q_proj.weight.device
                    layer.self_attn.global_integrator = layer.self_attn.global_integrator.to(layer_device)
                    print(" HiCI ")

        elif os.path.exists(single_file):
            state_dict = torch.load(single_file, map_location="cpu")
            hici_weights = {k: v for k, v in state_dict.items()
                           if "local_constructor" in k or "hierarchical" in k}

            if hici_weights:
                missing, unexpected = model.load_state_dict(hici_weights, strict=False)
                loaded = len(hici_weights) - len(missing)
                print(f"  Loaded {loaded} HiCI parameters")

                print("  Moving HiCI modules to correct device...")
                for layer in model.model.layers:
                    if hasattr(layer.self_attn, 'local_constructor'):
                        layer_device = layer.self_attn.q_proj.weight.device
                        layer.self_attn.local_constructor = layer.self_attn.local_constructor.to(layer_device)
                    if hasattr(layer.self_attn, 'global_integrator'):
                        layer_device = layer.self_attn.q_proj.weight.device
                        layer.self_attn.global_integrator = layer.self_attn.global_integrator.to(layer_device)
                print("  HiCI modules moved to correct device")
            else:
                print("  WARNING: No HiCI weights found! Model may not have been merged with merge_lora_weights_hici.py")

            del state_dict
            torch.cuda.empty_cache()
        else:
            print(f"  WARNING: Weight file not found: {single_file} or {index_file}")

        print("=" * 60 + "\n")

    total_test_points = args.max_tokens // args.interval
    all_accuries = {}
    for i in range(total_test_points):
        # This is a rough ratio to control the number of texts and tokens
        n_garbage = int(3.75 * (i + 1) * args.interval // 1024 * 1024)
        passed_tests = 0
        total_tokens = 0
        for j in range(args.num_tests):
            is_correct, len_tokens = passkey_retrieval_test(model, tokenizer, device, use_cache=not args.flash_attn, n_garbage=n_garbage, seed=j)
            passed_tests += is_correct
            total_tokens += len_tokens
        avg_tokens = total_tokens // args.num_tests
        accuracy = float(passed_tests) / args.num_tests
        print("accuracy on the token length %d is %f" % (avg_tokens, accuracy))
        all_accuries[str(avg_tokens)] = accuracy
    print("accuries over tokens", all_accuries)

if __name__ == "__main__":
    args = parse_config()
    main(args)
