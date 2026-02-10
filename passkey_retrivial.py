# Core code based on https://github.com/CStanKonrad/long_llama
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
from llama_attn_replace import replace_llama_attn

def update_rope_scaling(model, new_factor, max_position_embeddings):
    """Dynamically update RoPE scaling factor for all layers.
    """

    if hasattr(model, 'model'):
        llama_model = model.model
    else:
        llama_model = model

    llama_model.config.rope_scaling = {"type": "linear", "factor": new_factor}
    llama_model.config.max_position_embeddings = max_position_embeddings

    device = next(model.parameters()).device

    for layer in llama_model.layers:
        attn = layer.self_attn
        old_rotary_emb = attn.rotary_emb

        RotaryEmbClass = type(old_rotary_emb)

        try:
            # transformers >= 4.36 interface
            attn.rotary_emb = RotaryEmbClass(
                dim=attn.head_dim,
                max_position_embeddings=max_position_embeddings,
                scaling_factor=new_factor,
                base=getattr(old_rotary_emb, 'base', 10000.0),
            ).to(device=device)
        except TypeError:
            try:
                # transformers 4.34-4.35 interface
                attn.rotary_emb = RotaryEmbClass(
                    attn.head_dim,
                    max_position_embeddings=max_position_embeddings,
                    scaling_factor=new_factor,
                    base=getattr(old_rotary_emb, 'base', 10000.0),
                ).to(device=device)
            except TypeError:
                # Older versions: manually update inv_freq
                base = getattr(old_rotary_emb, 'base', 10000.0)
                inv_freq = 1.0 / (base ** (torch.arange(0, attn.head_dim, 2, dtype=torch.float32) / attn.head_dim))
                inv_freq = inv_freq / new_factor
                attn.rotary_emb.inv_freq = inv_freq.to(device=device)
                # Clear cache
                for attr in ['_cos_cached', '_sin_cached', 'cos_cached', 'sin_cached']:
                    if hasattr(attn.rotary_emb, attr):
                        setattr(attn.rotary_emb, attr, None)

    print(f"[RoPE] Updated scaling factor to {new_factor}, max_position_embeddings to {max_position_embeddings}")

def parse_config():
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument(
        "--base_model", type=str, default="/path/to/llama-7b-hf"
    )
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument(
        "--context_size", type=int, default=-1, help="context size during fine-tuning"
    )
    parser.add_argument(
        "--flash_attn", type=bool, default=True, help="whether to use flash attention 2"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=32000,
        help="maximum token length for evaluation",
    )
    parser.add_argument(
        "--interval", type=int, default=1000, help="interval for evaluation"
    )
    parser.add_argument(
        "--num_tests",
        type=int,
        default=10,
        help="number of repeat testing for each length",
    )

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
    assert len(garbage_inf) >= n_garbage, (
        f"garbage_inf length {len(garbage_inf)} < n_garbage {n_garbage}"
    )
    garbage_prefix = garbage_inf[:n_garbage_prefix]
    garbage_suffix = garbage_inf[:n_garbage_suffix]
    pass_key = random.randint(1, 50000)
    information_line = (
        f"The pass key is {pass_key}. Remember it. {pass_key} is the pass key."
    )
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

def passkey_retrieval_test(
    model, tokenizer, device, use_cache=False, n_garbage=60000, seed=666
):
    prompt, answer = generate_prompt_landmark(n_garbage, seed)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    len_token = input_ids.shape[-1]

    answer_ids = tokenizer(answer, return_tensors="pt").input_ids[:, 1:]  # drop BOS
    generation_output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=answer_ids.shape[-1],
        num_beams=1,
        use_cache=use_cache,
        pad_token_id=tokenizer.pad_token_id,
    )

    model_answer = generation_output[0, -answer_ids.shape[-1] :].cpu()

    is_correct = (model_answer == answer_ids[0]).all().item()
    # print(f"The correct answer is {tokenizer.decode(answer_ids[0].cpu())}")
    # print(f"The model answer is {tokenizer.decode(model_answer.cpu())}, is_correct : {is_correct}")
    return is_correct, len_token

def main(args):
    device = "cuda:0"
    torch.cuda.set_device(device)

    print("base model", args.base_model)

    if args.flash_attn:
        replace_llama_attn(use_full=True, inference=True)

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        args.base_model,
        cache_dir=args.cache_dir,
    )

    context_size = args.context_size
    orig_ctx_len = getattr(config, "max_position_embeddings", None)

    # LLaMA-2 original context length is 4096
    LLAMA2_ORIG_CTX = 4096

    # Case 1: config already has rope_scaling
    # Case 2: no rope_scaling but context_size > 4096
    existing_rope_scaling = getattr(config, "rope_scaling", None)

    if existing_rope_scaling is not None:
        print(
            f"[RoPE] Using existing rope_scaling from config: {existing_rope_scaling}"
        )

        config.max_position_embeddings = args.max_tokens
        print(
            f"[RoPE] Setting max_position_embeddings: {args.max_tokens} (for extrapolation)"
        )
    elif context_size > LLAMA2_ORIG_CTX:
        scaling_factor = float(math.ceil(context_size / LLAMA2_ORIG_CTX))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}

        config.max_position_embeddings = args.max_tokens
        print(
            f"[RoPE] Setting rope_scaling: type=linear, factor={scaling_factor} (based on context_size={context_size})"
        )
        print(
            f"[RoPE] Setting max_position_embeddings: {args.max_tokens} (for extrapolation)"
        )
    else:
        print(
            f"[RoPE] No rope_scaling needed (context_size={context_size} <= {LLAMA2_ORIG_CTX})"
        )
        config.max_position_embeddings = args.max_tokens

    print(
        f"[RoPE] Final config: max_position_embeddings={config.max_position_embeddings}, rope_scaling={config.rope_scaling}"
    )

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
        model_max_length=args.max_tokens,
        padding_side="right",
        use_fast=False,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    factor_within = float(math.ceil(context_size / LLAMA2_ORIG_CTX))
    factor_beyond = float(math.ceil(args.max_tokens / LLAMA2_ORIG_CTX))

    print(f"\n[Two-stage PI strategy]")
    print(f"  seq_len <= {context_size}: factor = {factor_within}")
    print(f"  seq_len >  {context_size}: factor = {factor_beyond}")

    current_factor = None

    total_test_points = args.max_tokens // args.interval
    all_accuries = {}
    for i in range(total_test_points):
        # This is a rough ratio to control the number of texts and tokens
        target_tokens = (i + 1) * args.interval
        n_garbage = int(3.75 * target_tokens // 1024 * 1024)

        if target_tokens <= context_size:
            needed_factor = factor_within
            max_pos = context_size
        else:
            needed_factor = factor_beyond
            max_pos = args.max_tokens

        # Update RoPE if factor changed
        if current_factor != needed_factor:
            update_rope_scaling(model, needed_factor, max_pos)
            current_factor = needed_factor

        passed_tests = 0
        total_tokens = 0
        for j in range(args.num_tests):
            is_correct, len_tokens = passkey_retrieval_test(
                model,
                tokenizer,
                device,
                use_cache=not args.flash_attn,
                n_garbage=n_garbage,
                seed=j,
            )
            passed_tests += is_correct
            total_tokens += len_tokens
        avg_tokens = total_tokens // args.num_tests
        accuracy = float(passed_tests) / args.num_tests

        marker = " [EXTRAPOLATION]" if avg_tokens > context_size else ""
        print("accuracy on the token length %d is %f%s (factor=%.1f)" % (avg_tokens, accuracy, marker, current_factor))
        all_accuries[str(avg_tokens)] = accuracy
    print("accuries over tokens", all_accuries)

if __name__ == "__main__":
    args = parse_config()
    main(args)
