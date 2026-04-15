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
import transformers
from peft import PeftModel


def parse_config():
    parser = argparse.ArgumentParser(
        description="Merge LoRA weights into the base model (LLaMA / Qwen3 / generic)"
    )
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--peft_model", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument(
        "--context_size", type=int, default=-1,
        help="Training context size. Used to set RoPE scaling when > max_position_embeddings."
    )
    parser.add_argument(
        "--torch_dtype", type=str, default="auto",
        choices=["auto", "float16", "bfloat16"],
        help="Model dtype. 'auto' reads from config (recommended). "
             "Use 'float16' for LLaMA-2, 'bfloat16' for Qwen3/LLaMA-3.",
    )
    parser.add_argument("--cache_dir", type=str, default=None)
    args = parser.parse_args()
    return args


def resolve_dtype(args, config):
    if args.torch_dtype == "auto":
        # Read from model config; fall back to bfloat16
        cfg_dtype = getattr(config, "torch_dtype", "bfloat16")
        dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}
        return dtype_map.get(str(cfg_dtype), torch.bfloat16)
    return torch.float16 if args.torch_dtype == "float16" else torch.bfloat16


def main(args):
    device = "cuda:0"
    torch.cuda.set_device(device)

    print("base model :", args.base_model)
    print("peft model :", args.peft_model)
    print("save path  :", args.save_path)

    # Load config (apply RoPE scaling before loading weights)
    config = transformers.AutoConfig.from_pretrained(
        args.base_model, cache_dir=args.cache_dir
    )
    if args.context_size > 0:
        orig_ctx_len = getattr(config, "max_position_embeddings", None)
        if orig_ctx_len and args.context_size > orig_ctx_len:
            scaling_factor = float(math.ceil(args.context_size / orig_ctx_len))
            config.rope_scaling = {"type": "linear", "factor": scaling_factor}
            print(f"RoPE scaling: factor={scaling_factor} ({orig_ctx_len} → {args.context_size})")

    torch_dtype = resolve_dtype(args, config)
    print(f"dtype: {torch_dtype}")

    # Load base model
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.base_model,
        config=config,
        cache_dir=args.cache_dir,
        torch_dtype=torch_dtype,
        device_map="auto",
    )

    # Load tokenizer: prefer the checkpoint's tokenizer (may have extra special tokens
    # added during training, e.g. [PAD] for LLaMA-2), fall back to base model.
    tokenizer_path = args.peft_model if os.path.isfile(
        os.path.join(args.peft_model, "tokenizer_config.json")
    ) else args.base_model
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_path,
        cache_dir=args.cache_dir,
        padding_side="right",
        use_fast=True,
    )

    # Load embed/norm weights from trainable_params.bin (strict=False so HiCI keys are ignored)
    trainable_params_path = os.path.join(args.peft_model, "trainable_params.bin")
    if os.path.isfile(trainable_params_path):
        tp = torch.load(trainable_params_path, map_location="cpu", weights_only=False)
        # Resize embeddings if the checkpoint extended the vocabulary
        for k, v in tp.items():
            if "embed_tokens" in k and v.shape[0] != config.vocab_size:
                model.resize_token_embeddings(v.shape[0])
                print(f"Resized token embeddings: {config.vocab_size} → {v.shape[0]}")
                break
        model.load_state_dict(tp, strict=False)
        print(f"Loaded trainable_params.bin ({len(tp)} keys)")
        del tp
    else:
        print("trainable_params.bin not found — skipping embed/norm weight loading")

    # Load LoRA and merge
    model = PeftModel.from_pretrained(
        model,
        args.peft_model,
        device_map="auto",
        torch_dtype=torch_dtype,
    )
    model = model.merge_and_unload()
    print("LoRA merged successfully")

    model.save_pretrained(args.save_path)
    tokenizer.save_pretrained(args.save_path)
    print(f"Saved to {args.save_path}")


if __name__ == "__main__":
    args = parse_config()
    main(args)
