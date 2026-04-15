import os
from datasets import load_dataset
import torch
import json
from transformers import (
    AutoTokenizer,
    LlamaTokenizer,
    LlamaForCausalLM,
    AutoModelForCausalLM,
)
from tqdm import tqdm
import numpy as np
import random
import argparse
from llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from llama_attn_hici_sft import (
    replace_llama_attn_hici_inference,
    register_hici_to_model,
)
import torch.distributed as dist
import torch.multiprocessing as mp

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=[
            "llama2-7b-chat-4k",
            "longchat-v1.5-7b-32k",
            "xgen-7b-8k",
            "internlm-7b-8k",
            "chatglm2-6b",
            "chatglm2-6b-32k",
            "chatglm3-6b-32k",
            "vicuna-v1.5-7b-16k",
            "longalpaca-7b-16k",
            "my-sft-16k",
            "hici-13b-16k",
            "hici-7b-sft-16k",
            "hici-7b-sft-re-16k",
            "hici-7b-sft-16k-yes",
            "hici-7b-chat-sft-16k-yes",
            "hici-7b-chat-sft-16k-no-5epoch",
        ],
    )
    parser.add_argument("--e", action="store_true", help="Evaluate on LongBench-E")
    parser.add_argument(
        "--use_hici_attn",
        action="store_true",
        default=False,
        help="Use HiCI hierarchical memory attention for hici models (default: use standard flash attention)",
    )
    parser.add_argument(
        "--output_suffix",
        type=str,
        default="",
        help="Suffix to append to output directory name (e.g., '-ori' -> pred/model-ori/)",
    )
    return parser.parse_args(args)

# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    if "chatglm3" in model_name:
        prompt = tokenizer.build_chat_input(prompt)
    elif "chatglm" in model_name:
        prompt = tokenizer.build_prompt(prompt)
    elif "longchat" in model_name or "vicuna" in model_name:
        from fastchat.model import get_conversation_template

        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    elif "llama2" in model_name:
        prompt = f"[INST]{prompt}[/INST]"
    elif "longalpaca" in model_name:
        prompt = f"[INST]{prompt}[/INST]"
    elif "my-sft" in model_name:
        prompt = f"[INST]{prompt}[/INST]"
    elif "hici" in model_name:
        prompt = f"[INST]{prompt}[/INST]"
    elif "xgen" in model_name:
        header = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
        )
        prompt = header + f" ### Human: {prompt}\n###"
    elif "internlm" in model_name:
        prompt = f"<|User|>:{prompt}<eoh>\n<|Bot|>:"
    return prompt

def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response

def get_pred(
    rank,
    world_size,
    data,
    max_length,
    max_gen,
    prompt_format,
    dataset,
    device,
    model_name,
    model2path,
    out_path,
    use_hici_attn=False,
):
    device = torch.device(f"cuda:{rank}")
    model, tokenizer = load_model_and_tokenizer(
        model2path[model_name],
        model_name,
        device,
        use_hici_attn=use_hici_attn,
    )
    for json_obj in tqdm(data):
        prompt = prompt_format.format(**json_obj)
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(
            prompt, truncation=False, return_tensors="pt"
        ).input_ids[0]
        if "chatglm3" in model_name:
            tokenized_prompt = tokenizer(
                prompt, truncation=False, return_tensors="pt", add_special_tokens=False
            ).input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length / 2)
            prompt = tokenizer.decode(
                tokenized_prompt[:half], skip_special_tokens=True
            ) + tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        if dataset not in [
            "trec",
            "triviaqa",
            "samsum",
            "lsht",
            "lcc",
            "repobench-p",
        ]:  # chat models are better off without build prompts on these tasks
            prompt = build_chat(tokenizer, prompt, model_name)
        if "chatglm3" in model_name:
            if dataset in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
                input = tokenizer(prompt, truncation=False, return_tensors="pt").to(
                    device
                )
            else:
                input = prompt.to(device)
        else:
            input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        context_length = input.input_ids.shape[-1]

        enable_kv_cache = True

        if (
            dataset == "samsum"
        ):  # prevent illegal output on samsum (model endlessly repeat "\nDialogue"), might be a prompting issue
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=context_length + 1,
                eos_token_id=[
                    tokenizer.eos_token_id,
                    tokenizer.encode("\n", add_special_tokens=False)[-1],
                ],
                use_cache=enable_kv_cache,
            )[0]
        else:
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                use_cache=enable_kv_cache,
            )[0]
        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        pred = post_process(pred, model_name)
        with open(out_path, "a", encoding="utf-8") as f:
            json.dump(
                {
                    "pred": pred,
                    "answers": json_obj["answers"],
                    "all_classes": json_obj["all_classes"],
                    "length": json_obj["length"],
                },
                f,
                ensure_ascii=False,
            )
            f.write("\n")
    if dist.is_initialized():
        dist.destroy_process_group()

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def load_model_and_tokenizer(
    path,
    model_name,
    device,
    use_hici_attn=False,
):
    if "chatglm" in model_name or "internlm" in model_name or "xgen" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            path, trust_remote_code=True, torch_dtype=torch.bfloat16
        ).to(device)
    elif "llama2" in model_name:
        replace_llama_attn_with_flash_attn()
        tokenizer = LlamaTokenizer.from_pretrained(path)
        model = LlamaForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).to(
            device
        )
    elif "longalpaca" in model_name:
        replace_llama_attn_with_flash_attn()
        tokenizer = LlamaTokenizer.from_pretrained(path)
        model = LlamaForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).to(
            device
        )
    elif "my-sft" in model_name:
        replace_llama_attn_with_flash_attn()
        tokenizer = LlamaTokenizer.from_pretrained(path)
        model = LlamaForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).to(
            device
        )
    elif "hici" in model_name:
        tokenizer = LlamaTokenizer.from_pretrained(path)

        if use_hici_attn:

            print("Using HiCI hierarchical attention")
            replace_llama_attn_hici_inference()
            model = LlamaForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16)

            register_hici_to_model(
                model,
                num_local_slots=7,
                global_slots=5,
                num_heads=8,
                use_bottleneck=True,
                bottleneck_dim=512,
                use_local_constructor_flash=False,
                use_global_integrator=True,
                use_shared_compressor=True,
                shared_compress_dim=128,
            )

            print("Loading trained HiCI weights from checkpoint...")
            import glob as glob_module

            checkpoint_path = os.path.join(path, "pytorch_model.bin")
            checkpoint_files = []

            if os.path.isfile(checkpoint_path):
                checkpoint_files = [checkpoint_path]
            else:

                index_path = os.path.join(path, "pytorch_model.bin.index.json")
                if os.path.isfile(index_path):
                    with open(index_path, "r") as f:
                        index = json.load(f)
                    shard_files = set(index.get("weight_map", {}).values())
                    checkpoint_files = [os.path.join(path, f) for f in shard_files]
                    checkpoint_files.sort()
                else:

                    shard_pattern = os.path.join(path, "pytorch_model-*.bin")
                    checkpoint_files = sorted(glob_module.glob(shard_pattern))

                    if not checkpoint_files:
                        safetensor_pattern = os.path.join(path, "model-*.safetensors")
                        checkpoint_files = sorted(glob_module.glob(safetensor_pattern))

            if checkpoint_files:
                total_hici_keys = 0

                model_hici_keys = set(
                    k
                    for k in model.state_dict().keys()
                    if "local_constructor" in k or "global_integrator" in k
                )
                print(
                    f"   Model has {len(model_hici_keys)} registered HiCI keys"
                )

                all_loaded_keys = set()
                for ckpt_file in checkpoint_files:
                    if os.path.isfile(ckpt_file):
                        if ckpt_file.endswith(".safetensors"):
                            from safetensors.torch import load_file

                            state_dict = load_file(ckpt_file)
                        else:
                            state_dict = torch.load(ckpt_file, map_location="cpu")

                        hici_state_dict = {
                            k: v
                            for k, v in state_dict.items()
                            if "local_constructor" in k or "global_integrator" in k
                        }

                        if hici_state_dict:
                            checkpoint_keys = set(hici_state_dict.keys())
                            matched_keys = checkpoint_keys & model_hici_keys
                            unmatched_keys = checkpoint_keys - model_hici_keys

                            if unmatched_keys and len(all_loaded_keys) == 0:
                                print(
                                    f"   {len(unmatched_keys)} keys in checkpoint but NOT in model (sample):"
                                )
                                for k in list(unmatched_keys)[:3]:
                                    print(f"       - {k}")

                            matched_state_dict = {
                                k: v
                                for k, v in hici_state_dict.items()
                                if k in model_hici_keys
                            }
                            if matched_state_dict:
                                model.load_state_dict(matched_state_dict, strict=False)
                                all_loaded_keys.update(matched_state_dict.keys())
                                print(
                                    f"   Loaded {len(matched_state_dict)} matched HiCI keys from {os.path.basename(ckpt_file)}"
                                )

                        del state_dict

                missing_keys = model_hici_keys - all_loaded_keys
                if missing_keys:
                    print(f"   {len(missing_keys)} model keys NOT loaded (sample):")
                    for k in list(missing_keys)[:3]:
                        print(f"       - {k}")

                print(
                    f"Total loaded: {len(all_loaded_keys)}/{len(model_hici_keys)} HiCI parameters"
                )

                if len(all_loaded_keys) == 0:
                    print(
                        "ERROR: No HiCI weights were loaded! Check if module structure matches."
                    )
            else:
                print(f"Warning: Could not find checkpoint files at {path}")
        else:

            print("Using standard Flash Attention")
            replace_llama_attn_with_flash_attn()
            model = LlamaForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16)

        model = model.to(device)
    elif "longchat" in model_name or "vicuna" in model_name:
        from fastchat.model import load_model

        replace_llama_attn_with_flash_attn()
        model, _ = load_model(
            path,
            device="cpu",
            num_gpus=0,
            load_8bit=False,
            cpu_offloading=False,
            debug=False,
        )
        model = model.to(device)
        model = model.bfloat16()
        tokenizer = AutoTokenizer.from_pretrained(
            path, trust_remote_code=True, use_fast=False
        )
    model = model.eval()
    return model, tokenizer

if __name__ == "__main__":
    seed_everything(42)
    args = parse_args()
    world_size = torch.cuda.device_count()
    mp.set_start_method("spawn", force=True)

    model2path = json.load(open("config/model2path.json", "r"))
    model2maxlen = json.load(open("config/model2maxlen.json", "r"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = args.model
    # define your model
    max_length = model2maxlen[model_name]
    if args.e:
        datasets = [
            "qasper",
            "multifieldqa_en",
            "hotpotqa",
            "2wikimqa",
            "gov_report",
            "multi_news",
            "trec",
            "triviaqa",
            "samsum",
            "passage_count",
            "passage_retrieval_en",
            "lcc",
            "repobench-p",
        ]
    else:
        datasets = [
            "narrativeqa",
            "qasper",
            "multifieldqa_en",
            "multifieldqa_zh",
            "hotpotqa",
            "2wikimqa",
            "musique",
            "dureader",
            "gov_report",
            "qmsum",
            "multi_news",
            "vcsum",
            "trec",
            "triviaqa",
            "samsum",
            "lsht",
            "passage_count",
            "passage_retrieval_en",
            "passage_retrieval_zh",
            "lcc",
            "repobench-p",
        ]
    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))

    dataset2expected_count = {
        "narrativeqa": 200,
        "qasper": 200,
        "multifieldqa_en": 150,
        "multifieldqa_zh": 200,
        "hotpotqa": 200,
        "2wikimqa": 200,
        "musique": 200,
        "dureader": 200,
        "gov_report": 200,
        "qmsum": 200,
        "multi_news": 200,
        "vcsum": 200,
        "trec": 200,
        "triviaqa": 200,
        "samsum": 200,
        "lsht": 200,
        "passage_count": 200,
        "passage_retrieval_en": 200,
        "passage_retrieval_zh": 200,
        "lcc": 500,
        "repobench-p": 500,
    }

    # predict on each dataset
    if not os.path.exists("pred"):
        os.makedirs("pred")
    if not os.path.exists("pred_e"):
        os.makedirs("pred_e")

    output_dir_name = f"{model_name}{args.output_suffix}"

    for dataset in datasets:
        if args.e:
            if not os.path.exists(f"pred_e/{output_dir_name}"):
                os.makedirs(f"pred_e/{output_dir_name}")
            out_path = f"pred_e/{output_dir_name}/{dataset}.jsonl"
        else:
            if not os.path.exists(f"pred/{output_dir_name}"):
                os.makedirs(f"pred/{output_dir_name}")
            out_path = f"pred/{output_dir_name}/{dataset}.jsonl"

        expected_count = dataset2expected_count.get(dataset, 0)
        if os.path.exists(out_path):
            with open(out_path, "r", encoding="utf-8") as f:
                current_count = sum(1 for _ in f)
            if current_count == expected_count:
                print(f"Skipping completed: {dataset} ({current_count}/{expected_count})")
                continue
            else:
                print(
                    f"Regenerating: {dataset} ({current_count}/{expected_count} incomplete)"
                )
                os.remove(out_path)
        else:
            print(f"Generating: {dataset}")

        if args.e:
            data = load_dataset("THUDM/LongBench", f"{dataset}_e", split="test")
        else:
            data = load_dataset("THUDM/LongBench", dataset, split="test")
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        data_all = [data_sample for data_sample in data]
        data_subsets = [data_all[i::world_size] for i in range(world_size)]
        processes = []
        for rank in range(world_size):
            p = mp.Process(
                target=get_pred,
                args=(
                    rank,
                    world_size,
                    data_subsets[rank],
                    max_length,
                    max_gen,
                    prompt_format,
                    dataset,
                    device,
                    model_name,
                    model2path,
                    out_path,
                    args.use_hici_attn,
                ),
            )
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
