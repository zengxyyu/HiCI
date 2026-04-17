# -*- coding:utf-8 -*-
"""
Multi-GPU distributed PPL evaluation for ChunkQwen3 (DCA on Qwen3).
Uses torchmetrics.Perplexity for consistent comparison with LLaMA3 baselines.

Usage:
    torchrun --nproc_per_node=4 ChunkLlama/ppl/test_ppl_distributed_qwen3.py \
        --model_path ./models/Qwen3-8B \
        --data_path ./data/pg19_qwen3/test.bin \
        --seq_len 65536 \
        --pretraining_length 40960
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.distributed as dist
from torch.nn import CrossEntropyLoss
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchmetrics import Accuracy
from torchmetrics.text import Perplexity
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoConfig

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from chunkqwen3_attn_replace import replace_with_chunkqwen3


class Pg19Dataset(Dataset):
    def __init__(self, data_path, seq_length, sliding_window=256, data_dtype="uint32"):
        self.seq_length = seq_length
        self.data = np.memmap(data_path, dtype=np.dtype(data_dtype), mode='r')
        self.start_indices = list(range(0, len(self.data) - seq_length, sliding_window))
        if self.start_indices:
            self.start_indices.pop()
        assert len(self.start_indices) > 0, "Dataset is empty"

    def __len__(self):
        return len(self.start_indices)

    def __getitem__(self, index):
        start = self.start_indices[index]
        input_id = torch.from_numpy(self.data[start:start + self.seq_length].astype(np.int64))
        return {"input_ids": input_id, "labels": input_id}

    def num_tokens(self):
        return len(self.data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_len', default=32768, type=int)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--pretraining_length', type=int, default=40960)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--no_chunk', action='store_true', help='Skip DCA, run original model baseline')
    args = parser.parse_args()

    # DDP setup
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    # Monkey-patch attention (skip for baseline)
    if not args.no_chunk:
        replace_with_chunkqwen3(args.pretraining_length)

    # Auto-detect data dtype from model vocab size
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    data_dtype = "uint32" if config.vocab_size > 65535 else "uint16"

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(local_rank)

    # Enable distributed evaluation (DDP requires at least one param with grad)
    [p.requires_grad_() for n, p in model.named_parameters() if "lm_head" in n]

    model = DDP(model, device_ids=[local_rank])
    model.eval()

    # Dataset + distributed sampler
    dataset = Pg19Dataset(args.data_path, args.seq_len, sliding_window=256, data_dtype=data_dtype)
    sampler = DistributedSampler(dataset, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, pin_memory=True)

    if local_rank == 0:
        print(f"Model: {args.model_path}")
        print(f"DCA: {'OFF (baseline)' if args.no_chunk else f'ON (pretraining_length={args.pretraining_length})'}")
        print(f"Test PPL on seq length {args.seq_len}")
        print(f"Num tokens: {dataset.num_tokens()}, Num examples: {len(dataset)}, GPUs: {dist.get_world_size()}")

    # Same metrics as LLaMA3 eval (test_ppl_distributed.py)
    accuracy = Accuracy(task="multiclass", num_classes=config.vocab_size).to(local_rank)
    perplexity = Perplexity(ignore_index=CrossEntropyLoss().ignore_index).to(local_rank)
    last_loss = 0.0

    iterator = tqdm(dataloader, desc="Eval") if local_rank == 0 else dataloader

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            input_ids = batch["input_ids"].to(local_rank)
            labels = batch["labels"].to(local_rank)

            outputs = model(input_ids=input_ids, labels=labels, use_cache=False)
            logits = outputs["logits"]

            shift_logits = logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            shift_preds = shift_logits.argmax(dim=-1)

            current_accuracy = accuracy.forward(preds=shift_preds, target=shift_labels)
            current_perplexity = perplexity.forward(preds=shift_logits, target=shift_labels)
            last_loss = outputs["loss"].item()

            if local_rank == 0 and i % 10 == 0:
                iterator.set_postfix(
                    accuracy=current_accuracy.item(),
                    perplexity=current_perplexity.item(),
                    loss=last_loss,
                )

    # Free model memory before cross-device compute()
    del model
    torch.cuda.empty_cache()

    result = {
        "accuracy": accuracy.compute().item(),
        "perplexity": perplexity.compute().item(),
        "loss": last_loss,
    }

    if local_rank == 0:
        print(result)
        args.ppl = result["perplexity"]
        with open("ppl.output.json", "a") as f:
            f.write(json.dumps(vars(args)) + "\n")

    dist.destroy_process_group()


if __name__ == '__main__':
    main()
