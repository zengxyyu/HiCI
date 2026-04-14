# Qwen3 HiCI Distributed Evaluation Script
# Adapted from eval_distributed_hici_qwen.py for Qwen3 (requires transformers >= 4.51)
#
# Usage:
#   torchrun --nproc_per_node=4 eval_distributed_hici_qwen3.py \
#       --base_model ./models/Qwen3-8B \
#       --peft_model ./checkpoints/Qwen3-8b-hici-48k/checkpoint-750 \
#       --seq_len 49152 \
#       --data_path pg19/test.bin \
#       --num_local_slots 8 \
#       --global_slots 4 \
#       --num_heads 8 \
#       --bottleneck_dim 512

import os
from dataclasses import dataclass, field
from typing import Optional

import math
import random
import transformers
from peft import PeftModel

from qwen3_attn_hici import (
    replace_qwen3_attn,
    register_hici_to_qwen3_model,
)
from torch.distributed import init_process_group, destroy_process_group
from torchmetrics import Accuracy
from torchmetrics.text import Perplexity
from torch.nn import CrossEntropyLoss

import inspect
from abc import ABC, abstractmethod
from typing import Union

from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers.modeling_utils import PreTrainedModel
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

import numpy as np
import torch


class Pg19Dataset(Dataset):
    def __init__(self, data_path: str, seq_length: int, sliding_window: int = 256):
        assert seq_length >= sliding_window, (
            f"Sliding window '{sliding_window}' must be smaller than sequence length '{seq_length}'"
        )
        self.seq_length = seq_length
        # Qwen3 vocab_size=151936 > uint16 max (65535), so use uint32
        self.data = np.memmap(data_path, dtype=np.uint32, mode="r")
        self.start_indices = list(range(0, len(self.data) - seq_length, sliding_window))
        assert len(self) > 0, "Dataset is empty"

    def __len__(self):
        return len(self.start_indices)

    def __getitem__(self, index) -> dict[str, torch.Tensor]:
        start = self.start_indices[index]
        end = start + self.seq_length
        input_id = torch.from_numpy(self.data[start:end].astype(np.int64))
        return {"input_ids": input_id, "labels": input_id}

    def num_tokens(self):
        return len(self.data)


class EvalMetric(ABC):
    @abstractmethod
    def add(self, logits, labels, model_output) -> dict:
        pass

    @abstractmethod
    def compute(self) -> dict:
        pass


class DistributedEvaluator:
    def __init__(self, model, batch_size, refresh_rate, gpu_id):
        self.gpu_id = gpu_id
        self.batch_size = batch_size
        self.refresh_rate = refresh_rate
        self.model = DDP(model, device_ids=[self.gpu_id])

    def evaluate(self, dataset, metric):
        data_loader = self._prepare_dataloader(dataset)
        self.model.eval()
        with torch.no_grad():
            if self.is_first_device():
                data_loader = tqdm(data_loader)
            for i, example_dict in enumerate(data_loader):
                sig = inspect.signature(self.model.forward)
                used = set(list(sig.parameters.keys()) + ["input_ids", "labels"])
                inputs = {
                    key: example_dict[key].to(self.gpu_id)
                    for key in used if key in example_dict
                }
                outputs = self.model(**inputs)
                metric_result = metric.add(
                    logits=outputs["logits"],
                    labels=inputs["labels"],
                    model_output=outputs,
                )
                if self.is_first_device() and (i % self.refresh_rate == 0):
                    data_loader.set_postfix(metric_result)
            # Free GPU memory before cross-device compute() to avoid OOM/SIGABRT
            del self.model
            torch.cuda.empty_cache()
            return metric.compute()

    def is_first_device(self):
        return self.gpu_id == 0

    def _prepare_dataloader(self, dataset):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            shuffle=False,
            sampler=DistributedSampler(dataset),
        )


class EvalMetricImpl(EvalMetric):
    def __init__(self, vocab_size, gpu_id):
        self.accuracy = Accuracy(task="multiclass", num_classes=vocab_size).to(gpu_id)
        self.perplexity = Perplexity(ignore_index=CrossEntropyLoss().ignore_index).to(gpu_id)
        self.last_loss = 0.0

    def add(self, logits, labels, model_output):
        shift_predictions = logits.argmax(dim=-1)[..., :-1]
        shift_labels = labels[..., 1:]

        current_accuracy = self.accuracy.forward(preds=shift_predictions, target=shift_labels)

        shift_logits = logits[..., :-1, :]
        current_perplexity = self.perplexity.forward(preds=shift_logits, target=shift_labels)

        self.last_loss = model_output["loss"].item()
        return {
            "accuracy": current_accuracy.item(),
            "perplexity": current_perplexity.item(),
            "loss": self.last_loss,
        }

    def compute(self):
        return {
            "accuracy": self.accuracy.compute().item(),
            "perplexity": self.perplexity.compute().item(),
            "loss": self.last_loss,
        }


@dataclass
class EvalArguments:
    batch_size: int = field(default=1)
    base_model: Optional[str] = field(default="./models/Qwen3-8B")
    seq_len: int = field(default=2048, metadata={"help": "context length during evaluation."})
    context_size: int = field(default=-1, metadata={"help": "context size during fine-tuning."})
    peft_model: Optional[str] = field(default=None)
    flash_attn: bool = field(default=True)
    data_path: str = field(default="./test.bin", metadata={"help": "test data path"})
    cache_dir: Optional[str] = field(default="./.cache")
    progress_bar_fresh_rate: int = field(default=10)
    num_local_slots: int = field(default=8)
    global_slots: int = field(default=4)
    use_local_constructor: bool = field(default=True)
    use_global_integrator: bool = field(default=True)
    use_flash_plus: Optional[bool] = field(default=False)
    use_local_constructor_flash: bool = field(default=False)
    use_hierarchical_forward: Optional[bool] = field(default=True)
    num_heads: int = field(default=8)
    use_bottleneck: bool = field(default=True)
    bottleneck_dim: int = field(default=512)
    eval_mode: Optional[str] = field(
        default=None,
        metadata={"help": "Evaluation mode: None (chunked), 'full', 'full_hierarchical'."},
    )


def run_eval(args: EvalArguments):
    # Shell passes "None" as string, convert to Python None
    if args.eval_mode in ("None", "none", ""):
        args.eval_mode = None

    torch_dtype = torch.bfloat16

    seed = 2
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    dataset = Pg19Dataset(args.data_path, seq_length=args.seq_len, sliding_window=256)

    # Determine if this is a pure baseline run (no HiCI, no peft)
    is_baseline = (args.eval_mode == "full" and not args.peft_model)

    if args.flash_attn:
        # Evaluation modes:
        # - None or "chunked": Chunked attention with HiCI memory (same as training) -- DEFAULT
        # - "full": Full attention without memory (standard transformer, baseline)
        # - "full_hierarchical": Full attention + hierarchical memory
        if is_baseline:
            # Pure Qwen3 baseline: do NOT replace attention.
            # Use official Qwen3 implementation with flash_attention_2 backend.
            pass
        elif args.eval_mode == "full":
            # Full attention with HiCI (has peft_model)
            replace_qwen3_attn(
                use_flash_attn=True,
                use_full=True,
                use_hierarchical_forward=False,
            )
        elif args.eval_mode == "full_hierarchical":
            # Full attention + hierarchical memory
            replace_qwen3_attn(
                use_flash_attn=True,
                use_full=False,
                use_hierarchical_forward=True,
            )
        else:
            # Default: chunked attention with HiCI memory (same as training)
            replace_qwen3_attn(
                use_flash_attn=True,
                use_full=False,
                use_hierarchical_forward=True,
            )

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        args.base_model, cache_dir=args.cache_dir, use_cache=False
    )

    context_size = args.context_size if args.context_size > 0 else args.seq_len
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and context_size > orig_ctx_len:
        scaling_factor = float(math.ceil(context_size / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}

    # Load model
    # For baseline: use official flash_attention_2 backend (no monkey-patching).
    # For HiCI: use eager (attention forward is replaced by replace_qwen3_attn).
    load_kwargs = dict(
        config=config,
        cache_dir=args.cache_dir,
        torch_dtype=torch_dtype,
    )
    if is_baseline and args.flash_attn:
        load_kwargs["attn_implementation"] = "flash_attention_2"
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.base_model, **load_kwargs
    )

    vocab_size = config.vocab_size

    # Check if we need to resize embeddings (if training added special tokens)
    if args.peft_model:
        trainable_params_path = os.path.join(args.peft_model, "trainable_params.bin")
        if os.path.isfile(trainable_params_path):
            tp = torch.load(trainable_params_path, map_location='cpu', weights_only=False)
            for k, v in tp.items():
                if 'embed_tokens' in k:
                    if v.shape[0] != vocab_size:
                        vocab_size = v.shape[0]
                        model.resize_token_embeddings(vocab_size)
                        print(f"Resized token embeddings to {vocab_size}")
                    break
            del tp

    # Determine whether HiCI memory modules are needed.
    # eval_mode="full" without peft_model = pure Qwen3 baseline, no memory needed.
    # All other cases (chunked, full_hierarchical, or any mode with peft_model) need memory.
    need_memory = not (args.eval_mode == "full" and not args.peft_model)

    if need_memory:
        # Register global memory modules (Qwen3)
        print(f"\n{'=' * 70}")
        print(f"Registering Global Memory for Qwen3 Evaluation")
        print(f"{'=' * 70}")
        register_hici_to_qwen3_model(
            model,
            num_local_slots=args.num_local_slots,
            global_slots=args.global_slots,
            num_heads=args.num_heads,
            use_bottleneck=args.use_bottleneck,
            bottleneck_dim=args.bottleneck_dim,
            use_local_constructor=args.use_local_constructor,
            use_global_integrator=args.use_global_integrator,
            use_flash_plus=args.use_flash_plus,
        )

        # Convert memory modules to eval dtype
        print(f"Converting memory modules to {torch_dtype}...")
        for layer in model.model.layers:
            if hasattr(layer.self_attn, "local_constructor"):
                layer.self_attn.local_constructor = layer.self_attn.local_constructor.to(torch_dtype)
            if hasattr(layer.self_attn, "global_integrator"):
                layer.self_attn.global_integrator = layer.self_attn.global_integrator.to(torch_dtype)

        print(f"Memory modules registration complete!")
        print(f"   Number of memory slots: {args.num_local_slots}")
        print(f"   Global slots: {args.global_slots}")
        print(f"   Num heads: {args.num_heads}")
        print(f"   Bottleneck dim: {args.bottleneck_dim}")
        print(f"   dtype: {torch_dtype}")
        print(f"{'=' * 70}\n")

        # Load checkpoint weights for full fine-tuning (non-PEFT)
        if not args.peft_model:
            print(f"\n{'=' * 70}")
            print(f"Loading memory weights from full fine-tuned checkpoint...")
            print(f"{'=' * 70}")

            checkpoint_path = os.path.join(args.base_model, "pytorch_model.bin")
            checkpoint_files = []

            if os.path.isfile(checkpoint_path):
                checkpoint_files = [checkpoint_path]
            else:
                index_path = os.path.join(args.base_model, "pytorch_model.bin.index.json")
                if os.path.isfile(index_path):
                    import json
                    with open(index_path, 'r') as f:
                        index = json.load(f)
                    shard_files = set(index.get("weight_map", {}).values())
                    checkpoint_files = sorted([os.path.join(args.base_model, f) for f in shard_files])
                else:
                    import glob
                    checkpoint_files = sorted(glob.glob(os.path.join(args.base_model, "pytorch_model-*.bin")))

            if checkpoint_files:
                total_memory_keys = 0
                for ckpt_file in checkpoint_files:
                    if os.path.isfile(ckpt_file):
                        state_dict = torch.load(ckpt_file, map_location=model.device)
                        memory_state_dict = {k: v for k, v in state_dict.items()
                                             if 'local_constructor' in k or 'global_integrator' in k}
                        if memory_state_dict:
                            model.load_state_dict(memory_state_dict, strict=False)
                            total_memory_keys += len(memory_state_dict)
                        del state_dict
                print(f"Loaded {total_memory_keys} memory-related parameters")
            else:
                print(f"No checkpoint files found at {args.base_model}")

            print(f"{'=' * 70}\n")
    else:
        print(f"\n{'=' * 70}")
        print(f"Pure Qwen3 baseline: eval_mode='full', no peft_model")
        print(f"Skipping memory module registration and checkpoint loading")
        print(f"{'=' * 70}\n")

    # Load PEFT model
    if args.peft_model:
        trainable_params = os.path.join(args.peft_model, "trainable_params.bin")
        if os.path.isfile(trainable_params):
            model.load_state_dict(
                torch.load(trainable_params, map_location=model.device, weights_only=False), strict=False
            )
        else:
            raise ValueError("Trainable input embedding and normalization are required.")

        model = PeftModel.from_pretrained(
            model,
            args.peft_model,
            torch_dtype=torch_dtype,
            offload_folder=args.cache_dir,
        )

    # Enable distributed evaluation
    [p.requires_grad_() for n, p in model.named_parameters() if "lm_head" in n]

    gpu_id = int(os.environ["LOCAL_RANK"])
    model.to(gpu_id)

    evaluator = DistributedEvaluator(
        model=model,
        batch_size=args.batch_size,
        refresh_rate=args.progress_bar_fresh_rate,
        gpu_id=gpu_id,
    )

    if evaluator.is_first_device():
        print("data path", args.data_path)
        print("base model", args.base_model)
        print("peft model", args.peft_model)
        eval_mode_desc = {
            None: "chunked (same as training)",
            "chunked": "chunked (same as training)",
            "full": "full attention (no memory)",
            "full_hierarchical": "full attention + hierarchical memory",
        }
        print(f"eval mode: {args.eval_mode} -> {eval_mode_desc.get(args.eval_mode, 'unknown')}")
        print(f"Num validation tokens: {dataset.num_tokens()}, Num validation examples: {len(dataset)}")

    eval_metric = EvalMetricImpl(vocab_size=config.vocab_size, gpu_id=gpu_id)
    result = evaluator.evaluate(dataset, eval_metric)
    if evaluator.is_first_device():
        print(result)


def ddp_setup():
    init_process_group(backend="nccl")


def main(cmd_args=None):
    ddp_setup()
    parser = transformers.HfArgumentParser((EvalArguments,))
    args = parser.parse_args_into_dataclasses(cmd_args)[0]
    try:
        run_eval(args)
    finally:
        destroy_process_group()


if __name__ == "__main__":
    main()
