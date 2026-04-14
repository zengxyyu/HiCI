# Written by Yukang Chen
# Some code based on https://github.com/epfml/landmark-attention
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
from dataclasses import dataclass, field
from typing import Optional

import math
import random
import transformers
from peft import PeftModel

# Use llama_attn_hici to support the HiCI memory mechanism
# from llama_attn_memory_inject import replace_llama_attn, register_hici_to_model
from llama_attn_hici import (
    replace_llama_attn,
    register_hici_to_model,
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
    def __init__(self, data_path: str, seq_length: int, sliding_window: int = 256, data_dtype: str = "uint16"):
        assert seq_length >= sliding_window, (
            f"Sliding window '{sliding_window}' must be smaller than sequence length '{seq_length}'"
        )

        self.seq_length = seq_length
        # uint16 for LLaMA-2 (vocab 32K), uint32 for LLaMA-3 (vocab 128K) / Qwen (vocab 152K)
        self.data = np.memmap(data_path, dtype=np.dtype(data_dtype), mode="r")
        self.start_indices = list(range(0, len(self.data) - seq_length, sliding_window))

        assert len(self) > 0, "Dataset is empty"

    def __len__(self):
        return len(self.start_indices)
        # return 1000

    def __getitem__(self, index) -> dict[str, torch.Tensor]:
        start = self.start_indices[index]
        end = start + self.seq_length

        input_id = torch.from_numpy(self.data[start:end].astype(np.int64))
        y = torch.from_numpy(self.data[start + 1 : end + 1].astype(np.int64))
        return {"input_ids": input_id, "labels": input_id, "ys": y}

    def num_tokens(self):
        return len(self.data)


class EvalMetric(ABC):
    @abstractmethod
    def add(
        self, logits: torch.FloatTensor, labels: torch.LongTensor, model_output: object
    ) -> dict[str, object]:
        pass

    @abstractmethod
    def compute(self) -> dict[str, object]:
        pass


class DistributedEvaluator:
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module],
        batch_size: int,
        refresh_rate: int,
        gpu_id: int,
    ):
        self.gpu_id = gpu_id
        self.batch_size = batch_size
        self.refresh_rate = refresh_rate

        self.model = DDP(model, device_ids=[self.gpu_id])

    def evaluate(self, dataset: Dataset, metric: EvalMetric) -> dict[str, object]:
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
                    for key in used
                    if key in example_dict
                }
                outputs = self.model(**inputs)
                metric_result = metric.add(
                    logits=outputs["logits"],
                    labels=inputs["labels"],
                    model_output=outputs,
                )

                if self.is_first_device() and (i % self.refresh_rate == 0):
                    data_loader.set_postfix(metric_result)
            return metric.compute()

    def is_first_device(self):
        return self.gpu_id == 0

    def _prepare_dataloader(self, dataset: Dataset):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            shuffle=False,
            sampler=DistributedSampler(dataset),
        )


class EvalMetricImpl(EvalMetric):
    def __init__(self, vocab_size: int, gpu_id: int):
        self.accuracy = Accuracy(task="multiclass", num_classes=vocab_size).to(gpu_id)
        self.perplexity = Perplexity(ignore_index=CrossEntropyLoss().ignore_index).to(
            gpu_id
        )
        self.last_loss = 0.0

    def add(
        self, logits: torch.FloatTensor, labels: torch.LongTensor, model_output: object
    ) -> dict[str, object]:
        shift_predictions = logits.argmax(dim=-1)[..., :-1]
        shift_labels = labels[..., 1:]

        current_accuracy = self.accuracy.forward(
            preds=shift_predictions, target=shift_labels
        )

        shift_logits = logits[..., :-1, :]
        current_perplexity = self.perplexity.forward(
            preds=shift_logits, target=shift_labels
        )

        self.last_loss = model_output["loss"].item()
        return {
            "accuracy": current_accuracy.item(),
            "perplexity": current_perplexity.item(),
            "loss": self.last_loss,
        }

    def compute(self) -> dict[str, object]:
        current_accuracy = self.accuracy.compute()
        current_perplexity = self.perplexity.compute()
        return {
            "accuracy": current_accuracy.item(),
            "perplexity": current_perplexity.item(),
            "loss": self.last_loss,
        }


@dataclass
class EvalArguments:
    batch_size: int = field(
        default=1,
        metadata={"help": "batch size."},
    )
    base_model: Optional[str] = field(default="meta-llama/Llama-2-7b-hf")
    seq_len: int = field(
        default=2048,
        metadata={"help": "context length during evaluation."},
    )
    context_size: int = field(
        default=-1,
        metadata={"help": "context size during fine-tuning."},
    )
    peft_model: Optional[str] = field(default=None)
    flash_attn: bool = field(
        default=True,
        metadata={"help": "Whether use flash attention."},
    )
    data_path: str = field(
        default="./test.bin",
        metadata={"help": "test data path"},
    )
    cache_dir: Optional[str] = field(default="./.cache")
    progress_bar_fresh_rate: int = field(
        default=10,
        metadata={"help": "progress bar metrics fresh rate."},
    )
    num_local_slots: int = field(
        default=8,
        metadata={
            "help": "Number of Global Representation Slots (must match training config)."
        },
    )
    global_slots: int = field(
        default=16,
        metadata={
            "help": "Number of Global Representation Slots for capturing document-level context (default: 16)."
        },
    )
    use_local_constructor: bool = field(
        default=True,
        metadata={"help": "Whether to use local memory attention."},
    )
    use_global_integrator: bool = field(
        default=True,
        metadata={"help": "Whether to use hierarchical memory attention."},
    )
    use_local_constructor_flash: bool = field(
        default=False,
        metadata={"help": "Whether to use flash attn in LocalConstructorFlash."},
    )
    use_hierarchical_forward: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use hierarchical forward (local + global memory)."},
    )
    num_heads: int = field(
        default=32,
        metadata={"help": "Number of attention heads in the memory module."},
    )
    use_bottleneck: bool = field(
        default=True,
        metadata={
            "help": "Whether to use bottleneck in hierarchical memory aggregator."
        },
    )
    bottleneck_dim: int = field(
        default=4096,
        metadata={"help": "Bottleneck dimension for memory compression."},
    )
    recurrence_size: Optional[int] = field(
        default=128,
        metadata={
            "help": "Number of tokens to carry from previous chunk (Transformer-XL style, default: 256)."
        },
    )
    eval_mode: Optional[str] = field(
        default=None,
        metadata={
            "help": "Evaluation mode: None (chunked, same as training) or 'full' (full attention, no memory)."
        },
    )


def run_eval(args: EvalArguments):
    torch_dtype = torch.float16

    seed = 2
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    if args.flash_attn:
        # Evaluation modes:
        # - None or "chunked": Chunked attention with memory (same as training)
        # - "full": Full attention without memory (LongLoRA style)
        # replace_llama_attn(
        #     use_flash_attn=True,
        #     use_full=True,
        #     eval_mode=args.eval_mode
        #     )
        replace_llama_attn(
            use_flash_attn=True,
            eval_mode=args.eval_mode,
            use_hierarchical_forward=args.use_hierarchical_forward,
        )

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        args.base_model, cache_dir=args.cache_dir, use_cache=False
    )

    # vocab_size > 65535 means token ids don't fit in uint16
    data_dtype = "uint32" if config.vocab_size > 65535 else "uint16"
    dataset = Pg19Dataset(args.data_path, seq_length=args.seq_len, sliding_window=256, data_dtype=data_dtype)

    context_size = args.context_size if args.context_size > 0 else args.seq_len
    orig_ctx_len = getattr(
        config, "max_position_embeddings", None
    )  # this value should be 4096 for LLaMA2 models
    if orig_ctx_len and context_size > orig_ctx_len:
        scaling_factor = float(math.ceil(context_size / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}

    # Load model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.base_model,
        config=config,
        cache_dir=args.cache_dir,
        torch_dtype=torch_dtype,
    )
    # Set vocab size based on model type
    # Llama-2: 32001, Llama-3: 128256 (or 128258 if special tokens were added during training)
    vocab_size = config.vocab_size
    if vocab_size >= 128000:  # Llama-3
        # Check checkpoint embedding size; special tokens may have been added
        trainable_params_path = os.path.join(args.peft_model, "trainable_params.bin") if args.peft_model else None
        if trainable_params_path and os.path.isfile(trainable_params_path):
            tp = torch.load(trainable_params_path, map_location='cpu', weights_only=False)
            for k, v in tp.items():
                if 'embed_tokens' in k:
                    vocab_size = v.shape[0]
                    break
            del tp
        model.resize_token_embeddings(vocab_size)
        print(f"📊 Resized token embeddings to {vocab_size} (Llama 3)")
    else:  # Llama 2
        model.resize_token_embeddings(32001)
        print(f"📊 Resized token embeddings to 32001 (Llama 2)")

    # Register global memory modules (CRITICAL: must be done before loading PEFT weights!)
    print(f"\n{'=' * 70}")
    print(f"Registering Global Memory for Evaluation")
    print(f"{'=' * 70}")
    # register_hici_to_model(
    #     model,
    #     num_local_slots=args.num_local_slots,
    #     global_slots=args.global_slots,
    #     num_heads=args.num_heads,
    #     use_bottleneck=args.use_bottleneck,
    #     bottleneck_dim=args.bottleneck_dim,
    #     use_local_constructor=args.use_local_constructor,
    #     use_global_integrator=args.use_global_integrator,
    #     # recurrence_size=args.recurrence_size
    # )
    register_hici_to_model(
        model,
        num_local_slots=args.num_local_slots,
        # recurrence_size=training_args.recurrence_size,
        global_slots=args.global_slots,
        # num_chunks=args.num_chunks,
        num_heads=args.num_heads,
        use_bottleneck=args.use_bottleneck,
        bottleneck_dim=args.bottleneck_dim,
        use_local_constructor=args.use_local_constructor,
        use_global_integrator=args.use_global_integrator,
        use_local_constructor_flash=args.use_local_constructor_flash,
    )

    # CRITICAL: Convert local_constructor and global_integrator to the same dtype as the model (fp16)
    # These modules are created in fp32 by default, but model is fp16
    print(f"Converting memory modules to {torch_dtype}...")
    for layer in model.model.layers:
        if hasattr(layer.self_attn, "local_constructor"):
            layer.self_attn.local_constructor = layer.self_attn.local_constructor.to(
                torch_dtype
            )
        if hasattr(layer.self_attn, "global_integrator"):
            layer.self_attn.global_integrator = (
                layer.self_attn.global_integrator.to(torch_dtype)
            )

    print(f"✅ Memory modules registration complete!")
    print(f"   Number of memory slots: {args.num_local_slots}")
    print(f"   dtype: {torch_dtype}")
    print(f"{'=' * 70}\n")

    # For full fine-tuning: reload checkpoint to restore memory module weights
    # Background: Memory modules are added dynamically via register_hici_to_model(),
    # not defined in LlamaAttention.__init__. HuggingFace's from_pretrained() ignores
    # these extra weights, so we need to explicitly reload them after registration.
    if not args.peft_model:
        print(f"\n{'=' * 70}")
        print(f"🔄 Loading memory weights from full fine-tuned checkpoint...")
        print(f"{'=' * 70}")

        # Try single file first, then check for sharded checkpoints
        checkpoint_path = os.path.join(args.base_model, "pytorch_model.bin")
        checkpoint_files = []

        if os.path.isfile(checkpoint_path):
            checkpoint_files = [checkpoint_path]
        else:
            # Check for sharded checkpoints via index.json
            index_path = os.path.join(args.base_model, "pytorch_model.bin.index.json")
            if os.path.isfile(index_path):
                import json
                with open(index_path, 'r') as f:
                    index = json.load(f)
                # Get unique shard files from weight_map
                shard_files = set(index.get("weight_map", {}).values())
                checkpoint_files = [os.path.join(args.base_model, f) for f in shard_files]
                checkpoint_files.sort()  # Ensure consistent order
                print(f"   Found {len(checkpoint_files)} shards via index.json")
            else:
                # Fallback: detect shards by pattern
                import glob
                shard_pattern = os.path.join(args.base_model, "pytorch_model-*.bin")
                checkpoint_files = sorted(glob.glob(shard_pattern))
                if checkpoint_files:
                    print(f"   Found {len(checkpoint_files)} shards via pattern matching")

        if checkpoint_files:
            total_memory_keys = 0
            for ckpt_file in checkpoint_files:
                if os.path.isfile(ckpt_file):
                    state_dict = torch.load(ckpt_file, map_location=model.device)

                    # Only load memory-related keys to avoid overwriting model weights
                    memory_state_dict = {k: v for k, v in state_dict.items()
                                        if 'local_constructor' in k or 'global_integrator' in k}

                    if memory_state_dict:
                        model.load_state_dict(memory_state_dict, strict=False)
                        total_memory_keys += len(memory_state_dict)
                        print(f"   Loaded {len(memory_state_dict)} memory keys from {os.path.basename(ckpt_file)}")

                    del state_dict  # Free memory

            print(f"✅ Loaded {total_memory_keys} total memory-related parameters")
        else:
            print(f"❌ Error: Could not find checkpoint file at {args.base_model}")
            print(f"   Tried: pytorch_model.bin, pytorch_model.bin.index.json, pytorch_model-*.bin")
            raise FileNotFoundError(f"Checkpoint not found at {args.base_model}")

        print(f"{'=' * 70}\n")

    if args.peft_model:
        trainable_params = os.path.join(args.peft_model, "trainable_params.bin")
        if os.path.isfile(trainable_params):
            model.load_state_dict(
                torch.load(trainable_params, map_location=model.device), strict=False
            )
        else:
            raise ValueError(
                "Trainable input embedding and normalization are required."
            )
        model = PeftModel.from_pretrained(
            model,
            args.peft_model,
            torch_dtype=torch_dtype,
            offload_folder=args.cache_dir,
        )

    # This is a hacky way to enable distributed evaluation. Otherwise, without any trainable parameters, we will not
    # be able to use DistributedDataParallel, although we don't update any parameters during evaluation.
    [
        p.requires_grad_()
        for n, p in model.named_parameters()
        if any([k in n for k in ["lm_head"]])
    ]

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
        }
        print(
            f"eval mode: {args.eval_mode} -> {eval_mode_desc.get(args.eval_mode, 'unknown')}"
        )
        print(
            f"Num validation tokens: {dataset.num_tokens()}, Num validation examples: {len(dataset)}"
        )

    eval_metric = EvalMetricImpl(vocab_size=config.vocab_size, gpu_id=gpu_id)
    result = evaluator.evaluate(dataset, eval_metric)
    if evaluator.is_first_device():
        print(result)

        # Save attention visualization stats (if collection is enabled)
        from llama_attn_hici import COLLECT_ATTENTION_FOR_VIZ, save_attention_stats
        if COLLECT_ATTENTION_FOR_VIZ:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"attention_stats_seq{args.seq_len}_{timestamp}.json"
            save_attention_stats(save_path)
            print(f"📊 Attention stats saved to {save_path}")


def ddp_setup():
    init_process_group(backend="nccl")


def main(cmd_args: list[str] = None):
    ddp_setup()
    parser = transformers.HfArgumentParser((EvalArguments,))
    args: EvalArguments = parser.parse_args_into_dataclasses(cmd_args)[0]
    try:
        run_eval(args)
    finally:
        destroy_process_group()


if __name__ == "__main__":
    main()
