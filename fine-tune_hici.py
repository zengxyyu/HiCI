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
import math
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, Optional, Sequence

import torch
import transformers
from torch.utils.data import Dataset
from transformers import Trainer, DataCollatorForLanguageModeling
from llama_attn_hici import (
    replace_llama_attn,
    register_hici_to_model,
)
from gptneox_attn_replace import replace_gpt_neox_attn
from peft import LoraConfig, get_peft_model
from torch.distributed import barrier

from datasets import load_dataset

IGNORE_INDEX = -100

# ============================================================================
# Custom Trainer with Layered Learning Rates
# ============================================================================


class LayeredLRTrainer(Trainer):
    """
    Custom Trainer that supports different learning rates for HiCI parameters.

    Usage:
        trainer = LayeredLRTrainer(
            model=model,
            args=training_args,  # Must have hici_lr set
            ...
        )
    """

    def create_optimizer(self):
        """
        Create optimizer with separate learning rates for different parameter groups.

        If args.hici_lr is set, HiCI parameters use that lr,
        while other parameters use args.learning_rate.
        """
        if self.optimizer is None:
            # Check if we need layered learning rates
            if self.args.hici_lr is not None:
                # Only print on rank 0
                is_main_process = self.args.local_rank <= 0
                if is_main_process:
                    print("\n" + "=" * 70)
                    print("Creating Optimizer with Layered Learning Rates")
                    print("=" * 70)

                # Separate parameters into groups
                # Collect local_constructor and global_integrator params separately
                local_constructor_params = []
                global_integrator_params = []
                other_params = []

                for n, p in self.model.named_parameters():
                    if not p.requires_grad:
                        continue

                    # Check if this is a HiCI module parameter
                    if "local_constructor" in n:
                        local_constructor_params.append(p)
                    elif "global_integrator" in n:
                        global_integrator_params.append(p)
                    else:
                        other_params.append(p)

                # Combine HiCI params for optimizer (may be empty if no HiCI modules)
                hici_params = local_constructor_params + global_integrator_params

                # Create parameter groups with different learning rates
                optimizer_grouped_parameters = [
                    {
                        "params": hici_params,
                        "lr": self.args.hici_lr,
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": other_params,
                        "lr": self.args.learning_rate,
                        "weight_decay": self.args.weight_decay,
                    },
                ]

                # Create optimizer
                optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
                    self.args
                )

                # Remove 'lr' from kwargs since we're specifying it per group
                optimizer_kwargs.pop("lr", None)

                self.optimizer = optimizer_cls(
                    optimizer_grouped_parameters, **optimizer_kwargs
                )

                # Print summary with detailed breakdown (only on rank 0)
                if is_main_process:
                    local_constructor_count = sum(p.numel() for p in local_constructor_params)
                    hierarchical_count = sum(
                        p.numel() for p in global_integrator_params
                    )
                    hici_count = local_constructor_count + hierarchical_count
                    other_count = sum(p.numel() for p in other_params)
                    total_count = hici_count + other_count

                    print(f"\n  📊 HiCI Module Parameters Breakdown:")
                    print(f"  " + "-" * 68)

                    if global_memory_count > 0:
                        print(f"    🧠 LocalConstructor:")
                        print(
                            f"       Count: {local_constructor_count:,} ({local_constructor_count / total_count * 100:.2f}%)"
                        )
                    else:
                        print(f"    🧠 LocalConstructor: Not enabled (0 parameters)")

                    if hierarchical_count > 0:
                        print(f"    🏗️  HierarchicalAggregator:")
                        print(
                            f"       Count: {hierarchical_count:,} ({hierarchical_count / total_count * 100:.2f}%)"
                        )
                    else:
                        print(
                            f"    🏗️  HierarchicalAggregator: Not enabled (0 parameters)"
                        )

                    print(f"  " + "-" * 68)
                    print(
                        f"    📦 Total HiCI Modules: {hici_count:,} ({hici_count / total_count * 100:.2f}%)"
                    )
                    print(f"    📝 Learning Rate: {self.args.hici_lr:.2e}")

                    print(f"\n  📚 Other Trainable Parameters:")
                    print(
                        f"    Count: {other_count:,} ({other_count / total_count * 100:.2f}%)"
                    )
                    print(f"    Learning Rate: {self.args.learning_rate:.2e}")

                    print(
                        f"\n  ⚡ Learning Rate Ratio: {self.args.hici_lr / self.args.learning_rate:.1f}x"
                    )
                    print("=" * 70 + "\n")

            else:
                # Use standard optimizer creation (only print on rank 0)
                if self.args.local_rank <= 0:
                    print("\n📌 Using uniform learning rate for all parameters")
                    print(f"   Learning Rate: {self.args.learning_rate:.2e}\n")
                return super().create_optimizer()

        return self.optimizer

    def training_step(self, model, inputs):
        """
        Perform a training step with separate gradient clipping for HiCI modules.

        If args.hici_grad_clip is set, HiCI module parameters get stricter clipping
        than other parameters (which use args.max_grad_norm).
        """
        # Call parent's training_step to do forward + backward
        loss = super().training_step(model, inputs)

        # Apply separate gradient clipping if configured
        if (
            self.args.hici_grad_clip is not None
            and self.args.hici_lr is not None
        ):
            # Separate parameters into groups
            hici_params = []
            other_params = []

            for name, param in model.named_parameters():
                if param.grad is not None:
                    # Check if this is a HiCI module parameter
                    if "local_constructor" in name or "global_integrator" in name:
                        hici_params.append(param)
                    else:
                        other_params.append(param)

            # Apply stricter gradient clipping to HiCI modules ONLY
            # Other parameters (embed, norm) are stable and don't need clipping
            if hici_params:
                torch.nn.utils.clip_grad_norm_(
                    hici_params, max_norm=self.args.hici_grad_clip
                )

            # Note: Other parameters don't use gradient clipping
            # This follows LongLoRA's original design which doesn't clip embed/norm gradients

            # Print gradient clipping info only once (on rank 0, first step)
            if not hasattr(self, "_grad_clip_printed"):
                is_main_process = self.args.local_rank <= 0
                if is_main_process:
                    print("\n" + "=" * 70)
                    print("Gradient Clipping Configuration")
                    print("=" * 70)
                    print(f"  🧠 HiCI Modules:")
                    print(f"     Max Gradient Norm: {self.args.hici_grad_clip}")
                    print(f"     Num Parameters: {len(hici_params)}")
                    print(f"\n  📚 Other Parameters (embed, norm):")
                    print(f"     Max Gradient Norm: None (no clipping)")
                    print(f"     Num Parameters: {len(other_params)}")
                    print("=" * 70 + "\n")
                self._grad_clip_printed = True

        return loss


DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="EleutherAI/pythia-1.4b-deduped")
    model_type: Optional[str] = field(default="llama")


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=8192 * 4,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    use_flash_attn: bool = field(
        default=True,
        metadata={"help": "Whether use flash attention for training."},
    )
    use_full_attn: bool = field(
        default=False,
        metadata={"help": "Whether to use plain, full-attention for training."},
    )
    low_rank_training: bool = field(
        default=True,
        metadata={"help": "Whether use low rank adaptation for training."},
    )
    use_yarn_rope: bool = field(
        default=False,
        metadata={"help": "Whether to use YaRN (Yet another RoPE extensioN) instead of PI. No Transformers upgrade needed."},
    )
    trainable_params: str = field(
        default="embed,norm",
        metadata={
            "help": "Additional trainable parameters except LoRA weights, if low rank training."
        },
    )
    num_local_slots: int = field(
        default=8,
        metadata={
            "help": "Number of Local Representation Slots for capturing chunk-level context (default: 8)."
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
        metadata={"help": "Whether to use LocalConstructor."},
    )
    use_global_integrator: bool = field(
        default=True,
        metadata={"help": "Whether to use GlobalIntegrator."},
    )
    use_local_constructor_flash: bool = field(
        default=False,
        metadata={"help": "Whether to use flash attn in LocalConstructorFlash."},
    )
    use_hierarchical_forward: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use the combined hierarchical forward (LocalConstructor + GlobalIntegrator)."},
    )
    use_llama_init: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Warm-initialize HiCI module Q/K/V projections from LLaMA pretrained weights."
        },
    )
    num_heads: int = field(
        default=32,
        metadata={"help": "Number of attention heads in HiCI module."},
    )
    use_bottleneck: bool = field(
        default=True,
        metadata={
            "help": "Whether to use bottleneck in GlobalIntegrator."
        },
    )
    bottleneck_dim: int = field(
        default=4096,
        metadata={"help": "Bottleneck dimension for HiCI compression."},
    )
    shared_compress_dim: int = field(
        default=128,
        metadata={
            "help": "Shared compressor intermediate dimension for GlobalIntegratorShared "
            "(only used when use_shared_compressor=True, default: 128). "
            "Recommended: 128 for 7B, 160 for 13B."
        },
    )
    recurrence_size: Optional[int] = field(
        default=128,
        metadata={
            "help": "Number of tokens to carry from previous chunk (Transformer-XL style, default: 256)."
        },
    )
    hici_lr: Optional[float] = field(
        default=None,
        metadata={
            "help": "Separate learning rate for HiCI parameters. "
            "If None, uses the same learning rate as other parameters. "
            "Recommended: 2e-4 to 5e-4 (10-25x base lr)."
        },
    )
    hici_grad_clip: Optional[float] = field(
        default=None,
        metadata={
            "help": "Separate gradient clipping for HiCI module parameters. "
            "If None, uses the same max_grad_norm as other parameters. "
            "Recommended: 0.1 to 0.3 (stricter than default 1.0). "
            "This helps prevent gradient explosion in HiCI modules."
        },
    )


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
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


def tokenize_fn(tokenizer, example):
    """
    Multiprocess-friendly tokenize function (supports num_proc > 1).

    Features:
    1. Smart document grouping (avoids hangs on very long docs)
    2. EOS concatenation + padding (100% data utilization)
    3. Returns Python lists (multiprocess-safe, avoids H100 hangs)
    """
    context_length = tokenizer.model_max_length
    MAX_CHARS_PER_SEGMENT = 200_000

    texts = example["text"]
    all_chunks = []

    # Group documents by character length to avoid processing very long docs together
    current_batch = []
    current_length = 0

    for text in texts:
        text_len = len(text)

        # If a single doc exceeds the segment limit, process it alone
        if text_len > MAX_CHARS_PER_SEGMENT:
            # Flush current batch first
            if current_batch:
                combined = tokenizer.eos_token.join(current_batch)
                chunks = _tokenize_and_chunk(tokenizer, combined, context_length)
                all_chunks.extend(chunks)
                current_batch = []
                current_length = 0

            # Process the oversized doc in segments
            num_segments = (
                text_len + MAX_CHARS_PER_SEGMENT - 1
            ) // MAX_CHARS_PER_SEGMENT
            for i in range(num_segments):
                start = i * MAX_CHARS_PER_SEGMENT
                end = min((i + 1) * MAX_CHARS_PER_SEGMENT, text_len)
                segment = text[start:end]
                chunks = _tokenize_and_chunk(tokenizer, segment, context_length)
                all_chunks.extend(chunks)

        # If adding this doc would exceed the threshold, flush the current batch first
        elif current_length + text_len > MAX_CHARS_PER_SEGMENT and current_batch:
            combined = tokenizer.eos_token.join(current_batch)
            chunks = _tokenize_and_chunk(tokenizer, combined, context_length)
            all_chunks.extend(chunks)

            current_batch = [text]
            current_length = text_len
        else:
            current_batch.append(text)
            current_length += text_len

    # Flush the last batch
    if current_batch:
        combined = tokenizer.eos_token.join(current_batch)
        chunks = _tokenize_and_chunk(tokenizer, combined, context_length)
        all_chunks.extend(chunks)

    return {"input_ids": all_chunks}


def _tokenize_and_chunk(tokenizer, text, context_length):
    """
    Tokenize text and split into fixed-length chunks.

    Returns:
        List[List[int]]: chunks of input_ids
    """
    # Tokenize (return Python list for multiprocess safety)
    outputs = tokenizer(
        text,
        truncation=False,
        return_tensors=None,  # Python list
        padding=False,
    )

    input_ids = outputs["input_ids"]

    # Pad to a multiple of context_length for 100% data utilization
    total_length = len(input_ids)
    if total_length % context_length != 0:
        padding_length = context_length - (total_length % context_length)
        pad_token_id = (
            tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        )
        input_ids = input_ids + [pad_token_id] * padding_length

    # Split into chunks
    num_chunks = len(input_ids) // context_length
    chunks = []
    for i in range(num_chunks):
        start = i * context_length
        end = start + context_length
        chunk = input_ids[start:end]
        chunks.append(chunk)

    return chunks


def train():
    # Set random seed for reproducibility (before any random operations)
    # Same as eval_distributed.py for consistency
    # import random
    # import numpy as np
    # seed = 42  # Using standard seed (eval uses 2, but 42 is more common)
    # torch.manual_seed(seed)
    # random.seed(seed)
    # np.random.seed(seed)

    parser = transformers.HfArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()

    # NOTE: May expand supported model types in the future
    if model_args.model_type == "gpt-neox":
        replace_gpt_neox_attn(training_args.use_flash_attn, training_args.use_full_attn)
    else:
        assert model_args.model_type == "llama", (
            "Only support llama and gpt-neox for now"
        )
        replace_llama_attn(
            use_flash_attn=training_args.use_flash_attn,
            use_hierarchical_forward=training_args.use_hierarchical_forward,
        )

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )

    orig_rope_scaling = getattr(config, "rope_scaling", None)
    if orig_rope_scaling is None:
        orig_rope_scaling = {"factor": 1}

    orig_rope_scaling_factor = (
        orig_rope_scaling["factor"] if "factor" in orig_rope_scaling.keys() else 1
    )
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len:
        orig_ctx_len *= orig_rope_scaling_factor
        if training_args.model_max_length > orig_ctx_len:
            scaling_factor = float(
                math.ceil(training_args.model_max_length / orig_ctx_len)
            )
            config.rope_scaling = {"type": "linear", "factor": scaling_factor}

    # Load model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        torch_dtype=torch.bfloat16,
    )

    # Replace RoPE with YaRN if requested
    if training_args.use_yarn_rope and training_args.model_max_length > orig_ctx_len:
        from yarn_rope_official import replace_rope_with_yarn

        print("\n" + "=" * 80)
        print("🚀 Replacing Position Interpolation (PI) with YaRN")
        print("=" * 80)
        print(f"  Original context length: {orig_ctx_len}")
        print(f"  Target context length:   {training_args.model_max_length}")
        print(f"  Scaling factor:          {scaling_factor:.1f}")
        print("  YaRN advantages:")
        print("    - Better perplexity (~3% improvement over PI)")
        print("    - Higher passkey accuracy (~6% improvement)")
        print("    - Faster convergence (75% fewer tokens)")
        print("=" * 80 + "\n")

        model = replace_rope_with_yarn(
            model,
            scaling_factor=scaling_factor,
            original_max_length=orig_ctx_len,
        )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
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

    # ========================================================================
    # Register HiCI Modules (CRITICAL: Before optimizer initialization!)
    # ========================================================================
    register_hici_to_model(
        model,
        num_local_slots=training_args.num_local_slots,
        # recurrence_size=training_args.recurrence_size,
        global_slots=training_args.global_slots,
        num_heads=training_args.num_heads,
        use_bottleneck=training_args.use_bottleneck,
        bottleneck_dim=training_args.bottleneck_dim,
        use_local_constructor=training_args.use_local_constructor,
        use_global_integrator=training_args.use_global_integrator,
        use_local_constructor_flash=training_args.use_local_constructor_flash,
        use_llama_init=training_args.use_llama_init,  # warm-init from LLaMA weights
        shared_compress_dim=training_args.shared_compress_dim,
        # ds_config_path=training_args.deepspeed,  # disabled: ZeRO-3 sharding incompatible with module init
    )
    print("=" * 70 + "\n")

    rank = int(os.environ.get("RANK", -1))
    if rank > 0:
        barrier()

    # dataset = load_dataset("ZengXiangyu/RedPajama-Data-1T-Sample", cache_dir=training_args.cache_dir)
    from datasets import load_from_disk

    dataset = load_from_disk("./cache/datasets")

    # Process dataset with adaptive batch sizes based on estimated token count
    # (1 char ≈ 0.3 tokens conservatively; batch * tokens <= 100K is the safety threshold)
    print("=" * 70)
    print("Processing dataset with adaptive batch sizes by estimated token count")
    print("=" * 70)

    print("\nStep 1: Grouping by estimated token count...")
    very_short_docs = dataset.filter(
        lambda x: len(x["text"]) < 20_000, num_proc=128
    )  # <6K tokens, batch=100
    short_docs = dataset.filter(
        lambda x: 20_000 <= len(x["text"]) < 100_000, num_proc=128
    )  # 6K-30K tokens, batch=20
    medium_docs = dataset.filter(
        lambda x: 100_000 <= len(x["text"]) < 300_000, num_proc=128
    )  # 30K-90K tokens, batch=3
    long_docs = dataset.filter(
        lambda x: len(x["text"]) >= 300_000, num_proc=128
    )  # >90K tokens, batch=1
    print(f"  Very short (<20K chars, ~<6K tokens): {len(very_short_docs['train']):,}")
    print(f"  Short (20K-100K chars, ~6K-30K tokens): {len(short_docs['train']):,}")
    print(f"  Medium (100K-300K chars, ~30K-90K tokens): {len(medium_docs['train']):,}")
    print(f"  Long (>=300K chars, ~>90K tokens): {len(long_docs['train']):,}")

    print("\nStep 2: Tokenizing with per-group batch sizes...")
    from datasets import concatenate_datasets

    very_short_processed = very_short_docs.map(
        partial(tokenize_fn, tokenizer),
        batched=True,
        batch_size=200,
        num_proc=128,
        remove_columns=["text", "meta"],
    )
    print(f"  Very short docs done")

    short_processed = short_docs.map(
        partial(tokenize_fn, tokenizer),
        batched=True,
        batch_size=40,
        num_proc=128,
        remove_columns=["text", "meta"],
    )
    print(f"  Short docs done")

    medium_processed = medium_docs.map(
        partial(tokenize_fn, tokenizer),
        batched=True,
        batch_size=5,
        num_proc=128,
        remove_columns=["text", "meta"],
    )
    print(f"  Medium docs done")

    long_processed = long_docs.map(
        partial(tokenize_fn, tokenizer),
        batched=True,
        batch_size=1,
        num_proc=128,
        remove_columns=["text", "meta"],
    )
    print(f"  Long docs done")

    print("\nStep 3: Concatenating and shuffling...")
    dataset = concatenate_datasets(
        [
            very_short_processed["train"],
            short_processed["train"],
            medium_processed["train"],
            long_processed["train"],
        ]
    )
    dataset = dataset.shuffle(seed=42)
    dataset = {"train": dataset}
    print(f"  Final samples: {len(dataset['train']):,}")
    print("=" * 70)

    if rank == 0:
        barrier()

    print(dataset)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # ========================================================================
    # Setup LoRA (if enabled)
    # Note: LoRA is configured AFTER HiCI module registration
    # This ensures HiCI parameters are already in model.parameters()
    # ========================================================================
    if training_args.low_rank_training:
        if model_args.model_type == "gpt-neox":
            # added `dense` to match with llama as the basic LoRA would only target 'query_key_value'
            targets = ["query_key_value", "dense"]
        else:
            targets = ["q_proj", "k_proj", "v_proj", "o_proj"]

        config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=targets,
            lora_dropout=0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)
        # enable trainable params
        [
            p.requires_grad_()
            for n, p in model.named_parameters()
            if any([k in n for k in training_args.trainable_params.split(",")])
        ]

    # ========================================================================
    # Verify and summarize trainable parameters
    # ========================================================================
    if rank == 0:
        print("\n" + "=" * 70)
        print("Trainable Parameters Summary")
        print("=" * 70)

    trainable_params_dict = {}
    for n, p in model.named_parameters():
        if p.requires_grad:
            category = None
            if "lora" in n.lower():
                category = "LoRA Adapters"
            elif (
                "local_constructor" in n
                or "global_integrator" in n
            ):
                category = "HiCI Modules"
            elif "embed" in n.lower():
                category = "Embeddings"
            elif "norm" in n.lower():
                category = "LayerNorm"
            else:
                category = "Other"

            if category not in trainable_params_dict:
                trainable_params_dict[category] = 0
            trainable_params_dict[category] += p.numel()

    total_trainable = sum(trainable_params_dict.values())
    total_params = sum(p.numel() for p in model.parameters())

    if rank == 0:
        for category, count in sorted(trainable_params_dict.items()):
            print(
                f"  {category:20s}: {count:15,} params ({count / total_trainable * 100:5.2f}%)"
            )

        if "HiCI Modules" in trainable_params_dict:
            global_memory_count = 0
            hierarchical_count = 0
            for n, p in model.named_parameters():
                if p.requires_grad:
                    if "local_constructor" in n:
                        global_memory_count += p.numel()
                    elif "global_integrator" in n:
                        hierarchical_count += p.numel()

            if global_memory_count > 0 or hierarchical_count > 0:
                print(f"    {'└─ LocalConstructor':20s}: {global_memory_count:15,} params")
                print(
                    f"    {'└─ HierarchicalAgg':20s}: {hierarchical_count:15,} params"
                )

        print(f"  {'─' * 20}   {'─' * 15}   {'─' * 7}")
        print(
            f"  {'Total Trainable':20s}: {total_trainable:15,} params ({total_trainable / total_params * 100:5.2f}% of total)"
        )
        print(f"  {'Total Params':20s}: {total_params:15,} params")

    # Warning if HiCI modules are not properly configured
    has_memory_in_trainable = "local_constructor" in training_args.trainable_params
    has_hierarchical_in_trainable = "hierarchical" in training_args.trainable_params
    has_hici_params = "HiCI Modules" in trainable_params_dict

    if rank == 0:
        if has_memory_in_trainable and not has_hici_params:
            print(
                "\n⚠️  WARNING: 'local_constructor' specified in --trainable_params but no HiCI parameters found!"
            )
        elif not has_memory_in_trainable and has_hici_params:
            print(
                "\n⚠️  WARNING: HiCI module parameters found but not in --trainable_params!"
            )
            print(
                "    Add '--trainable_params \"embed,norm,local_constructor,global_integrator\"' to enable training."
            )

        # Check if global_integrator is missing when use_global_integrator=True
        if training_args.use_global_integrator:
            if has_memory_in_trainable and not has_hierarchical_in_trainable:
                print(
                    "\n⚠️  WARNING: Using GlobalIntegrator but 'global_integrator' not in --trainable_params!"
                )
                print("    HierarchicalAggregator parameters may not be trained!")
                print(
                    "    Recommended: '--trainable_params \"embed,norm,local_constructor,global_integrator\"'"
                )

        print("=" * 70 + "\n")

    model.config.use_cache = False  # required for gradient checkpointing
    model.enable_input_require_grads()  # required for gradient checkpointing
    model.gradient_checkpointing_enable()  # enable gradient checkpointing

    # Log checkpoint resume status
    if rank == 0:
        import glob

        checkpoint_dirs = glob.glob(
            os.path.join(training_args.output_dir, "checkpoint-*")
        )
        if checkpoint_dirs:
            latest_checkpoint = max(
                checkpoint_dirs, key=lambda x: int(x.split("-")[-1])
            )

            print("\n" + "=" * 80)
            print("Existing checkpoint detected — resuming training")
            print("=" * 80)
            print(f"Output dir: {training_args.output_dir}")
            print(f"Found {len(checkpoint_dirs)} checkpoint(s)")
            print(f"HuggingFace Trainer will resume from: {latest_checkpoint}")
            print(f"   - Local slots: {training_args.num_local_slots}")
            print("=" * 80 + "\n")
        else:
            print("\n" + "=" * 80)
            print("Training from scratch (no existing checkpoints)")
            print("=" * 80)
            print(f"Output dir: {training_args.output_dir}")
            print(f"   - Local slots: {training_args.num_local_slots}")
            print("=" * 80 + "\n")

    # ========================================================================
    # Initialize Trainer (optimizer created here)
    # At this point, model.parameters() includes:
    # 1. Base model parameters (frozen if LoRA)
    # 2. LoRA adapters (trainable)
    # 3. HiCI parameters (trainable)
    # 4. Embeddings & LayerNorm (trainable if in trainable_params)
    #
    # Uses LayeredLRTrainer to support different learning rates for HiCI modules
    # ========================================================================
    trainer = LayeredLRTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=None,
        data_collator=data_collator,
    )

    # ========================================================================
    # Verify DeepSpeed Stage 3 sharding of HiCI modules
    # ========================================================================
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()

        hici_params_info = []
        total_hici_numel = 0

        for name, param in model.named_parameters():
            if "local_constructor" in name or "global_integrator" in name:
                hici_params_info.append({
                    "name": name.split(".")[-2] + "." + name.split(".")[-1],
                    "shape": tuple(param.shape),
                    "numel": param.numel(),
                    "device": str(param.device),
                })
                total_hici_numel += param.numel()

        # Print rank-by-rank to avoid interleaved output
        for print_rank in range(world_size):
            if rank == print_rank:
                if rank == 0:
                    print("\n" + "=" * 80)
                    print("DeepSpeed HiCI module sharding check")
                    print("=" * 80)
                    print(f"World Size: {world_size} GPUs")
                    print("-" * 80)

                print(f"\n[Rank {rank}] HiCI params: {total_hici_numel:,} ({total_hici_numel / 1e6:.2f}M)")

                for i, info in enumerate(hici_params_info[:3]):
                    print(f"  - {info['name']}: shape={info['shape']}, numel={info['numel']:,}, device={info['device']}")
                if len(hici_params_info) > 3:
                    print(f"  ... {len(hici_params_info)} HiCI params total")

                if rank == 0:
                    expected_sharded = total_hici_numel // world_size
                    print(f"\nSharding analysis:")
                    print(f"  Expected per GPU (if sharded): ~{expected_sharded:,} ({expected_sharded / 1e6:.2f}M)")
                    print(f"  Actual per GPU: {total_hici_numel:,} ({total_hici_numel / 1e6:.2f}M)")
                    if total_hici_numel > expected_sharded * 1.5:
                        print(f"  WARNING: HiCI modules may not be sharded — each GPU holds a full copy.")
                    else:
                        print(f"  HiCI modules correctly sharded")
                    print("=" * 80 + "\n")

            torch.distributed.barrier()
    # ========================================================================

    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
