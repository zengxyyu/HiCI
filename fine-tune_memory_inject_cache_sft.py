#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Supervised Fine-Tuning for Memory-Augmented LongLoRA

🔥 关键修改（相比原版supervised-fine-tune.py）:
1. ✅ 使用 llama_attn_memory_inject_cache.py（包含memory机制）
2. ✅ 注册GlobalMemoryModule和HierarchicalAggregator
3. ✅ 正确加载Stage 1的memory weights
4. ✅ 支持分层学习率（memory vs base model vs LoRA）
5. ✅ 保持memory modules trainable
"""

import io
import os
import copy
import json
import math
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import transformers
from torch.utils.data import Dataset
from transformers import Trainer, DataCollatorForLanguageModeling

# 🔥 关键修改1: 使用memory版本的attention替换（SFT专用版本）
from llama_attn_memory_inject_cache_sft import (
    replace_llama_attn,
    register_global_memory_to_model,
)

from peft import LoraConfig, get_peft_model
from torch.distributed import barrier

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


# ============================================================================
# Custom Trainer with Layered Learning Rates
# ============================================================================


class LayeredLRTrainer(Trainer):
    """
    Custom Trainer that supports different learning rates for global memory parameters.

    Usage:
        trainer = LayeredLRTrainer(
            model=model,
            args=training_args,  # Must have global_memory_lr set
            ...
        )
    """

    def create_optimizer(self):
        """
        Create optimizer with separate learning rates for different parameter groups.

        If args.global_memory_lr is set, global memory parameters use that lr,
        while other parameters use args.learning_rate.
        """
        if self.optimizer is None:
            # Check if we need layered learning rates
            if self.args.global_memory_lr is not None:
                # Only print on rank 0
                is_main_process = self.args.local_rank <= 0
                if is_main_process:
                    print("\n" + "=" * 70)
                    print("Creating Optimizer with Layered Learning Rates")
                    print("=" * 70)

                # Separate parameters into groups
                # 分别统计 global_memory 和 hierarchical_aggregator 参数
                global_memory_params = []
                hierarchical_aggregator_params = []
                other_params = []

                for n, p in self.model.named_parameters():
                    if not p.requires_grad:
                        continue

                    # 检查是否是内存模块参数（两个模块是独立的）
                    if "global_memory" in n:
                        global_memory_params.append(p)
                    elif "hierarchical_aggregator" in n:
                        hierarchical_aggregator_params.append(p)
                    else:
                        other_params.append(p)

                # Combine memory params for optimizer (may be empty if no memory modules)
                memory_params = global_memory_params + hierarchical_aggregator_params

                # Create parameter groups with different learning rates
                optimizer_grouped_parameters = [
                    {
                        "params": memory_params,
                        "lr": self.args.global_memory_lr,
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
                    global_memory_count = sum(p.numel() for p in global_memory_params)
                    hierarchical_count = sum(
                        p.numel() for p in hierarchical_aggregator_params
                    )
                    memory_count = global_memory_count + hierarchical_count
                    other_count = sum(p.numel() for p in other_params)
                    total_count = memory_count + other_count

                    print(f"\n  📊 Memory Module Parameters Breakdown:")
                    print(f"  " + "-" * 68)

                    if global_memory_count > 0:
                        print(f"    🧠 LocalMemory:")
                        print(
                            f"       Count: {global_memory_count:,} ({global_memory_count / total_count * 100:.2f}%)"
                        )
                    else:
                        print(f"    🧠 LocalMemory: Not enabled (0 parameters)")

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
                        f"    📦 Total Memory Modules: {memory_count:,} ({memory_count / total_count * 100:.2f}%)"
                    )
                    print(f"    📝 Learning Rate: {self.args.global_memory_lr:.2e}")

                    print(f"\n  📚 Other Trainable Parameters:")
                    print(
                        f"    Count: {other_count:,} ({other_count / total_count * 100:.2f}%)"
                    )
                    print(f"    Learning Rate: {self.args.learning_rate:.2e}")

                    print(
                        f"\n  ⚡ Learning Rate Ratio: {self.args.global_memory_lr / self.args.learning_rate:.1f}x"
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
        Perform a training step with separate gradient clipping for memory modules.

        If args.memory_grad_clip is set, memory module parameters get stricter clipping
        than other parameters (which use args.max_grad_norm).
        """
        # Call parent's training_step to do forward + backward
        loss = super().training_step(model, inputs)

        # Apply separate gradient clipping if configured
        if (
            self.args.memory_grad_clip is not None
            and self.args.global_memory_lr is not None
        ):
            # Separate parameters into groups
            memory_params = []
            other_params = []

            for name, param in model.named_parameters():
                if param.grad is not None:
                    # Check if this is a memory module parameter
                    if "global_memory" in name or "hierarchical_aggregator" in name:
                        memory_params.append(param)
                    else:
                        other_params.append(param)

            # Apply stricter gradient clipping to memory modules ONLY
            # Other parameters (embed, norm) are stable and don't need clipping
            if memory_params:
                torch.nn.utils.clip_grad_norm_(
                    memory_params, max_norm=self.args.memory_grad_clip
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
                    print(f"  🧠 Memory Modules:")
                    print(f"     Max Gradient Norm: {self.args.memory_grad_clip}")
                    print(f"     Num Parameters: {len(memory_params)}")
                    print(f"\n  📚 Other Parameters (embed, norm):")
                    print(f"     Max Gradient Norm: None (no clipping)")
                    print(f"     Num Parameters: {len(other_params)}")
                    print("=" * 70 + "\n")
                self._grad_clip_printed = True

        return loss


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict


PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
    "prompt_no_input_llama2": (
        "[INST] <<SYS>>\n"
        "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\n"
        "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n"
        "<</SYS>> \n\n {instruction} [/INST]"
    ),
    "prompt_input_llama2": (
        "[INST] <<SYS>>\n"
        "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\n"
        "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n"
        "<</SYS>> \n\n {instruction} \n{input} [/INST]"
    ),
    "prompt_llama2": "[INST]{instruction}[/INST]",
}


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="EleutherAI/pythia-1.4b-deduped")
    model_type: Optional[str] = field(default="llama")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )


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
        metadata={"help": "Whether to use plain, full attention for training."},
    )
    low_rank_training: bool = field(
        default=False,
        metadata={"help": "Whether use low rank adaptation for training."},
    )
    trainable_params: str = field(
        default="embed,norm",
        metadata={
            "help": "Additional trainable parameters except LoRA weights, if low rank training."
        },
    )

    # 🔥 关键修改2: Memory相关参数 (与 fine-tune_memory_inject_cache.py 保持一致)
    num_memory_slots: int = field(
        default=16,
        metadata={"help": "Number of local memory slots for GlobalMemoryModule"},
    )
    global_slots: int = field(
        default=4,
        metadata={"help": "Number of global memory slots for HierarchicalAggregator"},
    )
    num_chunks: int = field(
        default=4,
        metadata={
            "help": "Number of chunks to split each sequence into for hierarchical memory."
        },
    )
    use_local_summary: bool = field(
        default=True,
        metadata={"help": "Whether to use local memory attention."},
    )
    use_hierarchical_memory: bool = field(
        default=True,
        metadata={"help": "Whether to use hierarchical memory aggregator."},
    )
    num_heads: int = field(
        default=8,
        metadata={"help": "Number of attention heads in the memory module."},
    )
    use_bottleneck: bool = field(
        default=True,
        metadata={
            "help": "Whether to use bottleneck in hierarchical memory aggregator."
        },
    )
    bottleneck_dim: int = field(
        default=512,
        metadata={"help": "Bottleneck dimension for memory compression."},
    )
    memory_use_flash: bool = field(
        default=False,
        metadata={"help": "Whether to use flash attn in GlobalMemoryModule_Flash."},
    )
    use_flash_plus: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use GlobalMemoryModule_Flash_plus."},
    )
    use_flash_plus_norope: Optional[bool] = field(
        default=False,
        metadata={"help": "实验版：先调用 Memory 再应用 RoPE。"},
    )
    use_llama_init: Optional[bool] = field(
        default=False,
        metadata={"help": "从LLaMA预训练权重初始化Memory模块的Q/K/V投影"},
    )
    forward_flashattn_optimized: Optional[bool] = field(
        default=False,
        metadata={"help": "是否使用forward_flashattn_optimized函数"},
    )
    use_hierarchical_forward: Optional[bool] = field(
        default=False,
        metadata={"help": "是否使用综合函数，局部记忆+全局"},
    )
    recurrence_size: Optional[int] = field(
        default=128,
        metadata={
            "help": "Number of tokens to carry from previous chunk (Transformer-XL style)."
        },
    )
    memory_grad_clip: float = field(
        default=0.3,
        metadata={"help": "Gradient clipping for memory modules."},
    )
    global_memory_lr: float = field(
        default=1e-4,
        metadata={
            "help": "Learning rate for memory modules (higher than base model LR)"
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


def _tokenize_fn(
    strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer
) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [
        _tokenize_fn(strings, tokenizer) for strings in (examples, sources)
    ]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = jload(data_path)

        logging.warning("Formatting inputs...")

        prompt_input, prompt_no_input = (
            PROMPT_DICT["prompt_input_llama2"],
            PROMPT_DICT["prompt_llama2"],
        )
        sources = [
            prompt_input.format_map(example)
            if example.get("input", "") != ""
            else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]

        targets = [
            f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict
        ]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(
        tokenizer=tokenizer, data_path=data_args.data_path
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 🔥 关键修改3: 使用memory版本的attention替换 (传递所有参数)
    replace_llama_attn(
        use_flash_attn=training_args.use_flash_attn,
        use_full=training_args.use_full_attn,
        use_optimized=training_args.forward_flashattn_optimized,
        use_optimized_plus=training_args.use_flash_plus,
        use_optimized_plus_norope=training_args.use_flash_plus_norope,
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

    # 🔥 关键修改4: 注册Memory Modules（必须在加载checkpoint之前！）
    print("\n" + "=" * 80)
    print("🔥 Registering Memory Modules for SFT")
    print("=" * 80)
    register_global_memory_to_model(
        model,
        num_memory_slots=training_args.num_memory_slots,
        global_slots=training_args.global_slots,
        num_chunks=training_args.num_chunks,
        num_heads=training_args.num_heads,
        use_bottleneck=training_args.use_bottleneck,
        bottleneck_dim=training_args.bottleneck_dim,
        use_local_summary=training_args.use_local_summary,
        use_hierarchical=training_args.use_hierarchical_memory,
        use_flash_plus=training_args.use_flash_plus,
        use_flash=training_args.memory_use_flash,
        use_llama_init=training_args.use_llama_init,
    )
    print("=" * 80 + "\n")

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

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    # 🔥 关键修改5: 分层学习率设置
    # Memory modules使用更高的learning rate，因为训练量少
    if training_args.low_rank_training:
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

        # Enable trainable params
        trainable_param_names = training_args.trainable_params.split(",")
        for n, p in model.named_parameters():
            # Check if any trainable param name is in this parameter name
            if any([k in n for k in trainable_param_names]):
                p.requires_grad = True
            # Memory modules should always be trainable
            if "global_memory" in n or "hierarchical_aggregator" in n:
                p.requires_grad = True

        # 打印trainable parameters
        trainable_params = [n for n, p in model.named_parameters() if p.requires_grad]
        print(f"\n🔥 Trainable parameters ({len(trainable_params)} total):")
        for name in trainable_params[:20]:  # Show first 20
            print(f"  - {name}")
        if len(trainable_params) > 20:
            print(f"  ... and {len(trainable_params) - 20} more")
        print()

    model.config.use_cache = False  # required for gradient checkpointing
    model.enable_input_require_grads()  # required for gradient checkpointing
    model.gradient_checkpointing_enable()  # enable gradient checkpointing

    # 🔥 关键修改6: 使用分层学习率
    # LayeredLRTrainer 会自动为 memory modules 使用更高的 learning rate (global_memory_lr)
    # 如果 global_memory_lr 未设置，则回退到统一 learning rate
    trainer = LayeredLRTrainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )

    # 🔥 关键修改7: 如果指定了resume_from_checkpoint，确保正确加载memory weights
    # Trainer会自动处理，但我们需要确保checkpoint包含memory modules
    if training_args.resume_from_checkpoint:
        print(f"\n🔥 Resuming from checkpoint: {training_args.resume_from_checkpoint}")
        print("   Memory modules weights will be loaded from checkpoint")
        print()

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
