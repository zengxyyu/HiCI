# Qwen3 HiCI Training Script (requires transformers >= 4.51)
# Adapted from fine-tune_hici_qwen.py for native Qwen3 support
#
# Key differences from Qwen2 version (fine-tune_hici_qwen.py):
# 1. Imports from qwen3_attn_hici instead of qwen_attn_hici
# 2. Uses replace_qwen3_attn() — patches Qwen3Attention, not Qwen2Attention
# 3. Uses register_hici_to_qwen3_model()
# 4. model_type="qwen3" (Qwen3 has its own model class in transformers 4.51+)
# 5. Qwen3 has attention_bias=False (like LLaMA), QK-Norm, and GQA (8 KV heads)
#
# Usage:
#   torchrun --nproc_per_node=8 fine-tune_hici_qwen3.py \
#       --model_name_or_path Qwen/Qwen3-8B \
#       --output_dir ./checkpoints/qwen3-8b-hici-32k \
#       --model_max_length 32768 \
#       --use_flash_attn True \
#       --low_rank_training True \
#       --deepspeed ds_configs/stage2.json \
#       --max_steps 1000 \
#       --num_heads 32 \
#       --num_local_slots 8 \
#       --global_slots 16

import os
import math
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, Optional, Sequence

import torch
import transformers
from torch.utils.data import Dataset
from transformers import Trainer, DataCollatorForLanguageModeling
from qwen3_attn_hici import (
    replace_qwen3_attn,
    register_hici_to_qwen3_model,
)
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
    """

    def create_optimizer(self):
        if self.optimizer is None:
            if self.args.hici_lr is not None:
                is_main_process = self.args.local_rank <= 0
                if is_main_process:
                    print("\n" + "=" * 70)
                    print("Creating Optimizer with Layered Learning Rates")
                    print("=" * 70)

                local_constructor_params = []
                global_integrator_params = []
                other_params = []

                for n, p in self.model.named_parameters():
                    if not p.requires_grad:
                        continue
                    if "local_constructor" in n:
                        local_constructor_params.append(p)
                    elif "global_integrator" in n:
                        global_integrator_params.append(p)
                    else:
                        other_params.append(p)

                hici_params = local_constructor_params + global_integrator_params

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

                optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
                    self.args
                )
                optimizer_kwargs.pop("lr", None)
                self.optimizer = optimizer_cls(
                    optimizer_grouped_parameters, **optimizer_kwargs
                )

                if is_main_process:
                    local_constructor_count = sum(p.numel() for p in local_constructor_params)
                    global_integrator_count = sum(p.numel() for p in global_integrator_params)
                    hici_count = local_constructor_count + global_integrator_count
                    other_count = sum(p.numel() for p in other_params)
                    total_count = hici_count + other_count

                    print(f"\n  📊 HiCI Module Parameters Breakdown:")
                    print(f"  " + "-" * 68)
                    if local_constructor_count > 0:
                        print(f"    🧠 LocalConstructor: {local_constructor_count:,} ({local_constructor_count / total_count * 100:.2f}%)")
                    if global_integrator_count > 0:
                        print(f"    🏗️  GlobalIntegrator: {global_integrator_count:,} ({global_integrator_count / total_count * 100:.2f}%)")
                    print(f"  " + "-" * 68)
                    print(f"    📦 Total HiCI: {hici_count:,} LR={self.args.hici_lr:.2e}")
                    print(f"    📚 Other: {other_count:,} LR={self.args.learning_rate:.2e}")
                    print(f"    ⚡ LR Ratio: {self.args.hici_lr / self.args.learning_rate:.1f}x")
                    print("=" * 70 + "\n")
            else:
                if self.args.local_rank <= 0:
                    print("\n📌 Using uniform learning rate for all parameters")
                return super().create_optimizer()

        return self.optimizer

    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch)

        if (
            self.args.hici_grad_clip is not None
            and self.args.hici_lr is not None
        ):
            hici_params = []
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if "local_constructor" in name or "global_integrator" in name:
                        hici_params.append(param)

            if hici_params:
                torch.nn.utils.clip_grad_norm_(
                    hici_params, max_norm=self.args.hici_grad_clip
                )

            if not hasattr(self, "_grad_clip_printed"):
                if self.args.local_rank <= 0:
                    print(f"\n  🧠 HiCI gradient clip: {self.args.hici_grad_clip}")
                self._grad_clip_printed = True

        return loss


DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen3-8B")
    model_type: Optional[str] = field(default="qwen3")


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=32768,
        metadata={"help": "Maximum sequence length. Qwen3 native 32K context."},
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
    trainable_params: str = field(
        default="embed,norm",
        metadata={"help": "Additional trainable parameters except LoRA weights."},
    )
    num_local_slots: int = field(
        default=8,
        metadata={"help": "Number of local representation slots per segment (default: 8)."},
    )
    global_slots: int = field(
        default=16,
        metadata={"help": "Number of global context slots (default: 16)."},
    )
    use_local_constructor: bool = field(
        default=True,
        metadata={"help": "Whether to use LocalConstructor."},
    )
    use_global_integrator: bool = field(
        default=True,
        metadata={"help": "Whether to use GlobalIntegrator."},
    )
    use_local_constructor_flash: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to use LocalConstructorFlash (flash-attn, reuses K/V projections) instead of LocalConstructorMulti (standard PyTorch)."},
    )
    use_hierarchical_forward: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to use hierarchical forward (recommended)."},
    )
    use_attn_init: Optional[bool] = field(
        default=False,
        metadata={"help": "Initialize HiCI Q/K/V projections from pretrained weights."},
    )
    num_heads: int = field(
        default=32,
        metadata={"help": "Number of attention heads in HiCI module."},
    )
    use_bottleneck: bool = field(
        default=True,
        metadata={"help": "Whether to use bottleneck in GlobalIntegrator."},
    )
    bottleneck_dim: int = field(
        default=4096,
        metadata={"help": "Bottleneck dimension for GlobalIntegrator compression."},
    )
    hici_lr: Optional[float] = field(
        default=None,
        metadata={"help": "Separate learning rate for HiCI parameters. Recommended: 2e-4 to 5e-4."},
    )
    hici_grad_clip: Optional[float] = field(
        default=None,
        metadata={"help": "Gradient clipping for HiCI parameters. Recommended: 0.1 to 0.3."},
    )

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding."""
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def tokenize_fn(tokenizer, example):
    """
    Multi-process friendly tokenize function with smart grouping.
    """
    context_length = tokenizer.model_max_length
    MAX_CHARS_PER_SEGMENT = 200_000

    texts = example["text"]
    all_chunks = []

    current_batch = []
    current_length = 0

    for text in texts:
        text_len = len(text)

        if text_len > MAX_CHARS_PER_SEGMENT:
            if current_batch:
                combined = tokenizer.eos_token.join(current_batch)
                chunks = _tokenize_and_chunk(tokenizer, combined, context_length)
                all_chunks.extend(chunks)
                current_batch = []
                current_length = 0

            num_segments = (text_len + MAX_CHARS_PER_SEGMENT - 1) // MAX_CHARS_PER_SEGMENT
            for i in range(num_segments):
                start = i * MAX_CHARS_PER_SEGMENT
                end = min((i + 1) * MAX_CHARS_PER_SEGMENT, text_len)
                segment = text[start:end]
                chunks = _tokenize_and_chunk(tokenizer, segment, context_length)
                all_chunks.extend(chunks)

        elif current_length + text_len > MAX_CHARS_PER_SEGMENT and current_batch:
            combined = tokenizer.eos_token.join(current_batch)
            chunks = _tokenize_and_chunk(tokenizer, combined, context_length)
            all_chunks.extend(chunks)
            current_batch = [text]
            current_length = text_len
        else:
            current_batch.append(text)
            current_length += text_len

    if current_batch:
        combined = tokenizer.eos_token.join(current_batch)
        chunks = _tokenize_and_chunk(tokenizer, combined, context_length)
        all_chunks.extend(chunks)

    return {"input_ids": all_chunks}


def _tokenize_and_chunk(tokenizer, text, context_length):
    """Tokenize text and split into chunks."""
    outputs = tokenizer(
        text,
        truncation=False,
        return_tensors=None,
        padding=False,
    )

    input_ids = outputs["input_ids"]
    total_length = len(input_ids)
    if total_length % context_length != 0:
        padding_length = context_length - (total_length % context_length)
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        input_ids = input_ids + [pad_token_id] * padding_length

    num_chunks = len(input_ids) // context_length
    chunks = []
    for i in range(num_chunks):
        start = i * context_length
        end = start + context_length
        chunks.append(input_ids[start:end])

    return chunks


def train():
    parser = transformers.HfArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()

    # ========================================================================
    # Replace Qwen2 attention with HiCI
    # ========================================================================
    assert model_args.model_type == "qwen3", (
        f"This script only supports qwen3, got: {model_args.model_type}. "
        "For Qwen2/2.5, use fine-tune_hici_qwen.py instead."
    )
    replace_qwen3_attn(
        use_flash_attn=training_args.use_flash_attn,
        use_full=training_args.use_full_attn,
        use_hierarchical_forward=training_args.use_hierarchical_forward,
    )

    # ========================================================================
    # Set RoPE scaling factor (if target context > native context)
    # ========================================================================
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
            # scaling_factor = float(
            #     math.ceil(training_args.model_max_length / orig_ctx_len)
            # )
            scaling_factor = float(training_args.model_max_length / orig_ctx_len)
            config.rope_scaling = {"type": "linear", "factor": scaling_factor}
            rank = int(os.environ.get("RANK", 0))
            if rank == 0:
                print(f"\n🔧 RoPE scaling: {orig_ctx_len} -> {training_args.model_max_length} (factor={scaling_factor})")

    # ========================================================================
    # Load model and tokenizer
    # ========================================================================
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        torch_dtype=torch.bfloat16,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )

    # Qwen3 tokenizer typically has pad_token already set
    # But add special tokens if missing
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    if special_tokens_dict:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=special_tokens_dict,
            tokenizer=tokenizer,
            model=model,
        )

    # ========================================================================
    # Register HiCI modules (CRITICAL: Before optimizer initialization!)
    # ========================================================================
    register_hici_to_qwen3_model(
        model,
        num_local_slots=training_args.num_local_slots,
        global_slots=training_args.global_slots,
        num_heads=training_args.num_heads,
        use_bottleneck=training_args.use_bottleneck,
        bottleneck_dim=training_args.bottleneck_dim,
        use_local_constructor=training_args.use_local_constructor,
        use_global_integrator=training_args.use_global_integrator,
        use_local_constructor_flash=training_args.use_local_constructor_flash,
        use_attn_init=training_args.use_attn_init,
    )

    rank = int(os.environ.get("RANK", -1))
    if rank > 0:
        barrier()

    # ========================================================================
    # Load and process dataset
    # ========================================================================
    from datasets import load_from_disk, concatenate_datasets

    dataset_path = os.environ.get(
        "DATASET_PATH",
        "./cache/datasets",
    )

    if os.path.exists(dataset_path):
        dataset = load_from_disk(dataset_path)
    else:
        # Fallback: download RedPajama sample
        if rank <= 0:
            print(f"⚠️  Dataset not found at {dataset_path}, downloading RedPajama sample...")
        dataset = load_dataset(
            "togethercomputer/RedPajama-Data-1T-Sample",
            cache_dir=training_args.cache_dir,
        )

    # Smart grouping for tokenization
    # Only rank 0 uses parallel processing to avoid PyArrow cache conflicts
    proc_num = 128 if rank <= 0 else 1

    if rank <= 0:
        print("=" * 70)
        print("Processing dataset with smart grouping")
        print("=" * 70)

    very_short_docs = dataset.filter(lambda x: len(x["text"]) < 20_000, num_proc=proc_num)
    short_docs = dataset.filter(lambda x: 20_000 <= len(x["text"]) < 100_000, num_proc=proc_num)
    medium_docs = dataset.filter(lambda x: 100_000 <= len(x["text"]) < 300_000, num_proc=proc_num)
    long_docs = dataset.filter(lambda x: len(x["text"]) >= 300_000, num_proc=proc_num)

    # Get column names for removal
    remove_columns = [c for c in dataset["train"].column_names if c != "input_ids"]

    very_short_processed = very_short_docs.map(
        partial(tokenize_fn, tokenizer), batched=True, batch_size=200, num_proc=proc_num, remove_columns=remove_columns,
    )
    short_processed = short_docs.map(
        partial(tokenize_fn, tokenizer), batched=True, batch_size=40, num_proc=proc_num, remove_columns=remove_columns,
    )
    medium_processed = medium_docs.map(
        partial(tokenize_fn, tokenizer), batched=True, batch_size=5, num_proc=proc_num, remove_columns=remove_columns,
    )
    long_processed = long_docs.map(
        partial(tokenize_fn, tokenizer), batched=True, batch_size=1, num_proc=proc_num, remove_columns=remove_columns,
    )

    dataset = concatenate_datasets([
        very_short_processed["train"],
        short_processed["train"],
        medium_processed["train"],
        long_processed["train"],
    ])
    dataset = dataset.shuffle(seed=42)
    dataset = {"train": dataset}

    if rank <= 0:
        print(f"  Final samples: {len(dataset['train']):,}")
        print("=" * 70)

    if rank == 0:
        barrier()

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # ========================================================================
    # Setup LoRA
    # ========================================================================
    if training_args.low_rank_training:
        # Qwen2/3 uses same projection names as LLaMA
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

        # Enable trainable params (embed, norm, HiCI modules)
        [
            p.requires_grad_()
            for n, p in model.named_parameters()
            if any([k in n for k in training_args.trainable_params.split(",")])
        ]

    # ========================================================================
    # Verify trainable parameters
    # ========================================================================
    if rank <= 0:
        print("\n" + "=" * 70)
        print("Trainable Parameters Summary (Qwen3 HiCI)")
        print("=" * 70)

    trainable_params_dict = {}
    for n, p in model.named_parameters():
        if p.requires_grad:
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

    if rank <= 0:
        for category, count in sorted(trainable_params_dict.items()):
            print(f"  {category:20s}: {count:15,} params ({count / total_trainable * 100:5.2f}%)")

        if "HiCI Modules" in trainable_params_dict:
            lc_count = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and "local_constructor" in n)
            gi_count = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and "global_integrator" in n)
            if lc_count > 0:
                print(f"    {'└─ LocalConstructor':20s}: {lc_count:15,} params")
            if gi_count > 0:
                print(f"    {'└─ GlobalIntegrator':20s}: {gi_count:15,} params")

        print(f"  {'─' * 20}   {'─' * 15}   {'─' * 7}")
        print(f"  {'Total Trainable':20s}: {total_trainable:15,} params ({total_trainable / total_params * 100:.2f}% of total)")
        print(f"  {'Total Params':20s}: {total_params:15,} params")

    # Trainable params validation warnings
    has_hici_in_trainable = "local_constructor" in training_args.trainable_params or "global_integrator" in training_args.trainable_params
    has_hici_params = "HiCI Modules" in trainable_params_dict

    if rank <= 0:
        if has_hici_in_trainable and not has_hici_params:
            print(
                "\n⚠️  WARNING: 'local_constructor'/'global_integrator' specified in --trainable_params but no HiCI parameters found!"
            )
        elif not has_hici_in_trainable and has_hici_params:
            print(
                "\n⚠️  WARNING: HiCI module parameters found but not in --trainable_params!"
            )
            print(
                "    Add '--trainable_params \"embed,norm,local_constructor,global_integrator\"' to enable training."
            )

        has_global_integrator_in_trainable = "global_integrator" in training_args.trainable_params
        if training_args.use_global_integrator:
            if has_hici_in_trainable and not has_global_integrator_in_trainable:
                print(
                    "\n⚠️  WARNING: Using global integrator but 'global_integrator' not in --trainable_params!"
                )
                print("    GlobalIntegrator parameters may not be trained!")
                print(
                    "    Recommended: '--trainable_params \"embed,norm,local_constructor,global_integrator\"'"
                )

        print("=" * 70 + "\n")

    model.config.use_cache = False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    # Checkpoint resume detection
    if rank <= 0:
        import glob

        checkpoint_dirs = glob.glob(
            os.path.join(training_args.output_dir, "checkpoint-*")
        )
        if checkpoint_dirs:
            latest_checkpoint = max(
                checkpoint_dirs, key=lambda x: int(x.split("-")[-1])
            )
            print("\n" + "=" * 70)
            print("⚠️  Existing checkpoints detected")
            print("=" * 70)
            print(f"  Output dir: {training_args.output_dir}")
            print(f"  Found {len(checkpoint_dirs)} checkpoints")
            print(f"  Trainer will resume from: {latest_checkpoint}")
            print(f"  Local slots: {training_args.num_local_slots}")
            print("=" * 70 + "\n")
        else:
            print(f"✅ Training from scratch: {training_args.output_dir}")

    # ========================================================================
    # Initialize Trainer
    # ========================================================================
    trainer = LayeredLRTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=None,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
