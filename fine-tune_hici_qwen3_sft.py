# Qwen3 HiCI SFT Training Script (TRL SFTTrainer + Chat Messages Format)
#
# Supports UltraChat and any HuggingFace chat-messages dataset natively.
# Uses DataCollatorForCompletionOnlyLM to only compute loss on assistant tokens.
#
# Key differences from fine-tune_hici_qwen3.py (pre-training):
#   - SFT on instruction-following data (not next-token prediction on PG19)
#   - Uses TRL SFTTrainer with Qwen3 ChatML chat template
#   - Loss only on assistant responses (user/system tokens masked)
#   - Supports HuggingFace Hub datasets (UltraChat) or local JSON
#
# Prerequisites:
#   pip install trl>=0.12.0
#
# Usage:
#   torchrun --nproc_per_node=8 fine-tune_hici_qwen3_sft.py \
#       --model_name_or_path Qwen/Qwen3-8B \
#       --dataset_name HuggingFaceH4/ultrachat_200k \
#       --dataset_split train_sft \
#       --output_dir ./checkpoints/qwen3-8b-hici-sft \
#       --max_seq_length 8192 \
#       --per_device_train_batch_size 1 \
#       --gradient_accumulation_steps 8 \
#       --learning_rate 2e-5 \
#       --max_steps 1000 \
#       --bf16 True \
#       --gradient_checkpointing True \
#       --deepspeed ds_configs/stage2.json \
#       --use_flash_attn True \
#       --low_rank_training True \
#       --trainable_params "embed,norm,local_constructor,global_integrator" \
#       --num_local_slots 8 --global_slots 16 --num_heads 32 \
#       --bottleneck_dim 4096 \
#       --hici_lr 2e-4 --hici_grad_clip 0.3

import os
from dataclasses import dataclass, field
from typing import Optional

import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, load_from_disk

try:
    from trl import SFTTrainer, SFTConfig
except ImportError:
    raise ImportError(
        "TRL is required for this script. Install it with:\n"
        "  pip install trl>=0.29.0\n"
        "Or use fine-tune_hici_sft.py for the legacy instruction/input/output format."
    )

from qwen3_attn_hici import (
    replace_qwen3_attn,
    register_hici_to_qwen3_model,
)
from peft import LoraConfig, get_peft_model


# ============================================================================
# Arguments
# ============================================================================


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="Qwen/Qwen3-8B")
    model_type: str = field(default="qwen3")


@dataclass
class HiCIArguments:
    """HiCI-specific arguments separated from training config."""

    # --- Attention ---
    use_flash_attn: bool = field(default=True)
    use_full_attn: bool = field(default=False)
    use_hierarchical_forward: bool = field(default=True)

    # --- Memory modules ---
    num_local_slots: int = field(default=8)
    global_slots: int = field(default=16)
    num_chunks: int = field(default=4)
    num_heads: int = field(default=32)
    use_bottleneck: bool = field(default=True)
    bottleneck_dim: int = field(default=4096)
    use_local_constructor: bool = field(default=True)
    use_global_integrator: bool = field(default=True)
    use_flash_plus: bool = field(default=True)
    use_attn_init: bool = field(default=False)
    recurrence_size: int = field(default=128)

    # --- Layered learning rate ---
    hici_lr: Optional[float] = field(
        default=None,
        metadata={"help": "Separate LR for memory modules. Recommended: 2e-4."},
    )
    hici_grad_clip: Optional[float] = field(
        default=None,
        metadata={"help": "Gradient clipping for memory modules. Recommended: 0.3."},
    )

    # --- LoRA ---
    low_rank_training: bool = field(default=True)
    lora_r: int = field(default=8)
    lora_alpha: int = field(default=16)
    trainable_params: str = field(
        default="embed,norm,local_constructor,global_integrator",
    )

    # --- Dataset ---
    dataset_name: str = field(
        default="HuggingFaceH4/ultrachat_200k",
        metadata={"help": "HuggingFace dataset name (with messages column)."},
    )
    dataset_split: str = field(default="train_sft")
    data_path: Optional[str] = field(
        default=None,
        metadata={"help": "Local JSON/JSONL path (overrides dataset_name). "
                  "Must have a 'messages' column."},
    )
    max_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Max number of training samples (for debugging)."},
    )

    # --- RoPE context extension ---
    model_max_length: Optional[int] = field(
        default=None,
        metadata={"help": "If set and > native context, applies RoPE linear scaling. "
                  "E.g., 49152 for Qwen3-8B (native 32K) → 1.5x scaling."},
    )

    # --- Pre-trained memory weights ---
    pretrained_memory_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to trainable_params.bin from HiCI pre-training stage 1. "
                  "Loads memory module weights instead of random init."},
    )


# ============================================================================
# HiCI SFT Trainer (layered LR + memory gradient clipping)
# ============================================================================


class HiCISFTTrainer(SFTTrainer):
    """SFTTrainer with layered learning rates for HiCI memory modules."""

    def __init__(self, hici_args: HiCIArguments, **kwargs):
        self.hici_args = hici_args
        super().__init__(**kwargs)

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        if self.hici_args.hici_lr is None:
            return super().create_optimizer()

        is_main = self.args.local_rank <= 0

        memory_params = []
        other_params = []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if "local_constructor" in n or "global_integrator" in n:
                memory_params.append(p)
            else:
                other_params.append(p)

        optimizer_grouped_parameters = [
            {
                "params": memory_params,
                "lr": self.hici_args.hici_lr,
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": other_params,
                "lr": self.args.learning_rate,
                "weight_decay": self.args.weight_decay,
            },
        ]

        optimizer_cls, optimizer_kwargs = self.__class__.get_optimizer_cls_and_kwargs(
            self.args
        )
        optimizer_kwargs.pop("lr", None)
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        if is_main:
            mem_count = sum(p.numel() for p in memory_params)
            other_count = sum(p.numel() for p in other_params)
            total = mem_count + other_count
            print(f"\n{'='*70}")
            print(f"Layered Learning Rates")
            print(f"{'='*70}")
            print(f"  Memory modules: {mem_count:,} params, LR={self.hici_args.hici_lr:.2e}")
            print(f"  Other params:   {other_count:,} params, LR={self.args.learning_rate:.2e}")
            print(f"  LR ratio: {self.hici_args.hici_lr / self.args.learning_rate:.1f}x")
            print(f"{'='*70}\n")

        return self.optimizer

    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch)

        if self.hici_args.hici_grad_clip and self.hici_args.hici_lr:
            mem_params = [
                p for n, p in model.named_parameters()
                if p.grad is not None
                and ("local_constructor" in n or "global_integrator" in n)
            ]
            if mem_params:
                torch.nn.utils.clip_grad_norm_(
                    mem_params, max_norm=self.hici_args.hici_grad_clip
                )

        return loss


# ============================================================================
# Utilities
# ============================================================================


def smart_tokenizer_and_embedding_resize(special_tokens_dict, tokenizer, model):
    """Resize tokenizer and embedding if new special tokens are added."""
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data
        input_embeddings[-num_new_tokens:] = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings[-num_new_tokens:] = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )


def print_trainable_summary(model, rank):
    """Print categorized trainable parameter summary."""
    if rank > 0:
        return

    categories = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "lora" in n.lower():
            cat = "LoRA"
        elif "local_constructor" in n or "global_integrator" in n:
            cat = "Memory Modules"
        elif "embed" in n.lower():
            cat = "Embeddings"
        elif "norm" in n.lower():
            cat = "LayerNorm"
        else:
            cat = "Other"
        categories[cat] = categories.get(cat, 0) + p.numel()

    total_trainable = sum(categories.values())
    total_params = sum(p.numel() for p in model.parameters())

    print(f"\n{'='*70}")
    print("Trainable Parameters (Qwen3 HiCI SFT)")
    print(f"{'='*70}")
    for cat, count in sorted(categories.items()):
        print(f"  {cat:20s}: {count:>12,} ({count/total_trainable*100:5.2f}%)")
    print(f"  {'─'*20}  {'─'*12}  {'─'*7}")
    print(f"  {'Trainable':20s}: {total_trainable:>12,} ({total_trainable/total_params*100:.2f}% of {total_params:,})")
    print(f"{'='*70}\n")


# ============================================================================
# Main
# ============================================================================


def train():
    parser = transformers.HfArgumentParser((ModelArguments, HiCIArguments, SFTConfig))
    model_args, hici_args, training_args = parser.parse_args_into_dataclasses()

    rank = int(os.environ.get("RANK", 0))

    # ==================================================================
    # 1. Replace Qwen3 attention with HiCI
    # ==================================================================
    replace_qwen3_attn(
        use_flash_attn=hici_args.use_flash_attn,
        use_full=hici_args.use_full_attn,
        use_hierarchical_forward=hici_args.use_hierarchical_forward,
    )

    # ==================================================================
    # 2. Load config and optionally set RoPE scaling
    # ==================================================================
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)

    if hici_args.model_max_length:
        orig_rope_scaling = getattr(config, "rope_scaling", None) or {"factor": 1}
        orig_factor = orig_rope_scaling.get("factor", 1)
        orig_ctx = getattr(config, "max_position_embeddings", 32768) * orig_factor

        if hici_args.model_max_length > orig_ctx:
            scaling_factor = float(hici_args.model_max_length / orig_ctx)
            config.rope_scaling = {"type": "linear", "factor": scaling_factor}
            if rank == 0:
                print(f"RoPE scaling: {int(orig_ctx)} -> {hici_args.model_max_length} (factor={scaling_factor:.2f})")

    # ==================================================================
    # 3. Load model
    # ==================================================================
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        torch_dtype=torch.bfloat16,
    )

    # ==================================================================
    # 4. Register HiCI memory modules
    # ==================================================================
    register_hici_to_qwen3_model(
        model,
        num_local_slots=hici_args.num_local_slots,
        global_slots=hici_args.global_slots,
        num_chunks=hici_args.num_chunks,
        num_heads=hici_args.num_heads,
        use_bottleneck=hici_args.use_bottleneck,
        bottleneck_dim=hici_args.bottleneck_dim,
        use_local_constructor=hici_args.use_local_constructor,
        use_global_integrator=hici_args.use_global_integrator,
        use_flash_plus=hici_args.use_flash_plus,
        use_attn_init=hici_args.use_attn_init,
    )

    # Optionally load pre-trained memory weights from stage 1
    if hici_args.pretrained_memory_path:
        if rank == 0:
            print(f"Loading pre-trained memory weights: {hici_args.pretrained_memory_path}")
        state_dict = torch.load(hici_args.pretrained_memory_path, map_location="cpu")
        # Only load memory-related keys
        memory_keys = {
            k: v for k, v in state_dict.items()
            if "local_constructor" in k or "global_integrator" in k
        }
        missing, unexpected = model.load_state_dict(memory_keys, strict=False)
        if rank == 0:
            print(f"  Loaded {len(memory_keys)} memory params, {len(missing)} missing, {len(unexpected)} unexpected")

    # ==================================================================
    # 5. Load tokenizer
    # ==================================================================
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        padding_side="right",
        use_fast=True,
    )

    # Qwen3 should have pad_token already, but handle edge cases
    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            {"pad_token": "<|endoftext|>"},
            tokenizer, model,
        )

    # ==================================================================
    # 6. Load dataset
    # ==================================================================
    if hici_args.data_path:
        if rank == 0:
            print(f"Loading local dataset: {hici_args.data_path}")
        if os.path.isdir(hici_args.data_path) and os.path.exists(
            os.path.join(hici_args.data_path, "dataset_info.json")
        ):
            # Arrow format saved by datasets.save_to_disk()
            dataset = load_from_disk(hici_args.data_path)
        else:
            # JSON/JSONL file with 'messages' column
            dataset = load_dataset("json", data_files=hici_args.data_path, split="train")
    else:
        # HuggingFace Hub dataset
        if rank == 0:
            print(f"Loading dataset: {hici_args.dataset_name} (split={hici_args.dataset_split})")
        dataset = load_dataset(hici_args.dataset_name, split=hici_args.dataset_split)

    # Subsample for debugging
    if hici_args.max_samples and len(dataset) > hici_args.max_samples:
        dataset = dataset.select(range(hici_args.max_samples))
        if rank == 0:
            print(f"  Subsampled to {hici_args.max_samples} examples")

    # Normalize column name: support both 'messages' and 'conversations'
    if "conversations" in dataset.column_names and "messages" not in dataset.column_names:
        dataset = dataset.rename_column("conversations", "messages")

    if "messages" not in dataset.column_names:
        raise ValueError(
            f"Dataset must have 'messages' or 'conversations' column. "
            f"Found: {dataset.column_names}"
        )

    # Keep only messages column to avoid SFTTrainer confusion with extra columns
    extra_cols = [c for c in dataset.column_names if c != "messages"]
    if extra_cols:
        dataset = dataset.remove_columns(extra_cols)

    if rank == 0:
        print(f"  Dataset size: {len(dataset)} examples")
        print(f"  First example messages: {len(dataset[0]['messages'])} turns")

    # ==================================================================
    # 7. Setup LoRA
    # ==================================================================
    if hici_args.low_rank_training:
        lora_config = LoraConfig(
            r=hici_args.lora_r,
            lora_alpha=hici_args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

        # Enable additional trainable params (embed, norm, memory modules)
        trainable_keys = hici_args.trainable_params.split(",")
        for n, p in model.named_parameters():
            if any(k in n for k in trainable_keys):
                p.requires_grad_(True)
    else:
        lora_config = None

    print_trainable_summary(model, rank)

    model.config.use_cache = False
    model.enable_input_require_grads()

    # ==================================================================
    # 8. Enable completion-only loss (only compute loss on assistant tokens)
    # ==================================================================
    # trl >= 0.29 handles this via SFTConfig.completion_only_loss + chat template.
    # The tokenizer's chat_template (ChatML for Qwen3) marks assistant vs user turns,
    # and SFTTrainer automatically masks non-assistant tokens in the loss.
    training_args.completion_only_loss = True

    if rank == 0:
        # Verify chat template works
        test_msgs = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]
        test_text = tokenizer.apply_chat_template(test_msgs, tokenize=False)
        print(f"Chat template sample: {test_text!r}")

    # ==================================================================
    # 9. Checkpoint resume detection
    # ==================================================================
    if rank == 0:
        import glob
        checkpoint_dirs = glob.glob(
            os.path.join(training_args.output_dir, "checkpoint-*")
        )
        if checkpoint_dirs:
            latest = max(checkpoint_dirs, key=lambda x: int(x.split("-")[-1]))
            print(f"\nExisting checkpoints detected ({len(checkpoint_dirs)} found)")
            print(f"  Trainer will resume from: {latest}\n")
        else:
            print(f"Training from scratch: {training_args.output_dir}")

    # ==================================================================
    # 10. Create trainer and train
    # ==================================================================
    trainer = HiCISFTTrainer(
        hici_args=hici_args,
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
