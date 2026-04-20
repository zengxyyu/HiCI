#!/bin/bash
# bash train_fine_tune_hici_sft.sh 2>&1 | tee Train_out_sft/Llama-2-7b-16k-SFT-hici-16.txt
pkill -9 -f "fine-tune_hici_sft.py"
fuser -k 38493/tcp 2>/dev/null || echo "✅ Port 38493 not in use"
sleep 2

# Base configuration
# MODEL_PATH="/scratch/sh89/xz2053/projects/llm-memory/models/Llama-2-7b-hf"
MODEL_PATH="./models/Llama-2-7b-hf"
# MODEL_PATH="./models/Llama-2-7b-chat-hf"
# RESUME_CHECKPOINT="/scratch/sh89/xz2053/projects/llm-memory/checkpoints/Llama-2-7b-8k-hici-causal_gi-G4/checkpoint-1000"
RESUME_CHECKPOINT="./checkpoints/Llama-2-7b-hici-16k-none/checkpoint-1000"
OUTPUT_DIR="./checkpoints/Llama-2-7b-16k-SFT-hici-16"
MAX_LENGTH=16384  # SFT typically uses 8192 or 16384; 32768 not needed
# DATA_PATH="/scratch/sh89/xz2053/projects/llm-memory/data/sft/LongAlpaca-12k.json"
DATA_PATH="./data/sft/LongAlpaca-12k.json"
# DATA_PATH="./data/sft/LongAlpaca-16k-length/LongAlpaca-16k-length.json"

# Training hyperparameters
nproc_per_node=4
gradient_accumulation_steps=16
WARMUP_STEPS=20
NUM_EPOCHS=15
MAX_STEPS=3000  # -1 = auto-computed from epochs; or set a fixed value
low_rank_training=True  # whether to use low-rank training (LongLoRA)

# HiCI module configuration
use_local_constructor=True
use_global_integrator=True
NUM_LOCAL_SLOTS=8  # Local Representation Slots
global_slots=4     # Global Representation Slots
num_heads=8        # number of attention heads
use_bottleneck=True
bottleneck_dim=512        # bottleneck dimension
shared_compress_dim=128   # shared compress dim (128 for 7B, 160 for 13B)

# HiCI learning rate and gradient clipping
hici_lr=2e-4
hici_grad_clip=0.3

# Trainable parameters
# TRAINABLE_PARAMS="embed,norm"
TRAINABLE_PARAMS="embed,norm,local_constructor,global_integrator"

# LocalConstructor type
use_llama_init=False  # whether to init Q/K/V from LLaMA weights
use_local_constructor_flash=False
use_hierarchical_forward=True

deepspeed_config="ds_configs/stage2.json"  # Stage 2: 24GB VRAM; Stage 3: 16GB VRAM

echo "========================================"
echo "🔥 Supervised Fine-Tuning (SFT) with HiCI"
echo "========================================"
echo ""
echo "📦 Base configuration:"
echo "  - 📦 Base model:         $MODEL_PATH"
echo "  - 🔄 Resume checkpoint:  $RESUME_CHECKPOINT"
echo "  - 📁 Output dir:         $OUTPUT_DIR"
echo "  - 📊 Dataset:            $DATA_PATH"
echo "  - 🤖 GPUs:               $nproc_per_node"
echo "  - 📈 Grad accumulation:  $gradient_accumulation_steps"
echo "  - 📏 Max length:         $MAX_LENGTH"
echo "  - 🔄 Epochs:             $NUM_EPOCHS"
echo "  - 📈 Max steps:          $MAX_STEPS  (-1 = auto from epochs)"
echo "  - 🔥 Warmup steps:       $WARMUP_STEPS"
echo "  - ⚙️  Trainable params:   $TRAINABLE_PARAMS"
echo "  - 💾 DeepSpeed:          $deepspeed_config"
echo "  - 🎯 LoRA training:      $low_rank_training"
echo ""
echo "🧠 HiCI Configuration:"
echo "  - 📝 LocalConstructor:   $use_local_constructor"
echo "  - 🔁 GlobalIntegrator:   $use_global_integrator"
echo "  - 🌐 Global slots:       $global_slots"
echo "  - 🧠 Local slots:        $NUM_LOCAL_SLOTS"
echo "  - 💡 HiCI LR:            $hici_lr"
echo "  - 🤖 HiCI grad clip:     $hici_grad_clip"
echo "  - 🎯 Bottleneck:         $use_bottleneck"
echo "  - 📊 Bottleneck dim:     $bottleneck_dim"
echo "  - 🧩 Shared compress:    $shared_compress_dim"
echo "  - 🧠 Init from LLaMA:    $use_llama_init"
echo ""
echo "⚙️  Forward Function:"
echo "  - 🧠 LocalConstructorFlash: $use_local_constructor_flash"
echo "  - 📝 Hierarchical fwd:   $use_hierarchical_forward"
echo ""
echo "========================================"

# --resume_from_checkpoint $RESUME_CHECKPOINT \
torchrun --nproc_per_node $nproc_per_node \
      --master_port=38493 \
      fine-tune_hici_sft.py \
      --model_name_or_path $MODEL_PATH \
      --resume_from_checkpoint $RESUME_CHECKPOINT \
      --data_path $DATA_PATH \
      --bf16 True \
      --output_dir $OUTPUT_DIR \
      --cache_dir ./cache \
      --model_max_length $MAX_LENGTH \
      --use_flash_attn True \
      --low_rank_training $low_rank_training \
      --num_train_epochs $NUM_EPOCHS \
      --per_device_train_batch_size 1 \
      --per_device_eval_batch_size 2 \
      --gradient_accumulation_steps $gradient_accumulation_steps \
      --evaluation_strategy "no" \
      --save_strategy "steps" \
      --save_steps 500 \
      --save_total_limit 4 \
      --learning_rate 2e-5 \
      --weight_decay 0.0 \
      --warmup_steps $WARMUP_STEPS \
      --lr_scheduler_type "constant_with_warmup" \
      --logging_steps 1 \
      --deepspeed $deepspeed_config \
      --tf32 True \
      --max_steps $MAX_STEPS \
      --num_local_slots $NUM_LOCAL_SLOTS \
      --global_slots $global_slots \
      --use_local_constructor $use_local_constructor \
      --use_global_integrator $use_global_integrator \
      --num_heads $num_heads \
      --use_bottleneck $use_bottleneck \
      --bottleneck_dim $bottleneck_dim \
      --shared_compress_dim $shared_compress_dim \
      --trainable_params $TRAINABLE_PARAMS \
      --hici_lr $hici_lr \
      --hici_grad_clip $hici_grad_clip \
      --use_local_constructor_flash $use_local_constructor_flash \
      --use_llama_init $use_llama_init \
      --use_hierarchical_forward $use_hierarchical_forward \

echo ""
echo "========================================"
echo "✅ SFT Training Completed!"
echo "========================================"
echo "📁 Checkpoints saved to: $OUTPUT_DIR"
