#!/usr/bin/env python3
"""Extract training loss from trainer_state.json and plot, supports multi-model comparison"""

import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys
import os

# ============================================================
# ICML 2026 font settings (Times / STIX, matches LaTeX mathptmx)
# ============================================================
matplotlib.rcParams.update(
    {
        # Font: STIX (equivalent to LaTeX Times)
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        # Font size: ICML body 10pt, figure recommended 8-9pt
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        # Lines
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        # Remove top and right spines
        "axes.spines.top": False,
        "axes.spines.right": False,
        # PDF/SVG use vector fonts
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.unicode_minus": False,
    }
)

# ICML column width: single=3.25in, double=6.75in
ICML_SINGLE_COL = 3.25
ICML_DOUBLE_COL = 6.75

def load_losses(checkpoint_dir):
    """Load loss data from checkpoint directory"""
    json_path = os.path.join(checkpoint_dir, "trainer_state.json")
    with open(json_path) as f:
        state = json.load(f)
    logs = [e for e in state["log_history"] if "loss" in e and "train_loss" not in e]
    steps = np.array([e["step"] for e in logs])
    losses = np.array([e["loss"] for e in logs])
    return steps, losses

def smooth(data, window=1):
    """Moving average smoothing"""
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode="valid")

def plot_comparison(runs, output_dir="./attention_viz", smooth_window=15):
    """Compare training loss across multiple models.

    Args:
        runs: list of (name, checkpoint_dir) tuples
        output_dir: Output directory
        smooth_window: Smoothing window
    """
    os.makedirs(output_dir, exist_ok=True)

    # ============ Color config ============
    # HiCI (pink): 3 shades from light to dark pink
    # S²-Attn (blue): 3 shades from dark to medium blue
    colors = [
        # "#f9d2e7",  # HiCI 8k S=1024 - light pink
        # "#FDE2E4",  # HiCI 8k S=2048 - medium pink
        # "#C96E7B",  # HiCI 16k S=1024 - dark pink  FDE2E4 BDD7DE
        # "#274a78",  # S²-Attn 8k S=1024 - dark blue
        # "#d3e8ff",  # S²-Attn 8k S=2048 - medium blue
        # "#583722",  # S²-Attn 16k S=1024 - light blue
        "#7dba7f",  # HiCI 8k S=1024 - light pink
        "#b3d7ae",  # HiCI 8k S=2048 - medium pink ddeed9
        "#def2da",  # HiCI 16k S=1024 - dark pink  FDE2E4 BDD7DE
        "#225b91",  # S²-Attn 8k S=1024 - dark blue
        "#78aac8",  # S²-Attn 8k S=2048 - medium blue dae6f1 225b91
        "#e0effe",  # S²-Attn 16k S=1024 - light blue 225b91 dae6f1
    ]

    # ============ Figure: double column width ============
    fig, ax = plt.subplots(figsize=(ICML_DOUBLE_COL, 2.6))

    # First pass: draw lines and collect endpoints
    last_steps = None
    endpoints = []  # (final_loss, final_step, color, name)

    for i, (name, ckpt_dir) in enumerate(runs):
        steps, losses = load_losses(ckpt_dir)
        last_steps = steps
        color = colors[i % len(colors)]

        # Raw data (semi-transparent band)
        ax.plot(steps, losses, "-", alpha=0.10, color=color, linewidth=0.3)

        # Smoothed curve
        smooth_losses = smooth(losses, smooth_window)
        smooth_steps = steps[smooth_window - 1 :]
        ax.plot(
            smooth_steps,
            smooth_losses,
            "-",
            color=color,
            linewidth=1.4,
            label=name,
        )

        # Collect endpoint info
        endpoints.append((smooth_losses[-1], smooth_steps[-1], color, name))

        # Print statistics
        print(f"\n{name}:")
        print(f"  Steps: {len(steps)}, Loss: {losses[0]:.4f} -> {losses[-1]:.4f}")
        for s, e in [(1, 100), (900, 1000), (1900, 2000)]:
            mask = (steps >= s) & (steps <= e)
            if mask.any():
                print(f"  Steps {s:4d}-{e:4d}: avg={losses[mask].mean():.4f}")

    # Second pass: annotate only max and min endpoints
    final_losses = [ep[0] for ep in endpoints]
    min_idx = np.argmin(final_losses)
    max_idx = np.argmax(final_losses)

    for idx in [min_idx, max_idx]:
        final_loss, final_step, color, name = endpoints[idx]
        ax.annotate(
            f"{final_loss:.2f}",
            xy=(final_step, final_loss),
            xytext=(8, 0),
            textcoords="offset points",
            fontsize=7,
            color=color,
            va="center",
        )

    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Training Loss")
    ax.legend(loc="upper right", frameon=False)
    ax.grid(axis="y", alpha=0.15, linestyle="-", linewidth=0.3)
    ax.set_xlim(0, last_steps[-1] + 50)

    plt.tight_layout(pad=0.3)
    save_path = os.path.join(output_dir, "training_loss_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.savefig(save_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {save_path}")
    print(f"Saved: {save_path.replace('.png', '.pdf')}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  # Single model")
        print("  python plot_training_loss.py <checkpoint_dir>")
        print("  # Multi-model comparison")
        print("  python plot_training_loss.py <name1>:<dir1> <name2>:<dir2> ...")
        print()
        print("Example (HiCI pink vs S²-Attn blue, 6 models):")
        print("  python plot_training_loss.py \\")
        print("    'HiCI 8k S=1024':./checkpoints/Llama-2-7b-8k-FTM-NEW-84-..._G8 \\")
        print("    'HiCI 8k S=2048':./checkpoints/Llama-2-7b-8k-FTM-NEW-84-... \\")
        print(
            "    'HiCI 16k S=1024':./checkpoints/Llama-2-7b-16k-FTM-NEW-84-..._G16 \\"
        )
        print("    'S²-Attn 8k S=1024':./checkpoints/Llama-2-7b-8k-longlora-G8 \\")
        print("    'S²-Attn 8k S=2048':./checkpoints/Llama-2-7b-8k-longlora-G4 \\")
        print("    'S²-Attn 16k S=1024':./checkpoints/Llama-2-7b-16k-longlora-G16")
        sys.exit(1)

    runs = []
    for arg in sys.argv[1:]:
        if ":" in arg and not arg.startswith("/") and not arg.startswith("./"):
            # name:dir format
            name, dir_path = arg.split(":", 1)
            runs.append((name, dir_path))
        else:
            # Only path, use dir name
            name = os.path.basename(arg.rstrip("/"))
            runs.append((name, arg))

    plot_comparison(runs)
