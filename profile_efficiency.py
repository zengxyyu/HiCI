"""
Efficiency Profiling: HiCI vs LongLoRA vs Full Attention
Measures FLOPs, Peak GPU Memory, Throughput, and Wall-clock Time
across various context lengths.

Usage:
    python profile_efficiency.py --model_path ./models/Llama-2-7b-hf --context_lengths 8192 16384 32768 65536 102400
    python profile_efficiency.py --model_path ./models/Qwen3-8B --model_type qwen --context_lengths 8192 16384 32768

Output:
    - Console table with all metrics
    - CSV file: efficiency_profile_results.csv
    - LaTeX table: efficiency_profile_table.tex

Each method runs in a separate subprocess to avoid monkey-patch conflicts.
"""

import os
import sys
import subprocess
import argparse
import csv
import json
import time
import gc
from typing import Optional

import torch
from transformers import AutoConfig


# ============================================================================
# Argument Parsing
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Efficiency profiling for HiCI vs baselines")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_type", type=str, default="llama", choices=["llama", "qwen"])
    parser.add_argument("--context_lengths", type=int, nargs="+", default=[8192, 16384, 32768, 65536])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--warmup_steps", type=int, default=2)
    parser.add_argument("--measure_steps", type=int, default=3)
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--num_local_slots", type=int, default=8)
    parser.add_argument("--global_slots", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--bottleneck_dim", type=int, default=512)
    parser.add_argument("--compress_dim", type=int, default=512)
    parser.add_argument("--shared_compress_dim", type=int, default=128)
    parser.add_argument("--methods", type=str, nargs="+",
                        default=["full_attn", "longlora", "hici"],
                        choices=["full_attn", "longlora", "hici"])
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"])
    # Internal: run a single method+context (called by subprocess)
    parser.add_argument("--_run_single", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_method", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--_seq_len", type=int, help=argparse.SUPPRESS)
    return parser.parse_args()


# ============================================================================
# FLOPs Estimation (Analytical)
# ============================================================================

def estimate_flops_full_attention(seq_len, hidden_dim, num_layers, num_heads, intermediate_size, vocab_size, batch_size=1):
    flops = 0
    for _ in range(num_layers):
        flops += 3 * 2 * batch_size * seq_len * hidden_dim * hidden_dim
        flops += 2 * 2 * batch_size * seq_len * seq_len * hidden_dim
        flops += 2 * batch_size * seq_len * hidden_dim * hidden_dim
        flops += 2 * 2 * batch_size * seq_len * hidden_dim * intermediate_size
        flops += 2 * batch_size * seq_len * intermediate_size * hidden_dim
    flops += 2 * batch_size * seq_len * hidden_dim * vocab_size
    return flops


def estimate_flops_longlora(seq_len, hidden_dim, num_layers, num_heads, intermediate_size, vocab_size,
                            group_size_ratio=0.25, batch_size=1):
    segment_size = int(seq_len * group_size_ratio)
    num_segments = seq_len // segment_size if segment_size > 0 else 1
    flops = 0
    for _ in range(num_layers):
        flops += 3 * 2 * batch_size * seq_len * hidden_dim * hidden_dim
        flops += num_segments * 2 * 2 * batch_size * segment_size * segment_size * hidden_dim
        flops += 2 * batch_size * seq_len * hidden_dim * hidden_dim
        flops += 2 * 2 * batch_size * seq_len * hidden_dim * intermediate_size
        flops += 2 * batch_size * seq_len * intermediate_size * hidden_dim
    flops += 2 * batch_size * seq_len * hidden_dim * vocab_size
    return flops


def estimate_flops_hici(seq_len, hidden_dim, num_layers, num_heads, intermediate_size, vocab_size,
                        num_local_slots=8, global_slots=5, bottleneck_dim=512,
                        compress_dim=512, shared_compress_dim=128,
                        group_size_ratio=0.25, batch_size=1):
    segment_size = int(seq_len * group_size_ratio)
    num_segments = seq_len // segment_size if segment_size > 0 else 1
    flops = 0
    for _ in range(num_layers):
        # QKV + Output projections (same as LongLoRA)
        flops += 3 * 2 * batch_size * seq_len * hidden_dim * hidden_dim
        flops += 2 * batch_size * seq_len * hidden_dim * hidden_dim

        # Stage 1: Local Construction (bottleneck cross-attention per segment)
        flops += num_segments * 2 * batch_size * num_local_slots * hidden_dim * bottleneck_dim  # Q proj
        flops += num_segments * 2 * 2 * batch_size * segment_size * hidden_dim * bottleneck_dim  # K,V proj
        flops += num_segments * 2 * 2 * batch_size * num_local_slots * segment_size * bottleneck_dim  # attn
        flops += num_segments * 2 * batch_size * num_local_slots * bottleneck_dim * hidden_dim  # out proj

        # Stage 2: Global Integration
        flops += 5 * 2 * batch_size * num_segments * num_local_slots * hidden_dim * shared_compress_dim
        flops += 2 * batch_size * (shared_compress_dim * 5) * compress_dim
        flops += 2 * batch_size * global_slots * num_segments * num_local_slots * compress_dim
        flops += 2 * batch_size * global_slots * bottleneck_dim * hidden_dim

        # Stage 3: Top-down Broadcast (segment attention with memory context)
        effective_kv_len = segment_size + global_slots + num_local_slots + 128
        flops += num_segments * 2 * 2 * batch_size * segment_size * effective_kv_len * hidden_dim

        # MLP (same)
        flops += 2 * 2 * batch_size * seq_len * hidden_dim * intermediate_size
        flops += 2 * batch_size * seq_len * intermediate_size * hidden_dim

    flops += 2 * batch_size * seq_len * hidden_dim * vocab_size
    return flops


# ============================================================================
# Single Method Profiling (runs in subprocess)
# ============================================================================

def run_single_profile(args):
    """Profile a single method at a single context length. Called in subprocess."""
    import torch
    import time
    import gc
    from transformers import AutoModelForCausalLM

    method = args._method
    seq_len = args._seq_len
    torch_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]

    print(f"  [{method}] seq_len={seq_len}: Loading model...", flush=True)

    try:
        # Apply attention replacement
        if method == "full_attn":
            from llama_attn_replace import replace_llama_attn
            replace_llama_attn(use_flash_attn=True, use_full=True)
        elif method == "longlora":
            from llama_attn_replace import replace_llama_attn
            replace_llama_attn(use_flash_attn=True, use_full=False)
        elif method == "hici":
            from llama_attn_hici import replace_llama_attn, register_hici_to_model
            replace_llama_attn(use_flash_attn=True, use_full=False, use_hierarchical_forward=True)

        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=torch_dtype, trust_remote_code=True,
        ).cuda()

        if method == "hici":
            register_hici_to_model(
                model,
                num_local_slots=args.num_local_slots,
                global_slots=args.global_slots,
                num_heads=args.num_heads,
                use_bottleneck=True,
                bottleneck_dim=args.bottleneck_dim,
                use_local_constructor=True,
                use_global_integrator=True,
                use_local_constructor_flash=False,
                use_shared_compressor=True,
                compress_dim=args.compress_dim,
                shared_compress_dim=args.shared_compress_dim,
            )
            # Move all memory modules to GPU
            for layer in model.model.layers:
                if hasattr(layer.self_attn, 'local_constructor'):
                    layer.self_attn.local_constructor = layer.self_attn.local_constructor.cuda()
                if hasattr(layer.self_attn, 'global_integrator'):
                    layer.self_attn.global_integrator = layer.self_attn.global_integrator.cuda()

        model.gradient_checkpointing_enable()
        model.train()

        total_params = sum(p.numel() for p in model.parameters())

        # Dummy input
        config = model.config
        input_ids = torch.randint(0, config.vocab_size, (args.batch_size, seq_len)).cuda()
        labels = input_ids.clone()

        # Warmup
        print(f"  [{method}] seq_len={seq_len}: Warmup ({args.warmup_steps} steps)...", flush=True)
        for _ in range(args.warmup_steps):
            outputs = model(input_ids=input_ids, labels=labels)
            outputs.loss.backward()
            model.zero_grad(set_to_none=True)

        # Measure time & memory
        print(f"  [{method}] seq_len={seq_len}: Measuring ({args.measure_steps} steps)...", flush=True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        times = []
        for _ in range(args.measure_steps):
            torch.cuda.synchronize()
            start = time.perf_counter()
            outputs = model(input_ids=input_ids, labels=labels)
            outputs.loss.backward()
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        avg_time = sum(times) / len(times)
        tokens_per_sec = (args.batch_size * seq_len) / avg_time

        # Measure FLOPs via profiler
        print(f"  [{method}] seq_len={seq_len}: Measuring FLOPs...", flush=True)
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA],
            with_flops=True,
        ) as prof:
            outputs = model(input_ids=input_ids, labels=labels)
            outputs.loss.backward()
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize()

        empirical_flops = sum(e.flops for e in prof.key_averages() if e.flops and e.flops > 0)

        # Output as JSON
        result = {
            "method": method,
            "context_length": seq_len,
            "peak_memory_mb": round(peak_mem_mb, 1),
            "peak_memory_gb": round(peak_mem_mb / 1024, 2),
            "avg_time_sec": round(avg_time, 3),
            "tokens_per_sec": round(tokens_per_sec, 1),
            "empirical_tflops": round(empirical_flops / 1e12, 2),
            "total_params_m": round(total_params / 1e6, 1),
        }
        print(f"RESULT_JSON:{json.dumps(result)}", flush=True)

    except torch.cuda.OutOfMemoryError:
        result = {
            "method": method,
            "context_length": seq_len,
            "peak_memory_mb": "OOM", "peak_memory_gb": "OOM",
            "avg_time_sec": "OOM", "tokens_per_sec": "OOM",
            "empirical_tflops": "OOM", "total_params_m": "N/A",
        }
        print(f"RESULT_JSON:{json.dumps(result)}", flush=True)

    except Exception as e:
        print(f"ERROR: {method} seq_len={seq_len}: {e}", flush=True)
        result = {
            "method": method, "context_length": seq_len,
            "peak_memory_mb": "ERROR", "peak_memory_gb": "ERROR",
            "avg_time_sec": "ERROR", "tokens_per_sec": "ERROR",
            "empirical_tflops": "ERROR", "total_params_m": "N/A",
        }
        print(f"RESULT_JSON:{json.dumps(result)}", flush=True)


# ============================================================================
# Output Formatting
# ============================================================================

def print_results_table(all_results):
    print("\n" + "=" * 130)
    print(f"{'Method':<12} {'Context':<10} {'Peak Mem (GB)':<15} {'Time (s)':<12} "
          f"{'Throughput':<15} {'FLOPs-Emp (T)':<15} {'FLOPs-Ana (T)':<15} {'Params (M)':<12}")
    print("=" * 130)
    for r in all_results:
        print(f"{r['method']:<12} {r['context_length']:<10} {str(r.get('peak_memory_gb','')):<15} "
              f"{str(r.get('avg_time_sec','')):<12} {str(r.get('tokens_per_sec','')):<15} "
              f"{str(r.get('empirical_tflops','')):<15} {str(r.get('analytical_tflops','')):<15} "
              f"{str(r.get('total_params_m','')):<12}")
    print("=" * 130)


def save_csv(all_results, output_path):
    if not all_results:
        return
    keys = list(all_results[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nCSV saved to: {output_path}")


def save_latex_table(all_results, output_path):
    methods = sorted(set(r["method"] for r in all_results))
    context_lengths = sorted(set(r["context_length"] for r in all_results))

    lookup = {}
    for r in all_results:
        lookup[(r["method"], r["context_length"])] = r

    with open(output_path, "w") as f:
        n_ctx = len(context_lengths)
        col_spec = "l" + "rrr" * n_ctx

        f.write("\\begin{table}[t]\n\\centering\n")
        f.write("\\caption{Efficiency comparison across context lengths (LLaMA-2-7B, single GPU, bf16). "
                "FLOPs measured via \\texttt{torch.profiler}.}\n")
        f.write("\\label{tab:efficiency}\n")
        f.write("\\resizebox{\\textwidth}{!}{\n")
        f.write(f"\\begin{{tabular}}{{{col_spec}}}\n\\toprule\n")

        # Header
        header1 = "\\multirow{2}{*}{Method}"
        for ctx in context_lengths:
            label = f"{ctx // 1024}K" if ctx % 1024 == 0 else f"{ctx}"
            header1 += f" & \\multicolumn{{3}}{{c}}{{{label}}}"
        f.write(header1 + " \\\\\n")

        for i in range(n_ctx):
            start = 2 + i * 3
            f.write(f"\\cmidrule(lr){{{start}-{start+2}}}")
        f.write("\n")

        header2 = ""
        for _ in context_lengths:
            header2 += " & Mem (GB) & Tok/s & TFLOPs"
        f.write(header2 + " \\\\\n\\midrule\n")

        method_display = {"full_attn": "Full Attention", "longlora": "LongLoRA", "hici": "\\textbf{HiCI (Ours)}"}
        for method in ["full_attn", "longlora", "hici"]:
            if method not in methods:
                continue
            row = method_display.get(method, method)
            for ctx in context_lengths:
                r = lookup.get((method, ctx))
                if r and r.get("peak_memory_gb") not in ("OOM", "ERROR", None):
                    row += f" & {r['peak_memory_gb']} & {r['tokens_per_sec']} & {r.get('empirical_tflops', '--')}"
                else:
                    row += " & OOM & -- & --"
            f.write(row + " \\\\\n")

        f.write("\\bottomrule\n\\end{tabular}\n}\n\\end{table}\n")

    print(f"LaTeX table saved to: {output_path}")


# ============================================================================
# Main (Orchestrator)
# ============================================================================

def main():
    args = parse_args()

    # If called as subprocess worker, run single profile
    if args._run_single:
        run_single_profile(args)
        return

    print("=" * 60)
    print("Efficiency Profiling: HiCI vs LongLoRA vs Full Attention")
    print("=" * 60)

    # Model config
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    model_config = {
        "hidden_dim": config.hidden_size,
        "num_layers": config.num_hidden_layers,
        "num_heads": config.num_attention_heads,
        "intermediate_size": config.intermediate_size,
        "vocab_size": config.vocab_size,
    }
    print(f"\nModel: {args.model_path}")
    for k, v in model_config.items():
        print(f"  {k}: {v}")
    print(f"\nContext lengths: {args.context_lengths}")
    print(f"Methods: {args.methods}")
    print(f"Dtype: {args.dtype}")

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")

    os.makedirs(args.output_dir, exist_ok=True)
    all_results = []

    # Run each (method, context_length) in a separate subprocess
    for method in args.methods:
        for seq_len in args.context_lengths:
            print(f"\n{'─' * 60}")
            print(f"Profiling: {method} @ {seq_len} tokens")
            print(f"{'─' * 60}")

            cmd = [
                sys.executable, __file__,
                "--_run_single",
                "--_method", method,
                "--_seq_len", str(seq_len),
                "--model_path", args.model_path,
                "--model_type", args.model_type,
                "--batch_size", str(args.batch_size),
                "--warmup_steps", str(args.warmup_steps),
                "--measure_steps", str(args.measure_steps),
                "--dtype", args.dtype,
                "--num_local_slots", str(args.num_local_slots),
                "--global_slots", str(args.global_slots),
                "--num_heads", str(args.num_heads),
                "--bottleneck_dim", str(args.bottleneck_dim),
                "--compress_dim", str(args.compress_dim),
                "--shared_compress_dim", str(args.shared_compress_dim),
            ]

            try:
                proc = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=1800,
                    env={**os.environ, "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0")},
                )

                # Print subprocess output
                for line in proc.stdout.splitlines():
                    if line.startswith("RESULT_JSON:"):
                        result = json.loads(line[len("RESULT_JSON:"):])
                        # Add analytical FLOPs
                        if method == "full_attn":
                            ana = estimate_flops_full_attention(seq_len, **model_config, batch_size=args.batch_size)
                        elif method == "longlora":
                            ana = estimate_flops_longlora(seq_len, **model_config, batch_size=args.batch_size)
                        elif method == "hici":
                            ana = estimate_flops_hici(seq_len, **model_config,
                                                      num_local_slots=args.num_local_slots,
                                                      global_slots=args.global_slots,
                                                      bottleneck_dim=args.bottleneck_dim,
                                                      compress_dim=args.compress_dim,
                                                      shared_compress_dim=args.shared_compress_dim,
                                                      batch_size=args.batch_size)
                        result["analytical_tflops"] = round(ana / 1e12, 2)
                        all_results.append(result)

                        status = "OOM" if result.get("peak_memory_gb") == "OOM" else "OK"
                        if status == "OK":
                            print(f"  -> Mem: {result['peak_memory_gb']} GB | "
                                  f"Time: {result['avg_time_sec']}s | "
                                  f"Tok/s: {result['tokens_per_sec']} | "
                                  f"FLOPs(emp): {result['empirical_tflops']}T | "
                                  f"FLOPs(ana): {result['analytical_tflops']}T")
                        else:
                            print(f"  -> OOM")
                    else:
                        print(f"  {line}")

                if proc.stderr:
                    # Only print errors, skip warnings
                    for line in proc.stderr.splitlines():
                        if "error" in line.lower() or "traceback" in line.lower():
                            print(f"  STDERR: {line}")

                if not any(r["method"] == method and r["context_length"] == seq_len for r in all_results):
                    print(f"  -> No result captured, check stderr")
                    all_results.append({
                        "method": method, "context_length": seq_len,
                        "peak_memory_gb": "ERROR", "avg_time_sec": "ERROR",
                        "tokens_per_sec": "ERROR", "empirical_tflops": "ERROR",
                        "analytical_tflops": round(ana / 1e12, 2) if 'ana' in dir() else "N/A",
                        "total_params_m": "N/A",
                    })

            except subprocess.TimeoutExpired:
                print(f"  -> TIMEOUT (>30min)")
                all_results.append({
                    "method": method, "context_length": seq_len,
                    "peak_memory_gb": "TIMEOUT", "avg_time_sec": "TIMEOUT",
                    "tokens_per_sec": "TIMEOUT", "empirical_tflops": "TIMEOUT",
                    "analytical_tflops": "N/A", "total_params_m": "N/A",
                })

    # Summary
    print_results_table(all_results)

    csv_path = os.path.join(args.output_dir, "efficiency_profile_results.csv")
    tex_path = os.path.join(args.output_dir, "efficiency_profile_table.tex")
    save_csv(all_results, csv_path)
    save_latex_table(all_results, tex_path)

    # Analytical summary
    print("\n" + "=" * 60)
    print("Analytical FLOPs Summary (forward only)")
    print("=" * 60)
    for ctx in args.context_lengths:
        full = estimate_flops_full_attention(ctx, **model_config) / 1e12
        longlora = estimate_flops_longlora(ctx, **model_config) / 1e12
        hici = estimate_flops_hici(ctx, **model_config,
                                    num_local_slots=args.num_local_slots,
                                    global_slots=args.global_slots,
                                    bottleneck_dim=args.bottleneck_dim,
                                    compress_dim=args.compress_dim,
                                    shared_compress_dim=args.shared_compress_dim) / 1e12
        label = f"{ctx//1024}K" if ctx % 1024 == 0 else f"{ctx}"
        print(f"  {label}: Full={full:.2f}T  LongLoRA={longlora:.2f}T ({longlora/full*100:.1f}%)  "
              f"HiCI={hici:.2f}T ({hici/full*100:.1f}%)")

    print(f"\nResults saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
