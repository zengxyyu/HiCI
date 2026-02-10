#!/usr/bin/env python3
"""Compare LongBench evaluation results and output LaTeX tables"""

import json
import os
import argparse

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

TASK_CATEGORIES = {
    "Single-Doc QA": ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh"],
    "Multi-Doc QA": ["hotpotqa", "2wikimqa", "musique", "dureader"],
    "Summarization": ["gov_report", "qmsum", "multi_news", "vcsum"],
    "Few-shot": ["trec", "triviaqa", "samsum", "lsht"],
    "Synthetic": ["passage_count", "passage_retrieval_en", "passage_retrieval_zh"],
    "Code": ["lcc", "repobench-p"],
}

EN_TASKS = [
    "narrativeqa", "qasper", "multifieldqa_en",
    "hotpotqa", "2wikimqa", "musique",
    "gov_report", "qmsum", "multi_news",
    "trec", "triviaqa", "samsum",
    "passage_count", "passage_retrieval_en",
    "lcc", "repobench-p"
]

ZH_TASKS = [
    "multifieldqa_zh", "dureader", "vcsum", "lsht", "passage_retrieval_zh"
]

REFERENCE_MODELS = {
    "GPT-3.5-Turbo-16k": {
        "Single-Doc QA": 45.1, "Multi-Doc QA": 36.2, "Summarization": 23.9,
        "Few-shot": 57.6, "Synthetic": 51.0, "Code": 54.1,
        "EN": 44.0, "ZH": 44.5, "All": 44.7
    },
    "Llama2-7B-chat-4k": {
        "Single-Doc QA": 21.7, "Multi-Doc QA": 18.2, "Summarization": 18.5,
        "Few-shot": 49.9, "Synthetic": 4.1, "Code": 48.1,
        "EN": 31.0, "ZH": 14.3, "All": 26.8
    },
    "LongChat-7B-32k": {
        "Single-Doc QA": 28.8, "Multi-Doc QA": 20.3, "Summarization": 22.5,
        "Few-shot": 50.8, "Synthetic": 13.0, "Code": 54.1,
        "EN": 34.3, "ZH": 23.9, "All": 31.6
    },
    "Vicuna-v1.5-7B-16k": {
        "Single-Doc QA": 31.8, "Multi-Doc QA": 18.8, "Summarization": 23.2,
        "Few-shot": 56.8, "Synthetic": 5.3, "Code": 47.3,
        "EN": 31.9, "ZH": 26.4, "All": 30.5
    },
    "LongLoRA-7B-16k": {
        "Single-Doc QA": 28.7, "Multi-Doc QA": 28.1, "Summarization": 27.8,
        "Few-shot": 63.7, "Synthetic": 16.7, "Code": 56.0,
        "EN": 36.8, "ZH": None, "All": None
    },
}


def load_results(model_name, pred_dir="pred"):
    result_path = os.path.join(pred_dir, model_name, "result.json")
    if os.path.exists(result_path):
        with open(result_path, "r") as f:
            return json.load(f)
    return None


def calc_scores(results):
    """Calculate category and EN/ZH/All scores"""
    scores = {}

    for category, tasks in TASK_CATEGORIES.items():
        task_scores = [results.get(task, 0) for task in tasks]
        scores[category] = sum(task_scores) / len(task_scores)

    en_scores = [results.get(task, 0) for task in EN_TASKS if task in results]
    zh_scores = [results.get(task, 0) for task in ZH_TASKS if task in results]
    all_scores = list(results.values())

    scores["EN"] = sum(en_scores) / len(en_scores) if en_scores else 0
    scores["ZH"] = sum(zh_scores) / len(zh_scores) if zh_scores else 0
    scores["All"] = sum(all_scores) / len(all_scores) if all_scores else 0

    return scores


def print_latex_table(models_results, highlight_models=None):
    """Output LaTeX format table"""
    categories = ["Single-Doc QA", "Multi-Doc QA", "Summarization", "Few-shot", "Synthetic", "Code"]
    overall_keys = ["EN", "ZH", "All"]

    print(r"""\begin{table*}[t]
\caption{Results (\%) on LongBench~\citep{bai2023longbench}.
We report category-level scores and overall performance on English (EN), Chinese (ZH), and All tasks.
Best in \textbf{bold}, second \underline{underlined}.}
\label{tab:longbench}
\centering
\begin{small}
\begin{tabular}{l|cccccc|ccc}
\toprule
\multirow{2}{*}{Model}
  & \multirow{2}{*}{Single-Doc QA}
  & \multirow{2}{*}{Multi-Doc QA}
  & \multirow{2}{*}{Summ}
  & \multirow{2}{*}{Few-shot}
  & \multirow{2}{*}{Synthetic}
  & \multirow{2}{*}{Code}
  & \multicolumn{3}{c}{Overall} \\
\cmidrule(lr){8-10}
  &  &  &  &  &  &  & EN & ZH & All \\
\midrule""")

    for model, scores in REFERENCE_MODELS.items():
        cells = []
        for cat in categories:
            val = scores.get(cat)
            cells.append(f"{val:.1f}" if val is not None else "--")
        for key in overall_keys:
            val = scores.get(key)
            cells.append(f"{val:.1f}" if val is not None else "--")

        if model == "GPT-3.5-Turbo-16k":
            print(f"{model}")
            print(f"  & {' & '.join(cells)} \\\\")
            print(r"\midrule")
        else:
            print(f"{model}")
            print(f"  & {' & '.join(cells)} \\\\")

    if models_results:
        print(r"\midrule")

    for model, results in models_results.items():
        scores = calc_scores(results)
        cells = []
        for cat in categories:
            cells.append(f"{scores[cat]:.1f}")
        for key in overall_keys:
            cells.append(f"{scores[key]:.1f}")

        if highlight_models and model in highlight_models:
            print(r"\rowcolor{cyan!8}")
            print(f"\\textbf{{{model}}}")
        else:
            print(f"{model}")
        print(f"  & {' & '.join(cells)} \\\\")

    print(r"""\bottomrule
\end{tabular}
\end{small}
\end{table*}""")


def print_console_table(models_results):
    """Output console format table"""
    categories = ["Single-Doc QA", "Multi-Doc QA", "Summarization", "Few-shot", "Synthetic", "Code"]
    overall_keys = ["EN", "ZH", "All"]
    headers = ["Model", "Single", "Multi", "Summ", "Few-shot", "Synth", "Code", "EN", "ZH", "All"]

    rows = []

    for model, scores in REFERENCE_MODELS.items():
        row = [model[:22]]
        for cat in categories:
            val = scores.get(cat)
            row.append(f"{val:.1f}" if val is not None else "--")
        for key in overall_keys:
            val = scores.get(key)
            row.append(f"{val:.1f}" if val is not None else "--")
        rows.append(row)

    rows.append(["---"] * len(headers))

    for model, results in models_results.items():
        scores = calc_scores(results)
        row = [model[:22]]
        for cat in categories:
            row.append(f"{scores[cat]:.1f}")
        for key in overall_keys:
            row.append(f"{scores[key]:.1f}")
        rows.append(row)

    if HAS_TABULATE:
        print("\n" + tabulate(rows, headers=headers, tablefmt="simple_outline"))
    else:
        print("\n" + " | ".join(headers))
        print("-" * 120)
        for row in rows:
            print(" | ".join(str(c) for c in row))


def main():
    parser = argparse.ArgumentParser(description="Compare LongBench results")
    parser.add_argument(
        "--models", nargs="+",
        default=["longalpaca-7b-16k", "my-sft-16k", "hici-7b-sft-16k-full-wrong", "hici-7b-sft-re-16k"],
        help="Model names",
    )
    parser.add_argument("--pred_dir", default="pred", help="Prediction directory")
    parser.add_argument("--latex", action="store_true", help="Output LaTeX format")
    parser.add_argument("--highlight", nargs="+", default=[], help="Models to highlight (LaTeX)")
    parser.add_argument("--add", nargs="+", default=[], help="Additional models to append")
    args = parser.parse_args()

    if args.add:
        args.models = args.models + args.add

    models_results = {}
    for model in args.models:
        results = load_results(model, args.pred_dir)
        if results:
            models_results[model] = results
            print(f"[OK] {model}", file=__import__('sys').stderr)
        else:
            print(f"[--] {model} (not found)", file=__import__('sys').stderr)

    if args.latex:
        print_latex_table(models_results, args.highlight)
    else:
        print_console_table(models_results)


if __name__ == "__main__":
    main()
