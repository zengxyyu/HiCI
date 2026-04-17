#!/usr/bin/env python3
"""
Manual evaluation script — simple string matching.
Computes accuracy for the Topic Retrieval task.

Usage:
    python topic_retrieval_manual_eval.py <prediction_file>

Example:
    python topic_retrieval_manual_eval.py /path/to/5_response.txt
"""
import sys

if len(sys.argv) < 2:
    print("Usage: python topic_retrieval_manual_eval.py <prediction_file>")
    sys.exit(1)

with open(sys.argv[1], 'r') as f:
    lines = f.readlines()

correct = 0
total = len(lines)

print(f"\nFile   : {sys.argv[1]}")
print(f"Samples: {total}")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

for line in lines:
    parts = line.strip().split(', Predict: ')
    if len(parts) != 2:
        continue

    label   = parts[0].replace('Label: ', '').lower().strip()
    predict = parts[1].split(', prompt length:')[0].strip('[]\'"').lower().strip()

    # Correct if prediction contains the label
    if label in predict:
        correct += 1
        # print(f"✓ {label[:40]}")
    else:
        print(f"✗ Label  : {label}")
        print(f"  Predict: {predict}")
        print()

accuracy = correct / total if total > 0 else 0
print(f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print(f"Accuracy: {correct}/{total} = {accuracy:.2%}")
print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
