#!/usr/bin/env python3
"""
QA script for inspecting and validating reasoning datasets.

Usage:
    python qa_dataset.py <dataset.jsonl> [--samples N] [--random] [--stats]
"""
import json
import random
import argparse
from pathlib import Path
from collections import Counter
import re


def get_stats(samples: list) -> dict:
    """Calculate dataset statistics."""
    stats = {
        "total_samples": len(samples),
        "avg_instruction_len": 0,
        "avg_output_len": 0,
        "avg_think_len": 0,
        "avg_answer_len": 0,
        "has_latex_pct": 0,
        "healed_count": 0,
        "topics": Counter(),
    }

    think_lens = []
    answer_lens = []
    latex_count = 0

    for s in samples:
        instruction = s.get("instruction", "")
        output = s.get("output", "")

        stats["avg_instruction_len"] += len(instruction)
        stats["avg_output_len"] += len(output)

        if s.get("healed"):
            stats["healed_count"] += 1

        # Extract think/answer sections
        think_match = re.search(r'<think>(.*?)</think>', output, re.DOTALL | re.IGNORECASE)
        answer_match = re.search(r'<answer>(.*?)</answer>', output, re.DOTALL | re.IGNORECASE)

        if think_match:
            think_lens.append(len(think_match.group(1)))
        if answer_match:
            answer_lens.append(len(answer_match.group(1)))

        # Check for LaTeX
        if re.search(r'\$[^$]+\$|\\\[.*?\\\]', output, re.DOTALL):
            latex_count += 1

        # Extract topic keywords
        keywords = ["qubit", "entangle", "hamiltonian", "eigenvalue", "spin",
                    "operator", "wave", "quantum", "state", "circuit", "gate",
                    "measurement", "superposition", "decoherence", "density"]
        for kw in keywords:
            if kw in instruction.lower():
                stats["topics"][kw] += 1

    n = len(samples)
    stats["avg_instruction_len"] = stats["avg_instruction_len"] // n if n else 0
    stats["avg_output_len"] = stats["avg_output_len"] // n if n else 0
    stats["avg_think_len"] = sum(think_lens) // len(think_lens) if think_lens else 0
    stats["avg_answer_len"] = sum(answer_lens) // len(answer_lens) if answer_lens else 0
    stats["has_latex_pct"] = round(100 * latex_count / n, 1) if n else 0

    return stats


def display_sample(idx: int, sample: dict, truncate: int = 500):
    """Display a single sample in readable format."""
    print(f"\n{'='*70}")
    print(f"SAMPLE {idx}")
    print('='*70)

    instruction = sample.get("instruction", "N/A")
    output = sample.get("output", "N/A")

    print(f"\n[INSTRUCTION] ({len(instruction)} chars)")
    print("-"*40)
    print(instruction[:truncate] + ("..." if len(instruction) > truncate else ""))

    # Extract and display think section
    think_match = re.search(r'<think>(.*?)</think>', output, re.DOTALL | re.IGNORECASE)
    if think_match:
        think = think_match.group(1).strip()
        print(f"\n[REASONING] ({len(think)} chars)")
        print("-"*40)
        print(think[:truncate] + ("..." if len(think) > truncate else ""))

    # Extract and display answer section
    answer_match = re.search(r'<answer>(.*?)</answer>', output, re.DOTALL | re.IGNORECASE)
    if answer_match:
        answer = answer_match.group(1).strip()
        print(f"\n[ANSWER] ({len(answer)} chars)")
        print("-"*40)
        print(answer[:300])

    # Check for healed samples
    if sample.get("healed"):
        print(f"\n[HEALED] Methods: {sample.get('heal_methods', [])}")


def main():
    parser = argparse.ArgumentParser(description="QA inspection for reasoning datasets")
    parser.add_argument("dataset", help="Path to JSONL dataset")
    parser.add_argument("-n", "--samples", type=int, default=5, help="Number of samples to show")
    parser.add_argument("-r", "--random", action="store_true", help="Random sampling")
    parser.add_argument("-s", "--stats", action="store_true", help="Show statistics only")
    parser.add_argument("--offset", type=int, default=0, help="Start from this sample index")

    args = parser.parse_args()

    # Load dataset
    samples = []
    with open(args.dataset, 'r') as f:
        for line in f:
            samples.append(json.loads(line.strip()))

    print(f"Loaded {len(samples)} samples from {args.dataset}")

    # Show stats
    stats = get_stats(samples)
    print(f"\n{'='*70}")
    print("DATASET STATISTICS")
    print('='*70)
    print(f"Total samples:        {stats['total_samples']}")
    print(f"Healed samples:       {stats['healed_count']} ({100*stats['healed_count']/len(samples):.1f}%)")
    print(f"Avg instruction len:  {stats['avg_instruction_len']} chars")
    print(f"Avg output len:       {stats['avg_output_len']} chars")
    print(f"Avg reasoning len:    {stats['avg_think_len']} chars")
    print(f"Avg answer len:       {stats['avg_answer_len']} chars")
    print(f"Has LaTeX:            {stats['has_latex_pct']}%")
    print(f"\nTop topics:")
    for topic, count in stats['topics'].most_common(10):
        print(f"  {topic}: {count} ({100*count/len(samples):.1f}%)")

    if args.stats:
        return

    # Select samples to display
    if args.random:
        indices = random.sample(range(len(samples)), min(args.samples, len(samples)))
    else:
        start = args.offset
        indices = list(range(start, min(start + args.samples, len(samples))))

    for idx in indices:
        display_sample(idx, samples[idx])

    print(f"\n{'='*70}")
    print("QA CHECKLIST")
    print('='*70)
    print("[ ] Reasoning is coherent and step-by-step")
    print("[ ] LaTeX formatting is correct")
    print("[ ] Answer matches the reasoning conclusion")
    print("[ ] No hallucinated physics (check key equations)")
    print("[ ] Appropriate depth for the question complexity")


if __name__ == "__main__":
    main()
