#!/usr/bin/env python3
"""
analyse.py - Fingerprint Ollama responses and report on determinism skew.

Usage:
    python analyse.py [--input-dir DIR] [--verbose]
"""

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path


def hash_text(text: str) -> str:
    """SHA-256 of the exact text — catches any byte-level difference."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def hash_normalised(text: str) -> str:
    """SHA-256 after stripping whitespace/blank lines — catches semantic sameness."""
    normalised = "\n".join(line.rstrip() for line in text.splitlines() if line.strip())
    return hashlib.sha256(normalised.encode()).hexdigest()[:16]


def extract_code_block(text: str) -> str:
    """Pull out the first fenced code block if present, otherwise return full text."""
    match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()


def hash_code_only(text: str) -> str:
    """Hash just the extracted code block, ignoring any surrounding prose."""
    return hashlib.sha256(extract_code_block(text).encode()).hexdigest()[:16]


def bar(count: int, total: int, width: int = 40) -> str:
    filled = int(width * count / total) if total else 0
    return "█" * filled + "░" * (width - filled)


def print_distribution(label: str, counter: Counter, total: int, verbose: bool):
    unique = len(counter)
    most_common_count = counter.most_common(1)[0][1] if counter else 0
    print(f"\n{'═' * 60}")
    print(f"  {label}")
    print(f"{'═' * 60}")
    print(f"  Total runs:    {total}")
    print(f"  Unique hashes: {unique}")
    print(f"  Deterministic: {'YES ✓' if unique == 1 else f'NO ✗  ({unique} variants)'}")
    if unique > 1:
        pct = (most_common_count / total) * 100
        print(f"  Most common:   {most_common_count}/{total} runs ({pct:.1f}%)")
    print()

    if verbose or unique <= 20:
        print(f"  {'Hash':<18} {'Count':>6}  {'%':>6}  Distribution")
        print(f"  {'─'*18} {'─'*6}  {'─'*6}  {'─'*40}")
        for h, count in counter.most_common():
            pct = (count / total) * 100
            print(f"  {h:<18} {count:>6}  {pct:>5.1f}%  {bar(count, total)}")
    else:
        print(f"  (Showing top 20 of {unique} variants)")
        print(f"  {'Hash':<18} {'Count':>6}  {'%':>6}  Distribution")
        print(f"  {'─'*18} {'─'*6}  {'─'*6}  {'─'*40}")
        for h, count in counter.most_common(20):
            pct = (count / total) * 100
            print(f"  {h:<18} {count:>6}  {pct:>5.1f}%  {bar(count, total)}")


def main():
    parser = argparse.ArgumentParser(description="Analyse Ollama response determinism")
    parser.add_argument("--input-dir", default="./responses", help="Directory of collected responses")
    parser.add_argument("--verbose",   action="store_true", help="Always show all variants")
    parser.add_argument("--show-diff", action="store_true", help="Print first two differing responses side-by-side")
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    if not input_path.exists():
        print(f"[ERROR] Directory not found: {args.input_dir}")
        print("Run collect.py first.")
        return

    # Load metadata if present
    meta_path = input_path / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        print(f"\nRun info:")
        print(f"  Model:       {meta.get('model')}")
        print(f"  Temperature: {meta.get('temperature')}")
        print(f"  Seed:        {meta.get('seed')}")
        print(f"  Started:     {meta.get('started')}")
        print(f"  Prompt:      {meta.get('prompt')[:80]}...")

    # Collect all responses
    run_dirs = sorted(input_path.glob("run_*"))
    if not run_dirs:
        print("[ERROR] No run_XXXX directories found.")
        return

    exact_hashes    = Counter()
    normalised_hashes = Counter()
    code_hashes     = Counter()
    lengths         = []
    first_responses: dict[str, str] = {}  # hash -> text, for --show-diff

    for run_dir in run_dirs:
        resp_file = run_dir / "response.txt"
        if not resp_file.exists():
            continue
        text = resp_file.read_text()
        lengths.append(len(text))

        eh = hash_text(text)
        nh = hash_normalised(text)
        ch = hash_code_only(text)

        exact_hashes[eh] += 1
        normalised_hashes[nh] += 1
        code_hashes[ch] += 1

        if eh not in first_responses:
            first_responses[eh] = text

    total = len(run_dirs)

    print_distribution("EXACT match (byte-for-byte identical)",     exact_hashes,      total, args.verbose)
    print_distribution("NORMALISED match (whitespace-insensitive)", normalised_hashes, total, args.verbose)
    print_distribution("CODE BLOCK match (prose stripped)",         code_hashes,       total, args.verbose)

    # Length stats
    print(f"\n{'═' * 60}")
    print(f"  Response length (characters)")
    print(f"{'═' * 60}")
    print(f"  Min:    {min(lengths)}")
    print(f"  Max:    {max(lengths)}")
    print(f"  Mean:   {sum(lengths)//len(lengths)}")
    length_range = max(lengths) - min(lengths)
    print(f"  Range:  {length_range}  {'(identical lengths)' if length_range == 0 else ''}")

    # Optional diff
    if args.show_diff and len(first_responses) >= 2:
        variants = list(first_responses.values())
        a, b = variants[0], variants[1]
        print(f"\n{'═' * 60}")
        print("  DIFF: first two unique variants")
        print(f"{'═' * 60}")
        import difflib
        diff = list(difflib.unified_diff(
            a.splitlines(), b.splitlines(),
            fromfile="variant_1", tofile="variant_2", lineterm=""
        ))
        print("\n".join(diff[:80]))
        if len(diff) > 80:
            print(f"  ... ({len(diff) - 80} more diff lines)")

    print()


if __name__ == "__main__":
    main()