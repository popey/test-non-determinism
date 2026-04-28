#!/usr/bin/env python3
"""
collect.py - Call Ollama N times with the same prompt and save each response.

Usage:
    python collect.py [--model MODEL] [--runs N] [--temp TEMP] [--seed SEED]
                      [--output-dir DIR] [--prompt-file FILE] [--timeout SECS]
                      [--no-warmup]

Defaults:
    model:       llama3.1:8b
    runs:        1000
    temp:        0.0
    seed:        42
    output-dir:  ./responses
    timeout:     300

A warmup request is sent before run 1 by default to ensure the model is fully
loaded and the GPU is in a stable state. Pass --no-warmup to skip it.
"""

import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime

DEFAULT_PROMPT = (
    "Can you write a python program to calculate prime numbers up to 100000, "
    "and print the results as a markdown table. "
    "Output ONLY the code with no explanation or surrounding text."
)

OLLAMA_URL = "http://localhost:11434/api/generate"

# Sampling parameters pinned explicitly for maximum reproducibility.
# At temp=0 the model uses greedy decoding, but top_k/top_p/repeat_penalty
# are set to neutral values anyway so model defaults can't vary between runs.
FIXED_OPTIONS = {
    "temperature":    0.0,
    "top_k":          1,       # greedy — only the top token is ever considered
    "top_p":          1.0,     # no nucleus filtering
    "repeat_penalty": 1.0,     # no repetition penalty
    "num_predict":    4096,    # max tokens to generate
    "num_ctx":        8192,    # context window — pinned so it never varies
}


def call_ollama(model: str, prompt: str, seed: int, timeout: int) -> tuple[str, float]:
    options = {**FIXED_OPTIONS, "seed": seed}

    payload = json.dumps({
        "model":      model,
        "prompt":     prompt,
        "stream":     False,
        "keep_alive": -1,      # keep model loaded indefinitely between requests
        "options":    options,
    }).encode()

    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read())
            elapsed = time.monotonic() - t0
            return body.get("response", ""), elapsed
    except TimeoutError:
        elapsed = time.monotonic() - t0
        print(f"\n[ERROR] Request timed out after {elapsed:.0f}s.")
        print(f"Try increasing --timeout (currently {timeout}s).")
        sys.exit(1)
    except urllib.error.URLError as e:
        print(f"\n[ERROR] Could not reach Ollama: {e}")
        print("Is Ollama running? Try: ollama serve")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Collect Ollama responses for determinism testing")
    parser.add_argument("--model",       default="llama3.1:8b",    help="Ollama model name")
    parser.add_argument("--runs",        default=1000, type=int, help="Number of runs")
    parser.add_argument("--seed",        default=42,   type=int,   help="Fixed seed passed to Ollama")
    parser.add_argument("--output-dir",  default="./responses",    help="Directory to save responses")
    parser.add_argument("--timeout",     default=300,  type=int,   help="HTTP timeout in seconds (default: 300)")
    parser.add_argument("--prompt-file", default=None,             help="Path to a plain text file containing the prompt")
    parser.add_argument("--no-warmup",   action="store_true",      help="Skip the warmup request")
    args = parser.parse_args()

    # Load prompt
    if args.prompt_file:
        prompt_path = os.path.abspath(args.prompt_file)
        if not os.path.exists(prompt_path):
            print(f"[ERROR] Prompt file not found: {prompt_path}")
            sys.exit(1)
        with open(prompt_path) as f:
            prompt = f.read().strip()
        prompt_source = prompt_path
    else:
        prompt = DEFAULT_PROMPT
        prompt_source = "(built-in default)"

    os.makedirs(args.output_dir, exist_ok=True)

    meta = {
        "model":        args.model,
        "runs":         args.runs,
        "seed":         args.seed,
        "timeout":      args.timeout,
        "fixed_options": FIXED_OPTIONS,
        "prompt":       prompt,
        "prompt_source": prompt_source,
        "started":      datetime.now().isoformat(),
    }
    with open(os.path.join(args.output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Model:       {args.model}")
    print(f"Runs:        {args.runs}")
    print(f"Temperature: {FIXED_OPTIONS['temperature']} (top_k=1, greedy)")
    print(f"Seed:        {args.seed}")
    print(f"Timeout:     {args.timeout}s")
    print(f"Prompt:      {prompt_source}")
    print(f"Output:      {args.output_dir}")
    print(f"{'─' * 50}")

    # Warmup — loads the model into GPU memory and stabilises state
    if not args.no_warmup:
        print("Warming up (loading model)...", end=" ", flush=True)
        _, warmup_elapsed = call_ollama(args.model, "Say hello.", args.seed, args.timeout)
        print(f"done in {warmup_elapsed:.1f}s")
        print(f"{'─' * 50}")

    timings = []

    for i in range(1, args.runs + 1):
        run_dir = os.path.join(args.output_dir, f"run_{i:04d}")
        os.makedirs(run_dir, exist_ok=True)

        response, elapsed = call_ollama(args.model, prompt, args.seed, args.timeout)
        timings.append(elapsed)

        with open(os.path.join(run_dir, "response.txt"), "w") as f:
            f.write(response)

        with open(os.path.join(run_dir, "info.json"), "w") as f:
            json.dump({"run": i, "elapsed_s": round(elapsed, 3)}, f)

        avg = sum(timings) / len(timings)
        remaining = (args.runs - i) * avg
        print(
            f"\r[{i:>5}/{args.runs}] {elapsed:5.1f}s this run | "
            f"avg {avg:5.1f}s | "
            f"~{remaining/60:.1f}min remaining   ",
            end="",
            flush=True,
        )

    print(f"\n{'─' * 50}")
    print(f"Done. Responses saved to: {args.output_dir}")
    print(f"Total time: {sum(timings)/60:.1f} minutes")
    print(f"Now run:  python analyse.py --input-dir {args.output_dir}")


if __name__ == "__main__":
    main()
