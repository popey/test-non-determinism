# test-non-determinism

A small toolkit for empirically testing whether a locally-run LLM produces
deterministic output across repeated identical requests.

The common assumption is that LLMs are inherently non-deterministic. This is
true by default — sampling randomness, GPU thread scheduling, and batched
requests across shared infrastructure all introduce variance. However, with a
local model, a fixed seed, and greedy decoding (temperature=0), determinism
should be achievable. This project lets you verify that claim with real data.

## How it works

`collect.py` sends the same prompt to a locally-running [Ollama](https://ollama.com)
instance N times, saving each response to its own folder. It pins every
relevant sampling parameter and sends a warmup request first to ensure the
model is fully loaded before collection begins.

`analyse.py` fingerprints the saved responses three ways:

- **Exact** — byte-for-byte SHA-256 hash
- **Normalised** — hash after stripping trailing whitespace and blank lines
- **Code block only** — hash of just the extracted fenced code block, ignoring surrounding prose

It then reports how many unique variants were produced, with a histogram and
percentage breakdown.

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) running locally (`ollama serve`)
- At least one model pulled, e.g. `ollama pull llama3.1:8b`

No third-party Python packages required — only the standard library.

## Usage

### 1. Write a prompt

Create a plain text file containing your prompt, for example `prompt.txt`:

```
Write a complete Python implementation of a simple key-value store with the
following features:
- In-memory storage with optional persistence to a JSON file
- Support for set, get, delete, list-keys, and exists operations
- TTL (time-to-live) support per key, with automatic expiry
- A simple CLI interface using argparse
- Transaction support (begin, commit, rollback)
- Full type hints throughout
- A complete suite of unit tests using unittest

Output only the code, no explanation.
```

A more complex prompt with multiple valid implementation approaches will
produce more interesting results than a prompt with one obvious answer.

### 2. Collect responses

```bash
python collect.py \
  --model llama3.1:8b \
  --runs 100 \
  --prompt-file prompt.txt \
  --output-dir ./responses \
  --timeout 600
```

A warmup request is sent automatically before run 1 to load the model into
memory and stabilise GPU state. Skip it with `--no-warmup` if the model is
already loaded.

**All options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `llama3.2` | Ollama model name |
| `--runs` | `1000` | Number of requests to make |
| `--seed` | `42` | Fixed seed passed to Ollama |
| `--output-dir` | `./responses` | Where to save responses |
| `--timeout` | `300` | HTTP timeout per request (seconds) |
| `--prompt-file` | *(built-in)* | Path to prompt text file |
| `--no-warmup` | — | Skip the warmup request |

### 3. Analyse results

```bash
python analyse.py --input-dir ./responses
```

To also print a unified diff between the first two differing variants:

```bash
python analyse.py --input-dir ./responses --show-diff
```

**All options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--input-dir` | `./responses` | Directory of collected responses |
| `--verbose` | — | Always show all variants, not just top 20 |
| `--show-diff` | — | Print a unified diff of the first two differing variants |

## Output structure

```
responses/
├── meta.json          # run parameters: model, seed, prompt, options, timestamp
├── run_0001/
│   ├── response.txt   # raw model output
│   └── info.json      # run index and wall-clock duration
├── run_0002/
│   └── ...
└── ...
```

## What to look for

If the model is fully deterministic, `analyse.py` will report a single unique
hash across all three fingerprint types for all runs.

If you see exactly one divergent run (typically run 1) with a longer elapsed
time, this is almost always the model loading into GPU memory. The warmup
request should eliminate this. If it persists, try `--no-warmup` on a
subsequent run where the model is already loaded.

If you see persistent variance across all runs, the likely causes are:

- Non-deterministic CUDA kernels in the GPU backend (common with some
  versions of llama.cpp on certain hardware)
- The model being partially offloaded to CPU, introducing mixed
  floating-point execution paths
- Ollama receiving concurrent requests from another process

## Pinned sampling parameters

The following options are sent with every request to remove as many sources
of variance as possible:

| Parameter | Value | Reason |
|-----------|-------|--------|
| `temperature` | `0.0` | Disables sampling randomness |
| `top_k` | `1` | Greedy decoding — only the top token is ever selected |
| `top_p` | `1.0` | Disables nucleus sampling |
| `repeat_penalty` | `1.0` | Neutral — no repetition penalty applied |
| `num_ctx` | `8192` | Pinned context window size |
| `num_predict` | `4096` | Maximum tokens to generate |
| `keep_alive` | `-1` | Keeps the model loaded between requests |

## Licence

MIT