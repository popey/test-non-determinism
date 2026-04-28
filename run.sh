# Quick sanity check — 5 runs first
python collect.py --model llama3.1:8b --runs 5 \
  --prompt-file prompt_kvstore.txt \
  --output-dir ./responses_kv \
  --timeout 600
# Full run
python collect.py --model llama3.1:8b --runs 100 \
  --prompt-file prompt_kvstore.txt \
  --output-dir ./responses_kv \
  --timeout 600
# Analyse
python analyse.py --input-dir ./responses_kv --show-diff
