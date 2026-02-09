import json
import os
import random
import re
from collections import Counter
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(ROOT, "datasets", "qrecc", "qrecc-test.json")
OUTPUT_SAMPLE = os.path.join(ROOT, "results", "sample.jsonl")
STATS_PATH = os.path.join(ROOT, "results", "data_stats.json")
PLOT_PATH = os.path.join(ROOT, "results", "plots", "data_distributions.png")

SEED = 42
SAMPLE_SIZE = 120


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def safe_len(x):
    return len(x) if x is not None else 0


def tokenize(text: str):
    return re.findall(r"\w+|[^\w\s]", text.lower())


def main():
    set_seed(SEED)

    with open(DATA_PATH, "r") as f:
        data = json.load(f)

    total = len(data)
    with_context = [d for d in data if d.get("Context")]
    no_context = total - len(with_context)

    q_lens = [len(tokenize(d.get("Question", ""))) for d in data]
    r_lens = [len(tokenize(d.get("Rewrite", ""))) for d in data]
    ctx_lens = [safe_len(d.get("Context")) for d in data]

    missing_fields = 0
    for d in data:
        if not d.get("Question") or not d.get("Rewrite"):
            missing_fields += 1

    # Duplicate detection on (Context, Question)
    seen = Counter()
    for d in data:
        ctx = tuple(d.get("Context") or [])
        key = (ctx, d.get("Question", ""))
        seen[key] += 1
    duplicates = sum(v - 1 for v in seen.values() if v > 1)

    stats = {
        "timestamp": datetime.now().isoformat(),
        "total_examples": total,
        "with_context": len(with_context),
        "without_context": no_context,
        "missing_fields": missing_fields,
        "duplicates": duplicates,
        "question_len_mean": float(np.mean(q_lens)),
        "question_len_std": float(np.std(q_lens)),
        "rewrite_len_mean": float(np.mean(r_lens)),
        "rewrite_len_std": float(np.std(r_lens)),
        "context_len_mean": float(np.mean(ctx_lens)),
        "context_len_std": float(np.std(ctx_lens)),
    }

    os.makedirs(os.path.dirname(OUTPUT_SAMPLE), exist_ok=True)

    # Sample only from examples with context for intent-preservation testing
    if len(with_context) < SAMPLE_SIZE:
        sample = with_context
    else:
        sample = random.sample(with_context, SAMPLE_SIZE)

    with open(OUTPUT_SAMPLE, "w") as f:
        for d in sample:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    with open(STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)

    # Plot distributions
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].hist(ctx_lens, bins=30, color="#4C72B0")
    axes[0].set_title("Context Length (turns)")
    axes[0].set_xlabel("Turns")
    axes[0].set_ylabel("Count")

    axes[1].hist(q_lens, bins=30, color="#55A868")
    axes[1].set_title("Question Length (tokens)")
    axes[1].set_xlabel("Tokens")

    axes[2].hist(r_lens, bins=30, color="#C44E52")
    axes[2].set_title("Rewrite Length (tokens)")
    axes[2].set_xlabel("Tokens")

    plt.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)

    print("Wrote:", OUTPUT_SAMPLE)
    print("Wrote:", STATS_PATH)
    print("Wrote:", PLOT_PATH)


if __name__ == "__main__":
    main()
