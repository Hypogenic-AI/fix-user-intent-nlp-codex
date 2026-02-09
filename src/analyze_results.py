import json
import os
import re
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import seaborn as sns
from rouge_score import rouge_scorer
from scipy import stats
from sentence_transformers import SentenceTransformer
import torch
import matplotlib.pyplot as plt

from llm_utils import LLMClient, LLMConfig

ROOT = os.path.dirname(os.path.dirname(__file__))
OUTPUT_PATH = os.path.join(ROOT, "results", "model_outputs", "llm_outputs.jsonl")
METRICS_PATH = os.path.join(ROOT, "results", "metrics", "metrics.json")
JUDGE_PATH = os.path.join(ROOT, "results", "metrics", "judgments.jsonl")
PLOT_PATH = os.path.join(ROOT, "results", "plots", "method_comparison.png")

MODEL_JUDGE = os.getenv("MODEL_JUDGE", "gpt-4.1")

METHODS = ["no_rewrite", "direct_rewrite", "always_clarify", "gated_clarify"]


def tokenize(text: str):
    return re.findall(r"\w+|[^\w\s]", text.lower())


def sentence_bleu(reference: str, hypothesis: str) -> float:
    ref = tokenize(reference)
    hyp = tokenize(hypothesis)
    if not hyp:
        return 0.0
    # Simple BLEU-1 to avoid length issues
    overlap = sum(1 for t in hyp if t in ref)
    return overlap / max(len(hyp), 1)


def load_outputs():
    data = []
    with open(OUTPUT_PATH, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def load_judgments():
    if not os.path.exists(JUDGE_PATH):
        return {}
    out = {}
    with open(JUDGE_PATH, "r") as f:
        for line in f:
            obj = json.loads(line)
            out[(obj["example_id"], obj["method"])] = obj
    return out


def save_judgment(obj):
    with open(JUDGE_PATH, "a") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def run_judging(records):
    client = LLMClient(model=MODEL_JUDGE)
    cfg = LLMConfig(model=MODEL_JUDGE, temperature=0, max_tokens=200)

    system = (
        "You are evaluating whether a rewrite preserves the user's intent. "
        "Score from 1 (intent changed) to 5 (intent fully preserved). "
        "Return JSON only."
    )

    existing = load_judgments()

    for rec in records:
        example_id = rec["example_id"]
        context = rec["context"]
        question = rec["question"]
        gold = rec["gold_rewrite"]

        for method in METHODS:
            key = (example_id, method)
            if key in existing:
                continue

            rewrite = rec["outputs"][method]["rewrite"]
            user = (
                f"Conversation:\n{context}\n\nCurrent question: {question}\n\n"
                f"Candidate rewrite: {rewrite}\n\n"
                f"Gold rewrite (for reference only): {gold}\n\n"
                "Return JSON: {\"score\": 1-5, \"rationale\": \"short\"}"
            )
            resp = client.chat_json(system, user, cfg)
            obj = {
                "example_id": example_id,
                "method": method,
                "score": resp["data"]["score"],
                "rationale": resp["data"].get("rationale", ""),
                "usage": resp["usage"],
                "timestamp": datetime.now().isoformat(),
            }
            save_judgment(obj)


def compute_metrics(records):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    batch_size = 64 if device == "cuda" else 16

    metrics = defaultdict(list)

    golds = [r["gold_rewrite"] for r in records]
    gold_embs = model.encode(golds, batch_size=batch_size, convert_to_tensor=True, normalize_embeddings=True)

    for method in METHODS:
        hyps = [r["outputs"][method]["rewrite"] for r in records]
        hyp_embs = model.encode(hyps, batch_size=batch_size, convert_to_tensor=True, normalize_embeddings=True)

        for i, (gold, hyp) in enumerate(zip(golds, hyps)):
            bleu1 = sentence_bleu(gold, hyp)
            rouge = scorer.score(gold, hyp)["rougeL"].fmeasure
            sim = float((gold_embs[i] @ hyp_embs[i]).cpu())

            metrics["method"].append(method)
            metrics["bleu1"].append(bleu1)
            metrics["rougeL"].append(rouge)
            metrics["sbert_cosine"].append(sim)

    df = pd.DataFrame(metrics)
    return df


def summary_stats(df, judgments, judgments_ids):
    summary = {}
    for method in METHODS:
        subset = df[df["method"] == method]
        scores = [judgments.get((ex_id, method), {}).get("score")
                  for ex_id in judgments_ids]
        scores = [s for s in scores if s is not None]

        summary[method] = {
            "bleu1_mean": float(subset["bleu1"].mean()),
            "bleu1_std": float(subset["bleu1"].std()),
            "rougeL_mean": float(subset["rougeL"].mean()),
            "rougeL_std": float(subset["rougeL"].std()),
            "sbert_mean": float(subset["sbert_cosine"].mean()),
            "sbert_std": float(subset["sbert_cosine"].std()),
            "judge_mean": float(np.mean(scores)) if scores else None,
            "judge_std": float(np.std(scores)) if scores else None,
        }
    return summary


def compute_stats(df, judgments, judgments_ids):
    results = {}
    # Paired tests: direct_rewrite vs gated_clarify
    for metric in ["bleu1", "rougeL", "sbert_cosine"]:
        a = df[df["method"] == "direct_rewrite"][metric].values
        b = df[df["method"] == "gated_clarify"][metric].values
        t_stat, p_val = stats.ttest_rel(b, a)
        d = (b.mean() - a.mean()) / (np.std(b - a) + 1e-9)
        results[metric] = {
            "t_stat": float(t_stat),
            "p_val": float(p_val),
            "cohens_d": float(d),
        }

    # Judge scores
    judge_pairs = []
    for ex_id in judgments_ids:
        s_a = judgments.get((ex_id, "direct_rewrite"), {}).get("score")
        s_b = judgments.get((ex_id, "gated_clarify"), {}).get("score")
        if s_a is not None and s_b is not None:
            judge_pairs.append((s_a, s_b))

    if judge_pairs:
        a = np.array([p[0] for p in judge_pairs])
        b = np.array([p[1] for p in judge_pairs])
        t_stat, p_val = stats.ttest_rel(b, a)
        d = (b.mean() - a.mean()) / (np.std(b - a) + 1e-9)
        results["judge_score"] = {
            "t_stat": float(t_stat),
            "p_val": float(p_val),
            "cohens_d": float(d),
            "n": int(len(judge_pairs)),
        }

    return results


def plot_metrics(df):
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    sns.barplot(data=df, x="method", y="bleu1", ax=axes[0])
    axes[0].set_title("BLEU-1 vs Gold Rewrite")
    axes[0].set_xlabel("Method")
    axes[0].set_ylabel("Score")

    sns.barplot(data=df, x="method", y="rougeL", ax=axes[1])
    axes[1].set_title("ROUGE-L vs Gold Rewrite")
    axes[1].set_xlabel("Method")
    axes[1].set_ylabel("Score")

    sns.barplot(data=df, x="method", y="sbert_cosine", ax=axes[2])
    axes[2].set_title("SBERT Cosine Similarity")
    axes[2].set_xlabel("Method")
    axes[2].set_ylabel("Score")

    plt.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)


def main():
    records = load_outputs()

    # Use a deterministic list of example IDs for judging
    judgments_ids = [r["example_id"] for r in records]

    run_judging(records)
    judgments = load_judgments()

    df = compute_metrics(records)
    summary = summary_stats(df, judgments, judgments_ids)
    stats_results = compute_stats(df, judgments, judgments_ids)

    clar_rate = {
        "always_clarify": 1.0,
        "gated_clarify": float(np.mean([r["outputs"]["gated_clarify"].get("used_clarification", False) for r in records])),
        "direct_rewrite": 0.0,
        "no_rewrite": 0.0,
    }

    output = {
        "summary": summary,
        "pairwise_stats": stats_results,
        "clarification_rate": clar_rate,
        "timestamp": datetime.now().isoformat(),
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(output, f, indent=2)

    plot_metrics(df)

    print("Wrote:", METRICS_PATH)
    print("Wrote:", PLOT_PATH)


if __name__ == "__main__":
    main()
