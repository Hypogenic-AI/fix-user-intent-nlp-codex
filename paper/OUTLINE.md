# Outline: “Do You Mean…?” Fixing User Intent Without Annoying Them

## Title
- Do You Mean? Clarification Gating Does Not Improve Intent Preservation in Conversational Rewriting

## Abstract
- Problem: LLM rewrites can shift user intent; clarification is a possible mitigation.
- Gap: Clarification policies are often uncalibrated; trade-off between intent fidelity and user burden is under-measured.
- Approach: Compare four methods (no rewrite, direct rewrite, always-clarify, gated-clarify) on QReCC test sample; evaluate BLEU-1, ROUGE-L, SBERT, LLM-judge; measure clarification rate.
- Key results: Gated clarification lowers clarification rate to 56.7% but does not improve intent preservation; LLM-judge worse vs direct rewrite (p=0.046).
- Significance: Naive gating reduces questions without improving intent fidelity; need better ambiguity detection.

## Introduction
- Hook: Assistants rewrite to help, but small changes can change intent and harm trust.
- Importance: Intent preservation is central to search/assistant UX; over-clarifying adds friction.
- Gap: Prior work evaluates rewrites via downstream metrics; clarification strategies often studied separately.
- Approach: Unified evaluation of rewrite fidelity + clarification burden; gated clarification policy.
- Quantitative preview: Direct rewrites highest LLM-judge (4.725); gated clarify 4.567 with 56.7% clarification rate; no statistical gain.
- Contributions (3–4 bullets)
  - We define an intent-preservation evaluation combining gold rewrite similarity, SBERT, and LLM-judge.
  - We compare direct rewriting with always-clarify and gated-clarify on QReCC test sample.
  - We quantify the clarification-rate vs intent-fidelity trade-off and show naive gating hurts LLM-judge.

## Related Work
- Conversational rewriting datasets and methods: CANARD, QReCC; generative reformulation (ConvGQR), RL-based (CONQRR), term-based (QuReTeC).
- Clarification question work: ClarQ/ClariQ.
- Reformulation sensitivity: Vakulenko et al.
- Positioning: Our work links rewrite evaluation with clarification burden; unlike prior work, we quantify intent preservation directly and test gating.

## Methodology
- Problem formulation: Given context C and user question q, produce rewrite r that preserves intent of gold rewrite g; optionally ask clarification.
- Methods: No rewrite, direct rewrite, always-clarify, gated-clarify.
- Clarification gate: LLM predicts ambiguity; if high, ask question and answer (synthetic from gold intent) then rewrite.
- Data: QReCC test split; sample 120 examples with non-empty context; note duplicates.
- Metrics: BLEU-1, ROUGE-L, SBERT cosine, LLM-judge 1–5, clarification rate.
- Implementation: GPT-4.1 for rewrite/clarify/judge; MiniLM-L6-v2 for embeddings; temperature 0; hardware 2x RTX 3090.

## Results
- Main table: mean ± std for four methods, highlight best results in each metric; show clarification rate.
- Statistical test table or paragraph: gated vs direct (t, p, d); only LLM-judge significant and worse.
- Figures: method comparison plot and data distribution plot; captions explain.

## Discussion
- Interpretation: Direct rewrite already resolves context; clarifications can introduce noise.
- Clarification burden: gating reduces questions but does not improve fidelity.
- Limitations: synthetic clarifications; LLM-judge bias; small sample; single dataset.
- Broader implications: Need calibrated ambiguity detectors and human-in-loop evaluation; consider user annoyance signals.

## Conclusion
- Summary of contributions and findings.
- Key takeaway: Naive clarification gating does not improve intent preservation on QReCC.
- Future work: human clarifications, supervised ambiguity detector, other datasets.

## Figures/Tables
- Table 1: Main results (mean ± std) across methods.
- Figure 1: Method comparison plot (results/plots/method_comparison.png).
- Figure 2: Data distribution plot (results/plots/data_distributions.png).

## Citations
- CANARD (Elgohary et al., 2019)
- QReCC (Anantha et al., 2021)
- ClarQ (Kumar & Black, 2020)
- QuReTeC (Voskarides et al., 2020)
- Reformulation sensitivity (Vakulenko et al., 2020)
- ConvGQR (Mo et al., 2023)
- CONQRR (Wu et al., 2021)
- Optional: ClariQ (if cited)
