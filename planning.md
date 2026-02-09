## Motivation & Novelty Assessment

### Why This Research Matters
Autocomplete and intent-detection systems frequently rewrite user inputs to be “helpful,” but small changes can shift intent and harm outcomes. Measuring intent preservation directly and designing clarification policies that only ask when necessary can improve user trust, reduce friction, and increase task success across search, assistants, and enterprise workflows.

### Gap in Existing Work
Prior work emphasizes rewrite quality or downstream retrieval/QA metrics, often treating intent preservation as an indirect proxy. Clarification is typically studied separately from rewriting, with limited evidence on when clarifications help versus annoy, and few studies quantify the trade-off between intent preservation and clarification burden.

### Our Novel Contribution
We introduce an intent-preservation evaluation that combines gold rewrite similarity with LLM-judged intent fidelity, and we test a clarification-gating policy that asks questions only when ambiguity risk is high. This bridges rewrite evaluation and clarification strategy in a single experimental framework.

### Experiment Justification
- Experiment 1: Establish how often LLM rewrites preserve intent compared to gold rewrites and direct LLM judging. This directly measures the core problem: “helpful” corrections that alter intent.
- Experiment 2: Test whether clarification gating improves intent preservation without excessive questioning, quantifying the trade-off between clarification rate and intent fidelity.

## Research Question
How often do modern LLM-based rewrite systems alter user intent, and can a gated clarification policy improve intent preservation without excessive user burden?

## Background and Motivation
Conversational query rewriting is widely used to disambiguate context-dependent questions, but existing evaluations rarely measure intent preservation explicitly. Clarification-question research shows improvements when questions are asked, yet user annoyance and over-asking remain under-studied. This work targets the practical gap: when to rewrite versus when to ask.

## Hypothesis Decomposition
1. LLM rewrites sometimes diverge from the user’s intended goal even when fluent and grammatical.
2. A clarification-gated policy improves intent preservation compared to direct rewriting.
3. Clarification-gated policy reduces question volume versus always-clarify while retaining most of the intent gains.

## Proposed Methodology

### Approach
Use QReCC to measure intent preservation of LLM rewrites against gold rewrites and LLM-judge scoring. Implement a clarification-gated policy that asks a question only when predicted ambiguity is high. Compare against no-clarification and always-clarify baselines.

### Experimental Steps
1. Load QReCC test set and sample a manageable subset (e.g., 200 examples) with conversational context.
2. Generate LLM rewrites (baseline) for each example.
3. Score intent preservation with metrics: BLEU/ROUGE against gold rewrite, semantic similarity (SBERT), and LLM-judge fidelity score.
4. Build a clarification-gated policy: LLM predicts ambiguity risk; if high, ask a clarification question and generate a refined rewrite using the answer.
5. Evaluate clarification rate and intent-preservation improvements; compare against “never clarify” and “always clarify.”

### Baselines
- No-rewrite baseline: original user question as-is.
- Direct LLM rewrite (no clarification).
- Always-clarify: ask a clarification question for every example before rewriting.
- Clarification-gated: ask only when ambiguity is predicted high.

### Evaluation Metrics
- Rewrite similarity: BLEU, ROUGE-L vs gold rewrite.
- Semantic similarity: cosine similarity of sentence embeddings (SBERT) between model rewrite and gold.
- LLM-judge intent fidelity: 1–5 rating (with rubric).
- Clarification burden: % of examples that trigger a question.

### Statistical Analysis Plan
- Paired t-test or Wilcoxon signed-rank test on per-example scores between methods.
- Bootstrap confidence intervals for mean differences.
- Significance threshold: alpha = 0.05, Holm-Bonferroni for multiple comparisons.

## Expected Outcomes
If the hypothesis holds, clarification-gated policy improves LLM-judge fidelity and semantic similarity over direct rewrite, while keeping clarification rates substantially lower than always-clarify.

## Timeline and Milestones
- Phase 0–1 (planning): 30–45 min
- Phase 2 (setup + data prep): 30 min
- Phase 3–4 (implementation + experiments): 2–3 hours
- Phase 5 (analysis): 45 min
- Phase 6 (documentation): 45 min

## Potential Challenges
- API cost/latency for LLM evaluations.
- Clarification simulation requires careful prompting to avoid leakage.
- Ambiguity prediction could be noisy; mitigate with conservative thresholding.

## Success Criteria
- Statistically significant improvement in intent preservation metrics for clarification-gated vs direct rewrite.
- Clarification rate at least 40% lower than always-clarify with minimal loss in intent fidelity.
