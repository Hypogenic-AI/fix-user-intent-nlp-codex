# “Do You Mean…?”: Fixing User Intent Without Annoying Them

## 1. Executive Summary
We evaluated how often LLM rewrites preserve user intent and whether clarification gating improves intent preservation without excessive questioning on QReCC conversational queries. On a 120-example context-heavy sample, a gated clarification policy reduced clarification rate to 56.7% but did not improve intent preservation over direct rewrites; LLM-judged intent fidelity was slightly worse (p=0.046). Practically, this suggests that naive gating based on perceived ambiguity can reduce questions but does not automatically improve intent fidelity.

## 2. Goal
**Hypothesis**: Autocomplete/intent-rewrite systems sometimes alter intent; a clarification-gated policy improves intent preservation without spamming users.

**Why it matters**: User trust in assistants/search depends on preserving intent. Over-correcting queries can be infuriating and harmful; over-clarifying increases friction.

**Expected impact**: A clear evaluation framework for intent preservation and a measured approach to asking clarifying questions.

## 3. Data Construction

### Dataset Description
- **Source**: QReCC test split (`datasets/qrecc/qrecc-test.json`)
- **Size**: 16,451 total examples
- **Task**: conversational question rewriting with gold rewrites
- **Selection**: sampled 120 examples with non-empty context
- **Known biases**: QReCC is constructed from multiple sources; contexts can be stylized or web-sourced.

### Example Samples
| Context (truncated) | Question | Gold Rewrite |
|---|---|---|
| Pete Maravich history in Atlanta Hawks… | How did he play in his next season? | How did Pete Maravich play in his fourth season with the Atlanta Hawks? |
| Will Forte SNL history… | did he have any notable characters that he played? | Did Will Forte have any notable characters that he played on SNL? |
| 529 plan overview… | What are the types of plans | What are the types of 529 plans? |

### Data Quality
- Missing values: **0**
- Duplicates: **90** (by Context+Question)
- Avg question length: **7.33 tokens** (std 2.31)
- Avg rewrite length: **11.40 tokens** (std 5.15)
- Avg context length: **5.83 turns** (std 4.63)

### Preprocessing Steps
1. Filtered to examples with non-empty context to focus on intent preservation in context.
2. Sampled 120 examples using a fixed random seed (42).

### Train/Val/Test Splits
- Used **test split only** for evaluation. No training performed.

## 4. Experiment Description

### Methodology

#### High-Level Approach
We compare four methods on the same sample: no rewrite, direct LLM rewrite, always-clarify then rewrite, and gated-clarify (ask only when ambiguity is predicted high). Intent preservation is measured via similarity to gold rewrites, semantic similarity, and LLM-judge scoring.

#### Why This Method?
Direct rewrite baselines are common in conversational QA. Clarification is a known alternative, but its use is often uncalibrated. Our approach tests whether a simple gating policy can improve intent fidelity without excessive questioning.

### Implementation Details

#### Tools and Libraries
- openai: 2.17.0
- sentence-transformers: 5.2.2
- torch: 2.10.0+cu128
- numpy: 2.4.2
- pandas: 3.0.0
- rouge-score: 0.1.2
- scipy: 1.17.0

#### Models
- **Rewrite + Clarification + Judge**: `gpt-4.1` (temperature=0)
- **Semantic similarity**: `sentence-transformers/all-MiniLM-L6-v2`

#### Hyperparameters
| Parameter | Value | Selection Method |
|---|---|---|
| temperature | 0.0 | fixed for determinism |
| max_tokens | 200 | fixed |
| sample_size | 120 | resource-aware |
| SBERT batch_size | 64 (GPU) | based on 24GB GPU |

#### Training Procedure or Analysis Pipeline
1. Generate rewrites and clarification decisions via LLM prompts.
2. For always-clarify and gated-clarify, ask a clarification question and generate a user answer consistent with gold intent.
3. Compute BLEU-1, ROUGE-L, SBERT cosine similarity vs gold rewrite.
4. LLM judge rates intent preservation (1–5) per method.

### Experimental Protocol

#### Reproducibility Information
- Runs: 1 (temperature=0)
- Random seed: 42
- Hardware: 2x NVIDIA RTX 3090 (24GB each); GPU used for embeddings
- Execution time: ~1–2 hours for LLM calls + evaluation

#### Evaluation Metrics
- **BLEU-1 / ROUGE-L**: overlap with gold rewrite; proxy for rewrite fidelity.
- **SBERT cosine**: semantic similarity to gold rewrite; captures paraphrase intent.
- **LLM-judge score (1–5)**: direct intent preservation judgment.
- **Clarification rate**: fraction of examples that trigger a clarification question.

### Raw Results

#### Tables (mean ± std)
| Method | BLEU-1 | ROUGE-L | SBERT Cosine | LLM Judge | Clarification Rate |
|---|---|---|---|---|---|
| No rewrite | 0.873 ± 0.113 | 0.650 ± 0.185 | 0.648 ± 0.170 | 4.708 ± 0.712 | 0.0 |
| Direct rewrite | 0.565 ± 0.191 | 0.608 ± 0.195 | 0.851 ± 0.123 | 4.725 ± 0.658 | 0.0 |
| Always clarify | 0.529 ± 0.184 | 0.553 ± 0.184 | 0.843 ± 0.118 | 4.450 ± 0.729 | 1.0 |
| Gated clarify | 0.568 ± 0.203 | 0.601 ± 0.204 | 0.851 ± 0.124 | 4.567 ± 0.692 | 0.567 |

#### Statistical Tests (Gated vs Direct)
- BLEU-1: t=0.259, p=0.796 (ns)
- ROUGE-L: t=-0.450, p=0.654 (ns)
- SBERT: t=-0.031, p=0.975 (ns)
- LLM-judge: t=-2.017, p=0.046, d=-0.185 (gated slightly worse)

#### Visualizations
- Data distributions: `results/plots/data_distributions.png`
- Method comparison: `results/plots/method_comparison.png`

#### Output Locations
- Model outputs: `results/model_outputs/llm_outputs.jsonl`
- Metrics: `results/metrics/metrics.json`
- Plots: `results/plots/`

## 5. Result Analysis

### Key Findings
1. Direct LLM rewrites are semantically close to gold rewrites (SBERT ~0.85) and have the highest LLM-judge intent scores.
2. Always-clarify reduces intent fidelity and increases burden (100% clarification rate), suggesting over-asking harms user experience.
3. Gated clarification reduced questions to 56.7% but did not improve intent preservation; LLM-judge scores were slightly worse than direct rewrite.

### Hypothesis Testing Results
- The hypothesis that gating improves intent preservation **was not supported**.
- Gating reduced clarification rate but did not improve intent fidelity.

### Comparison to Baselines
Direct rewrites outperformed both always-clarify and gated-clarify on LLM-judge scores. No-rewrite achieved high lexical overlap due to many gold rewrites being identical to the original question; however, semantic similarity and LLM-judge favored direct rewrites for context-heavy cases.

### Surprises and Insights
- The LLM judge rated direct rewrites higher than clarification-based methods, suggesting clarification may introduce noise when the model already resolves context correctly.
- The gating policy often triggered on ambiguity but did not yield better rewrites, indicating a mismatch between “perceived ambiguity” and actual intent errors.

### Error Analysis
Common failure modes (qualitative inspection in outputs):
- Clarification questions focused on irrelevant details.
- Clarification answers (synthetic) occasionally simplified or drifted from the gold rewrite.
- For certain context-heavy questions, the direct rewrite already resolved intent well, making clarification redundant.

### Limitations
- Clarification answers were synthetic (LLM-generated from gold rewrite), not from real users.
- LLM-judge is a proxy for human evaluation and may share biases with the rewrite model.
- Evaluation on a single dataset and a 120-example sample.

## 6. Conclusions
The study confirms that direct LLM rewrites generally preserve intent well in QReCC-like contexts, and naive clarification gating does not improve intent preservation. Clarification can reduce user burden if gated, but careful design is required to avoid worsening intent fidelity.

## 7. Next Steps
1. Replace synthetic clarification answers with human-in-the-loop or dataset-sourced clarifications.
2. Train a dedicated ambiguity detector using supervised labels to improve gating accuracy.
3. Extend evaluation to other datasets (e.g., CANARD, ClariQ) and add human judgments.

## References
- Elgohary et al., 2019 (CANARD)
- Anantha et al., 2021 (QReCC)
- Kumar & Black, 2020 (ClarQ)
- Voskarides et al., 2020 (QuReTeC)
- Vakulenko et al., 2020 (Reformulation sensitivity)
- Mo et al., 2023 (ConvGQR)
- Wu et al., 2021 (CONQRR)
