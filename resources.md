# Resources Catalog

## Summary
This document catalogs all resources gathered for the research project, including papers, datasets, and code repositories.

### Papers
Total papers downloaded: 7

| Title | Authors | Year | File | Key Info |
|-------|---------|------|------|----------|
| Can You Unpack That? Learning to Rewrite Questions-in-Context | Elgohary et al. | 2019 | papers/2019_canard_question_rewriting.pdf | CANARD dataset, question rewriting |
| Open-Domain Question Answering Goes Conversational via Question Rewriting | Anantha et al. | 2021 | papers/2021_qrecc_question_rewriting.pdf | QReCC dataset and baselines |
| ClarQ: A large-scale and diverse dataset for Clarification Question Generation | Kumar & Black | 2020 | papers/2020_clariq_clarifying_questions.pdf | Clarification question dataset |
| Query Resolution for Conversational Search with Limited Supervision | Voskarides et al. | 2020 | papers/2020_quretec_conversational_search.pdf | QuReTeC term-level query resolution |
| A Wrong Answer or a Wrong Question? | Vakulenko et al. | 2020 | papers/2020_wrong_answer_wrong_question.pdf | QA sensitivity to reformulation |
| ConvGQR: Generative Query Reformulation for Conversational Search | Mo et al. | 2023 | papers/2023_convgqr_conversational_query_rewriting.pdf | PLM-based query reformulation |
| CONQRR: Conversational Query Rewriting for Retrieval with RL | Wu et al. | 2021 | papers/2021_conqrr_conversational_qa.pdf | RL-based query rewriting |

See papers/README.md for detailed descriptions.

### Datasets
Total datasets downloaded: 2

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| QReCC | HF `svakulenk0/qrecc` | train 63,501; test 16,451 | question rewriting + QA | datasets/qrecc/ | Direct file download to avoid schema errors |
| CANARD-QuReTeC Gold | HF `uva-irlab/canard_quretec` | train 20,181; dev 2,196; test 3,373 | term-level query resolution | datasets/canard_quretec/ | Direct download (HF script unsupported) |

See datasets/README.md for detailed descriptions.

### Code Repositories
Total repositories cloned: 2

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| ClariQ | github.com/aliannejadi/ClariQ | Clarifying question dataset + eval | code/ClariQ/ | Contains TSV data + eval script |
| ml-qrecc | github.com/apple/ml-qrecc | QReCC dataset + eval | code/ml-qrecc/ | Contains retrieval/QA eval scripts |

See code/README.md for detailed descriptions.

## Resource Gathering Notes

### Search Strategy
- Attempted paper-finder (timed out), then manual search via ACL Anthology/arXiv.
- Queried GitHub for dataset and baseline repositories.
- Used HuggingFace dataset hosting for direct downloads.

### Selection Criteria
- Direct relevance to conversational query rewriting or clarification questions
- Publicly accessible PDFs and data
- Widely cited or representative methods in the field

### Challenges Encountered
- Paper-finder service unavailable (timed out).
- Some GitHub repositories referenced in papers were not accessible.
- HF datasets with legacy loading scripts required direct downloads.

### Gaps and Workarounds
- ClarQ/ClariQ large-scale dataset not fully downloaded; repository cloned for data access.
- For missing repos, relied on paper descriptions and available datasets.

## Recommendations for Experiment Design

1. **Primary dataset(s)**: QReCC for rewrite supervision; CANARD-QuReTeC for term-level resolution.
2. **Baseline methods**: seq2seq rewrite, QuReTeC term addition, no-rewrite baseline.
3. **Evaluation metrics**: retrieval MRR/nDCG + QA F1, plus rewrite fidelity metrics.
4. **Code to adapt/reuse**: ml-qrecc eval scripts; ClariQ data format for clarification modeling.

## Experiment Usage (This Study)
- Dataset used: QReCC test split (`datasets/qrecc/qrecc-test.json`)
- Sample constructed: `results/sample.jsonl` (120 examples with conversational context)
- Outputs: `results/model_outputs/llm_outputs.jsonl`, `results/metrics/metrics.json`
- Plots: `results/plots/data_distributions.png`, `results/plots/method_comparison.png`
