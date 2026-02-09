# Literature Review: Fixing User Intent Without Annoying Them

## Review Scope

### Research Question
How can we design intent-preserving query rewriting and clarification mechanisms that improve user outcomes without excessive or annoying questioning?

### Inclusion Criteria
- Papers on conversational question rewriting or query reformulation
- Papers on clarification question generation / asking when to clarify
- Datasets for conversational QA/search with rewrite or clarification supervision
- Empirical evaluations of rewriting vs QA robustness

### Exclusion Criteria
- Non-conversational query expansion without context
- Datasets without conversational context or clarification relevance
- Pure dialogue generation without intent preservation objectives

### Time Frame
2019–2023 (with emphasis on 2020–2023)

### Sources
- ACL Anthology
- arXiv
- HuggingFace datasets
- GitHub repositories

## Search Log

| Date | Query | Source | Results | Notes |
|------|-------|--------|---------|-------|
| 2026-02-09 | "question rewriting conversational" | ACL/arXiv | 10+ | Selected CANARD, QReCC, ConvGQR |
| 2026-02-09 | "clarifying questions dataset" | ACL/GitHub | 5+ | Selected ClarQ paper, ClariQ repo |
| 2026-02-09 | "conversational query rewriting RL" | arXiv | 5+ | Selected CONQRR |

## Paper Summaries

### Paper 1: Can You Unpack That? Learning to Rewrite Questions-in-Context
- **Authors**: Ahmed Elgohary, Denis Peskov, Jordan Boyd-Graber
- **Year**: 2019
- **Source**: EMNLP-IJCNLP (ACL Anthology D19-1605)
- **Key Contribution**: Introduces CANARD dataset and the task of question-in-context rewriting.
- **Methodology**: Seq2Seq rewriting of conversational questions into self-contained forms.
- **Datasets Used**: CANARD (built from QuAC).
- **Results**: Demonstrates feasibility of rewriting models for context-dependent questions.
- **Code Available**: Not found publicly.
- **Relevance to Our Research**: Core benchmark for intent-preserving question rewriting.

### Paper 2: Open-Domain Question Answering Goes Conversational via Question Rewriting
- **Authors**: Raviteja Anantha, Svitlana Vakulenko, Zhucheng Tu, Shayne Longpre, Stephen Pulman, Srinivas Chappidi
- **Year**: 2021
- **Source**: NAACL-HLT (ACL Anthology 2021.naacl-main.44)
- **Key Contribution**: Introduces QReCC dataset and baseline pipeline for conversational QA via rewriting.
- **Methodology**: Rewriting + retrieval + reading comprehension pipeline.
- **Datasets Used**: QReCC (new), built from TREC CAsT, QuAC, NQ.
- **Results**: Establishes baselines and highlights difficulty gap to human performance.
- **Code Available**: Yes (ml-qrecc).
- **Relevance to Our Research**: Primary dataset for studying intent preservation in rewriting.

### Paper 3: ClarQ: A large-scale and diverse dataset for Clarification Question Generation
- **Authors**: Vaibhav Kumar, Alan W Black
- **Year**: 2020
- **Source**: ACL (ACL Anthology 2020.acl-main.651)
- **Key Contribution**: Introduces ClarQ dataset with large-scale clarification questions from StackExchange.
- **Methodology**: Bootstrapped dataset creation with classifier-based filtering.
- **Datasets Used**: ClarQ dataset.
- **Results**: Demonstrates downstream QA gains from clarification data.
- **Code Available**: Not specified.
- **Relevance to Our Research**: Provides data for asking clarifying questions without over-asking.

### Paper 4: Query Resolution for Conversational Search with Limited Supervision
- **Authors**: Nikos Voskarides, Dan Li, Pengjie Ren, Evangelos Kanoulas, Maarten de Rijke
- **Year**: 2020
- **Source**: arXiv:2005.11723
- **Key Contribution**: QuReTeC term-level query resolution.
- **Methodology**: Binary term classification to add context terms to current query; distant supervision.
- **Datasets Used**: Conversational search collections with relevance labels.
- **Results**: Improves retrieval without retraining underlying search engine.
- **Code Available**: Repository link not accessible.
- **Relevance to Our Research**: Alternative to full rewrite; preserves intent by controlled term addition.

### Paper 5: A Wrong Answer or a Wrong Question? An Intricate Relationship between Question Reformulation and Answer Selection in Conversational QA
- **Authors**: Svitlana Vakulenko, Shayne Longpre, Zhucheng Tu, Raviteja Anantha
- **Year**: 2020
- **Source**: arXiv:2010.06835
- **Key Contribution**: Analysis framework for QA sensitivity to question reformulation.
- **Methodology**: Uses rewrites to probe robustness of answer selection and retrieval.
- **Datasets Used**: TREC CAsT, QuAC (CANARD).
- **Results**: Retrieval is sensitive to minor reformulations; RC less so.
- **Code Available**: Not specified.
- **Relevance to Our Research**: Directly evaluates intent preservation and impact of rewrites.

### Paper 6: ConvGQR: Generative Query Reformulation for Conversational Search
- **Authors**: Fengran Mo, Kelong Mao, Yutao Zhu, Yihong Wu, Kaiyu Huang, Jian-Yun Nie
- **Year**: 2023
- **Source**: ACL 2023 (ACL Anthology 2023.acl-long.274)
- **Key Contribution**: Generative PLM-based query reformulation with answer-aware signals.
- **Methodology**: Two PLMs for query rewriting and answer generation; optimize reformulation quality.
- **Datasets Used**: Conversational search benchmarks.
- **Results**: Improved retrieval over rewrite baselines.
- **Code Available**: Not found publicly.
- **Relevance to Our Research**: Modern generative rewrite approach with intent-aware signals.

### Paper 7: CONQRR: Conversational Query Rewriting for Retrieval with Reinforcement Learning
- **Authors**: Zeqiu Wu, Yi Luan, Hannah Rashkin, David Reitter, Hannaneh Hajishirzi, Mari Ostendorf, Gaurav Singh Tomar
- **Year**: 2021
- **Source**: arXiv:2112.08558
- **Key Contribution**: RL-based rewriting to optimize retrieval effectiveness.
- **Methodology**: Reward shaping for retrieval metrics; adapts to off-the-shelf retrievers.
- **Datasets Used**: Open-domain CQA datasets from multiple sources.
- **Results**: Competitive retrieval improvements and robustness.
- **Code Available**: Not found publicly.
- **Relevance to Our Research**: Trains toward end-task metrics, useful for intent preservation.

## Common Methodologies
- **Seq2Seq rewriting**: CANARD, QReCC baselines
- **Term selection / query resolution**: QuReTeC
- **Reinforcement learning for retrieval-oriented rewriting**: CONQRR
- **Generative PLM reformulation**: ConvGQR
- **Clarification question generation / selection**: ClarQ/ClariQ

## Standard Baselines
- Seq2Seq rewrite model (BART/T5-style)
- Rewrite + retrieval + reader pipeline (QReCC)
- Term addition via classifier (QuReTeC)
- No-rewrite baselines (raw question)

## Evaluation Metrics
- **Rewriting quality**: BLEU/ROUGE (if gold rewrites), or retrieval metrics downstream
- **Retrieval**: MRR, nDCG, Recall@k
- **QA**: Exact Match, F1
- **Clarification**: classification accuracy/F1 for clarification need, Recall@k for question selection

## Datasets in the Literature
- **QReCC**: conversational rewriting + QA
- **CANARD**: question-in-context rewriting
- **ClarQ/ClariQ**: clarification question generation and selection
- **QuAC / TREC CAsT**: source conversational datasets

## Gaps and Opportunities
- Intent preservation is often inferred indirectly via retrieval/QA metrics rather than measured explicitly.
- Limited evaluation of user annoyance or query burden; few datasets include user preference signals.
- Clarification strategies are typically evaluated separately from rewriting pipelines.

## Recommendations for Our Experiment
- **Recommended datasets**: QReCC (primary), CANARD-QuReTeC (term-level resolution), ClariQ/ClarQ (clarification modeling)
- **Recommended baselines**: raw question, seq2seq rewrite, QuReTeC term selection
- **Recommended metrics**: retrieval MRR/nDCG + QA F1; add intent-preservation proxies (rewrite fidelity vs. answer correctness)
- **Methodological considerations**: evaluate whether clarification improves downstream performance without increasing interaction length excessively
