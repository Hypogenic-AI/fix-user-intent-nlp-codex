# Fixing User Intent Without Annoying Them

This project evaluates intent preservation in conversational query rewriting and tests whether clarification gating reduces user burden without harming intent fidelity.

## Key Findings
- Direct LLM rewrites preserved intent well on QReCC (SBERT ~0.85; LLM-judge ~4.73/5).
- Always-clarify increased burden and slightly reduced intent fidelity.
- Gated clarification reduced questions to 56.7% but did not improve intent preservation over direct rewriting.

## How To Reproduce
1. Create/activate environment:
```bash
source .venv/bin/activate
```
2. Prepare data sample:
```bash
python src/data_prep.py
```
3. Run LLM experiments (requires `OPENAI_API_KEY` or `OPENROUTER_API_KEY`):
```bash
python src/run_experiments.py
```
4. Analyze results and generate plots:
```bash
python src/analyze_results.py
```

## File Structure
- `planning.md`: research plan and motivation
- `src/data_prep.py`: data sampling + stats
- `src/run_experiments.py`: LLM rewrites + clarifications
- `src/analyze_results.py`: metrics, judging, plots
- `results/`: outputs, metrics, plots
- `REPORT.md`: full report

## Report
See `REPORT.md` for methodology, results, and discussion.
