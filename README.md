# LLM Evaluation Pipeline

Automated RAG quality gate that scores every PR against a golden Q&A set using Gemini as the judge.

---

## Why this exists

Shipping a RAG system without measuring faithfulness, relevance, and recall is shipping blind — you cannot tell whether a model change improved or degraded answer quality. This pipeline runs on every pull request and blocks merges when any metric drops below its defined threshold, turning LLM quality from a subjective guess into a measurable, enforceable signal.

---

## How it works

```
PR opened
  → GitHub Actions triggers eval.yml
      → Load golden_set.json (25 Q&A pairs)
      → Score each sample: faithfulness, relevance, recall, precision
      → Aggregate scores
      → All pass? → PR can merge  (green check)
      → Any fail? → PR blocked    (red X) + results posted as PR comment
```

---

## Metrics

| Metric             | What it measures                                          | Threshold |
| ------------------ | --------------------------------------------------------- | --------- |
| faithfulness       | Fraction of answer claims grounded in retrieved context   | 0.85      |
| answer_relevance   | How well the answer addresses the question asked          | 0.80      |
| context_recall     | Coverage of ground truth by the retrieved chunks          | 0.75      |
| context_precision  | Signal-to-noise ratio of the retrieved context            | 0.78      |

---

## Setup

```bash
git clone https://github.com/kotesh/llm-eval-pipeline.git
cd llm-eval-pipeline

pip install -r requirements.txt

cp .env.example .env
# Open .env and fill in GEMINI_API_KEY from https://aistudio.google.com

python -m evals.runner
```

---

## GitHub Actions setup

1. Go to your repo → **Settings → Secrets and variables → Actions**
2. Click **New repository secret**
3. Name: `GEMINI_API_KEY`, value: your key from Google AI Studio

Every PR to `main` now runs the eval suite automatically. The job posts a results table as a PR comment and fails the check if any metric is below threshold.

---

## Running locally with custom answers

```python
from evals.runner import EvalRunner

runner = EvalRunner(config_path="config.yml")

# Provide answers your RAG system generated
answers = [
    {"id": "q001", "answer": "MongoDB Atlas Vector Search stores embeddings as BSON arrays..."},
    {"id": "q002", "answer": "Redis TTL expiration uses EXPIRE and PEXPIRE commands..."},
    # ... one entry per golden set sample
]

passed = runner.run(answers=answers)
print("All metrics passed!" if passed else "Some metrics failed — check results/eval_results.json")
```

---

## Sample output

```
RAG Evaluation Result: PASSED

╭──────────────────────┬───────────┬───────────┬────────╮
│ Metric               │ Avg Score │ Threshold │ Status │
├──────────────────────┼───────────┼───────────┼────────┤
│ faithfulness         │    0.9120 │      0.85 │  PASS  │
│ answer_relevance     │    0.8874 │      0.80 │  PASS  │
│ context_recall       │    0.8310 │      0.75 │  PASS  │
│ context_precision    │    0.8056 │      0.78 │  PASS  │
╰──────────────────────┴───────────┴───────────┴────────╯
```

---

## Running tests

```bash
pytest tests/ -v --cov=evals
```

<!-- docs: verified pipeline output -->
