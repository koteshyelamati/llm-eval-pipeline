import logging
import os
import warnings
from dataclasses import dataclass

import yaml
from datasets import Dataset
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from ragas import evaluate
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)
from ragas.llms import _LangchainLLMWrapper
from ragas.embeddings import _LangchainEmbeddingsWrapper

from evals.dataset import EvalSample

logger = logging.getLogger(__name__)

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yml")


def _load_thresholds() -> dict:
    with open(_CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg["thresholds"]


@dataclass
class MetricResult:
    name: str
    score: float
    passed: bool
    threshold: float
    details: dict


def _sample_to_dataset(sample: EvalSample) -> Dataset:
    return Dataset.from_dict(
        {
            "question": [sample.question],
            "answer": [sample.answer],
            "contexts": [sample.contexts],
            "ground_truth": [sample.ground_truth],
        }
    )


class EvalMetrics:
    def __init__(self, judge_model: str, gemini_api_key: str):
        self._thresholds = _load_thresholds()

        lc_llm = ChatGoogleGenerativeAI(
            model=judge_model,
            google_api_key=gemini_api_key,
            temperature=0,
        )
        lc_emb = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=gemini_api_key,
        )
        self._ragas_llm = _LangchainLLMWrapper(lc_llm)
        self._ragas_emb = _LangchainEmbeddingsWrapper(lc_emb)

    def _run_ragas_metric(self, metric, sample: EvalSample) -> float:
        dataset = _sample_to_dataset(sample)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = evaluate(dataset, metrics=[metric])
        scores = result.to_pandas()
        col = scores.columns[-1]
        value = scores[col].iloc[0]
        return float(value) if value is not None else 0.0

    def score_faithfulness(self, sample: EvalSample) -> MetricResult:
        threshold = self._thresholds["faithfulness"]
        metric = Faithfulness(llm=self._ragas_llm)
        score = self._run_ragas_metric(metric, sample)
        return MetricResult(
            name="faithfulness",
            score=score,
            passed=score >= threshold,
            threshold=threshold,
            details={"question": sample.question, "answer_length": len(sample.answer)},
        )

    def score_answer_relevance(self, sample: EvalSample) -> MetricResult:
        threshold = self._thresholds["answer_relevance"]
        metric = AnswerRelevancy(llm=self._ragas_llm, embeddings=self._ragas_emb)
        score = self._run_ragas_metric(metric, sample)
        return MetricResult(
            name="answer_relevance",
            score=score,
            passed=score >= threshold,
            threshold=threshold,
            details={"question": sample.question},
        )

    def score_context_recall(self, sample: EvalSample) -> MetricResult:
        threshold = self._thresholds["context_recall"]
        metric = ContextRecall(llm=self._ragas_llm)
        score = self._run_ragas_metric(metric, sample)
        return MetricResult(
            name="context_recall",
            score=score,
            passed=score >= threshold,
            threshold=threshold,
            details={"num_contexts": len(sample.contexts)},
        )

    def score_context_precision(self, sample: EvalSample) -> MetricResult:
        threshold = self._thresholds["context_precision"]
        metric = ContextPrecision(llm=self._ragas_llm)
        score = self._run_ragas_metric(metric, sample)
        return MetricResult(
            name="context_precision",
            score=score,
            passed=score >= threshold,
            threshold=threshold,
            details={"num_contexts": len(sample.contexts)},
        )

    def score_all(self, sample: EvalSample) -> list[MetricResult]:
        scorers = [
            ("faithfulness", self.score_faithfulness),
            ("answer_relevance", self.score_answer_relevance),
            ("context_recall", self.score_context_recall),
            ("context_precision", self.score_context_precision),
        ]
        results = []
        for name, fn in scorers:
            try:
                results.append(fn(sample))
            except Exception as exc:
                logger.warning("Metric '%s' failed for sample '%s': %s", name, sample.id, exc)
        return results
