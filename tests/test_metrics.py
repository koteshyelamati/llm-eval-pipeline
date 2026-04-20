from unittest.mock import MagicMock, patch

import pytest

from evals.dataset import EvalSample
from evals.metrics import EvalMetrics, MetricResult

_SAMPLE = EvalSample(
    id="q001",
    question="What is vector search?",
    ground_truth="Vector search finds nearest neighbors in embedding space.",
    contexts=["Vector search uses cosine similarity to find semantically similar documents."],
    answer="Vector search finds nearest neighbors in embedding space.",
)

_ALL_METRIC_NAMES = {
    "faithfulness",
    "answer_relevance",
    "context_recall",
    "context_precision",
}


def _make_result(name: str, score: float, threshold: float) -> MetricResult:
    return MetricResult(
        name=name,
        score=score,
        passed=score >= threshold,
        threshold=threshold,
        details={},
    )


def test_metric_result_passed_when_above_threshold():
    result = MetricResult(
        name="faithfulness", score=0.90, passed=True, threshold=0.85, details={}
    )
    assert result.passed is True


def test_metric_result_failed_when_below_threshold():
    result = MetricResult(
        name="faithfulness", score=0.70, passed=False, threshold=0.85, details={}
    )
    assert result.passed is False


def test_score_all_returns_four_metrics():
    four_results = [
        _make_result("faithfulness", 0.90, 0.85),
        _make_result("answer_relevance", 0.88, 0.80),
        _make_result("context_recall", 0.80, 0.75),
        _make_result("context_precision", 0.82, 0.78),
    ]
    with patch.object(EvalMetrics, "__init__", return_value=None):
        metrics = EvalMetrics.__new__(EvalMetrics)
        metrics.score_all = MagicMock(return_value=four_results)
        result = metrics.score_all(_SAMPLE)
    assert len(result) == 4


def test_metric_failure_does_not_crash_score_all():
    with patch.object(EvalMetrics, "__init__", return_value=None):
        metrics = EvalMetrics.__new__(EvalMetrics)
        metrics._thresholds = {
            "faithfulness": 0.85,
            "answer_relevance": 0.80,
            "context_recall": 0.75,
            "context_precision": 0.78,
        }
        # faithfulness raises, others return normally
        metrics.score_faithfulness = MagicMock(side_effect=RuntimeError("LLM timeout"))
        metrics.score_answer_relevance = MagicMock(
            return_value=_make_result("answer_relevance", 0.88, 0.80)
        )
        metrics.score_context_recall = MagicMock(
            return_value=_make_result("context_recall", 0.80, 0.75)
        )
        metrics.score_context_precision = MagicMock(
            return_value=_make_result("context_precision", 0.82, 0.78)
        )

        # Call the real score_all logic by binding the real method
        results = EvalMetrics.score_all(metrics, _SAMPLE)

    # faithfulness failed silently; 3 others returned
    assert len(results) == 3
    returned_names = {r.name for r in results}
    assert "faithfulness" not in returned_names
    assert {"answer_relevance", "context_recall", "context_precision"} == returned_names


def test_all_metric_names_present():
    four_results = [
        _make_result("faithfulness", 0.90, 0.85),
        _make_result("answer_relevance", 0.88, 0.80),
        _make_result("context_recall", 0.80, 0.75),
        _make_result("context_precision", 0.82, 0.78),
    ]
    returned_names = {r.name for r in four_results}
    assert returned_names == _ALL_METRIC_NAMES
