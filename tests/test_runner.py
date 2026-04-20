import json
import os
from unittest.mock import MagicMock, patch

import pytest

from evals.dataset import EvalSample
from evals.metrics import MetricResult

_THRESHOLDS = {
    "faithfulness": 0.85,
    "answer_relevance": 0.80,
    "context_recall": 0.75,
    "context_precision": 0.78,
}

_SAMPLE = EvalSample(
    id="q001",
    question="What is vector search?",
    ground_truth="Vector search finds nearest neighbors in embedding space.",
    contexts=["Vector search uses cosine similarity."],
    answer="",
)

_CFG = {
    "thresholds": _THRESHOLDS,
    "models": {"judge": "gemini-2.5-flash", "embedding": "gemini-embedding-001"},
    "settings": {
        "golden_set_path": "datasets/golden_set.json",
        "results_output": "results/eval_results.json",
        "fail_on_threshold": True,
    },
}


def _passing_results() -> list[MetricResult]:
    return [
        MetricResult("faithfulness", 0.90, True, 0.85, {}),
        MetricResult("answer_relevance", 0.88, True, 0.80, {}),
        MetricResult("context_recall", 0.80, True, 0.75, {}),
        MetricResult("context_precision", 0.82, True, 0.78, {}),
    ]


def _failing_results() -> list[MetricResult]:
    return [
        MetricResult("faithfulness", 0.60, False, 0.85, {}),  # below threshold
        MetricResult("answer_relevance", 0.88, True, 0.80, {}),
        MetricResult("context_recall", 0.80, True, 0.75, {}),
        MetricResult("context_precision", 0.82, True, 0.78, {}),
    ]


def _build_runner(metric_results):
    """Build an EvalRunner with all external I/O mocked out."""
    from evals.runner import EvalRunner

    with patch.dict(os.environ, {"GEMINI_API_KEY": "fake-key"}, clear=False):
        with patch("evals.runner.EvalMetrics") as MockMetrics, \
             patch("evals.runner.DatasetLoader") as MockLoader, \
             patch("evals.runner.EvalReporter") as MockReporter, \
             patch("builtins.open", MagicMock()), \
             patch("yaml.safe_load", return_value=_CFG):

            mock_loader_inst = MockLoader.return_value
            mock_loader_inst.load.return_value = [_SAMPLE]

            mock_metrics_inst = MockMetrics.return_value
            mock_metrics_inst.score_all.return_value = metric_results

            mock_reporter_inst = MockReporter.return_value
            mock_reporter_inst.write_json_results = MagicMock()
            mock_reporter_inst.format_markdown_table.return_value = "## markdown"
            mock_reporter_inst.post_pr_comment = MagicMock()

            runner = EvalRunner.__new__(EvalRunner)
            runner._cfg = _CFG
            runner._loader = mock_loader_inst
            runner._metrics = mock_metrics_inst
            runner._reporter = mock_reporter_inst
            runner._config_path = "config.yml"

    return runner


def test_runner_returns_true_when_all_pass(tmp_path):
    runner = _build_runner(_passing_results())
    with patch("evals.runner.console"), \
         patch.dict(os.environ, {"GEMINI_API_KEY": "fake-key"}, clear=False):
        result = runner.run()
    assert result is True


def test_runner_returns_false_when_any_fail(tmp_path):
    runner = _build_runner(_failing_results())
    with patch("evals.runner.console"), \
         patch.dict(os.environ, {"GEMINI_API_KEY": "fake-key"}, clear=False):
        result = runner.run()
    assert result is False


def test_runner_fails_fast_without_gemini_key():
    from evals.runner import EvalRunner

    env_without_key = {k: v for k, v in os.environ.items() if k != "GEMINI_API_KEY"}
    with patch.dict(os.environ, env_without_key, clear=True), \
         patch("builtins.open", MagicMock()), \
         patch("yaml.safe_load", return_value=_CFG):
        with pytest.raises((EnvironmentError, SystemExit, ValueError)):
            EvalRunner(config_path="config.yml")


def test_results_written_to_disk(tmp_path):
    output_path = str(tmp_path / "results" / "eval_results.json")
    cfg = {**_CFG, "settings": {**_CFG["settings"], "results_output": output_path}}

    runner = _build_runner(_passing_results())
    runner._cfg = cfg

    written = {}

    def fake_write(data, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f)
        written["path"] = path

    runner._reporter.write_json_results = fake_write

    with patch("evals.runner.console"), \
         patch.dict(os.environ, {"GEMINI_API_KEY": "fake-key"}, clear=False):
        runner.run()

    assert os.path.exists(written["path"])
    with open(written["path"]) as f:
        data = json.load(f)
    assert "overall_passed" in data


def test_failed_metrics_listed_in_results():
    runner = _build_runner(_failing_results())

    captured = {}

    def fake_write(data, path):
        captured["data"] = data

    runner._reporter.write_json_results = fake_write

    with patch("evals.runner.console"), \
         patch.dict(os.environ, {"GEMINI_API_KEY": "fake-key"}, clear=False):
        runner.run()

    assert "faithfulness" in captured["data"]["failed_metrics"]
