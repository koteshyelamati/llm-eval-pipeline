import pytest

from evals.dataset import DatasetLoader, EvalSample

GOLDEN_SET_PATH = "datasets/golden_set.json"


def test_load_local_success():
    loader = DatasetLoader()
    samples = loader.load_local(GOLDEN_SET_PATH)
    assert len(samples) >= 25
    first = samples[0]
    assert first.id
    assert first.question
    assert first.ground_truth
    assert isinstance(first.contexts, list)
    assert len(first.contexts) > 0
    assert isinstance(first.answer, str)


def test_load_local_missing_file():
    loader = DatasetLoader()
    missing = "datasets/does_not_exist.json"
    with pytest.raises(FileNotFoundError) as exc_info:
        loader.load_local(missing)
    assert missing in str(exc_info.value)


def test_load_returns_eval_samples():
    loader = DatasetLoader()
    samples = loader.load_local(GOLDEN_SET_PATH)
    assert all(isinstance(s, EvalSample) for s in samples)


def test_sample_has_required_fields():
    loader = DatasetLoader()
    samples = loader.load_local(GOLDEN_SET_PATH)
    for sample in samples:
        assert sample.id, f"Empty id on sample: {sample}"
        assert sample.question, f"Empty question on sample {sample.id}"
        assert sample.ground_truth, f"Empty ground_truth on sample {sample.id}"
        assert sample.contexts, f"Empty contexts on sample {sample.id}"
        assert all(c for c in sample.contexts), f"Blank context entry on sample {sample.id}"
