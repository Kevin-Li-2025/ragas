import pytest

from ragas.metrics._context_precision import NonLLMContextPrecisionWithReference
from ragas.metrics._context_recall import NonLLMContextRecall


def test_non_llm_context_metrics_include_scores_at_threshold():
    recall = NonLLMContextRecall(threshold=0.5)
    precision = NonLLMContextPrecisionWithReference(threshold=0.5)

    assert recall._compute_score([0.5]) == 1.0
    assert precision._calculate_average_precision([1]) == pytest.approx(1.0)
