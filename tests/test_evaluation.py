import pytest
import numpy as np
from src.evaluation import (
    compute_classification_metrics,
    compute_regression_metrics,
    compute_negative_triage_metrics,
    compute_two_star_error_breakdown,
    get_historical_benchmarks
)


def test_compute_classification_metrics():
    y_true = [1, 2, 3, 4, 5, 5, 1]
    y_pred = [1, 2, 3, 4, 5, 4, 1]
    metrics = compute_classification_metrics(y_true, y_pred)
    assert metrics["accuracy"] > 0.8
    assert "macro_f1" in metrics
    assert "confusion_matrix" in metrics
    assert metrics["sample_size"] == 7


def test_compute_regression_metrics():
    y_true = [1.0, 2.0, 3.0, 4.0, 5.0]
    y_pred = [1.1, 1.9, 3.2, 3.8, 4.9]
    metrics = compute_regression_metrics(y_true, y_pred)
    assert metrics["mae"] < 0.3
    assert metrics["rmse"] < 0.3
    assert metrics["r2"] > 0.9


def test_compute_negative_triage_metrics():
    # True ratings: [1, 2, 3, 4, 5] -> true negative binary: [1, 1, 0, 0, 0]
    y_true = [1, 2, 3, 4, 5]
    probs = [0.9, 0.7, 0.2, 0.1, 0.05]
    # At threshold 0.456, predictions are [1, 1, 0, 0, 0]
    metrics = compute_negative_triage_metrics(y_true, probs, threshold=0.456)
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["true_positives"] == 2
    assert metrics["true_negatives"] == 3


def test_compute_two_star_error_breakdown():
    y_true = [2, 2, 2, 2]
    y_pred = [1, 2, 4, 5]
    breakdown = compute_two_star_error_breakdown(y_true, y_pred)
    assert breakdown["two_star_count"] == 4
    assert breakdown["two_star_recall"] == 0.25
    assert breakdown["distribution"][1]["count"] == 1
    assert breakdown["distribution"][2]["count"] == 1


def test_historical_benchmarks():
    benchmarks = get_historical_benchmarks()
    assert "models_summary" in benchmarks
    assert len(benchmarks["models_summary"]) >= 4
    assert "negative_triage_reference" in benchmarks
