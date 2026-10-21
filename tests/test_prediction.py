import pytest
import pandas as pd
import numpy as np
from src.prediction import (
    predict_single_review,
    predict_batch,
    rating_to_sentiment,
    determine_priority_level
)
from src.model_loader import load_artifacts


@pytest.fixture(scope="module")
def artifacts():
    return load_artifacts()


def test_rating_to_sentiment():
    assert rating_to_sentiment(1) == "Negative"
    assert rating_to_sentiment(2) == "Negative"
    assert rating_to_sentiment(3) == "Neutral"
    assert rating_to_sentiment(4) == "Positive"
    assert rating_to_sentiment(5) == "Positive"


def test_determine_priority_level():
    assert determine_priority_level(0.8, 1, 0.456) == "High Priority"
    assert determine_priority_level(0.1, 5, 0.456) == "Low Priority"
    assert determine_priority_level(0.35, 3, 0.456) == "Medium Priority"


def test_predict_single_review_positive(artifacts):
    res = predict_single_review(
        review_text="This is an absolutely fantastic product! Five stars all the way.",
        review_title="Loved it",
        artifacts=artifacts
    )
    assert res["is_valid"] is True
    assert res["predicted_rating"] in [4, 5]
    assert res["predicted_sentiment"] == "Positive"
    assert res["negative_probability"] < 0.456
    assert res["is_negative_review"] is False
    assert res["priority_level"] == "Low Priority"
    assert res["regression_rating"] >= 3.5


def test_predict_single_review_negative(artifacts):
    res = predict_single_review(
        review_text="Horrible quality, stopped working on day one. Dangerous burning smell. Do not buy!",
        review_title="Terrible defect",
        artifacts=artifacts
    )
    assert res["is_valid"] is True
    assert res["predicted_rating"] in [1, 2]
    assert res["predicted_sentiment"] == "Negative"
    assert res["negative_probability"] >= 0.456
    assert res["is_negative_review"] is True
    assert res["priority_level"] == "High Priority"
    assert "guidance" in res


def test_predict_single_review_empty_text(artifacts):
    res = predict_single_review("", artifacts=artifacts)
    assert res["is_valid"] is False
    assert "error" in res


def test_predict_batch(artifacts):
    df = pd.DataFrame({
        "reviews_text": [
            "Super durable and great sound, highly recommended!",
            "Fell apart after two days, complete waste of money.",
            "Average product, does what it says."
        ],
        "reviews_title": ["Great", "Broken", "Okay"]
    })
    results = predict_batch(df, text_col="reviews_text", title_col="reviews_title", artifacts=artifacts)
    assert len(results) == 3
    assert "predicted_rating" in results.columns
    assert "negative_probability" in results.columns
    assert "is_negative_predicted" in results.columns
    assert "priority_level" in results.columns
    assert results.iloc[0]["predicted_rating"] >= 4
    assert bool(results.iloc[1]["is_negative_predicted"]) is True
