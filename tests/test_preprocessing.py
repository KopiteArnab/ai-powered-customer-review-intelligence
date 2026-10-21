import pytest
import pandas as pd
from src.preprocessing import (
    combine_title_and_text,
    clean_review_series,
    normalize_review_for_grouping,
    validate_and_clean_dataset
)


def test_combine_title_and_text_both_present():
    title = "Excellent value"
    text = "Works great right out of the box."
    result = combine_title_and_text(text, title)
    assert result == "Excellent value. Works great right out of the box."


def test_combine_title_and_text_text_only():
    text = "Loved this product, completely satisfied."
    result = combine_title_and_text(text, "")
    assert result == "Loved this product, completely satisfied."


def test_combine_title_and_text_handles_none_and_nan():
    assert combine_title_and_text(None, None) == ""
    assert combine_title_and_text("Just text", float("nan")) == "Just text"
    assert combine_title_and_text("", "Just title") == "Just title"


def test_clean_review_series_vectorized():
    texts = pd.Series(["Great sound", "Died quickly", None, ""])
    titles = pd.Series(["Love it", "Horrible", "Title only", ""])
    cleaned = clean_review_series(texts, titles)
    assert cleaned.iloc[0] == "Love it. Great sound"
    assert cleaned.iloc[1] == "Horrible. Died quickly"
    assert cleaned.iloc[2] == "Title only"


def test_normalize_review_for_grouping():
    t1 = "Check out https://example.com for MORE info!!!"
    t2 = "check out for more info"
    norm = normalize_review_for_grouping(t1)
    assert norm == t2


def test_validate_and_clean_dataset():
    df = pd.DataFrame({
        "reviews_text": ["Good item", "   ", None, "Terrible quality"],
        "reviews_title": ["Nice", "", "", "Bad"],
        "reviews_rating": [5, 4, 3, 1]
    })
    cleaned = validate_and_clean_dataset(df, text_col="reviews_text", title_col="reviews_title", rating_col="reviews_rating")
    assert len(cleaned) == 2
    assert "model_text" in cleaned.columns
    assert "is_valid_ground_truth" in cleaned.columns
    assert cleaned.iloc[0]["model_text"] == "Nice. Good item"
    assert cleaned.iloc[1]["model_text"] == "Bad. Terrible quality"
