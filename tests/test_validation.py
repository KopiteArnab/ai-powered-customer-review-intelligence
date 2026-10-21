import pytest
import pandas as pd
from src.validation import (
    validate_uploaded_csv,
    detect_candidate_columns,
    validate_rating_values
)


def test_validate_uploaded_csv_valid():
    csv_bytes = b"reviews_text,reviews_rating\nGood,5\nBad,1"
    valid, msg, df = validate_uploaded_csv(csv_bytes, "test.csv")
    assert valid is True
    assert df is not None
    assert len(df) == 2


def test_validate_uploaded_csv_invalid_extension():
    valid, msg, df = validate_uploaded_csv(b"some content", "data.xlsx")
    assert valid is False
    assert "Unsupported file format" in msg


def test_validate_uploaded_csv_empty():
    valid, msg, df = validate_uploaded_csv(b"", "empty.csv")
    assert valid is False


def test_detect_candidate_columns():
    df = pd.DataFrame(columns=["name", "reviews_rating", "reviews_text", "reviews_date", "random_col"])
    candidates = detect_candidate_columns(df)
    assert candidates["text_col"] == "reviews_text"
    assert candidates["rating_col"] == "reviews_rating"
    assert candidates["date_col"] == "reviews_date"
    assert candidates["product_col"] == "name"


def test_validate_rating_values():
    s = pd.Series([1, 2, 3, 4, 5, 0, 6, None, "bad"])
    report = validate_rating_values(s)
    assert report["total_count"] == 9
    assert report["valid_count"] == 5
    assert report["out_of_bounds_count"] == 2
    assert report["missing_count"] == 2
