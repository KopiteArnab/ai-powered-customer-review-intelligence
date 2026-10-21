"""
ReviewIQ Data Validation and Guardrails Module.
Enforces security, schema validation, file size constraints, and error explanations.
"""

from typing import Tuple, Optional, Dict, Any, List
import io
import pandas as pd
import numpy as np

MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB limit
MAX_BATCH_ROWS = 100_000


def validate_uploaded_csv(
    file_bytes: bytes,
    filename: str,
    max_size_mb: int = 50
) -> Tuple[bool, str, Optional[pd.DataFrame]]:
    """
    Validates uploaded CSV file for size, format, readability, and content.
    Returns (is_valid, message, dataframe).
    """
    # 1. Format check
    if not filename.lower().endswith(".csv"):
        return False, "Unsupported file format. Please upload a standard CSV (.csv) file.", None

    # 2. Size check
    max_bytes = max_size_mb * 1024 * 1024
    if len(file_bytes) > max_bytes:
        return (
            False,
            f"File exceeds maximum allowed size ({max_size_mb} MB). Please upload a smaller file.",
            None
        )

    if len(file_bytes) == 0:
        return False, "Uploaded file is completely empty.", None

    # 3. Readability check with encoding fallbacks
    df = None
    for encoding in ["utf-8", "latin-1", "iso-8859-1", "cp1252"]:
        try:
            df = pd.read_csv(io.BytesIO(file_bytes), encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
        except Exception as e:
            return False, f"Failed to parse CSV: {str(e)}", None

    if df is None:
        return False, "Failed to parse CSV file with standard character encodings.", None

    # 4. Content checks
    if df.empty:
        return False, "Uploaded CSV contains no rows of data.", None

    if len(df.columns) < 1:
        return False, "Uploaded CSV contains no recognized columns.", None

    if len(df) > MAX_BATCH_ROWS:
        return (
            False,
            f"Dataset has {len(df):,} rows, exceeding the batch limit of {MAX_BATCH_ROWS:,} rows. "
            "Please upload a subset for interactive analysis.",
            None
        )

    return True, f"File successfully loaded with {len(df):,} rows and {len(df.columns)} columns.", df


def detect_candidate_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """
    Inspects DataFrame column names and automatically detects review text,
    title, rating, date, and product columns using pattern heuristics.
    """
    cols = [str(c) for c in df.columns]
    cols_lower = [c.lower() for c in cols]

    candidates: Dict[str, Optional[str]] = {
        "text_col": None,
        "title_col": None,
        "rating_col": None,
        "date_col": None,
        "product_col": None
    }

    # Text column heuristics
    text_patterns = [
        "reviews_text", "review_text", "reviewtext", "review_body", "reviews_body",
        "review", "text", "body", "content", "comment", "comments", "feedback"
    ]
    for pattern in text_patterns:
        if pattern in cols_lower:
            candidates["text_col"] = cols[cols_lower.index(pattern)]
            break

    # Title column heuristics
    title_patterns = [
        "reviews_title", "review_title", "reviewtitle", "title", "headline",
        "summary", "subject", "header"
    ]
    for pattern in title_patterns:
        if pattern in cols_lower:
            candidates["title_col"] = cols[cols_lower.index(pattern)]
            break

    # Rating column heuristics
    rating_patterns = [
        "reviews_rating", "review_rating", "rating", "ratings", "stars",
        "score", "star_rating", "overall"
    ]
    for pattern in rating_patterns:
        if pattern in cols_lower:
            candidates["rating_col"] = cols[cols_lower.index(pattern)]
            break

    # Date column heuristics
    date_patterns = [
        "reviews_date", "review_date", "date", "created_at", "timestamp",
        "reviewdate", "time", "submission_date"
    ]
    for pattern in date_patterns:
        if pattern in cols_lower:
            candidates["date_col"] = cols[cols_lower.index(pattern)]
            break

    # Product/Brand column heuristics
    product_patterns = [
        "name", "product_name", "product", "item_name", "item", "brand",
        "manufacturer", "asin", "sku"
    ]
    for pattern in product_patterns:
        if pattern in cols_lower:
            candidates["product_col"] = cols[cols_lower.index(pattern)]
            break

    return candidates


def validate_rating_values(series: pd.Series) -> Dict[str, Any]:
    """Validates whether a column contains acceptable 1-5 star ratings."""
    num_series = pd.to_numeric(series, errors="coerce")
    total_count = len(series)
    valid_mask = num_series.between(1, 5) & num_series.notna()
    valid_count = int(valid_mask.sum())
    missing_or_non_numeric = int(num_series.isna().sum())
    out_of_bounds = int((~valid_mask & num_series.notna()).sum())

    return {
        "total_count": total_count,
        "valid_count": valid_count,
        "valid_share": round(valid_count / total_count * 100, 1) if total_count > 0 else 0.0,
        "missing_count": missing_or_non_numeric,
        "out_of_bounds_count": out_of_bounds,
        "is_usable_for_eval": valid_count >= 5
    }
