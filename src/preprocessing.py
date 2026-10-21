"""
ReviewIQ Preprocessing Module.
Preserves the exact text combination and cleaning logic from the notebook.
"""

import re
from typing import Optional, Union, List
import pandas as pd
import numpy as np


def combine_title_and_text(text: Union[str, float, None], title: Union[str, float, None] = "") -> str:
    """
    Combines optional review title and review text into the single model input string.
    Matches notebook logic: title + ". " + text, with leading periods and whitespaces stripped.
    """
    clean_title = "" if (title is None or pd.isna(title)) else str(title).strip()
    clean_text = "" if (text is None or pd.isna(text)) else str(text).strip()

    if clean_title and clean_text:
        combined = f"{clean_title}. {clean_text}".strip()
    elif clean_text:
        combined = clean_text
    elif clean_title:
        combined = clean_title
    else:
        combined = ""

    # Clean leading periods or spaces (e.g. if title was empty or just punctuation)
    combined = re.sub(r"^\.\s*", "", combined)
    return combined.strip()


def clean_review_series(
    text_series: pd.Series,
    title_series: Optional[pd.Series] = None
) -> pd.Series:
    """
    Vectorized transformation of text and optional title series into model_text series.
    """
    clean_text = text_series.fillna("").astype(str).str.strip()
    if title_series is not None and not title_series.empty:
        clean_title = title_series.fillna("").astype(str).str.strip()
        # If title is present, join with '. '
        combined = (clean_title + ". " + clean_text).str.strip()
        combined = combined.str.replace(r"^\.\s*", "", regex=True)
        combined = combined.str.replace(r"\.\s*$", "", regex=True).str.strip()
        return combined
    return clean_text


def normalize_review_for_grouping(text: Union[str, float, None]) -> str:
    """
    Normalizes review body for duplicate detection and grouping.
    Preserves exact regex cleaning from notebook cell 6.
    """
    if text is None or pd.isna(text):
        return ""
    text_str = str(text).lower().strip()
    text_str = re.sub(r"https?://\S+|www\.\S+", " URL ", text_str)
    text_str = re.sub(r"[^a-z0-9]+", " ", text_str)
    return re.sub(r"\s+", " ", text_str).strip()


def validate_and_clean_dataset(
    df: pd.DataFrame,
    text_col: str,
    title_col: Optional[str] = None,
    rating_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Validates and cleans uploaded DataFrame.
    Filters invalid text rows and cleans rating column if provided.
    Does NOT require a rating column for inference.
    """
    data = df.copy()
    if text_col not in data.columns:
        raise ValueError(f"Review text column '{text_col}' not found in DataFrame.")

    # Convert text column to string and strip
    data[text_col] = data[text_col].fillna("").astype(str).str.strip()
    
    # Filter out empty review text
    initial_len = len(data)
    valid_mask = data[text_col].ne("")
    data = data[valid_mask].copy()

    # Title handling
    if title_col and title_col in data.columns:
        titles = data[title_col].fillna("").astype(str).str.strip()
    else:
        titles = None

    data["model_text"] = clean_review_series(data[text_col], titles)

    # Optional rating column validation
    if rating_col and rating_col in data.columns:
        data[rating_col] = pd.to_numeric(data[rating_col], errors="coerce")
        # Identify valid 1-5 ratings for metrics, but keep rows even if rating is missing for inference
        data["is_valid_ground_truth"] = data[rating_col].between(1, 5) & data[rating_col].notna()
        # Binary target: 1 = negative (1-2 stars), 0 = non-negative (3-5 stars)
        data["actual_negative"] = np.where(
            data["is_valid_ground_truth"],
            (data[rating_col] <= 2).astype(int),
            np.nan
        )
    else:
        data["is_valid_ground_truth"] = False
        data["actual_negative"] = np.nan

    return data
