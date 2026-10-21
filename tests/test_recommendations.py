import pytest
import pandas as pd
from src.recommendations import generate_business_recommendations


def test_generate_business_recommendations():
    df = pd.DataFrame({
        "reviews_text": [
            "Terrible broken product, leaks everywhere.",
            "Defective and stopped working after one day.",
            "Lacks durability, flimsy plastic.",
            "Loved it, great performance!",
            "Fantastic quality, highly recommended."
        ],
        "name": ["SKU-A", "SKU-A", "SKU-A", "SKU-B", "SKU-B"],
        "predicted_rating": [1, 1, 2, 5, 5],
        "negative_probability": [0.92, 0.88, 0.75, 0.05, 0.02]
    })
    recs = generate_business_recommendations(
        df,
        text_col="reviews_text",
        pred_rating_col="predicted_rating",
        neg_prob_col="negative_probability",
        product_col="name"
    )
    assert len(recs) >= 2
    types = [r["type"] for r in recs]
    assert "Rule-Based Operational Rule" in types
    for r in recs:
        assert "owner" in r
        assert "evidence" in r
        assert "suggested_action" in r
