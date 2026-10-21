"""
ReviewIQ Prediction Module.
Implements inference for single reviews and batches using trained artifacts.
Includes 5-class Logistic Regression, Ridge Regression, Binary Negative Detector,
Cumulative Ordinal Logistic Regression, and optional Hugging Face Zero-Shot classification.
"""

import os
from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd

from src.preprocessing import combine_title_and_text, clean_review_series
from src.model_loader import ModelArtifacts, load_artifacts


def rating_to_sentiment(rating: Union[int, float]) -> str:
    """Converts 1-5 rating into standard 3-tier sentiment."""
    try:
        r = float(rating)
        if r <= 2.0:
            return "Negative"
        elif r <= 3.0:
            return "Neutral"
        else:
            return "Positive"
    except (ValueError, TypeError):
        return "Unknown"


def determine_priority_level(
    negative_probability: float,
    predicted_rating: int,
    negative_threshold: float = 0.456
) -> str:
    """
    Transparent, rule-based priority categorization for support triage:
    - High Priority: negative probability >= threshold OR predicted rating <= 2
    - Medium Priority: negative probability >= (threshold * 0.70) OR predicted rating == 3
    - Low Priority: all other positive reviews
    """
    if negative_probability >= negative_threshold or predicted_rating in (1, 2):
        return "High Priority"
    elif negative_probability >= (negative_threshold * 0.70) or predicted_rating == 3:
        return "Medium Priority"
    else:
        return "Low Priority"


def generate_support_guidance(
    predicted_rating: int,
    negative_probability: float,
    priority: str
) -> Dict[str, str]:
    """Generates concrete operational recommendations for customer support."""
    if priority == "High Priority":
        if predicted_rating == 1:
            return {
                "action": "Immediate Human Outreach & Defection Prevention",
                "sla": "Within 2-4 Hours",
                "suggested_response": "Issue direct apology, offer no-hassle refund or express replacement, and flag root cause (defect/breakage/safety) in CRM.",
                "routing": "Tier 2 Support / Escalation Specialist"
            }
        else:
            return {
                "action": "Resolution & Feedback Capture",
                "sla": "Within 12 Hours",
                "suggested_response": "Acknowledge friction point, investigate product usability or delivery delay, and provide partial credit or troubleshooting guide.",
                "routing": "Tier 1 Senior Representative"
            }
    elif priority == "Medium Priority":
        return {
            "action": "Proactive Engagement & Experience Monitoring",
            "sla": "Within 24-48 Hours",
            "suggested_response": "Send polite inquiry to clarify product expectations or offer tips on optimal product usage.",
            "routing": "Customer Success / Retention Team"
        }
    else:
        return {
            "action": "Advocate Nurturing & Review Retention",
            "sla": "Automated / Standard",
            "suggested_response": "Send automated thank-you note, invite to loyalty/referral rewards program, and route positive quotes to marketing.",
            "routing": "Automated Engagement / Marketing"
        }


def predict_single_review(
    review_text: str,
    review_title: str = "",
    artifacts: Optional[ModelArtifacts] = None,
    negative_threshold: Optional[float] = None
) -> Dict[str, Any]:
    """
    Executes full multi-model prediction on a single customer review.
    Preserves exact feature construction and classification/regression logic from notebook.
    """
    if artifacts is None:
        artifacts = load_artifacts()

    threshold = negative_threshold if negative_threshold is not None else artifacts.negative_threshold
    model_input = combine_title_and_text(review_text, review_title)

    if not model_input:
        return {
            "error": "Review text is empty. Please enter customer review content.",
            "is_valid": False
        }

    results: Dict[str, Any] = {
        "is_valid": True,
        "input_text": model_input,
        "negative_threshold": threshold,
        "limitations": (
            "Model predictions reflect linguistic sentiment patterns in text only, "
            "not verified product defect status or future customer behavior. "
            "Ratings can be influenced by external context not present in the review text."
        )
    }

    # 1. 5-Class Rating Classifier
    if artifacts.classification_model is not None:
        try:
            clf = artifacts.classification_model
            preds = clf.predict([model_input])[0]
            probs = clf.predict_proba([model_input])[0]
            classes = clf.named_steps["classifier"].classes_

            results["predicted_rating"] = int(preds)
            results["predicted_sentiment"] = rating_to_sentiment(results["predicted_rating"])

            # Class probability distribution
            prob_dist = {int(c): float(p) for c, p in zip(classes, probs)}
            results["rating_probabilities"] = prob_dist
            results["model_confidence"] = float(prob_dist.get(int(preds), max(probs)))
            results["primary_model_name"] = "TF-IDF + Balanced Logistic Regression"
        except Exception as e:
            results["classification_error"] = str(e)
            results["predicted_rating"] = 3
            results["predicted_sentiment"] = "Unknown"
            results["model_confidence"] = 0.0
    else:
        results["predicted_rating"] = 3
        results["predicted_sentiment"] = "Unavailable"
        results["model_confidence"] = 0.0
        results["classification_error"] = "Classification model artifact not loaded."

    # 2. Continuous Rating Regressor
    if artifacts.regression_model is not None:
        try:
            reg = artifacts.regression_model
            raw_reg = float(reg.predict([model_input])[0])
            results["regression_rating"] = round(float(np.clip(raw_reg, 1.0, 5.0)), 2)
            results["regression_raw"] = round(raw_reg, 2)
            results["regression_model_name"] = "TF-IDF + Ridge Regression"
        except Exception as e:
            results["regression_error"] = str(e)
            results["regression_rating"] = None
    else:
        results["regression_rating"] = None

    # 3. Dedicated Binary Negative Review Classifier
    if artifacts.negative_review_model is not None:
        try:
            neg_clf = artifacts.negative_review_model
            neg_prob = float(neg_clf.predict_proba([model_input])[0, 1])
            results["negative_probability"] = round(neg_prob, 4)
            results["is_negative_review"] = bool(neg_prob >= threshold)
            results["negative_model_name"] = "TF-IDF + Balanced Logistic Binary Classifier"
        except Exception as e:
            results["negative_error"] = str(e)
            results["negative_probability"] = 0.0
            results["is_negative_review"] = False
    else:
        # Fallback to rating definition (<= 2)
        results["negative_probability"] = 1.0 if results["predicted_rating"] <= 2 else 0.0
        results["is_negative_review"] = bool(results["predicted_rating"] <= 2)

    # 4. Cumulative Ordinal Classifier (Optional / Diagnostic)
    if artifacts.ordinal_models is not None and artifacts.ordinal_vectorizer is not None:
        try:
            ord_vec = artifacts.ordinal_vectorizer
            ord_models = artifacts.ordinal_models
            X_tfidf = ord_vec.transform([model_input])

            cumulative = np.column_stack([
                ord_models[cutoff].predict_proba(X_tfidf)[:, 1]
                for cutoff in [1, 2, 3, 4]
            ])
            cumulative = np.minimum.accumulate(cumulative, axis=1)
            class_probs = np.column_stack([
                1.0 - cumulative[:, 0],
                cumulative[:, 0] - cumulative[:, 1],
                cumulative[:, 1] - cumulative[:, 2],
                cumulative[:, 2] - cumulative[:, 3],
                cumulative[:, 3]
            ])
            class_probs = np.clip(class_probs, 0.0, 1.0)
            row_sums = class_probs.sum(axis=1, keepdims=True)
            norm_probs = (class_probs / np.where(row_sums == 0, 1.0, row_sums))[0]

            ord_rating = int(np.argmax(norm_probs) + 1)
            results["ordinal_predicted_rating"] = ord_rating
            results["ordinal_probabilities"] = {i + 1: float(norm_probs[i]) for i in range(5)}
            results["ordinal_model_name"] = "TF-IDF + Cumulative Ordinal Logistic Regression"
        except Exception as e:
            results["ordinal_error"] = str(e)

    # 5. Priority and Guidance
    priority = determine_priority_level(
        results.get("negative_probability", 0.0),
        results["predicted_rating"],
        threshold
    )
    results["priority_level"] = priority
    results["guidance"] = generate_support_guidance(
        results["predicted_rating"],
        results.get("negative_probability", 0.0),
        priority
    )

    return results


def predict_batch(
    df: pd.DataFrame,
    text_col: str,
    title_col: Optional[str] = None,
    artifacts: Optional[ModelArtifacts] = None,
    negative_threshold: Optional[float] = None,
    batch_size: int = 2000
) -> pd.DataFrame:
    """
    Executes vectorized batch inference over a customer review DataFrame.
    Returns a copy of the original DataFrame augmented with prediction columns.
    Never fits or retrains on uploaded data.
    """
    if artifacts is None:
        artifacts = load_artifacts()

    threshold = negative_threshold if negative_threshold is not None else artifacts.negative_threshold
    data = df.copy()

    # Build model_text
    title_series = data[title_col] if (title_col and title_col in data.columns) else None
    model_texts = clean_review_series(data[text_col], title_series)
    data["_model_input"] = model_texts

    # Identify non-empty rows
    valid_mask = data["_model_input"].astype(str).str.strip().ne("")
    valid_indices = data.index[valid_mask]
    clean_inputs = data.loc[valid_indices, "_model_input"].tolist()

    # Initialize output columns
    data["predicted_rating"] = np.nan
    data["predicted_confidence"] = np.nan
    data["predicted_sentiment"] = "Unknown"
    data["negative_probability"] = np.nan
    data["is_negative_predicted"] = False
    data["priority_level"] = "Low Priority"
    data["regression_rating"] = np.nan

    if not clean_inputs:
        data.drop(columns=["_model_input"], inplace=True)
        return data

    # 1. 5-Class Rating Classifier
    if artifacts.classification_model is not None:
        try:
            clf = artifacts.classification_model
            preds = clf.predict(clean_inputs)
            probs = clf.predict_proba(clean_inputs)
            max_probs = np.max(probs, axis=1)

            data.loc[valid_indices, "predicted_rating"] = preds.astype(int)
            data.loc[valid_indices, "predicted_confidence"] = np.round(max_probs, 4)
            data.loc[valid_indices, "predicted_sentiment"] = [rating_to_sentiment(p) for p in preds]
        except Exception as e:
            pass

    # 2. Continuous Rating Regressor
    if artifacts.regression_model is not None:
        try:
            reg = artifacts.regression_model
            reg_preds = reg.predict(clean_inputs)
            data.loc[valid_indices, "regression_rating"] = np.round(np.clip(reg_preds, 1.0, 5.0), 2)
        except Exception:
            pass

    # 3. Dedicated Negative Review Classifier
    if artifacts.negative_review_model is not None:
        try:
            neg_clf = artifacts.negative_review_model
            neg_probs = neg_clf.predict_proba(clean_inputs)[:, 1]
            data.loc[valid_indices, "negative_probability"] = np.round(neg_probs, 4)
            data.loc[valid_indices, "is_negative_predicted"] = (neg_probs >= threshold)
        except Exception:
            pass
    else:
        # Fallback if binary model is unavailable
        is_neg = data.loc[valid_indices, "predicted_rating"] <= 2
        data.loc[valid_indices, "is_negative_predicted"] = is_neg
        data.loc[valid_indices, "negative_probability"] = np.where(is_neg, 1.0, 0.0)

    # 4. Priority Level Assignment
    neg_p = data.loc[valid_indices, "negative_probability"].fillna(0.0).values
    ratings = data.loc[valid_indices, "predicted_rating"].fillna(3).values

    priorities = [
        determine_priority_level(p, int(r), threshold)
        for p, r in zip(neg_p, ratings)
    ]
    data.loc[valid_indices, "priority_level"] = priorities

    # Clean up internal column
    data.drop(columns=["_model_input"], inplace=True)
    return data


def predict_zero_shot_hf(
    text: str,
    model_name: str = "facebook/bart-large-mnli",
    hf_token: Optional[str] = None
) -> Dict[str, Any]:
    """
    Optional Hugging Face Zero-Shot rating classification.
    Only called if explicitly triggered by the user.
    """
    try:
        from transformers import pipeline

        token = hf_token or os.getenv("HF_TOKEN")
        zero_shot = pipeline(
            "zero-shot-classification",
            model=model_name,
            token=token
        )

        rating_labels = [
            "a one-star customer review expressing a very negative experience",
            "a two-star customer review expressing a mostly negative experience",
            "a three-star customer review expressing a mixed or neutral experience",
            "a four-star customer review expressing a positive experience",
            "a five-star customer review expressing a very positive experience"
        ]
        label_to_rating = {label: i + 1 for i, label in enumerate(rating_labels)}

        result = zero_shot(str(text)[:2500], candidate_labels=rating_labels, multi_label=False)
        best_label = result["labels"][0]
        predicted_rating = label_to_rating[best_label]
        best_score = float(result["scores"][0])

        return {
            "success": True,
            "predicted_rating": predicted_rating,
            "best_label": best_label,
            "confidence": round(best_score, 4),
            "all_scores": {
                label_to_rating[l]: round(float(s), 4)
                for l, s in zip(result["labels"], result["scores"])
            },
            "model_name": model_name
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Hugging Face zero-shot pipeline is optional and requires online access/dependencies."
        }
