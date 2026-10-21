"""
ReviewIQ Artifact Export and Retraining Script.
Reproduces the complete training, validation-threshold selection, evaluation,
and artifact export pipeline directly from reviews.csv, strictly preserving
the notebook's duplicate-group safeguards and random seeds.
"""

import os
import re
import json
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score, fbeta_score, mean_absolute_error,
    mean_squared_error, r2_score, precision_score, recall_score,
    average_precision_score, precision_recall_curve
)
import joblib

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
warnings.filterwarnings("ignore")


def normalize_review_for_grouping(text):
    text = str(text).lower().strip()
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def run_pipeline(data_path="reviews.csv", output_dir="review_rating_artifacts"):
    print(f"[ReviewIQ] Starting artifact export using {data_path}...")
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    if not Path(data_path).exists():
        raise FileNotFoundError(f"Dataset '{data_path}' not found.")

    df = pd.read_csv(data_path)
    print(f"Loaded raw dataset: {df.shape}")

    data = df.copy()
    data["reviews_rating"] = pd.to_numeric(data["reviews_rating"], errors="coerce")
    data = data.dropna(subset=["reviews_rating", "reviews_text"]).copy()
    data["reviews_text"] = data["reviews_text"].astype(str).str.strip()
    data = data[data["reviews_text"].ne("")].copy()
    data = data[data["reviews_rating"].between(1, 5)].copy()
    data = data[data["reviews_rating"].mod(1).eq(0)].copy()
    data["reviews_rating"] = data["reviews_rating"].astype(int)

    if "reviews_title" in data.columns:
        title = data["reviews_title"].fillna("").astype(str).str.strip()
    else:
        title = pd.Series("", index=data.index)

    data["model_text"] = (title + ". " + data["reviews_text"].astype(str)).str.strip()
    data["model_text"] = data["model_text"].str.replace(r"^\.\s*", "", regex=True)

    data["duplicate_group"] = data["reviews_text"].map(normalize_review_for_grouping)
    blank_group = data["duplicate_group"].eq("")
    data.loc[blank_group, "duplicate_group"] = [f"blank_review_{i}" for i in data.index[blank_group]]

    print(f"Cleaned dataset rows: {len(data):,}")
    print(f"Unique duplicate groups: {data['duplicate_group'].nunique():,}")

    X = data["model_text"].fillna("").astype(str)
    y = data["reviews_rating"].astype(int)
    groups = data["duplicate_group"]

    group_splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_idx, test_idx = next(group_splitter.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    groups_train, groups_test = groups.iloc[train_idx], groups.iloc[test_idx]

    overlap_groups = set(groups_train).intersection(set(groups_test))
    assert not overlap_groups, f"Leakage detected: {len(overlap_groups)} groups overlap"
    print(f"Train rows: {len(X_train):,}, Test rows: {len(X_test):,}")

    # Baseline
    majority_clf = DummyClassifier(strategy="most_frequent")
    majority_clf.fit(np.zeros((len(y_train), 1)), y_train)
    y_pred_majority = majority_clf.predict(np.zeros((len(y_test), 1)))

    baseline_classification = {
        "model": "Majority-class baseline",
        "evaluation_set": "full held-out test",
        "n_evaluated": len(y_test),
        "accuracy": accuracy_score(y_test, y_pred_majority),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred_majority),
        "macro_f1": f1_score(y_test, y_pred_majority, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_test, y_pred_majority, average="weighted", zero_division=0)
    }

    # 1. 5-Class Classifier
    print("Training 5-class TF-IDF + Logistic Regression...")
    classification_model = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=2,
            max_features=150_000,
            sublinear_tf=True
        )),
        ("classifier", LogisticRegression(
            max_iter=1500,
            class_weight="balanced",
            random_state=SEED
        ))
    ])
    classification_model.fit(X_train, y_train)
    y_pred_class = classification_model.predict(X_test)

    class_metrics = pd.DataFrame(classification_report(
        y_test, y_pred_class, labels=[1, 2, 3, 4, 5], output_dict=True, zero_division=0
    )).T

    classification_comparison_rows = [
        baseline_classification,
        {
            "model": "TF-IDF + balanced Logistic Regression",
            "evaluation_set": "full held-out test",
            "n_evaluated": len(y_test),
            "accuracy": accuracy_score(y_test, y_pred_class),
            "balanced_accuracy": balanced_accuracy_score(y_test, y_pred_class),
            "macro_f1": f1_score(y_test, y_pred_class, labels=[1, 2, 3, 4, 5], average="macro", zero_division=0),
            "weighted_f1": f1_score(y_test, y_pred_class, labels=[1, 2, 3, 4, 5], average="weighted", zero_division=0)
        }
    ]

    # 2. Ordinal Model
    print("Training Cumulative Ordinal Logistic Regression...")
    ordinal_vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_features=150_000,
        sublinear_tf=True
    )
    X_train_tfidf_ordinal = ordinal_vectorizer.fit_transform(X_train)
    X_test_tfidf_ordinal = ordinal_vectorizer.transform(X_test)

    ordinal_binary_models = {}
    for cutoff in [1, 2, 3, 4]:
        y_train_binary = (y_train > cutoff).astype(int)
        binary_model = LogisticRegression(
            max_iter=1500,
            class_weight="balanced",
            random_state=SEED
        )
        binary_model.fit(X_train_tfidf_ordinal, y_train_binary)
        ordinal_binary_models[cutoff] = binary_model

    def ordinal_predict_proba(X_tfidf):
        cumulative = np.column_stack([
            ordinal_binary_models[cutoff].predict_proba(X_tfidf)[:, 1]
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
        return class_probs / np.where(row_sums == 0, 1.0, row_sums)

    ord_probs = ordinal_predict_proba(X_test_tfidf_ordinal)
    y_pred_ordinal = np.argmax(ord_probs, axis=1) + 1

    ordinal_class_report = pd.DataFrame(classification_report(
        y_test, y_pred_ordinal, labels=[1, 2, 3, 4, 5], output_dict=True, zero_division=0
    )).T

    classification_comparison_rows.append({
        "model": "TF-IDF + cumulative ordinal Logistic Regression",
        "evaluation_set": "full held-out test",
        "n_evaluated": len(y_test),
        "accuracy": accuracy_score(y_test, y_pred_ordinal),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred_ordinal),
        "macro_f1": f1_score(y_test, y_pred_ordinal, labels=[1, 2, 3, 4, 5], average="macro", zero_division=0),
        "weighted_f1": f1_score(y_test, y_pred_ordinal, labels=[1, 2, 3, 4, 5], average="weighted", zero_division=0)
    })

    # 3. Negative Review Binary Classifier & Validation Threshold Selection
    print("Selecting validation threshold and training Negative Review Classifier...")
    y_negative_train = y_train.le(2).astype(int)
    y_negative_test = y_test.le(2).astype(int)

    neg_splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED + 1)
    fit_rel_idx, val_rel_idx = next(neg_splitter.split(X_train, y_negative_train, groups=groups_train))
    X_neg_fit, X_neg_val = X_train.iloc[fit_rel_idx], X_train.iloc[val_rel_idx]
    y_neg_fit, y_neg_val = y_negative_train.iloc[fit_rel_idx], y_negative_train.iloc[val_rel_idx]

    def make_neg_model():
        return Pipeline([
            ("tfidf", TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                ngram_range=(1, 2),
                min_df=2,
                max_features=150_000,
                sublinear_tf=True
            )),
            ("classifier", LogisticRegression(
                max_iter=1500,
                class_weight="balanced",
                random_state=SEED
            ))
        ])

    thresh_model = make_neg_model()
    thresh_model.fit(X_neg_fit, y_neg_fit)
    val_probs = thresh_model.predict_proba(X_neg_val)[:, 1]
    val_p, val_r, val_th = precision_recall_curve(y_neg_val, val_probs)

    val_f2 = (5 * val_p[:-1] * val_r[:-1]) / (4 * val_p[:-1] + val_r[:-1] + 1e-12)
    best_idx = int(np.nanargmax(val_f2))
    NEGATIVE_THRESHOLD = float(val_th[best_idx])
    print(f"Validation Selected Negative Threshold (max F2): {NEGATIVE_THRESHOLD:.5f}")

    negative_review_model = make_neg_model()
    negative_review_model.fit(X_train, y_negative_train)
    neg_test_probs = negative_review_model.predict_proba(X_test)[:, 1]
    y_pred_neg = (neg_test_probs >= NEGATIVE_THRESHOLD).astype(int)

    negative_metrics = {
        "model": "TF-IDF + balanced Logistic Regression (negative vs non-negative)",
        "threshold_selected_on_validation": NEGATIVE_THRESHOLD,
        "negative_precision": precision_score(y_negative_test, y_pred_neg, zero_division=0),
        "negative_recall": recall_score(y_negative_test, y_pred_neg, zero_division=0),
        "negative_f1": f1_score(y_negative_test, y_pred_neg, zero_division=0),
        "negative_f2": fbeta_score(y_negative_test, y_pred_neg, beta=2, zero_division=0),
        "average_precision": average_precision_score(y_negative_test, neg_test_probs),
        "balanced_accuracy": balanced_accuracy_score(y_negative_test, y_pred_neg),
        "test_negative_prevalence": float(y_negative_test.mean())
    }

    # 4. Continuous Rating Regressor
    print("Training Ridge Regressor...")
    regression_model = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=2,
            max_features=150_000,
            sublinear_tf=True
        )),
        ("regressor", Ridge(alpha=2.0))
    ])
    regression_model.fit(X_train, y_train)
    y_pred_reg = regression_model.predict(X_test)

    # Median baseline
    median_reg = DummyRegressor(strategy="median")
    median_reg.fit(np.zeros((len(y_train), 1)), y_train)
    y_pred_med = median_reg.predict(np.zeros((len(y_test), 1)))

    regression_comparison_rows = [
        {
            "model": "Median-rating baseline",
            "evaluation_set": "full held-out test",
            "n_evaluated": len(y_test),
            "MAE": mean_absolute_error(y_test, y_pred_med),
            "RMSE": mean_squared_error(y_test, y_pred_med) ** 0.5,
            "R2": r2_score(y_test, y_pred_med)
        },
        {
            "model": "TF-IDF + Ridge Regression",
            "evaluation_set": "full held-out test",
            "n_evaluated": len(y_test),
            "MAE": mean_absolute_error(y_test, y_pred_reg),
            "RMSE": mean_squared_error(y_test, y_pred_reg) ** 0.5,
            "R2": r2_score(y_test, y_pred_reg)
        },
        {
            "model": "Ordinal classifier (star prediction)",
            "evaluation_set": "full held-out test",
            "n_evaluated": len(y_test),
            "MAE": mean_absolute_error(y_test, y_pred_ordinal),
            "RMSE": mean_squared_error(y_test, y_pred_ordinal) ** 0.5,
            "R2": r2_score(y_test, y_pred_ordinal)
        }
    ]

    # Save artifacts
    print(f"Exporting model artifacts to {output_path}...")
    joblib.dump(classification_model, output_path / "tfidf_logistic_rating_classifier.joblib")
    joblib.dump(regression_model, output_path / "tfidf_ridge_rating_regressor.joblib")
    joblib.dump(ordinal_vectorizer, output_path / "ordinal_tfidf_vectorizer.joblib")
    joblib.dump(ordinal_binary_models, output_path / "ordinal_cumulative_logistic_models.joblib")
    joblib.dump(negative_review_model, output_path / "tfidf_negative_review_classifier.joblib")

    pd.DataFrame(classification_comparison_rows).to_csv(output_path / "classification_model_comparison.csv", index=False)
    pd.DataFrame(regression_comparison_rows).to_csv(output_path / "regression_model_comparison.csv", index=False)
    class_metrics.to_csv(output_path / "classification_report.csv")
    ordinal_class_report.to_csv(output_path / "ordinal_classification_report.csv")
    pd.DataFrame([negative_metrics]).to_csv(output_path / "negative_review_metrics.csv", index=False)

    metadata = {
        "random_seed": SEED,
        "dataset_rows_after_cleaning": int(len(data)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "target": "reviews_rating",
        "input_text": "reviews_title + reviews_text",
        "split_strategy": "StratifiedGroupKFold grouped by normalized review body",
        "duplicate_group_overlap": 0,
        "negative_review_definition": "reviews_rating <= 2",
        "negative_review_threshold": NEGATIVE_THRESHOLD,
        "negative_review_threshold_selection": "maximum validation F2; duplicate-group-aware fit/validation split",
        "negative_review_validation_metrics": {
            "selected_threshold": NEGATIVE_THRESHOLD,
            "selection_method": "Selected by maximum validation F2 (beta=2; recall-weighted)",
            "validation_precision": precision_score(y_neg_val, (val_probs >= NEGATIVE_THRESHOLD).astype(int), zero_division=0),
            "validation_recall": recall_score(y_neg_val, (val_probs >= NEGATIVE_THRESHOLD).astype(int), zero_division=0),
            "validation_f2": fbeta_score(y_neg_val, (val_probs >= NEGATIVE_THRESHOLD).astype(int), beta=2, zero_division=0),
            "fit_rows_for_threshold_selection": len(X_neg_fit),
            "validation_rows": len(X_neg_val)
        },
        "rating_values": [1, 2, 3, 4, 5]
    }
    with open(output_path / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("[ReviewIQ] Artifact export complete!")


if __name__ == "__main__":
    run_pipeline()
