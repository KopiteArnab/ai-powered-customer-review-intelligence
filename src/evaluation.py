"""
ReviewIQ Evaluation Module.
Computes live evaluation metrics (classification, regression, negative review triage)
and parses historical notebook evaluation artifacts.
Distinguishes live evaluation from historical benchmarks.
"""

from typing import Dict, Any, List, Optional, Tuple, Sequence, Union
import numpy as np
import pandas as pd

Union_List_Or_Array = Union[List, np.ndarray, pd.Series, Sequence]
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score, fbeta_score, mean_absolute_error,
    mean_squared_error, r2_score, precision_score, recall_score,
    average_precision_score
)

from src.model_loader import ModelArtifacts, load_artifacts


def compute_classification_metrics(
    y_true: Union_List_Or_Array,
    y_pred: Union_List_Or_Array
) -> Dict[str, Any]:
    """Computes comprehensive 5-class rating classification metrics."""
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)

    acc = float(accuracy_score(y_t, y_p))
    bal_acc = float(balanced_accuracy_score(y_t, y_p))
    macro_f1 = float(f1_score(y_t, y_p, labels=[1, 2, 3, 4, 5], average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_t, y_p, labels=[1, 2, 3, 4, 5], average="weighted", zero_division=0))

    cm = confusion_matrix(y_t, y_p, labels=[1, 2, 3, 4, 5])
    report_dict = classification_report(
        y_t, y_p, labels=[1, 2, 3, 4, 5], output_dict=True, zero_division=0
    )

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "confusion_matrix": cm,
        "classification_report": report_dict,
        "sample_size": len(y_t)
    }


def compute_regression_metrics(
    y_true: Union_List_Or_Array,
    y_pred: Union_List_Or_Array
) -> Dict[str, Any]:
    """Computes numeric rating regression metrics (MAE, RMSE, R2)."""
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)

    mae = float(mean_absolute_error(y_t, y_p))
    rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))
    r2 = float(r2_score(y_t, y_p))

    # Clipped MAE
    y_p_clipped = np.clip(y_p, 1.0, 5.0)
    mae_clipped = float(mean_absolute_error(y_t, y_p_clipped))

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mae_clipped": mae_clipped,
        "sample_size": len(y_t)
    }


def compute_negative_triage_metrics(
    y_true_ratings: Union_List_Or_Array,
    neg_probabilities: Union_List_Or_Array,
    threshold: float = 0.456
) -> Dict[str, Any]:
    """
    Computes precision, recall, F1, F2, and confusion matrix for binary negative review triage
    at a specific decision threshold.
    True Negative = rating 1 or 2.
    """
    y_ratings = np.asarray(y_true_ratings, dtype=float)
    probs = np.asarray(neg_probabilities, dtype=float)

    # Ground truth: 1 if rating <= 2, else 0
    y_true_binary = (y_ratings <= 2).astype(int)
    y_pred_binary = (probs >= threshold).astype(int)

    precision = float(precision_score(y_true_binary, y_pred_binary, zero_division=0))
    recall = float(recall_score(y_true_binary, y_pred_binary, zero_division=0))
    f1 = float(f1_score(y_true_binary, y_pred_binary, zero_division=0))
    f2 = float(fbeta_score(y_true_binary, y_pred_binary, beta=2, zero_division=0))

    try:
        avg_precision = float(average_precision_score(y_true_binary, probs))
    except Exception:
        avg_precision = 0.0

    cm = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1])
    # [ [TN, FP], [FN, TP] ]
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f2": f2,
        "average_precision": avg_precision,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "confusion_matrix": cm,
        "sample_size": len(y_true_binary),
        "negative_prevalence": float(y_true_binary.mean())
    }


def compute_two_star_error_breakdown(
    y_true: Union_List_Or_Array,
    y_pred: Union_List_Or_Array
) -> Dict[str, Any]:
    """
    Analyzes why two-star reviews are frequently confused with other star ratings.
    Calculates 2-star recall and destination distribution.
    """
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)

    mask = (y_t == 2)
    total_two_star = int(mask.sum())
    if total_two_star == 0:
        return {"two_star_count": 0, "two_star_recall": 0.0, "distribution": {}}

    preds_for_two_star = y_p[mask]
    two_star_recall = float((preds_for_two_star == 2).sum() / total_two_star)

    dist = {}
    for star in [1, 2, 3, 4, 5]:
        count = int((preds_for_two_star == star).sum())
        dist[star] = {
            "count": count,
            "share": round(count / total_two_star * 100, 2)
        }

    return {
        "two_star_count": total_two_star,
        "two_star_recall": two_star_recall,
        "distribution": dist
    }


def get_historical_benchmarks(artifacts: Optional[ModelArtifacts] = None) -> Dict[str, Any]:
    """
    Retrieves and summarizes the project's recorded evaluation results from disk.
    Explicitly marked as 'Historical Notebook Results'.
    """
    if artifacts is None:
        artifacts = load_artifacts()

    benchmarks = {
        "models_summary": [
            {
                "Model": "TF-IDF + Balanced Logistic Regression",
                "Type": "Classification (5-class)",
                "Accuracy": "69.90%",
                "Macro F1": "0.5289",
                "Balanced Acc": "55.42%",
                "Notes": "Primary rating classifier; handles minority classes via balanced weights."
            },
            {
                "Model": "TF-IDF + Cumulative Ordinal Logistic",
                "Type": "Ordinal Classification",
                "Accuracy": "73.88%",
                "Macro F1": "0.4801",
                "Balanced Acc": "50.71%",
                "Notes": "Higher top-line accuracy but severely degrades 2-star recall (10.14%)."
            },
            {
                "Model": "TF-IDF + Ridge Regression",
                "Type": "Numeric Regression",
                "MAE": "0.4654 stars",
                "RMSE": "0.6865 stars",
                "R²": "0.5708",
                "Notes": "Predicts continuous score; effective for overall sentiment calibration."
            },
            {
                "Model": "Negative-Review Binary Detector",
                "Type": "Binary Classification (1-2 vs 3-5)",
                "Recall (F2 optimized)": "85.90%",
                "Precision": "60.49%",
                "F1 / F2": "0.710 / 0.792",
                "Notes": "Threshold=0.456 selected on validation set for high recall triage."
            },
            {
                "Model": "Majority-Class Baseline",
                "Type": "Baseline",
                "Accuracy": "68.49%",
                "Macro F1": "0.1626",
                "Balanced Acc": "20.00%",
                "Notes": "Always predicts 5 stars; high accuracy due to heavy class imbalance."
            }
        ],
        "negative_triage_reference": {
            "validation_threshold": float(artifacts.negative_threshold),
            "test_recall": 0.8590,
            "test_precision": 0.6049,
            "test_f1": 0.7101,
            "test_f2": 0.7920,
            "test_prevalence": 0.0626
        },
        "raw_tables": {
            "model_comparison": artifacts.model_comparison,
            "classification_comparison": artifacts.classification_comparison,
            "regression_comparison": artifacts.regression_comparison,
            "classification_report": artifacts.classification_report,
            "ordinal_classification_report": artifacts.ordinal_classification_report,
            "negative_metrics": artifacts.negative_metrics
        }
    }
    return benchmarks
