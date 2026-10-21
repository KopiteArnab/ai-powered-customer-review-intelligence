"""
ReviewIQ Model Loader Module.
Locates and safely loads model artifacts, evaluation reports, and metadata.
Gracefully handles missing or optional models without crashing.
"""

import os
import json
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import joblib
import pandas as pd

# Suppress sklearn unpickling version warnings cleanly
warnings.filterwarnings("ignore", category=UserWarning)

# Candidate artifact directories
CANDIDATE_DIRS = [
    Path("review_rating_artifacts"),
    Path("models"),
    Path("artifacts"),
    Path(__file__).resolve().parent.parent / "review_rating_artifacts",
    Path(__file__).resolve().parent.parent / "models",
    Path(__file__).resolve().parent.parent / "artifacts",
]

DEFAULT_METADATA = {
    "negative_review_threshold": 0.45638056644861674,
    "negative_review_definition": "reviews_rating <= 2",
    "target": "reviews_rating",
    "rating_values": [1, 2, 3, 4, 5],
    "model_notes": "Default configuration based on notebook validation"
}


def find_artifact_dir() -> Optional[Path]:
    """Find the directory containing the project artifacts."""
    for candidate in CANDIDATE_DIRS:
        if candidate.exists() and candidate.is_dir():
            if (candidate / "tfidf_logistic_rating_classifier.joblib").exists():
                return candidate
    # Return first existing dir or default
    for candidate in CANDIDATE_DIRS:
        if candidate.exists() and candidate.is_dir():
            return candidate
    return Path("review_rating_artifacts")


class ModelArtifacts:
    """Container holding all loaded models, vectorizers, metadata, and reports."""

    def __init__(self, artifact_dir: Optional[Path] = None):
        self.artifact_dir = artifact_dir or find_artifact_dir()
        self.metadata: Dict[str, Any] = {}
        self.negative_threshold: float = DEFAULT_METADATA["negative_review_threshold"]

        # Models
        self.classification_model = None
        self.negative_review_model = None
        self.regression_model = None
        self.ordinal_models = None
        self.ordinal_vectorizer = None

        # Evaluation DataFrames
        self.model_comparison: Optional[pd.DataFrame] = None
        self.classification_comparison: Optional[pd.DataFrame] = None
        self.regression_comparison: Optional[pd.DataFrame] = None
        self.classification_report: Optional[pd.DataFrame] = None
        self.ordinal_classification_report: Optional[pd.DataFrame] = None
        self.negative_metrics: Optional[pd.DataFrame] = None

        # Status tracking
        self.loaded_status: Dict[str, bool] = {
            "classification_model": False,
            "negative_review_model": False,
            "regression_model": False,
            "ordinal_model": False,
            "metadata": False,
            "evaluation_reports": False
        }
        self.errors: Dict[str, str] = {}

        self._load_all()

    def _load_all(self):
        if not self.artifact_dir or not self.artifact_dir.exists():
            self.errors["artifact_dir"] = f"Directory not found: {self.artifact_dir}"
            self.metadata = DEFAULT_METADATA
            return

        # 1. Load Metadata
        meta_path = self.artifact_dir / "run_metadata.json"
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
                self.negative_threshold = float(
                    self.metadata.get("negative_review_threshold", DEFAULT_METADATA["negative_review_threshold"])
                )
                self.loaded_status["metadata"] = True
            except Exception as e:
                self.errors["metadata"] = str(e)
                self.metadata = DEFAULT_METADATA
        else:
            self.metadata = DEFAULT_METADATA

        # 2. 5-Class Rating Classifier
        clf_path = self.artifact_dir / "tfidf_logistic_rating_classifier.joblib"
        if clf_path.exists():
            try:
                self.classification_model = joblib.load(clf_path)
                self.loaded_status["classification_model"] = True
            except Exception as e:
                self.errors["classification_model"] = str(e)

        # 3. Dedicated Negative Review Detector
        neg_path = self.artifact_dir / "tfidf_negative_review_classifier.joblib"
        if neg_path.exists():
            try:
                self.negative_review_model = joblib.load(neg_path)
                self.loaded_status["negative_review_model"] = True
            except Exception as e:
                self.errors["negative_review_model"] = str(e)

        # 4. Continuous Rating Regressor
        reg_path = self.artifact_dir / "tfidf_ridge_rating_regressor.joblib"
        if reg_path.exists():
            try:
                self.regression_model = joblib.load(reg_path)
                self.loaded_status["regression_model"] = True
            except Exception as e:
                self.errors["regression_model"] = str(e)

        # 5. Cumulative Ordinal Classifier
        ord_m_path = self.artifact_dir / "ordinal_cumulative_logistic_models.joblib"
        ord_v_path = self.artifact_dir / "ordinal_tfidf_vectorizer.joblib"
        if ord_m_path.exists() and ord_v_path.exists():
            try:
                self.ordinal_models = joblib.load(ord_m_path)
                self.ordinal_vectorizer = joblib.load(ord_v_path)
                self.loaded_status["ordinal_model"] = True
            except Exception as e:
                self.errors["ordinal_model"] = str(e)

        # 6. Load Evaluation CSVs
        try:
            reports_loaded = 0
            for name, attr in [
                ("model_comparison.csv", "model_comparison"),
                ("classification_model_comparison.csv", "classification_comparison"),
                ("regression_model_comparison.csv", "regression_comparison"),
                ("classification_report.csv", "classification_report"),
                ("ordinal_classification_report.csv", "ordinal_classification_report"),
                ("negative_review_metrics.csv", "negative_metrics")
            ]:
                csv_path = self.artifact_dir / name
                if csv_path.exists():
                    setattr(self, attr, pd.read_csv(csv_path))
                    reports_loaded += 1
            if reports_loaded > 0:
                self.loaded_status["evaluation_reports"] = True
        except Exception as e:
            self.errors["evaluation_reports"] = str(e)

    def is_operational(self) -> bool:
        """Returns True if at least the primary classification or negative model is loaded."""
        return self.loaded_status["classification_model"] or self.loaded_status["negative_review_model"]


# Cached loader helper for Streamlit
_global_artifacts: Optional[ModelArtifacts] = None


def load_artifacts(force_reload: bool = False) -> ModelArtifacts:
    """
    Returns the singleton ModelArtifacts instance.
    Uses st.cache_resource if running within Streamlit.
    """
    global _global_artifacts
    try:
        import streamlit as st
        # If running inside streamlit, use cache_resource
        @st.cache_resource(show_spinner=False)
        def _get_cached_artifacts():
            return ModelArtifacts()
        return _get_cached_artifacts()
    except Exception:
        # Outside Streamlit or fallback
        if _global_artifacts is None or force_reload:
            _global_artifacts = ModelArtifacts()
        return _global_artifacts
