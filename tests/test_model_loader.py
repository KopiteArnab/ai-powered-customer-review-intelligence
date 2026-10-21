import pytest
from pathlib import Path
from src.model_loader import ModelArtifacts, load_artifacts, find_artifact_dir


def test_artifact_dir_found():
    art_dir = find_artifact_dir()
    assert art_dir is not None
    assert art_dir.exists()


def test_model_artifacts_loading():
    artifacts = load_artifacts(force_reload=True)
    assert artifacts.is_operational()
    assert artifacts.loaded_status["classification_model"] is True
    assert artifacts.loaded_status["negative_review_model"] is True
    assert artifacts.loaded_status["regression_model"] is True
    assert artifacts.classification_model is not None
    assert artifacts.negative_review_model is not None
    assert artifacts.regression_model is not None


def test_metadata_values():
    artifacts = load_artifacts()
    assert isinstance(artifacts.negative_threshold, float)
    assert 0.0 < artifacts.negative_threshold < 1.0
    # Expected threshold from notebook validation
    assert abs(artifacts.negative_threshold - 0.45638) < 0.01


def test_graceful_missing_dir_handling():
    fake_artifacts = ModelArtifacts(artifact_dir=Path("non_existent_folder_xyz"))
    assert not fake_artifacts.is_operational()
    assert fake_artifacts.classification_model is None
    assert "artifact_dir" in fake_artifacts.errors
