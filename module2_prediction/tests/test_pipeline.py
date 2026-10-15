from pathlib import Path

from module2_prediction.config import TARGET_COLUMNS
from module2_prediction.pipeline import run

CLEANED_DATA_PATH = Path("data/processed/cleaned_features.csv")


def test_run_selects_model_and_reports_test_metrics():
    results = run(CLEANED_DATA_PATH)

    assert set(results.keys()) == set(TARGET_COLUMNS)
    for info in results.values():
        assert info["model"] in {"RandomForest", "XGBoost", "LightGBM"}
        assert "test_r2" in info
        assert "test_rmse" in info
        assert "test_mae" in info
