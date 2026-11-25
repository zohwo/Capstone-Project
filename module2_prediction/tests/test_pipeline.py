from pathlib import Path

from module2_prediction.config import MODEL_HYPERPARAMETER_GRIDS, TARGET_COLUMNS
from module2_prediction.pipeline import run

CLEANED_DATA_PATH = Path("data/processed/cleaned_features.csv")


def test_run_selects_model_and_reports_test_metrics(tmp_path):
    from module1_preprocessing.pipeline import run as preprocess
    preprocess(Path("data/raw/영풍그룹_금속회수율최적화_RawMaterialData_학생배포용.xlsx"),
               tmp_path / "cleaned_features.csv")
    results = run(tmp_path / "pre_impute_data.csv")

    assert set(results.keys()) == set(TARGET_COLUMNS)
    for info in results.values():
        assert info["model"] in MODEL_HYPERPARAMETER_GRIDS
        assert "test_r2" in info
        assert "test_rmse" in info
        assert "test_mae" in info
