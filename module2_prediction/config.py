from pathlib import Path

TARGET_COLUMNS = ["Ni_회수율_pct", "Co_회수율_pct"]

COMPOSITION_COLUMNS = [
    "Ni_함량_pct",
    "Co_함량_pct",
    "Li_함량_pct",
    "Mn_함량_pct",
    "Cu_불순물_pct",
    "Fe_불순물_pct",
    "Al_불순물_pct",
]

PROCESS_COLUMNS = ["침출온도_C", "침출시간_hr"]

DERIVED_COLUMNS = [
    "acid_to_ore_ratio",
    "leach_intensity",
    "total_impurity_pct",
    "target_purity_ratio",
]

FEATURE_COLUMNS = COMPOSITION_COLUMNS + PROCESS_COLUMNS + DERIVED_COLUMNS

RAW_BLEND_INPUT_COLUMNS = ["원료투입량_kg", "황산투입량_L"]

CLEANED_DATA_PATH = Path("data/processed/cleaned_features.csv")

MODEL_HYPERPARAMETER_GRIDS = {
    "RandomForest": [
        {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1},
        {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 1},
        {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 1},
        {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 2},
        {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 4},
        {"n_estimators": 100, "max_depth": 10, "min_samples_leaf": 1},
        {"n_estimators": 200, "max_depth": 10, "min_samples_leaf": 2},
        {"n_estimators": 100, "max_depth": 5, "min_samples_leaf": 1},
        {"n_estimators": 200, "max_depth": 5, "min_samples_leaf": 1},
    ],
    "XGBoost": [
        {"n_estimators": 100, "max_depth": 2, "learning_rate": 0.1},
        {"n_estimators": 200, "max_depth": 2, "learning_rate": 0.1},
        {"n_estimators": 300, "max_depth": 2, "learning_rate": 0.05},
        {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.05},
        {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.03},
        {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.03},
        {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.1, "subsample": 0.8, "colsample_bytree": 0.8},
        {"n_estimators": 200, "max_depth": 2, "learning_rate": 0.1, "subsample": 0.8, "colsample_bytree": 0.8},
        {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.05},
        {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.03},
    ],
    "LightGBM": [
        {"n_estimators": 100, "max_depth": 2, "learning_rate": 0.1},
        {"n_estimators": 200, "max_depth": 2, "learning_rate": 0.1},
        {"n_estimators": 300, "max_depth": 2, "learning_rate": 0.05},
        {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.05},
        {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.03},
        {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.03},
        {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.1, "subsample": 0.8, "colsample_bytree": 0.8},
        {"n_estimators": 200, "max_depth": 2, "learning_rate": 0.1, "subsample": 0.8, "colsample_bytree": 0.8},
        {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.05},
        {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.03},
    ],
}
