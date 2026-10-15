# 모듈 2(회수율 예측) + 레포 재구조화 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `src/`를 `module1_preprocessing/`으로 이름을 바꿔 과제 요구 레포 구조에 맞추고, 그 패턴으로 `module2_prediction/`(모델 학습·튜닝·평가·해석·시나리오 예측)을 새로 구축한다.

**Architecture:** 모듈 1은 지금 코드를 그대로, 폴더명과 import 경로만 바꿔 이동한다. 모듈 2는 모듈 1의 코드를 import하지 않고 `data/processed/cleaned_features.csv`(`split` 컬럼 포함)만 읽는 완전히 독립된 패키지로, 모듈 1과 동일한 leak-safe 원칙(Validation으로 선택, Test는 1회 평가)을 따른다.

**Tech Stack:** pandas, scikit-learn, xgboost, lightgbm, shap, pytest, nbformat/jupyter

## Global Constraints

- 기존 모듈 1 테스트 46개는 경로만 바뀐 채 전부 통과해야 한다
- 모듈 2는 `from module1_preprocessing...`를 import하지 않는다 (완전히 독립)
- 모듈 2는 feature 선택을 다시 하지 않는다 — `FEATURE_COLUMNS`는 모듈 1이 확정한 `base_plus_derived`(조성 7 + 공정조건 2 + 파생변수 4, 총 13개) 그대로 고정
- 모델 선정(모델 종류 + 하이퍼파라미터)은 Validation 성능만으로 결정, Test는 최종 1회 평가에만 사용
- 타겟(Ni_회수율_pct, Co_회수율_pct)마다 완전히 독립된 모델을 학습
- `notebooks/`는 과제 명세대로 top-level 유지 (모듈 폴더 안에 안 넣음)
- `tests/`, `scripts/`, `reports/figures/`는 각 모듈 폴더 안으로 이동

---

## Part A: 모듈 1 마이그레이션 (`src/` → `module1_preprocessing/`)

### Task 1: 패키지 디렉터리 이동 + 내부 import 경로 수정

**Files:**
- Move: `src/*.py` → `module1_preprocessing/*.py`
- Modify: `module1_preprocessing/pipeline.py`, `module1_preprocessing/validation.py`, `module1_preprocessing/imputation.py`, `module1_preprocessing/preprocessing.py`

**Interfaces:**
- Consumes: 없음 (순수 이동 작업)
- Produces: `module1_preprocessing.config`, `.preprocessing`, `.splitting`, `.imputation`, `.features`, `.validation`, `.pipeline` — 기존 `src.*`와 동일한 함수명·시그니처

- [ ] **Step 1: git mv로 디렉터리 이동**

```bash
git mv src module1_preprocessing
```

- [ ] **Step 2: 각 파일의 `from src.` import를 `from module1_preprocessing.`로 변경**

`module1_preprocessing/preprocessing.py`에서:
```python
from src.config import RAW_SHEET_NAME
```
→
```python
from module1_preprocessing.config import RAW_SHEET_NAME
```

`module1_preprocessing/imputation.py`에서:
```python
from src.config import NORMAL_RANGES, COMPOSITION_COLUMNS, PROCESS_COLUMNS
```
→
```python
from module1_preprocessing.config import NORMAL_RANGES, COMPOSITION_COLUMNS, PROCESS_COLUMNS
```

`module1_preprocessing/validation.py`에서:
```python
from src.config import NORMAL_RANGES
from src.features import engineer_features
from src.imputation import impute_missing_from_reference
```
→
```python
from module1_preprocessing.config import NORMAL_RANGES
from module1_preprocessing.features import engineer_features
from module1_preprocessing.imputation import impute_missing_from_reference
```

`module1_preprocessing/pipeline.py`에서:
```python
from src.config import COMPOSITION_COLUMNS, DERIVED_FEATURES, PROCESS_COLUMNS, TARGET_COLUMNS
from src.features import engineer_features
from src.imputation import replace_outliers_with_nan, impute_missing_from_reference, interpolate_analysis_date
from src.preprocessing import clean_schema, load_raw, resolve_duplicate_batches
from src.splitting import temporal_train_val_test_split
from src.validation import check_ranges, check_missing, evaluate_final_model, run_ablation_test
```
→
```python
from module1_preprocessing.config import COMPOSITION_COLUMNS, DERIVED_FEATURES, PROCESS_COLUMNS, TARGET_COLUMNS
from module1_preprocessing.features import engineer_features
from module1_preprocessing.imputation import replace_outliers_with_nan, impute_missing_from_reference, interpolate_analysis_date
from module1_preprocessing.preprocessing import clean_schema, load_raw, resolve_duplicate_batches
from module1_preprocessing.splitting import temporal_train_val_test_split
from module1_preprocessing.validation import check_ranges, check_missing, evaluate_final_model, run_ablation_test
```

`module1_preprocessing/pipeline.py`의 docstring에서 `src/validation.py`를 `module1_preprocessing/validation.py`로 수정:
```python
    배치를 평가하는 `module1_preprocessing/validation.py`의 `run_ablation_test`와 동일한 원칙이다.
```

`module1_preprocessing/features.py`, `module1_preprocessing/splitting.py`, `module1_preprocessing/config.py`는 `src.` import가 없으므로 수정 불필요.

- [ ] **Step 3: import가 정상적으로 풀리는지 확인**

Run: `cd /home/llsowy/Downloads/capstone && python3 -c "from module1_preprocessing.pipeline import run; print('ok')"`
Expected: `ok` 출력, `ModuleNotFoundError` 없음

- [ ] **Step 4: Commit**

```bash
git add module1_preprocessing/
git commit -m "refactor: rename src/ to module1_preprocessing/ for repo structure compliance"
```

---

### Task 2: tests/scripts/reports 이동, pytest.ini·.gitignore 수정, 전체 검증

**Files:**
- Move: `tests/*.py` → `module1_preprocessing/tests/*.py`
- Move: `scripts/build_notebook.py` → `module1_preprocessing/scripts/build_notebook.py`
- Move: `reports/figures/` → `module1_preprocessing/reports/figures/`
- Modify: `pytest.ini`, `.gitignore`

**Interfaces:**
- Consumes: Task 1에서 만든 `module1_preprocessing.*` 모듈들
- Produces: `python3 -m pytest module1_preprocessing/tests`로 46개 테스트 실행 가능, `python3 module1_preprocessing/scripts/build_notebook.py`로 `notebooks/01_eda_and_validation.ipynb` 생성 가능

- [ ] **Step 1: tests, scripts, reports/figures 이동**

```bash
git mv tests module1_preprocessing/tests
git mv scripts module1_preprocessing/scripts
mkdir -p module1_preprocessing/reports
git mv reports/figures module1_preprocessing/reports/figures
rmdir reports
```

- [ ] **Step 2: 테스트 파일들의 import 경로 수정**

각 `module1_preprocessing/tests/test_*.py`에서 `from src.` → `from module1_preprocessing.`로 변경:

`test_features.py`: `from src.features import engineer_features` → `from module1_preprocessing.features import engineer_features`

`test_imputation.py`: `from src.imputation import replace_outliers_with_nan` → `from module1_preprocessing.imputation import replace_outliers_with_nan`, 그리고 파일 중간의 `from src.imputation import impute_missing_from_reference` → `from module1_preprocessing.imputation import impute_missing_from_reference`, `from src.imputation import interpolate_analysis_date` → `from module1_preprocessing.imputation import interpolate_analysis_date`

`test_pipeline.py`: `from src.config import NORMAL_RANGES` → `from module1_preprocessing.config import NORMAL_RANGES`, `from src.pipeline import run` → `from module1_preprocessing.pipeline import run`

`test_preprocessing.py`: `from src.preprocessing import load_raw, clean_schema, resolve_duplicate_batches` → `from module1_preprocessing.preprocessing import load_raw, clean_schema, resolve_duplicate_batches`

`test_splitting.py`: `from src.splitting import temporal_train_val_test_split` → `from module1_preprocessing.splitting import temporal_train_val_test_split`

`test_validation.py`: `from src.config import COMPOSITION_COLUMNS, PROCESS_COLUMNS` → `from module1_preprocessing.config import COMPOSITION_COLUMNS, PROCESS_COLUMNS`, `from src.splitting import temporal_train_val_test_split` → `from module1_preprocessing.splitting import temporal_train_val_test_split`, `from src.validation import check_ranges, ...` → `from module1_preprocessing.validation import check_ranges, ...` (3개 import 줄 전부)

- [ ] **Step 3: `module1_preprocessing/scripts/build_notebook.py`의 import 문자열 수정**

노트북 셀 생성 코드 안의 문자열에서 `from src.` → `from module1_preprocessing.`로 변경 (import 셀의 5개 줄):
```python
    "from module1_preprocessing.preprocessing import load_raw, clean_schema, resolve_duplicate_batches\n"
    "from module1_preprocessing.imputation import replace_outliers_with_nan, impute_missing_from_reference, interpolate_analysis_date\n"
    "from module1_preprocessing.features import engineer_features\n"
    "from module1_preprocessing.splitting import temporal_train_val_test_split\n"
    "from module1_preprocessing.validation import (\n"
```
그리고 뒤쪽 `from module1_preprocessing.config import COMPOSITION_COLUMNS, PROCESS_COLUMNS, TARGET_COLUMNS\n\n` 줄도 동일하게 변경.

마크다운 텍스트 중 `replace_outliers_with_nan`를 설명하는 문장에 `src.`가 코드 인용이 아니라 그냥 함수명만 있으면 수정 불필요 — `grep -n "src\."` 결과에 남은 것만 고친다.

- [ ] **Step 4: pytest.ini 수정**

`pytest.ini` 현재:
```ini
[pytest]
testpaths = tests
pythonpath = .
```
변경:
```ini
[pytest]
testpaths = module1_preprocessing/tests
pythonpath = .
```
(module2_prediction/tests는 Task 9에서 추가로 반영)

- [ ] **Step 5: .gitignore 수정**

`reports/figures/` 줄을:
```
module1_preprocessing/reports/figures/
```
로 변경.

- [ ] **Step 6: 전체 테스트 실행**

Run: `cd /home/llsowy/Downloads/capstone && python3 -m pytest -q`
Expected: `46 passed`

- [ ] **Step 7: 파이프라인 직접 실행 확인**

Run: `python3 -m module1_preprocessing.pipeline`
Expected: 에러 없이 끝까지 실행, `data/processed/cleaned_features.csv` 생성, 마지막 줄에 "데이터 정제 및 특성공학 완료" 출력

- [ ] **Step 8: 노트북 재생성 + 실행 확인**

Run:
```bash
python3 module1_preprocessing/scripts/build_notebook.py
jupyter nbconvert --to notebook --execute notebooks/01_eda_and_validation.ipynb --output executed_check.ipynb
```
Expected: 에러 없이 완료. 확인 후 `rm notebooks/executed_check.ipynb`로 임시 파일 삭제.

- [ ] **Step 9: grep으로 남은 `src.` 참조 확인**

Run: `grep -rn "from src\.\|import src\." module1_preprocessing/ notebooks/`
Expected: 결과 없음 (전부 수정됨)

- [ ] **Step 10: Commit**

```bash
git add -A
git commit -m "refactor: move tests/scripts/reports into module1_preprocessing/, update pytest.ini and .gitignore"
```

---

## Part B: `module2_prediction/` 신규 구축

### Task 3: `config.py` + `data_loading.py`

**Files:**
- Create: `module2_prediction/config.py`
- Create: `module2_prediction/data_loading.py`
- Test: `module2_prediction/tests/test_data_loading.py`

**Interfaces:**
- Produces:
  - `config.TARGET_COLUMNS: list[str]`, `config.COMPOSITION_COLUMNS: list[str]`, `config.PROCESS_COLUMNS: list[str]`, `config.DERIVED_COLUMNS: list[str]`, `config.FEATURE_COLUMNS: list[str]`, `config.RAW_BLEND_INPUT_COLUMNS: list[str]`, `config.CLEANED_DATA_PATH: Path`, `config.MODEL_HYPERPARAMETER_GRIDS: dict[str, list[dict]]`
  - `data_loading.load_cleaned_features(path) -> pd.DataFrame`
  - `data_loading.split_by_column(df) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]` (train, validation, test 순서)

- [ ] **Step 1: `module2_prediction/config.py` 작성**

```python
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
        {"n_estimators": 100, "max_depth": None},
        {"n_estimators": 200, "max_depth": None},
        {"n_estimators": 100, "max_depth": 5},
        {"n_estimators": 200, "max_depth": 5},
    ],
    "XGBoost": [
        {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1},
        {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.1},
        {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.05},
        {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.05},
    ],
    "LightGBM": [
        {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1},
        {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.1},
        {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.05},
        {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.05},
    ],
}
```

- [ ] **Step 2: 실패하는 테스트 작성 (`data_loading.py`가 아직 없음)**

`module2_prediction/tests/test_data_loading.py`:
```python
import pandas as pd
import pytest

from module2_prediction.data_loading import load_cleaned_features, split_by_column


def test_load_cleaned_features_reads_csv(tmp_path):
    csv_path = tmp_path / "cleaned.csv"
    pd.DataFrame({"a": [1, 2], "split": ["train", "test"]}).to_csv(csv_path, index=False)

    df = load_cleaned_features(csv_path)

    assert df.shape == (2, 2)


def test_split_by_column_separates_train_validation_test():
    df = pd.DataFrame({
        "value": [1, 2, 3, 4, 5, 6],
        "split": ["train", "train", "validation", "validation", "test", "test"],
    })

    train_df, val_df, test_df = split_by_column(df)

    assert train_df["value"].tolist() == [1, 2]
    assert val_df["value"].tolist() == [3, 4]
    assert test_df["value"].tolist() == [5, 6]


def test_split_by_column_raises_without_split_column():
    df = pd.DataFrame({"value": [1, 2]})

    with pytest.raises(ValueError):
        split_by_column(df)
```

- [ ] **Step 3: 테스트 실패 확인**

Run: `python3 -m pytest module2_prediction/tests/test_data_loading.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'module2_prediction.data_loading'`

- [ ] **Step 4: `module2_prediction/data_loading.py` 구현**

```python
import pandas as pd

from module2_prediction.config import CLEANED_DATA_PATH


def load_cleaned_features(path=CLEANED_DATA_PATH) -> pd.DataFrame:
    """module1_preprocessing의 산출물(cleaned_features.csv)을 읽는다."""
    return pd.read_csv(path)


def split_by_column(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """df의 `split` 컬럼(train/validation/test) 기준으로 세 DataFrame으로 나눈다."""
    if "split" not in df.columns:
        raise ValueError("`split` 컬럼이 없습니다 — module1_preprocessing의 산출물이 맞는지 확인하세요.")

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "validation"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    return train_df, val_df, test_df
```

- [ ] **Step 5: 테스트 통과 확인**

Run: `python3 -m pytest module2_prediction/tests/test_data_loading.py -v`
Expected: `3 passed`

- [ ] **Step 6: Commit**

```bash
git add module2_prediction/config.py module2_prediction/data_loading.py module2_prediction/tests/test_data_loading.py
git commit -m "feat(module2): add config and data_loading"
```

---

### Task 4: `modeling.py` — 모델 비교 (Validation 기준)

**Files:**
- Create: `module2_prediction/modeling.py`
- Test: `module2_prediction/tests/test_modeling.py`

**Interfaces:**
- Consumes: `config.FEATURE_COLUMNS`, `config.TARGET_COLUMNS`
- Produces:
  - `modeling.MODEL_FACTORIES: dict[str, Callable[..., estimator]]` (키: `"RandomForest"`, `"XGBoost"`, `"LightGBM"`)
  - `modeling._fit_and_evaluate(model, train_df, val_df, feature_columns, target) -> dict` (키: `val_r2`, `val_rmse`, `val_mae`)
  - `modeling.compare_models(train_df, val_df, model_params=None, feature_columns=FEATURE_COLUMNS, target_columns=TARGET_COLUMNS) -> dict[tuple[str,str], dict]`

- [ ] **Step 1: 테스트 fixture 및 실패 테스트 작성**

`module2_prediction/tests/test_modeling.py`:
```python
import numpy as np
import pandas as pd
import pytest

from module2_prediction.modeling import compare_models


def _make_fixture(n=60, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "Ni_함량_pct": rng.normal(20, 3, n),
        "Co_함량_pct": rng.normal(8, 2, n),
        "Li_함량_pct": rng.normal(4, 1, n),
        "Mn_함량_pct": rng.normal(10, 2, n),
        "Cu_불순물_pct": rng.normal(2, 0.5, n),
        "Fe_불순물_pct": rng.normal(1.5, 0.5, n),
        "Al_불순물_pct": rng.normal(2.5, 0.5, n),
        "침출온도_C": rng.normal(65, 10, n),
        "침출시간_hr": rng.normal(4, 1, n),
        "acid_to_ore_ratio": rng.normal(7, 0.5, n),
        "leach_intensity": rng.normal(260, 40, n),
        "total_impurity_pct": rng.normal(6, 1, n),
        "target_purity_ratio": rng.normal(0.8, 0.05, n),
    })
    df["Ni_회수율_pct"] = 50 + df["Ni_함량_pct"] * 1.5 + rng.normal(0, 2, n)
    df["Co_회수율_pct"] = 50 + df["Co_함량_pct"] * 1.5 + rng.normal(0, 2, n)
    return df


def test_compare_models_returns_metrics_for_each_model_and_target():
    df = _make_fixture()
    train_df, val_df = df.iloc[:40], df.iloc[40:]

    results = compare_models(train_df, val_df)

    assert set(results.keys()) == {
        ("RandomForest", "Ni_회수율_pct"), ("RandomForest", "Co_회수율_pct"),
        ("XGBoost", "Ni_회수율_pct"), ("XGBoost", "Co_회수율_pct"),
        ("LightGBM", "Ni_회수율_pct"), ("LightGBM", "Co_회수율_pct"),
    }
    for metrics in results.values():
        for key in ["val_r2", "val_rmse", "val_mae"]:
            assert key in metrics
            assert isinstance(metrics[key], float)


def test_compare_models_applies_custom_hyperparameters():
    df = _make_fixture()
    train_df, val_df = df.iloc[:40], df.iloc[40:]

    results = compare_models(
        train_df, val_df,
        model_params={"RandomForest": {"n_estimators": 10}},
    )

    assert ("RandomForest", "Ni_회수율_pct") in results


def test_compare_models_raises_when_target_has_missing_values():
    df = _make_fixture()
    df.loc[0, "Ni_회수율_pct"] = None
    train_df, val_df = df.iloc[:40], df.iloc[40:]

    with pytest.raises(ValueError):
        compare_models(train_df, val_df)
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `python3 -m pytest module2_prediction/tests/test_modeling.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'module2_prediction.modeling'`

- [ ] **Step 3: `module2_prediction/modeling.py` 구현**

```python
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from module2_prediction.config import FEATURE_COLUMNS, TARGET_COLUMNS

MODEL_FACTORIES = {
    "RandomForest": lambda **params: RandomForestRegressor(random_state=42, **params),
    "XGBoost": lambda **params: XGBRegressor(random_state=42, verbosity=0, **params),
    "LightGBM": lambda **params: LGBMRegressor(random_state=42, verbose=-1, **params),
}


def _fit_and_evaluate(model, train_df, val_df, feature_columns, target) -> dict:
    """model을 train_df로 학습해 val_df로 평가한다 (Validation 지표만)."""
    model.fit(train_df[feature_columns], train_df[target])
    preds = model.predict(val_df[feature_columns])

    return {
        "val_r2": float(r2_score(val_df[target], preds)),
        "val_rmse": float(np.sqrt(mean_squared_error(val_df[target], preds))),
        "val_mae": float(mean_absolute_error(val_df[target], preds)),
    }


def compare_models(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    model_params: dict | None = None,
    feature_columns: list = FEATURE_COLUMNS,
    target_columns: list = TARGET_COLUMNS,
) -> dict:
    """모델 종류별로 Validation 성능을 비교한다 (feature는 고정, Test는 안 봄).

    model_params: {모델명: 하이퍼파라미터 dict}. 생략하면 각 모델 기본값을 쓴다.
    반환값: {(모델명, target): {val_r2, val_rmse, val_mae}}
    """
    model_params = model_params or {}

    for target in target_columns:
        if train_df[target].isna().any() or val_df[target].isna().any():
            raise ValueError(f"목표변수 '{target}'에 결측치가 있습니다.")

    results = {}
    for model_name, make_model in MODEL_FACTORIES.items():
        params = model_params.get(model_name, {})
        for target in target_columns:
            model = make_model(**params)
            results[(model_name, target)] = _fit_and_evaluate(
                model, train_df, val_df, feature_columns, target
            )

    return results
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `python3 -m pytest module2_prediction/tests/test_modeling.py -v`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add module2_prediction/modeling.py module2_prediction/tests/test_modeling.py
git commit -m "feat(module2): add model comparison (RF/XGBoost/LightGBM, Validation-based)"
```

---

### Task 5: `tuning.py` — 하이퍼파라미터 탐색 (Validation 기준)

**Files:**
- Create: `module2_prediction/tuning.py`
- Test: `module2_prediction/tests/test_tuning.py`

**Interfaces:**
- Consumes: `modeling.MODEL_FACTORIES`, `modeling._fit_and_evaluate`, `config.MODEL_HYPERPARAMETER_GRIDS`, `config.FEATURE_COLUMNS`, `config.TARGET_COLUMNS`
- Produces: `tuning.tune_models(train_df, val_df, feature_columns=FEATURE_COLUMNS, target_columns=TARGET_COLUMNS) -> dict[tuple[str,str], dict]` — 값은 `{"params": dict, "metrics": {val_r2, val_rmse, val_mae}}`

- [ ] **Step 1: 실패하는 테스트 작성**

`module2_prediction/tests/test_tuning.py`:
```python
import numpy as np
import pandas as pd

from module2_prediction.tuning import tune_models


def _make_fixture(n=60, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "Ni_함량_pct": rng.normal(20, 3, n),
        "Co_함량_pct": rng.normal(8, 2, n),
        "Li_함량_pct": rng.normal(4, 1, n),
        "Mn_함량_pct": rng.normal(10, 2, n),
        "Cu_불순물_pct": rng.normal(2, 0.5, n),
        "Fe_불순물_pct": rng.normal(1.5, 0.5, n),
        "Al_불순물_pct": rng.normal(2.5, 0.5, n),
        "침출온도_C": rng.normal(65, 10, n),
        "침출시간_hr": rng.normal(4, 1, n),
        "acid_to_ore_ratio": rng.normal(7, 0.5, n),
        "leach_intensity": rng.normal(260, 40, n),
        "total_impurity_pct": rng.normal(6, 1, n),
        "target_purity_ratio": rng.normal(0.8, 0.05, n),
    })
    df["Ni_회수율_pct"] = 50 + df["Ni_함량_pct"] * 1.5 + rng.normal(0, 2, n)
    df["Co_회수율_pct"] = 50 + df["Co_함량_pct"] * 1.5 + rng.normal(0, 2, n)
    return df


def test_tune_models_returns_best_params_per_model_and_target():
    df = _make_fixture()
    train_df, val_df = df.iloc[:40], df.iloc[40:]

    best = tune_models(train_df, val_df)

    assert set(best.keys()) == {
        ("RandomForest", "Ni_회수율_pct"), ("RandomForest", "Co_회수율_pct"),
        ("XGBoost", "Ni_회수율_pct"), ("XGBoost", "Co_회수율_pct"),
        ("LightGBM", "Ni_회수율_pct"), ("LightGBM", "Co_회수율_pct"),
    }
    for entry in best.values():
        assert "params" in entry
        assert "val_r2" in entry["metrics"]
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `python3 -m pytest module2_prediction/tests/test_tuning.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'module2_prediction.tuning'`

- [ ] **Step 3: `module2_prediction/tuning.py` 구현**

```python
from module2_prediction.config import FEATURE_COLUMNS, MODEL_HYPERPARAMETER_GRIDS, TARGET_COLUMNS
from module2_prediction.modeling import MODEL_FACTORIES, _fit_and_evaluate


def tune_models(
    train_df, val_df, feature_columns: list = FEATURE_COLUMNS, target_columns: list = TARGET_COLUMNS
) -> dict:
    """MODEL_HYPERPARAMETER_GRIDS의 각 조합을 Validation으로 평가해 모델·타겟별 최적 하이퍼파라미터를 고른다.

    Test는 전혀 보지 않는다. 반환값: {(모델명, target): {"params": {...}, "metrics": {val_r2, val_rmse, val_mae}}}
    """
    for target in target_columns:
        if train_df[target].isna().any() or val_df[target].isna().any():
            raise ValueError(f"목표변수 '{target}'에 결측치가 있습니다.")

    best = {}
    for model_name, grid in MODEL_HYPERPARAMETER_GRIDS.items():
        make_model = MODEL_FACTORIES[model_name]
        for params in grid:
            for target in target_columns:
                model = make_model(**params)
                metrics = _fit_and_evaluate(model, train_df, val_df, feature_columns, target)

                key = (model_name, target)
                if key not in best or metrics["val_r2"] > best[key]["metrics"]["val_r2"]:
                    best[key] = {"params": params, "metrics": metrics}

    return best
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `python3 -m pytest module2_prediction/tests/test_tuning.py -v`
Expected: `1 passed`

- [ ] **Step 5: Commit**

```bash
git add module2_prediction/tuning.py module2_prediction/tests/test_tuning.py
git commit -m "feat(module2): add Validation-based hyperparameter tuning"
```

---

### Task 6: `evaluation.py` — 최종 모델 Test 1회 평가

**Files:**
- Create: `module2_prediction/evaluation.py`
- Test: `module2_prediction/tests/test_evaluation.py`

**Interfaces:**
- Consumes: `modeling.MODEL_FACTORIES`, `config.FEATURE_COLUMNS`, `config.TARGET_COLUMNS`
- Produces: `evaluation.evaluate_final_model(train_df, val_df, test_df, model_name: str, model_params: dict, feature_columns=FEATURE_COLUMNS, target_columns=TARGET_COLUMNS) -> dict[str, dict]` — 값은 `{test_r2, test_rmse, test_mae}`

- [ ] **Step 1: 실패하는 테스트 작성**

`module2_prediction/tests/test_evaluation.py`:
```python
import numpy as np
import pandas as pd
import pytest

from module2_prediction.evaluation import evaluate_final_model


def _make_fixture(n=60, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "Ni_함량_pct": rng.normal(20, 3, n),
        "Co_함량_pct": rng.normal(8, 2, n),
        "Li_함량_pct": rng.normal(4, 1, n),
        "Mn_함량_pct": rng.normal(10, 2, n),
        "Cu_불순물_pct": rng.normal(2, 0.5, n),
        "Fe_불순물_pct": rng.normal(1.5, 0.5, n),
        "Al_불순물_pct": rng.normal(2.5, 0.5, n),
        "침출온도_C": rng.normal(65, 10, n),
        "침출시간_hr": rng.normal(4, 1, n),
        "acid_to_ore_ratio": rng.normal(7, 0.5, n),
        "leach_intensity": rng.normal(260, 40, n),
        "total_impurity_pct": rng.normal(6, 1, n),
        "target_purity_ratio": rng.normal(0.8, 0.05, n),
    })
    df["Ni_회수율_pct"] = 50 + df["Ni_함량_pct"] * 1.5 + rng.normal(0, 2, n)
    df["Co_회수율_pct"] = 50 + df["Co_함량_pct"] * 1.5 + rng.normal(0, 2, n)
    return df


def test_evaluate_final_model_returns_test_metrics():
    df = _make_fixture()
    train_df, val_df, test_df = df.iloc[:30], df.iloc[30:45], df.iloc[45:]

    results = evaluate_final_model(
        train_df, val_df, test_df,
        model_name="RandomForest", model_params={"n_estimators": 50},
    )

    assert set(results.keys()) == {"Ni_회수율_pct", "Co_회수율_pct"}
    for metrics in results.values():
        for key in ["test_r2", "test_rmse", "test_mae"]:
            assert key in metrics


def test_evaluate_final_model_raises_when_target_has_missing_values():
    df = _make_fixture()
    df.loc[0, "Ni_회수율_pct"] = None
    train_df, val_df, test_df = df.iloc[:30], df.iloc[30:45], df.iloc[45:]

    with pytest.raises(ValueError):
        evaluate_final_model(
            train_df, val_df, test_df,
            model_name="RandomForest", model_params={},
        )
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `python3 -m pytest module2_prediction/tests/test_evaluation.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'module2_prediction.evaluation'`

- [ ] **Step 3: `module2_prediction/evaluation.py` 구현**

```python
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from module2_prediction.config import FEATURE_COLUMNS, TARGET_COLUMNS
from module2_prediction.modeling import MODEL_FACTORIES


def evaluate_final_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_name: str,
    model_params: dict,
    feature_columns: list = FEATURE_COLUMNS,
    target_columns: list = TARGET_COLUMNS,
) -> dict:
    """선택된 모델(model_name, model_params)로 Train+Validation을 합쳐 재학습한 뒤 Test에 딱 한 번 적용한다."""
    for target in target_columns:
        if (
            train_df[target].isna().any()
            or val_df[target].isna().any()
            or test_df[target].isna().any()
        ):
            raise ValueError(f"목표변수 '{target}'에 결측치가 있습니다.")

    train_val_df = pd.concat([train_df, val_df], ignore_index=True)
    make_model = MODEL_FACTORIES[model_name]

    results = {}
    for target in target_columns:
        model = make_model(**model_params)
        model.fit(train_val_df[feature_columns], train_val_df[target])
        preds = model.predict(test_df[feature_columns])

        results[target] = {
            "test_r2": float(r2_score(test_df[target], preds)),
            "test_rmse": float(np.sqrt(mean_squared_error(test_df[target], preds))),
            "test_mae": float(mean_absolute_error(test_df[target], preds)),
        }

    return results
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `python3 -m pytest module2_prediction/tests/test_evaluation.py -v`
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add module2_prediction/evaluation.py module2_prediction/tests/test_evaluation.py
git commit -m "feat(module2): add final model Test evaluation (Train+Val refit, Test once)"
```

---

### Task 7: `interpretation.py` — Feature Importance / SHAP

**Files:**
- Create: `module2_prediction/interpretation.py`
- Test: `module2_prediction/tests/test_interpretation.py`
- Modify: `requirements.txt`

**Interfaces:**
- Produces: `interpretation.get_feature_importance(model, feature_columns=FEATURE_COLUMNS) -> pd.Series` (내림차순 정렬), `interpretation.compute_shap_values(model, X: pd.DataFrame)` (TreeExplainer shap values)

- [ ] **Step 1: shap 설치 및 requirements.txt 추가**

Run: `/home/llsowy/anaconda3/bin/pip install shap`
Expected: `Successfully installed shap-...`

`requirements.txt`에 추가:
```
shap>=0.44
```

- [ ] **Step 2: 실패하는 테스트 작성**

`module2_prediction/tests/test_interpretation.py`:
```python
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from module2_prediction.interpretation import compute_shap_values, get_feature_importance


def _make_fixture(n=40, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "feature_a": rng.normal(0, 1, n),
        "feature_b": rng.normal(0, 1, n),
    })
    y = X["feature_a"] * 2 + rng.normal(0, 0.1, n)
    return X, y


def test_get_feature_importance_returns_sorted_series():
    X, y = _make_fixture()
    model = RandomForestRegressor(random_state=42).fit(X, y)

    importance = get_feature_importance(model, feature_columns=["feature_a", "feature_b"])

    assert importance.iloc[0] >= importance.iloc[1]
    assert importance["feature_a"] > importance["feature_b"]


def test_get_feature_importance_raises_for_unsupported_model():
    class DummyModel:
        pass

    with pytest.raises(ValueError):
        get_feature_importance(DummyModel(), feature_columns=["feature_a"])


def test_compute_shap_values_returns_array_matching_input_rows():
    X, y = _make_fixture()
    model = RandomForestRegressor(random_state=42).fit(X, y)

    shap_values = compute_shap_values(model, X)

    assert shap_values.shape[0] == len(X)
```

- [ ] **Step 3: 테스트 실패 확인**

Run: `python3 -m pytest module2_prediction/tests/test_interpretation.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'module2_prediction.interpretation'`

- [ ] **Step 4: `module2_prediction/interpretation.py` 구현**

```python
import pandas as pd
import shap

from module2_prediction.config import FEATURE_COLUMNS


def get_feature_importance(model, feature_columns: list = FEATURE_COLUMNS) -> pd.Series:
    """학습된 모델의 feature_importances_를 feature명과 묶어 내림차순으로 반환한다."""
    if not hasattr(model, "feature_importances_"):
        raise ValueError(f"{type(model).__name__}은 feature_importances_를 지원하지 않습니다.")

    return pd.Series(model.feature_importances_, index=feature_columns).sort_values(ascending=False)


def compute_shap_values(model, X: pd.DataFrame):
    """TreeExplainer로 SHAP value를 계산한다 (RandomForest/XGBoost/LightGBM 전용)."""
    explainer = shap.TreeExplainer(model)
    return explainer.shap_values(X)
```

- [ ] **Step 5: 테스트 통과 확인**

Run: `python3 -m pytest module2_prediction/tests/test_interpretation.py -v`
Expected: `3 passed`

- [ ] **Step 6: Commit**

```bash
git add module2_prediction/interpretation.py module2_prediction/tests/test_interpretation.py requirements.txt
git commit -m "feat(module2): add feature importance and SHAP interpretation"
```

---

### Task 8: `scenario.py` — 공정조건별 변화 분석 + 7:3 혼합 시나리오

**Files:**
- Create: `module2_prediction/scenario.py`
- Test: `module2_prediction/tests/test_scenario.py`

**Interfaces:**
- Consumes: `config.COMPOSITION_COLUMNS`, `config.PROCESS_COLUMNS`, `config.RAW_BLEND_INPUT_COLUMNS`, `config.FEATURE_COLUMNS`
- Produces:
  - `scenario.partial_dependence(model, base_row: pd.Series, feature: str, values, feature_columns=FEATURE_COLUMNS) -> pd.DataFrame` (컬럼: `[feature, "prediction"]`)
  - `scenario.predict_blend_scenario(model, train_df, grade_a: str, grade_b: str, ratio_a: float, feature_columns=FEATURE_COLUMNS, grade_column="원료등급") -> float`

- [ ] **Step 1: 실패하는 테스트 작성**

`module2_prediction/tests/test_scenario.py`:
```python
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from module2_prediction.config import FEATURE_COLUMNS
from module2_prediction.scenario import partial_dependence, predict_blend_scenario


def _make_fixture(n=60, seed=0):
    rng = np.random.default_rng(seed)
    grades = np.array(["A"] * (n // 2) + ["C"] * (n - n // 2))
    df = pd.DataFrame({
        "원료등급": grades,
        "Ni_함량_pct": np.where(grades == "A", rng.normal(25, 1, n), rng.normal(12, 1, n)),
        "Co_함량_pct": rng.normal(8, 2, n),
        "Li_함량_pct": rng.normal(4, 1, n),
        "Mn_함량_pct": rng.normal(10, 2, n),
        "Cu_불순물_pct": np.where(grades == "A", rng.normal(1, 0.2, n), rng.normal(4, 0.2, n)),
        "Fe_불순물_pct": rng.normal(1.5, 0.5, n),
        "Al_불순물_pct": rng.normal(2.5, 0.5, n),
        "침출온도_C": rng.normal(65, 10, n),
        "침출시간_hr": rng.normal(4, 1, n),
        "원료투입량_kg": rng.normal(140, 10, n),
        "황산투입량_L": rng.normal(900, 50, n),
    })
    df["acid_to_ore_ratio"] = df["황산투입량_L"] / df["원료투입량_kg"]
    df["leach_intensity"] = df["침출온도_C"] * df["침출시간_hr"]
    df["total_impurity_pct"] = df["Cu_불순물_pct"] + df["Fe_불순물_pct"] + df["Al_불순물_pct"]
    target_metals = df["Ni_함량_pct"] + df["Co_함량_pct"]
    df["target_purity_ratio"] = target_metals / (target_metals + df["total_impurity_pct"])
    df["Ni_회수율_pct"] = 50 + df["Ni_함량_pct"] * 1.5 + rng.normal(0, 1, n)
    return df


def test_predict_blend_scenario_returns_float():
    df = _make_fixture()
    model = RandomForestRegressor(random_state=42).fit(df[FEATURE_COLUMNS], df["Ni_회수율_pct"])

    prediction = predict_blend_scenario(model, df, grade_a="A", grade_b="C", ratio_a=0.7)

    assert isinstance(prediction, float)
    assert not np.isnan(prediction)


def test_predict_blend_scenario_rejects_invalid_ratio():
    df = _make_fixture()
    model = RandomForestRegressor(random_state=42).fit(df[FEATURE_COLUMNS], df["Ni_회수율_pct"])

    with pytest.raises(ValueError):
        predict_blend_scenario(model, df, grade_a="A", grade_b="C", ratio_a=1.5)


def test_predict_blend_scenario_rejects_unknown_grade():
    df = _make_fixture()
    model = RandomForestRegressor(random_state=42).fit(df[FEATURE_COLUMNS], df["Ni_회수율_pct"])

    with pytest.raises(ValueError):
        predict_blend_scenario(model, df, grade_a="A", grade_b="Z", ratio_a=0.5)


def test_partial_dependence_varies_predictions_across_values():
    df = _make_fixture()
    model = RandomForestRegressor(random_state=42).fit(df[FEATURE_COLUMNS], df["Ni_회수율_pct"])
    base_row = df.iloc[0]

    result = partial_dependence(model, base_row, feature="침출온도_C", values=[40, 60, 80])

    assert result.shape == (3, 2)
    assert list(result["침출온도_C"]) == [40, 60, 80]
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `python3 -m pytest module2_prediction/tests/test_scenario.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'module2_prediction.scenario'`

- [ ] **Step 3: `module2_prediction/scenario.py` 구현**

```python
import pandas as pd

from module2_prediction.config import (
    COMPOSITION_COLUMNS,
    FEATURE_COLUMNS,
    PROCESS_COLUMNS,
    RAW_BLEND_INPUT_COLUMNS,
)


def partial_dependence(
    model, base_row: pd.Series, feature: str, values, feature_columns: list = FEATURE_COLUMNS
) -> pd.DataFrame:
    """base_row를 고정한 채 feature 값만 바꿔가며 예측값이 어떻게 변하는지 확인한다.

    반환값: columns=[feature, "prediction"]인 DataFrame, len(values)행
    """
    rows = []
    for value in values:
        row = base_row.copy()
        row[feature] = value
        rows.append(row[feature_columns])

    X = pd.DataFrame(rows)
    predictions = model.predict(X)

    return pd.DataFrame({feature: list(values), "prediction": predictions})


def _engineer_blend_features(blend: dict) -> dict:
    """블렌딩된 원본 컬럼(조성·공정·원료/황산투입량)으로부터 파생변수 4종을 계산한다.

    module1_preprocessing.features.engineer_features와 같은 공식이지만, 모듈 2는
    모듈 1을 import하지 않는 원칙에 따라 여기서 독립적으로 재구현한다.
    """
    blend = dict(blend)
    blend["acid_to_ore_ratio"] = blend["황산투입량_L"] / blend["원료투입량_kg"]
    blend["leach_intensity"] = blend["침출온도_C"] * blend["침출시간_hr"]
    blend["total_impurity_pct"] = (
        blend["Cu_불순물_pct"] + blend["Fe_불순물_pct"] + blend["Al_불순물_pct"]
    )
    target_metals = blend["Ni_함량_pct"] + blend["Co_함량_pct"]
    blend["target_purity_ratio"] = target_metals / (target_metals + blend["total_impurity_pct"])
    return blend


def predict_blend_scenario(
    model,
    train_df: pd.DataFrame,
    grade_a: str,
    grade_b: str,
    ratio_a: float,
    feature_columns: list = FEATURE_COLUMNS,
    grade_column: str = "원료등급",
) -> float:
    """grade_a:grade_b를 ratio_a:(1-ratio_a) 비율로 혼합했을 때의 예상 값을 예측한다.

    조성·공정·원료/황산투입량은 train_df에서 각 등급의 중앙값을 ratio_a:(1-ratio_a)로
    가중평균해 가상의 배치를 구성한 뒤, 파생변수를 재계산해 예측한다.
    """
    if not 0 <= ratio_a <= 1:
        raise ValueError("ratio_a는 0~1 사이여야 합니다.")

    blend_source_columns = COMPOSITION_COLUMNS + PROCESS_COLUMNS + RAW_BLEND_INPUT_COLUMNS

    median_a = train_df.loc[train_df[grade_column] == grade_a, blend_source_columns].median()
    median_b = train_df.loc[train_df[grade_column] == grade_b, blend_source_columns].median()

    if median_a.isna().any():
        raise ValueError(f"'{grade_a}' 등급 데이터가 train_df에 없습니다.")
    if median_b.isna().any():
        raise ValueError(f"'{grade_b}' 등급 데이터가 train_df에 없습니다.")

    blend = (median_a * ratio_a + median_b * (1 - ratio_a)).to_dict()
    blend = _engineer_blend_features(blend)

    X = pd.DataFrame([blend])[feature_columns]
    return float(model.predict(X)[0])
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `python3 -m pytest module2_prediction/tests/test_scenario.py -v`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add module2_prediction/scenario.py module2_prediction/tests/test_scenario.py
git commit -m "feat(module2): add partial dependence and 7:3 blend scenario prediction"
```

---

### Task 9: `pipeline.py` — 엔트리포인트, pytest.ini 최종 반영

**Files:**
- Create: `module2_prediction/pipeline.py`
- Test: `module2_prediction/tests/test_pipeline.py`
- Modify: `pytest.ini`

**Interfaces:**
- Consumes: `data_loading.load_cleaned_features`, `data_loading.split_by_column`, `tuning.tune_models`, `evaluation.evaluate_final_model`
- Produces: `pipeline.run(cleaned_data_path) -> dict[str, dict]` — 타겟별 `{"model": str, "params": dict, "val_r2": float, "test_r2": float, "test_rmse": float, "test_mae": float}`

- [ ] **Step 1: 실패하는 통합 테스트 작성**

`module2_prediction/tests/test_pipeline.py`:
```python
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
```

- [ ] **Step 2: 테스트 실패 확인 (`cleaned_features.csv`가 없으면 먼저 생성)**

Run: `ls data/processed/cleaned_features.csv || python3 -m module1_preprocessing.pipeline`
Run: `python3 -m pytest module2_prediction/tests/test_pipeline.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'module2_prediction.pipeline'`

- [ ] **Step 3: `module2_prediction/pipeline.py` 구현**

```python
from pathlib import Path

from module2_prediction.config import TARGET_COLUMNS
from module2_prediction.data_loading import load_cleaned_features, split_by_column
from module2_prediction.evaluation import evaluate_final_model
from module2_prediction.tuning import tune_models


def run(cleaned_data_path: str | Path) -> dict:
    """모델 종류·하이퍼파라미터를 Validation으로 선정하고 Test로 1회 평가한다."""
    df = load_cleaned_features(cleaned_data_path)
    train_df, val_df, test_df = split_by_column(df)

    tuned = tune_models(train_df, val_df)

    best_per_target = {
        target: max(
            (key for key in tuned if key[1] == target),
            key=lambda key: tuned[key]["metrics"]["val_r2"],
        )
        for target in TARGET_COLUMNS
    }

    final_results = {}
    for target, (model_name, _) in best_per_target.items():
        params = tuned[(model_name, target)]["params"]
        test_metrics = evaluate_final_model(
            train_df, val_df, test_df,
            model_name=model_name, model_params=params,
            target_columns=[target],
        )[target]

        final_results[target] = {
            "model": model_name,
            "params": params,
            "val_r2": tuned[(model_name, target)]["metrics"]["val_r2"],
            **test_metrics,
        }

    return final_results


if __name__ == "__main__":
    results = run(Path("data/processed/cleaned_features.csv"))
    print("모듈 2 최종 모델 선정 결과:")
    for target, info in results.items():
        print(f"  {target}: {info}")
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `python3 -m pytest module2_prediction/tests/test_pipeline.py -v`
Expected: `1 passed`

- [ ] **Step 5: pytest.ini를 두 모듈 다 찾도록 수정**

```ini
[pytest]
testpaths = module1_preprocessing/tests module2_prediction/tests
pythonpath = .
```

- [ ] **Step 6: 전체 테스트 스위트 재확인**

Run: `python3 -m pytest -q`
Expected: 모듈 1 46개 + 모듈 2(Task 3~9 테스트 합) 전부 통과, 실패 없음

- [ ] **Step 7: 파이프라인 직접 실행**

Run: `python3 -m module2_prediction.pipeline`
Expected: 에러 없이 "모듈 2 최종 모델 선정 결과:" 출력 후 Ni_회수율_pct/Co_회수율_pct 각각의 모델명·params·val_r2·test_r2 등 출력

- [ ] **Step 8: Commit**

```bash
git add module2_prediction/pipeline.py module2_prediction/tests/test_pipeline.py pytest.ini
git commit -m "feat(module2): add pipeline entrypoint, wire both modules into pytest.ini"
```

---

### Task 10: `module2_prediction/scripts/build_notebook.py` → `notebooks/02_model_training.ipynb`

**Files:**
- Create: `module2_prediction/scripts/build_notebook.py`
- Create (generated, not hand-written): `notebooks/02_model_training.ipynb`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: 모든 `module2_prediction.*` public 함수
- Produces: 실행 가능한 `notebooks/02_model_training.ipynb`

- [ ] **Step 1: `module2_prediction/reports/figures/` 디렉터리 생성 및 .gitignore 반영**

```bash
mkdir -p module2_prediction/reports/figures
```

`.gitignore`의 `module1_preprocessing/reports/figures/` 줄 다음에 추가:
```
module2_prediction/reports/figures/
```

- [ ] **Step 2: `module2_prediction/scripts/build_notebook.py` 작성**

```python
from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell(
    "# 모듈 2: 회수율 예측 모델 학습·평가\n\n"
    "- 모듈 1의 산출물(`cleaned_features.csv`)만 읽어서 시작, 모듈 1 코드는 import하지 않음\n"
    "- feature는 모듈 1에서 이미 확정됨(`base_plus_derived`) — 모듈 2는 모델 종류·하이퍼파라미터만 탐색"
))

cells.append(nbf.v4.new_code_cell(
    "import sys\n"
    "sys.path.append('..')\n\n"
    "import pandas as pd\n"
    "from pathlib import Path\n\n"
    "from module2_prediction.config import FEATURE_COLUMNS, TARGET_COLUMNS\n"
    "from module2_prediction.data_loading import load_cleaned_features, split_by_column\n"
    "from module2_prediction.modeling import compare_models, MODEL_FACTORIES\n"
    "from module2_prediction.tuning import tune_models\n"
    "from module2_prediction.evaluation import evaluate_final_model\n"
    "from module2_prediction.interpretation import get_feature_importance, compute_shap_values\n"
    "from module2_prediction.scenario import partial_dependence, predict_blend_scenario\n\n"
    "CLEANED_DATA_PATH = Path('../data/processed/cleaned_features.csv')\n"
    "FIGURES_DIR = Path('../module2_prediction/reports/figures')\n"
    "FIGURES_DIR.mkdir(parents=True, exist_ok=True)"
))

cells.append(nbf.v4.new_markdown_cell("## 1. 데이터 로드"))

cells.append(nbf.v4.new_code_cell(
    "df = load_cleaned_features(CLEANED_DATA_PATH)\n"
    "train_df, val_df, test_df = split_by_column(df)\n"
    "print(f'Train: {len(train_df)}행 / Validation: {len(val_df)}행 / Test: {len(test_df)}행')"
))

cells.append(nbf.v4.new_markdown_cell(
    "## 2. 모델 비교 (기본값, Validation 기준)\n\n"
    "RandomForest, XGBoost, LightGBM을 동일한 feature set으로 비교한다. Test는 안 본다."
))

cells.append(nbf.v4.new_code_cell(
    "default_results = compare_models(train_df, val_df)\n"
    "pd.DataFrame(default_results).T"
))

cells.append(nbf.v4.new_markdown_cell(
    "## 3. 하이퍼파라미터 튜닝 (Validation 기준)\n\n"
    "`config.MODEL_HYPERPARAMETER_GRIDS`의 각 조합을 Validation으로 평가해 모델·타겟별 최적값을 고른다."
))

cells.append(nbf.v4.new_code_cell(
    "tuned = tune_models(train_df, val_df)\n"
    "pd.DataFrame({key: v['metrics'] for key, v in tuned.items()}).T"
))

cells.append(nbf.v4.new_markdown_cell("## 4. 최종 모델 선정 (타겟별 Validation R² 최고 모델)"))

cells.append(nbf.v4.new_code_cell(
    "best_per_target = {\n"
    "    target: max(\n"
    "        (key for key in tuned if key[1] == target),\n"
    "        key=lambda key: tuned[key]['metrics']['val_r2']\n"
    "    )\n"
    "    for target in TARGET_COLUMNS\n"
    "}\n"
    "for target, (model_name, _) in best_per_target.items():\n"
    "    p = tuned[(model_name, target)]\n"
    "    print(f'{target}: {model_name}, params={p[\"params\"]}, val_r2={p[\"metrics\"][\"val_r2\"]:.4f}')"
))

cells.append(nbf.v4.new_markdown_cell("## 5. 최종 Test 평가 (1회, Train+Validation 재학습)"))

cells.append(nbf.v4.new_code_cell(
    "final_results = {}\n"
    "for target, (model_name, _) in best_per_target.items():\n"
    "    params = tuned[(model_name, target)]['params']\n"
    "    result = evaluate_final_model(\n"
    "        train_df, val_df, test_df,\n"
    "        model_name=model_name, model_params=params,\n"
    "        target_columns=[target],\n"
    "    )\n"
    "    final_results[target] = result[target]\n\n"
    "pd.DataFrame(final_results).T"
))

cells.append(nbf.v4.new_markdown_cell(
    "## 6. Feature Importance / SHAP\n\n"
    "최종 선정 모델을 Train+Validation으로 다시 학습해 해석용으로 사용한다."
))

cells.append(nbf.v4.new_code_cell(
    "train_val_df = pd.concat([train_df, val_df], ignore_index=True)\n\n"
    "interpretation_models = {}\n"
    "for target, (model_name, _) in best_per_target.items():\n"
    "    params = tuned[(model_name, target)]['params']\n"
    "    model = MODEL_FACTORIES[model_name](**params)\n"
    "    model.fit(train_val_df[FEATURE_COLUMNS], train_val_df[target])\n"
    "    interpretation_models[target] = model\n\n"
    "    importance = get_feature_importance(model)\n"
    "    print(f'--- {target} ({model_name}) Feature Importance ---')\n"
    "    print(importance)"
))

cells.append(nbf.v4.new_code_cell(
    "import shap\n"
    "import matplotlib.pyplot as plt\n\n"
    "target = TARGET_COLUMNS[0]\n"
    "model = interpretation_models[target]\n"
    "shap_values = compute_shap_values(model, train_val_df[FEATURE_COLUMNS])\n"
    "shap.summary_plot(shap_values, train_val_df[FEATURE_COLUMNS], show=False)\n"
    "plt.savefig(FIGURES_DIR / 'shap_summary.png', bbox_inches='tight')\n"
    "plt.close()"
))

cells.append(nbf.v4.new_markdown_cell(
    "## 7. 공정조건별 변화 분석 (Partial Dependence)\n\n"
    "침출온도_C를 바꿔가며 예측값이 어떻게 변하는지 확인한다 (나머지 변수는 고정)."
))

cells.append(nbf.v4.new_code_cell(
    "target = TARGET_COLUMNS[0]\n"
    "model = interpretation_models[target]\n"
    "base_row = train_val_df.iloc[0]\n\n"
    "pd_result = partial_dependence(model, base_row, feature='침출온도_C', values=list(range(40, 96, 5)))\n"
    "pd_result.plot(x='침출온도_C', y='prediction', marker='o', title=f'{target} vs 침출온도_C')"
))

cells.append(nbf.v4.new_markdown_cell(
    "## 8. 7:3 혼합 시나리오 예측\n\n"
    "A급 원료 70% + C급 원료 30%를 혼합했을 때의 예상 Ni 회수율을 예측한다."
))

cells.append(nbf.v4.new_code_cell(
    "target = TARGET_COLUMNS[0]\n"
    "model = interpretation_models[target]\n\n"
    "blend_prediction = predict_blend_scenario(model, train_val_df, grade_a='A', grade_b='C', ratio_a=0.7)\n"
    "print(f'A:C = 7:3 혼합 예상 {target}: {blend_prediction:.2f}%')"
))

nb["cells"] = cells

Path("notebooks").mkdir(exist_ok=True)
with open("notebooks/02_model_training.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("노트북 생성 완료: notebooks/02_model_training.ipynb")
```

- [ ] **Step 3: 노트북 생성 + 실행 확인**

Run (repo root에서):
```bash
python3 module2_prediction/scripts/build_notebook.py
jupyter nbconvert --to notebook --execute notebooks/02_model_training.ipynb --output executed_check.ipynb
```
Expected: 에러 없이 완료. 확인 후 `rm notebooks/executed_check.ipynb`.

- [ ] **Step 4: 전체 테스트 재확인**

Run: `python3 -m pytest -q`
Expected: 전부 통과, 실패 없음

- [ ] **Step 5: Commit**

```bash
git add module2_prediction/scripts/build_notebook.py notebooks/02_model_training.ipynb .gitignore
git commit -m "feat(module2): add notebook generator for model training/interpretation/scenario"
```

---

## 완료 기준

- [ ] `python3 -m pytest -q` 전체 통과 (모듈 1 46개 + 모듈 2 신규 테스트)
- [ ] `python3 -m module1_preprocessing.pipeline` 에러 없이 실행
- [ ] `python3 -m module2_prediction.pipeline` 에러 없이 실행
- [ ] `notebooks/01_eda_and_validation.ipynb`, `notebooks/02_model_training.ipynb` 둘 다 `jupyter nbconvert --execute`로 에러 없이 실행
- [ ] `grep -rn "from src\."` 결과 없음 (레포 전체)
- [ ] 레포 최상위가 `data/`, `module1_preprocessing/`, `module2_prediction/`, `notebooks/`, `docs/`, `README.md`, `requirements.txt`, `pytest.ini`, `.gitignore`로 구성됨
