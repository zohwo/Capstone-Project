from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

matplotlib.rcParams["font.family"] = "Noto Sans CJK JP"
matplotlib.rcParams["axes.unicode_minus"] = False
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import NORMAL_RANGES
from src.features import engineer_features
from src.imputation import impute_missing_from_reference


def check_ranges(df: pd.DataFrame, ranges: dict = NORMAL_RANGES) -> dict:
    """각 컬럼 값이 데이터사전에 정의된 정상 범위를 벗어나는지 검증한다.

    반환값: {컬럼명: [위반 행 인덱스, ...]}
    위반이 없으면 빈 dict를 반환한다.
    """

    violations = {}

    for col, (low, high) in ranges.items():
        if col not in df.columns:
            continue

        out_of_range = df[(df[col] < low) | (df[col] > high)]

        if not out_of_range.empty:
            violations[col] = out_of_range.index.tolist()

    return violations


def plot_missing_before_after(
    before: pd.DataFrame, after: pd.DataFrame, output_dir: str | Path
) -> Path:
    """처리 전/후의 컬럼별 결측 비율(%)을 비교한 막대그래프를 저장한다."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    before_pct = before.isna().mean() * 100
    after_pct = after.isna().mean() * 100
    columns = [c for c in before_pct.index if c in after_pct.index]

    fig, ax = plt.subplots(figsize=(max(6, len(columns) * 0.8), 5))
    x = range(len(columns))
    ax.bar([i - 0.2 for i in x], before_pct[columns], width=0.4, label="처리 전")
    ax.bar([i + 0.2 for i in x], after_pct[columns], width=0.4, label="처리 후")
    ax.set_xticks(list(x))
    ax.set_xticklabels(columns, rotation=90)
    ax.set_ylabel("결측 비율 (%)")
    ax.legend()
    fig.tight_layout()

    output_path = output_dir / "missing_before_after.png"
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def check_missing(df: pd.DataFrame, columns: list | None = None) -> dict:
    """컬럼별 결측치 개수를 센다.

    반환값: {컬럼명: 결측 개수} — 결측이 있는 컬럼만 포함, 없으면 빈 dict.
    """

    columns = columns if columns is not None else df.columns

    if not all(col in df.columns for col in columns):
        raise ValueError("존재하지 않는 컬럼이 포함되어 있습니다.")

    missing = df[columns].isna().sum()

    return {col: int(count) for col, count in missing.items() if count > 0}


def plot_distributions(df: pd.DataFrame, columns: list, output_dir: str | Path) -> Path:
    """지정 컬럼들의 분포를 박스플롯으로 저장한다 (이상치 확인용)."""

    if not all(col in df.columns for col in columns):
        raise ValueError("존재하지 않는 컬럼이 포함되어 있습니다.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, len(columns), figsize=(5 * len(columns), 4))
    if len(columns) == 1:
        axes = [axes]
    for ax, col in zip(axes, columns):
        ax.boxplot(df[col].dropna())
        ax.set_title(col)
    fig.tight_layout()

    output_path = output_dir / "distributions.png"
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_correlation_heatmap(
    df: pd.DataFrame, feature_columns: list, target_columns: list, output_dir: str | Path
) -> Path:
    """feature(파생변수 포함) vs 목표변수 상관관계 히트맵을 저장한다."""

    columns = feature_columns + target_columns

    if not all(col in df.columns for col in columns):
        raise ValueError("존재하지 않는 컬럼이 포함되어 있습니다.")

    non_numeric = [
        col for col in columns
        if not pd.api.types.is_numeric_dtype(df[col])
    ]

    if non_numeric:
        raise ValueError(
            f"상관분석에는 수치형 컬럼만 사용할 수 있습니다: {non_numeric}"
        )
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    corr = df[columns].corr().loc[feature_columns, target_columns]

    fig, ax = plt.subplots(figsize=(6, max(4, len(feature_columns) * 0.4)))

    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    fig.tight_layout()

    output_path = output_dir / "correlation_heatmap.png"
    fig.savefig(output_path)
    plt.close(fig)

    return output_path


def _prepare_split(reference_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
    """reference_df 통계로 target_df의 결측치를 대치한 뒤 파생변수를 생성한다.

    결측치 대치와 파생변수 생성을 순서대로 적용하며,
    Validation/Test에는 Train 기준 통계를 사용해 데이터 누수를 방지한다.
    """

    imputed = impute_missing_from_reference(reference_df, target_df)
    return engineer_features(imputed)


def run_ablation_test(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_sets: dict,
    target_columns: list,
    random_state: int = 42,
) -> dict:
    """Train/Validation만 사용해 피처 조합별 성능을 비교한다.

    - 결측치 대치: Train 통계만 사용해 Train/Validation에 적용한다.
    - 파생변수 생성: 결측치 대치 후 각 split에 동일하게 적용한다.
    - Validation 성능만 비교해 최종 피처 조합을 선택한다.
    """

    for target in target_columns:
        if train_df[target].isna().any() or val_df[target].isna().any():
            raise ValueError(f"목표변수 '{target}'에 결측치가 있습니다.")

    train_prepared = _prepare_split(train_df, train_df)
    val_prepared = _prepare_split(train_df, val_df)

    results = {}

    for case_name, feature_columns in feature_sets.items():
        X_train = pd.get_dummies(train_prepared[feature_columns])

        X_val = pd.get_dummies(
            val_prepared[feature_columns]
        ).reindex(
            columns=X_train.columns,
            fill_value=0
        )

        for target in target_columns:
            y_train = train_df[target]
            y_val = val_df[target]

            model = RandomForestRegressor(
                random_state=random_state
            )

            model.fit(X_train, y_train)
            val_preds = model.predict(X_val)

            results[(case_name, target)] = {
                "val_r2": float(r2_score(y_val, val_preds)),
                "val_rmse": float(
                    np.sqrt(mean_squared_error(y_val, val_preds))
                ),
                "val_mae": float(
                    mean_absolute_error(y_val, val_preds)
                ),
            }

    return results


def plot_ablation_results(results: dict, output_dir: str | Path) -> Path:
    """Ablation test의 Validation R²를 케이스별 막대그래프로 저장한다."""

    if not results:
        raise ValueError("Ablation test 결과가 비어 있습니다.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    case_names = list(dict.fromkeys(case for case, _ in results.keys()))
    target_names = list(dict.fromkeys(target for _, target in results.keys()))

    fig, ax = plt.subplots(figsize=(max(6, len(case_names) * 1.8), 5))
    x = range(len(case_names))
    width = 0.8 / len(target_names)

    for i, target in enumerate(target_names):
        values = [
            results[(case, target)]["val_r2"]
            for case in case_names
        ]

        offsets = [
            xi + (i - (len(target_names) - 1) / 2) * width
            for xi in x
        ]

        ax.bar(
            offsets,
            values,
            width=width,
            label=target
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(case_names, rotation=20, ha="right")
    ax.set_ylabel("Validation R²")
    ax.legend()
    fig.tight_layout()

    output_path = output_dir / "ablation_results.png"
    fig.savefig(output_path)
    plt.close(fig)

    return output_path

def plot_grade_distribution(
    df: pd.DataFrame, output_dir: str | Path, grade_column: str = "원료등급"
) -> Path:
    """범주형 컬럼(기본: 원료등급)의 클래스 분포를 막대그래프로 저장한다 (데이터 불균형 확인용)."""

    if grade_column not in df.columns:
        raise ValueError(f"존재하지 않는 컬럼입니다: {grade_column}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    counts = df[grade_column].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(counts.index.astype(str), counts.values)
    ax.set_ylabel("배치 수")
    ax.set_xlabel(grade_column)
    for i, v in enumerate(counts.to_numpy()):
        ax.text(i, v, str(v), ha="center", va="bottom")
    fig.tight_layout()

    output_path = output_dir / "grade_distribution.png"
    fig.savefig(output_path)
    plt.close(fig)
    return output_path

def evaluate_final_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list,
    target_columns: list,
    random_state: int = 42,
) -> dict:
    """선택된 최종 피처 조합으로 Test 성능을 한 번만 평가한다.

    - Train과 Validation을 합쳐 최종 학습 데이터로 사용한다.
    - 결측치 대치 통계도 Train+Validation에서 다시 계산한다.
    - Test는 최종 성능 평가에만 사용한다.
    """

    for target in target_columns:
        if (
            train_df[target].isna().any()
            or val_df[target].isna().any()
            or test_df[target].isna().any()
        ):
            raise ValueError(f"목표변수 '{target}'에 결측치가 있습니다.")

    train_val_df = pd.concat([train_df, val_df])

    train_val_prepared = _prepare_split(
        train_val_df,
        train_val_df
    )
    test_prepared = _prepare_split(
        train_val_df,
        test_df
    )

    X_train_val = pd.get_dummies(
        train_val_prepared[feature_columns]
    )

    X_test = pd.get_dummies(
        test_prepared[feature_columns]
    ).reindex(
        columns=X_train_val.columns,
        fill_value=0
    )

    results = {}

    for target in target_columns:
        y_train_val = train_val_df[target]
        y_test = test_df[target]

        model = RandomForestRegressor(
            random_state=random_state
        )

        model.fit(X_train_val, y_train_val)
        test_preds = model.predict(X_test)

        results[target] = {
            "test_r2": float(
                r2_score(y_test, test_preds)
            ),
            "test_rmse": float(
                np.sqrt(mean_squared_error(y_test, test_preds))
            ),
            "test_mae": float(
                mean_absolute_error(y_test, test_preds)
            ),
        }

    return results