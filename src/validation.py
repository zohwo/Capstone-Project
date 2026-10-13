from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# matplotlib's font manager only registers the first face of the Noto Sans CJK
# .ttc collection under the "CJK JP" name, but that face still contains the
# Hangul glyphs needed to render Korean labels correctly.
matplotlib.rcParams["font.family"] = "Noto Sans CJK JP"
matplotlib.rcParams["axes.unicode_minus"] = False
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import NORMAL_RANGES
from src.features import engineer_features
from src.imputation import impute_missing_from_reference
from src.splitting import temporal_train_val_test_split


def check_ranges(df: pd.DataFrame, ranges: dict = NORMAL_RANGES) -> dict:
    """정제 후 데이터가 데이터사전의 정상 범위를 만족하는지 검증한다.

    반환값: {컬럼명: [위반 행 인덱스, ...]} — 위반이 없으면 빈 dict
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
    """컬럼별 결측 비율을 처리 전/후로 비교하는 막대그래프를 저장한다."""
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


def plot_distributions(df: pd.DataFrame, columns: list, output_dir: str | Path) -> Path:
    """지정 컬럼들의 분포를 박스플롯으로 저장한다 (이상치 확인용)."""
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
    """피처(파생변수 포함) vs 목표변수 상관관계 히트맵을 저장한다."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    corr = df[feature_columns + target_columns].corr().loc[feature_columns, target_columns]

    fig, ax = plt.subplots(figsize=(6, max(4, len(feature_columns) * 0.4)))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    fig.tight_layout()

    output_path = output_dir / "correlation_heatmap.png"
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def _prepare_split(reference_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
    """reference_df 통계로 target_df의 결측치를 leak-safe하게 대치한 뒤 파생변수를 생성한다.

    결측치 대치(`impute_missing_from_reference`, 6단계)와 파생변수 생성
    (`engineer_features`, 7단계)을 분리된 단계로 순서대로 적용한다.
    """
    imputed = impute_missing_from_reference(reference_df, target_df)
    return engineer_features(imputed)


def run_ablation_test(
    df: pd.DataFrame,
    feature_sets: dict,
    target_columns: list,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    random_state: int = 42,
) -> dict:
    """피처 조합별로 시간순 Train/Validation/Test 성능을 비교해 파생변수·원료등급의
    성능 기여를 정량 비교한다.

    df는 결측치 대치 전(flag_outliers까지만 적용된) 데이터를 받는다. 배치ID 순번(분석일자
    순서의 근사)으로 시간순 정렬한 뒤, Train 구간 통계로만 결측치를 대치해 Validation/Test
    구간에 적용하므로 미래 데이터를 미리 들여다보지 않는 leak-safe 평가가 된다. Validation은
    피처 조합 간 비교(이 함수의 목적)에 쓰고, Test는 최종 확인용으로 함께 보고한다.
    """
    train_i, val_i, test_i = temporal_train_val_test_split(df, train_frac, val_frac)

    results = {}
    for case_name, feature_columns in feature_sets.items():
        train_prepared = _prepare_split(df.loc[train_i], df.loc[train_i])
        val_prepared = _prepare_split(df.loc[train_i], df.loc[val_i])
        test_prepared = _prepare_split(df.loc[train_i], df.loc[test_i])

        X_train = pd.get_dummies(train_prepared[feature_columns])
        X_val = pd.get_dummies(val_prepared[feature_columns]).reindex(
            columns=X_train.columns, fill_value=0
        )
        X_test = pd.get_dummies(test_prepared[feature_columns]).reindex(
            columns=X_train.columns, fill_value=0
        )

        for target in target_columns:
            y_train = df.loc[train_i, target]
            y_val = df.loc[val_i, target]
            y_test = df.loc[test_i, target]

            model = RandomForestRegressor(random_state=random_state)
            model.fit(X_train, y_train)

            val_preds = model.predict(X_val)
            test_preds = model.predict(X_test)

            results[(case_name, target)] = {
                "val_r2": float(r2_score(y_val, val_preds)),
                "val_rmse": float(np.sqrt(mean_squared_error(y_val, val_preds))),
                "val_mae": float(mean_absolute_error(y_val, val_preds)),
                "test_r2": float(r2_score(y_test, test_preds)),
                "test_rmse": float(np.sqrt(mean_squared_error(y_test, test_preds))),
                "test_mae": float(mean_absolute_error(y_test, test_preds)),
            }

    return results


def plot_ablation_results(results: dict, output_dir: str | Path) -> Path:
    """ablation test 결과(Validation/Test R²)를 케이스별 막대그래프로 저장한다.

    Validation과 Test를 나란히 보여줘서, 케이스 간 R² 차이가 두 구간 모두에서
    일관되게 나타나는 실제 효과인지 한눈에 비교할 수 있게 한다.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    case_names = list(dict.fromkeys(case for case, _ in results.keys()))
    target_names = list(dict.fromkeys(target for _, target in results.keys()))
    series = [(target, split) for target in target_names for split in ("val", "test")]

    fig, ax = plt.subplots(figsize=(max(6, len(case_names) * 1.8), 5))
    x = range(len(case_names))
    width = 0.8 / len(series)

    for i, (target, split) in enumerate(series):
        values = [results[(case, target)][f"{split}_r2"] for case in case_names]
        offsets = [xi + (i - (len(series) - 1) / 2) * width for xi in x]
        ax.bar(offsets, values, width=width, label=f"{target} ({split})")

    ax.set_xticks(list(x))
    ax.set_xticklabels(case_names, rotation=20, ha="right")
    ax.set_ylabel("R²")
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
