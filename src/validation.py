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
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_validate

from src.config import NORMAL_RANGES


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


def run_ablation_test(
    df: pd.DataFrame,
    feature_sets: dict,
    target_columns: list,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict:
    """피처 조합별로 교차검증 R²/RMSE를 계산해 파생변수·원료등급의 성능 기여를 정량 비교한다."""
    results = {}

    for case_name, feature_columns in feature_sets.items():
        X = pd.get_dummies(df[feature_columns])

        for target in target_columns:
            y = df[target]
            model = RandomForestRegressor(random_state=random_state)
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            scores = cross_validate(
                model, X, y, cv=cv, scoring=["r2", "neg_root_mean_squared_error"]
            )
            results[(case_name, target)] = {
                "r2_mean": float(scores["test_r2"].mean()),
                "r2_std": float(scores["test_r2"].std()),
                "rmse_mean": float(-scores["test_neg_root_mean_squared_error"].mean()),
            }

    return results


def plot_ablation_results(results: dict, output_dir: str | Path) -> Path:
    """ablation test 결과(R² 평균 ± fold 표준편차)를 케이스별 막대그래프로 저장한다.

    오차막대(fold 표준편차)를 같이 그려서, 케이스 간 R² 차이가 fold 노이즈보다
    큰 실제 효과인지 한눈에 비교할 수 있게 한다.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    case_names = list(dict.fromkeys(case for case, _ in results.keys()))
    target_names = list(dict.fromkeys(target for _, target in results.keys()))

    fig, ax = plt.subplots(figsize=(max(6, len(case_names) * 1.5), 5))
    x = range(len(case_names))
    width = 0.8 / len(target_names)

    for i, target in enumerate(target_names):
        means = [results[(case, target)]["r2_mean"] for case in case_names]
        stds = [results[(case, target)]["r2_std"] for case in case_names]
        offsets = [xi + (i - (len(target_names) - 1) / 2) * width for xi in x]
        ax.bar(offsets, means, width=width, yerr=stds, capsize=4, label=target)

    ax.set_xticks(list(x))
    ax.set_xticklabels(case_names, rotation=20, ha="right")
    ax.set_ylabel("R² (mean ± fold std)")
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
