from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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
