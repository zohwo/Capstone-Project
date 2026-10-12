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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

from src.config import COMPOSITION_COLUMNS, NORMAL_RANGES, PROCESS_COLUMNS
from src.features import engineer_features


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


def _fold_safe_prepare(train_df: pd.DataFrame, apply_df: pd.DataFrame) -> pd.DataFrame:
    """train_df의 통계(그룹별/전체 중앙값, 회귀계수)만으로 apply_df의 결측치를 대치하고
    파생변수를 생성한다. apply_df가 train_df 자신이면 일반적인 전처리와 동일하고,
    apply_df가 별도의 test 구간이면 train 통계만 새어나가게 해서 data leakage를 막는다.
    """
    out = apply_df.copy()

    for col in COMPOSITION_COLUMNS:
        group_medians = train_df.groupby("원료등급")[col].median()
        overall_median = train_df[col].median()
        missing = out[col].isna()
        if missing.any():
            out.loc[missing, col] = (
                out.loc[missing, "원료등급"].map(group_medians).fillna(overall_median)
            )

    for col in PROCESS_COLUMNS:
        out[col] = out[col].fillna(train_df[col].median())

    train_missing = train_df["황산투입량_L"].isna()
    train_acid = train_df.loc[~train_missing, ["원료투입량_kg", "황산투입량_L"]].dropna()
    if not train_acid.empty:
        model = LinearRegression().fit(
            train_acid[["원료투입량_kg"]], train_acid["황산투입량_L"]
        )
        acid_missing = out["황산투입량_L"].isna()
        if acid_missing.any():
            out.loc[acid_missing, "황산투입량_L"] = model.predict(
                out.loc[acid_missing, ["원료투입량_kg"]]
            )

    return engineer_features(out)


def run_ablation_test(
    df: pd.DataFrame,
    feature_sets: dict,
    target_columns: list,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict:
    """피처 조합별로 교차검증 R²/RMSE를 계산해 파생변수·원료등급의 성능 기여를 정량 비교한다.

    df는 결측치 대치 전(flag_outliers까지만 적용된) 데이터를 받는다. 각 fold마다 train
    구간의 통계로만 결측치를 대치하고 파생변수를 만들어 train/test에 각각 적용하므로,
    fold 간 data leakage 없이 평가한다.
    """
    results = {}
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_indices = list(kf.split(df))

    for case_name, feature_columns in feature_sets.items():
        for target in target_columns:
            r2_scores = []
            rmse_scores = []

            for train_pos, test_pos in fold_indices:
                train_i = df.index[train_pos]
                test_i = df.index[test_pos]

                train_prepared = _fold_safe_prepare(df.loc[train_i], df.loc[train_i])
                test_prepared = _fold_safe_prepare(df.loc[train_i], df.loc[test_i])

                X_train = pd.get_dummies(train_prepared[feature_columns])
                X_test = pd.get_dummies(test_prepared[feature_columns]).reindex(
                    columns=X_train.columns, fill_value=0
                )
                y_train = df.loc[train_i, target]
                y_test = df.loc[test_i, target]

                model = RandomForestRegressor(random_state=random_state)
                model.fit(X_train, y_train)
                preds = model.predict(X_test)

                r2_scores.append(r2_score(y_test, preds))
                rmse_scores.append(float(np.sqrt(mean_squared_error(y_test, preds))))

            results[(case_name, target)] = {
                "r2_mean": float(np.mean(r2_scores)),
                "r2_std": float(np.std(r2_scores)),
                "rmse_mean": float(np.mean(rmse_scores)),
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
