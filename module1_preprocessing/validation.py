import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import missingno as msno
import pandas as pd
import seaborn as sns

matplotlib.rcParams["font.family"] = "Noto Sans CJK JP"
matplotlib.rcParams["axes.unicode_minus"] = False
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold

from module1_preprocessing.config import NORMAL_RANGES, COMPOSITION_COLUMNS
from module1_preprocessing.imputation import accepted_range_mask
from module1_preprocessing.transformation import validate_targets, prepare_split, validate_model_features


def check_composition_sum(df: pd.DataFrame) -> pd.DataFrame:
    """같은 시료·같은 질량 기준의 원소 wt% 7종 합계를 점검한다.

    100% 초과는 검토 표시이며 오류 확정이나 자동 보정이 아니다.
    미측정 성분이 있으므로 100% 미만은 허용한다. 결측·무한대·개별
    0~100% 위반이 있는 행은 합계 판정을 보류한다(NA).
    1e-9 %p는 부동소수점 계산 오차만 무시하며 분석 불확도가 아니다.
    """
    values = df[COMPOSITION_COLUMNS].apply(pd.to_numeric, errors="raise")
    valid = (np.isfinite(values) & values.ge(0) & values.le(100)).all(axis=1)
    result = pd.DataFrame(index=df.index)
    result["composition_sum_pct"] = values.sum(axis=1).where(valid)
    flag = (result["composition_sum_pct"] > 100 + 1e-9).astype("boolean")
    result["composition_sum_review_flag"] = flag.where(valid, pd.NA)
    return result


def check_ranges(
    df: pd.DataFrame, ranges: dict = NORMAL_RANGES,
    reference_exception_ranges: dict | None = None,
) -> dict:
    """각 컬럼의 정상범위를 검증한다. 예외 구간을 전달하면 해당 값은 허용한다.
    결측 여부는 check_missing으로 별도 검증한다.

    반환값: {컬럼명: [위반 행 인덱스, ...]}
    위반이 없으면 빈 dict를 반환
    """

    violations = {}

    for col, (low, high) in ranges.items():
        if col not in df.columns:
            continue

        accepted = accepted_range_mask(
            df[col], (low, high), (reference_exception_ranges or {}).get(col, ())
        )
        out_of_range = df[df[col].notna() & ~accepted]

        if not out_of_range.empty:
            violations[col] = out_of_range.index.tolist()

    return violations


def format_range_violations(df: pd.DataFrame, violations: dict) -> str:
    """범위 이탈을 변수명·실제 배치ID·관측값으로 표시함"""
    sections = []
    for column, indices in violations.items():
        values = df.loc[df.index.isin(indices), ["배치ID", column]]
        if values.empty:
            continue
        table = values.rename(columns={column: "관측값"}).to_string(index=False)
        sections.append(f"변수명: {column}\n{'─' * 40}\n{table}")
    return "\n\n".join(sections) if sections else "없음"


def plot_missing_before_after(
    before: pd.DataFrame, after: pd.DataFrame, output_dir: str | Path,
    columns: list | None = None,
) -> Path:
    """처리 전/후의 컬럼별 결측 비율(%)을 비교한 막대그래프를 저장
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    before_pct = before.isna().mean() * 100
    after_pct = after.isna().mean() * 100
    shared_columns = [c for c in before_pct.index if c in after_pct.index]
    columns = [c for c in columns if c in shared_columns] if columns is not None else shared_columns

    fig, ax = plt.subplots(figsize=(max(9, len(columns) * 0.85), 4.5))
    positions = np.arange(len(columns))
    labels = [c.rsplit('_', 1)[0].replace('_', ' ')
              if c.endswith(('_pct', '_C', '_hr', '_kg', '_L')) else c
              for c in columns]
    labels = [label.replace(' ', '\n') if ' ' in label else
              {'침출온도': '침출\n온도', '침출시간': '침출\n시간',
               '원료투입량': '원료\n투입량', '황산투입량': '황산\n투입량',
               '분석일자': '분석\n일자'}.get(label, label) for label in labels]
    for offset, values, label, color in [
        (-0.18, before_pct[columns], '대치 전', '#427AA1'),
        (0.18, after_pct[columns], '대치 후', '#E89B52'),
    ]:
        bars = ax.bar(positions + offset, values, width=0.32, label=label, color=color)
        if label == '대치 전':
            value_labels = ['0%' if v == 0 else f'{v:.1f}%' for v in values]
            ax.bar_label(bars, labels=value_labels, padding=4, fontsize=9)
    ax.set_xticks(positions, labels)
    ax.set_ylabel('결측률 (%)')
    ax.set_title('결측치 대치 전·후 비교')
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 20))
    ax.yaxis.grid(True, alpha=0.2)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(loc='upper right', ncol=2)
    fig.tight_layout()

    output_path = output_dir / "missing_before_after.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_missingno_matrix(
    df: pd.DataFrame, output_dir: str | Path, columns: list | None = None
) -> Path:
    """missingno matrix로 행 단위 결측 패턴(어느 행에서 어느 컬럼이 동시에 비는지)을 저장

    columns를 지정하지 않으면 df의 모든 컬럼을 그린다 — `_dict_range_flag`처럼
    결측이 원천적으로 발생할 수 없는 컬럼까지 섞이면 그림만 넓어지고 흐려지므로,
    실제 결측이 있을 수 있는 원변수만 보고 싶으면 columns로 좁혀서 넘긴다.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if columns is not None:
        df = df[columns]

    ax = msno.matrix(df)
    fig = ax.get_figure()
    fig.tight_layout()

    output_path = output_dir / "missingno_matrix.png"
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_missingno_bar(df: pd.DataFrame, output_dir: str | Path) -> Path:
    """missingno bar로 컬럼별 결측 비율을 저장"""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ax = msno.bar(df)
    fig = ax.get_figure()
    fig.tight_layout()

    output_path = output_dir / "missingno_bar.png"
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_missingno_heatmap(df: pd.DataFrame, output_dir: str | Path) -> Path:
    """missingno heatmap으로 컬럼 쌍 간 결측 동시발생 상관관계를 저장"""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ax = msno.heatmap(df)
    fig = ax.get_figure()
    fig.tight_layout()

    output_path = output_dir / "missingno_heatmap.png"
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


_COLUMN_SUFFIX_UNITS = {
    "_pct": "%",
    "_C": "°C",
    "_hr": "h",
    "_kg": "kg",
    "_L": "L",
}


def _infer_unit_label(col: str) -> str:
    """컬럼명 접미사(_pct, _C 등)로 단위를 추정해 'col (단위)' 형태의 축 레이블을 만듦"""
    for suffix, unit in _COLUMN_SUFFIX_UNITS.items():
        if col.endswith(suffix):
            return f"{col} ({unit})"
    return col


def plot_numeric_histograms(df: pd.DataFrame, columns: list, output_dir: str | Path) -> Path:
    """수치형 컬럼들의 히스토그램 저장"""

    if not all(col in df.columns for col in columns):
        raise ValueError("존재하지 않는 컬럼이 포함되어 있습니다.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ncols = min(3, len(columns))
    nrows = math.ceil(len(columns) / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax, col in zip(axes, columns):
        ax.hist(df[col].dropna(), bins=20, color="steelblue", edgecolor="white")
        ax.set_title(col)
        ax.set_xlabel(_infer_unit_label(col))
        ax.set_ylabel("빈도수(행 개수)")

    for ax in axes[len(columns):]:
        ax.axis("off")

    fig.tight_layout()

    output_path = output_dir / "numeric_histograms.png"
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def check_missing(df: pd.DataFrame, columns: list | None = None) -> dict:
    """컬럼별 결측치 개수 셈
    반환값: {컬럼명: 결측 개수} — 결측이 있는 컬럼만 포함, 없으면 빈 dict
    """

    columns = columns if columns is not None else df.columns

    if not all(col in df.columns for col in columns):
        raise ValueError("존재하지 않는 컬럼이 포함되어 있습니다.")

    missing = df[columns].isna().sum()

    return {col: int(count) for col, count in missing.items() if count > 0}


def plot_distributions(df: pd.DataFrame, columns: list, output_dir: str | Path) -> Path:
    """지정 컬럼들의 분포를 박스플롯으로 저장 (이상치 확인용)"""

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
    df: pd.DataFrame, feature_columns: list, target_columns: list, output_dir: str | Path,
    display_labels: dict | None = None,
) -> Path:
    """feature(파생변수 포함) vs 목표변수 상관관계 히트맵을 저장

    display_labels: {원래 컬럼명: 표시용 라벨} — 상관계수 계산에는 영향 없이,
    그래프에 표시되는 행 이름만 바꾼다(예: 영문 변수명을 한글로).
    """

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

    if display_labels:
        corr = corr.rename(index=display_labels)

    fig, ax = plt.subplots(figsize=(6, max(4, len(feature_columns) * 0.4)))

    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, center=0, ax=ax)
    fig.tight_layout()

    output_path = output_dir / "correlation_heatmap.png"
    fig.savefig(output_path)
    plt.close(fig)

    return output_path


def plot_ablation_results(results: dict, output_dir: str | Path) -> Path:
    """Ablation test의 평균 CV MAE를 케이스별 막대그래프로 저장함"""

    if not results:
        raise ValueError("Ablation test 결과가 비어 있습니다.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    case_names = list(dict.fromkeys(case for case, _ in results.keys()))
    target_names = list(dict.fromkeys(target for _, target in results.keys()))
    case_labels = {"base": "원변수", "base_plus_grade": "원변수+등급",
                   "base_plus_derived": "원변수+파생", "all_features": "전체 변수"}
    target_labels = {"Ni_회수율_pct": "Ni 회수율", "Co_회수율_pct": "Co 회수율"}

    fig, ax = plt.subplots(figsize=(max(6, len(case_names) * 1.8), 5))
    x = range(len(case_names))
    width = 0.8 / len(target_names)

    for i, target in enumerate(target_names):
        values = [
            results[(case, target)]["cv_mae"]
            for case in case_names
        ]

        offsets = [
            xi + (i - (len(target_names) - 1) / 2) * width
            for xi in x
        ]

        bars = ax.bar(
            offsets,
            values,
            width=width,
            label=target_labels.get(target, target)
        )
        ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)

    ax.set_xticks(list(x))
    ax.set_xticklabels([case_labels.get(c, c) for c in case_names], rotation=0, ha="center")
    ax.set_ylabel("CV MAE (%p)")
    ax.set_title("특징 조합에 따른 Ni·Co 회수율 예측 오차 비교")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    fig.tight_layout()

    output_path = output_dir / "ablation_results.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    return output_path

def plot_grade_distribution(
    df: pd.DataFrame, output_dir: str | Path, grade_column: str = "원료등급"
) -> Path:
    """범주형 컬럼(기본: 원료등급)의 클래스 분포를 막대그래프로 저장 (데이터 불균형 확인용)"""

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


def run_cv_ablation_test(
    pool_df: pd.DataFrame,
    feature_sets: dict,
    target_columns: list,
    n_splits: int = 5,
    stratify_column: str = "원료등급",
    random_state: int = 42,
    model_factory=None,
) -> dict:
    """Stratified K-Fold CV로 피처 조합별 평균 성능을 비교

    단일 Train/Validation 분할은 Validation 행 수가 적어(약 63행) 특정 분할의
    운에 따라 결과가 흔들릴 수 있음. pool_df(보통 Train+Validation)를
    stratify_column 비율을 유지하며 n_splits개 fold로 나누고, fold마다
    결측치 대치 통계를 그 fold의 train 조각에서만 계산해 누수 없이 평가한 뒤 fold별 성능을 평균함
    Test는 pool_df에 포함하지 않음
    model_factory(random_state) -> 학습 전 모델 인스턴스. 지정 안 하면 RandomForestRegressor
    """

    if "split" in pool_df and not pool_df["split"].isin(["train", "validation"]).all():
        raise ValueError("피처 선택에는 개발 데이터만 허용됩니다. Test를 제외하세요.")

    if not isinstance(n_splits, int) or n_splits < 2:
        raise ValueError("CV 분할 수는 2 이상의 정수여야 함")
    if stratify_column not in pool_df or pool_df[stratify_column].isna().any():
        raise ValueError("층화 기준 컬럼이 없거나 결측값이 있음")
    if pool_df[stratify_column].value_counts().empty or pool_df[stratify_column].value_counts().min() < n_splits:
        raise ValueError("층화 CV에는 등급별로 분할 수 이상의 표본이 필요함")
    if "배치ID" in pool_df and (pool_df["배치ID"].isna().any() or pool_df["배치ID"].duplicated().any()):
        raise ValueError("CV 입력의 배치ID 누락 또는 중복 있음")

    if model_factory is None:
        model_factory = lambda rs: RandomForestRegressor(random_state=rs)

    for target in target_columns:
        if pool_df[target].isna().any():
            raise ValueError(f"목표변수 '{target}'에 결측치가 있습니다.")

    validate_targets(pool_df, target_columns)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_indices = list(skf.split(pool_df, pool_df[stratify_column]))

    results = {}

    for case_name, feature_columns in feature_sets.items():
        for target in target_columns:
            fold_r2, fold_mse, fold_mae = [], [], []

            for train_pos, val_pos in fold_indices:
                fold_train = pool_df.iloc[train_pos]
                fold_val = pool_df.iloc[val_pos]

                train_prepared = prepare_split(fold_train, fold_train)
                val_prepared = prepare_split(fold_train, fold_val)

                X_train = pd.get_dummies(train_prepared[feature_columns])
                X_val = pd.get_dummies(
                    val_prepared[feature_columns]
                ).reindex(columns=X_train.columns, fill_value=0)

                validate_model_features(X_train, list(X_train.columns))
                validate_model_features(X_val, list(X_train.columns))
                model = model_factory(random_state)
                model.fit(X_train, fold_train[target])
                preds = model.predict(X_val)

                y_val = fold_val[target]
                fold_r2.append(r2_score(y_val, preds))
                fold_mse.append(mean_squared_error(y_val, preds))
                fold_mae.append(mean_absolute_error(y_val, preds))

            mean_mse = float(np.mean(fold_mse))

            results[(case_name, target)] = {
                "cv_r2": float(np.mean(fold_r2)),
                "cv_mse": mean_mse,
                "cv_rmse": float(np.sqrt(mean_mse)),
                "cv_mae": float(np.mean(fold_mae)),
                "cv_fold_rmse_mean": float(np.mean(np.sqrt(fold_mse))),
                "cv_rmse_std": float(np.std(np.sqrt(fold_mse), ddof=1)),
                "cv_mae_std": float(np.std(fold_mae, ddof=1)),
            }

    return results
