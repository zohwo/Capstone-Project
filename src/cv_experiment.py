"""
실행:
    python -m src.cv_experiment                  # 시간순(확장 창) 교차검증 5폴드 (기본, 주 평가)
    python -m src.cv_experiment --mode kfold     # 섞은 5겹 x 3회 반복 (안정성 참고용)

- Train+Validation(357행 안팎)만 사용한다. Test는 읽지도, 평가하지도 않는다.
- 폴드마다 '폴드-Train'만으로 대치·파생변수를 계산한다 (팀의 _prepare_split 재사용, 누수 방지).
- 결과 (data/ 는 git에 포함되지 않음):
    data/processed/cv_folds.csv        폴드별 점수
    data/processed/cv_summary.csv      모델·피처 조합별 평균 ± 표준편차
    data/processed/cv_paired_diff.csv  XGBoost 대비 폴드별 RMSE 차이 (우연 여부 확인용)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import RepeatedKFold, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import COMPOSITION_COLUMNS, DERIVED_FEATURES, PROCESS_COLUMNS, TARGET_COLUMNS
from src.imputation import interpolate_analysis_date, replace_outliers_with_nan
from src.preprocessing import clean_schema, load_raw, resolve_duplicate_batches
from src.splitting import temporal_train_val_test_split
from src.validation import _prepare_split
from src.xgb_experiment import DEFAULT_INPUT, _features, _metrics, make_model

DEFAULT_OUT_DIR = Path("data/processed")


def make_cv_model(model_name: str, random_state: int = 42):
    """rf / xgb 는 기존 설정 그대로, lgbm / ridge 는 비교용 (모두 튜닝 전 초기값)."""

    if model_name in ("rf", "xgb"):
        return make_model(model_name, random_state)

    if model_name == "lgbm":
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )

    if model_name == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))

    raise ValueError(f"지원하지 않는 모델입니다: {model_name}")


def available_models(requested) -> list:
    """설치되지 않은 라이브러리의 모델은 건너뛴다."""

    models = []
    for name in requested:
        try:
            make_cv_model(name)
            models.append(name)
        except ImportError:
            print(f"[건너뜀] {name}: 라이브러리가 설치되어 있지 않습니다 (pip install lightgbm)")
    return models


def load_train_val(input_path) -> pd.DataFrame:
    """팀 pipeline.run 과 같은 정제까지만 수행하고, Train+Validation 행만 시간순으로 돌려준다."""

    df = load_raw(input_path)
    df = clean_schema(df)
    df = resolve_duplicate_batches(df)
    df = replace_outliers_with_nan(df)
    df = interpolate_analysis_date(df)

    train_i, val_i, _test_i = temporal_train_val_test_split(df)  # Test 인덱스는 버린다
    return pd.concat([df.loc[train_i], df.loc[val_i]]).reset_index(drop=True)


def make_splits(n_rows: int, mode: str, n_splits: int, n_repeats: int, seed: int) -> list:
    index = np.arange(n_rows)

    if mode == "time":
        # 앞 구간으로 학습 → 바로 뒤 구간으로 평가 (학습 구간이 점점 늘어남)
        return list(TimeSeriesSplit(n_splits=n_splits).split(index))

    if mode == "kfold":
        # 시간 순서를 무시하고 섞는다. 이웃 배치가 학습에 들어가 점수가 낙관적일 수 있다.
        return list(RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed).split(index))

    raise ValueError("mode 는 'time' 또는 'kfold' 여야 합니다.")


def run_cv(
    mode="time",
    model_names=("rf", "xgb", "lgbm", "ridge"),
    n_splits=5,
    n_repeats=3,
    seed=42,
    input_path=DEFAULT_INPUT,
    out_dir=DEFAULT_OUT_DIR,
):
    tv = load_train_val(input_path)
    print(f"교차검증 대상(Train+Validation): {len(tv)}행, 모드: {mode}")

    model_names = available_models(model_names)

    base = COMPOSITION_COLUMNS + PROCESS_COLUMNS
    feature_sets = {"base": base, "base_plus_derived": base + DERIVED_FEATURES}

    splits = make_splits(len(tv), mode, n_splits, n_repeats, seed)

    rows = []
    for fold, (tr_i, va_i) in enumerate(splits):
        tr, va = tv.iloc[tr_i], tv.iloc[va_i]

        # 대치·파생변수는 폴드-Train 통계만 사용
        tr_prep = _prepare_split(tr, tr)
        va_prep = _prepare_split(tr, va)

        for model_name in model_names:
            for target in TARGET_COLUMNS:
                for case, cols in feature_sets.items():
                    x_tr = _features(tr_prep, cols)
                    x_va = _features(va_prep, cols, x_tr.columns)

                    model = make_cv_model(model_name, seed).fit(x_tr, tr[target])
                    scores = _metrics(va[target], model.predict(x_va))

                    rows.append({
                        "mode": mode, "fold": fold, "n_train": len(tr), "n_val": len(va),
                        "model": model_name, "target": target, "feature_set": case, **scores,
                    })

        print(f"  폴드 {fold + 1}/{len(splits)} 완료 (train {len(tr)} / val {len(va)})")

    folds = pd.DataFrame(rows)

    summary = (
        folds.groupby(["target", "model", "feature_set"])
        .agg(
            rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
            mae_mean=("mae", "mean"),
            r2_mean=("r2", "mean"), r2_std=("r2", "std"),
            n_folds=("fold", "count"),
        )
        .reset_index()
        .sort_values(["target", "rmse_mean"])
        .round(4)
    )

    paired = paired_difference(folds)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    folds.to_csv(out_dir / "cv_folds.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / "cv_summary.csv", index=False, encoding="utf-8-sig")
    paired.to_csv(out_dir / "cv_paired_diff.csv", index=False, encoding="utf-8-sig")

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print("\n=== 모델·피처 조합별 요약 (RMSE 낮은 순) ===")
        print(summary.to_string(index=False))
        print("\n=== XGBoost 대비 폴드별 RMSE 차이 (base_plus_derived) ===")
        print("rmse_diff_mean > 0 이면 해당 모델이 XGBoost보다 나쁨, xgb_better_folds 는 XGBoost가 더 나았던 폴드 수")
        print(paired.to_string(index=False))

    print(f"\n결과 저장: {out_dir}/cv_folds.csv, cv_summary.csv, cv_paired_diff.csv")
    return folds, summary, paired


def paired_difference(folds: pd.DataFrame, ref_model: str = "xgb", feature_set: str = "base_plus_derived"):
    """같은 폴드끼리 RMSE 차이를 계산한다. 차이가 폴드 간 흔들림(표준편차)보다 작으면 우연으로 본다."""

    sub = folds[folds["feature_set"] == feature_set]
    if ref_model not in set(sub["model"]):
        return pd.DataFrame()

    out = []
    for target, g in sub.groupby("target"):
        pivot = g.pivot_table(index="fold", columns="model", values="rmse")
        for model_name in pivot.columns:
            if model_name == ref_model:
                continue
            diff = pivot[model_name] - pivot[ref_model]
            out.append({
                "target": target, "model": model_name, "vs": ref_model,
                "rmse_diff_mean": round(float(diff.mean()), 4),
                "rmse_diff_std": round(float(diff.std()), 4),
                "xgb_better_folds": int((diff > 0).sum()),
                "n_folds": int(len(diff)),
            })
    return pd.DataFrame(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["time", "kfold"], default="time")
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    run_cv(mode=args.mode, n_splits=args.splits, n_repeats=args.repeats)
