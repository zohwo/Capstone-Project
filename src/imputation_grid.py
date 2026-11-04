"""
이상치 판정 x 대치 방식 2x2 비교 (교차검증)

실행 (프로젝트 루트에서):
    python -m src.imputation_grid                     # XGBoost, 섞은 5겹 x 3회
    python -m src.imputation_grid --repeats 1         # 빠른 확인
    python -m src.imputation_grid --model rf          # RandomForest

비교 축
    판정(rule)    team  = 정상범위 밖 전부 NaN (팀 코드)
                  mine  = 명백한 오류 + (경미 초과 & 등급별 IQR 밖) NaN (outlier_experiment.replace_outliers_mine)
    대치(imputer) median_reg = 성분 등급별 중앙값, 온도·시간 전체 중앙값, 황산투입량은 원료투입량 선형회귀
                               (옛 팀 코드 방식, 이 파일 안에 복사해 둠)
                  iterative  = 입력 11개 전체 반복회귀 (Features13 방식, 이 파일 안에 복사해 둠)
    피처 세트     base11, base11 + DERIVED_FEATURES (src.config 의 현재 값)

네 조합이 같은 폴드를 쓰므로 폴드별 짝 비교가 가능하다.
Train+Validation 만 사용하고 Test 는 읽지 않는다.
결과 (data/processed, git 제외): imputation_grid_folds.csv, imputation_grid_summary.csv, imputation_grid_paired.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler

from src.config import COMPOSITION_COLUMNS, DERIVED_FEATURES, NORMAL_RANGES, PROCESS_COLUMNS, TARGET_COLUMNS
from src.features import engineer_features
from src.imputation import interpolate_analysis_date, replace_outliers_with_nan
from src.outlier_experiment import replace_outliers_mine
from src.preprocessing import clean_schema, load_raw, resolve_duplicate_batches
from src.splitting import temporal_train_val_test_split

DEFAULT_INPUT = Path("data/raw/영풍그룹_금속회수율최적화_RawMaterialData_학생배포용.xlsx")
DEFAULT_OUT_DIR = Path("data/processed")
RULES = ("team", "mine")
IMPUTERS = ("median_reg", "iterative")


# ---------------------------------------------------------------- 대치 함수 (두 버전을 그대로 복사)
def impute_median_reg(reference_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
    """옛 팀 코드: 성분=등급별 중앙값, 온도·시간=전체 중앙값, 황산=원료투입량 선형회귀. 원료투입량은 대치 안 함."""

    out = target_df.copy()

    for col in COMPOSITION_COLUMNS:
        group_medians = reference_df.groupby("원료등급")[col].median()
        overall_median = reference_df[col].median()
        missing = out[col].isna()
        if missing.any():
            out.loc[missing, col] = out.loc[missing, "원료등급"].map(group_medians).fillna(overall_median)

    for col in PROCESS_COLUMNS:
        out[col] = out[col].fillna(reference_df[col].median())

    ref_acid = reference_df[["원료투입량_kg", "황산투입량_L"]].dropna()
    if not ref_acid.empty:
        model = LinearRegression().fit(ref_acid[["원료투입량_kg"]], ref_acid["황산투입량_L"])
        acid_missing = out["황산투입량_L"].isna() & out["원료투입량_kg"].notna()
        if acid_missing.any():
            out.loc[acid_missing, "황산투입량_L"] = model.predict(out.loc[acid_missing, ["원료투입량_kg"]])
    return out


def impute_iterative(reference_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
    """Features13 팀 코드: 입력 11개를 표준화한 뒤 IterativeImputer(BayesianRidge)로 대치, Train 관측 범위로 제한."""

    columns = list(NORMAL_RANGES)
    reference = reference_df[columns].astype(float)
    raw = target_df[columns].astype(float)

    out = target_df.copy()
    if raw.empty:
        return out
    scaler = StandardScaler().fit(reference)
    imputer = IterativeImputer(max_iter=30, random_state=42, tol=1e-3, sample_posterior=False)
    imputer.fit(scaler.transform(reference))
    estimates = scaler.inverse_transform(imputer.transform(scaler.transform(raw)))
    filled = pd.DataFrame(estimates, index=raw.index, columns=columns).clip(reference.min(), reference.max(), axis=1)
    for col in columns:
        missing = raw[col].isna()
        out.loc[missing, col] = filled.loc[missing, col]
    return out


IMPUTE_FUNCS = {"median_reg": impute_median_reg, "iterative": impute_iterative}


def prepare(imputer: str, reference_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
    return engineer_features(IMPUTE_FUNCS[imputer](reference_df, target_df))


# ---------------------------------------------------------------- 데이터와 모델
def load_by_rule(input_path) -> dict:
    """판정 기준별로 정제한 Train+Validation 데이터 (행 순서 동일, Test 제외)."""

    df = load_raw(input_path)
    df = clean_schema(df)
    df = resolve_duplicate_batches(df)
    train_i, val_i, _test_i = temporal_train_val_test_split(df)

    out = {}
    for rule in RULES:
        d = replace_outliers_with_nan(df) if rule == "team" else replace_outliers_mine(df, train_i)
        d = interpolate_analysis_date(d)
        out[rule] = pd.concat([d.loc[train_i], d.loc[val_i]]).reset_index(drop=True)
    return out


def make_model(name: str, seed: int = 42):
    if name == "xgb":
        from xgboost import XGBRegressor

        return XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8,
                            colsample_bytree=0.8, random_state=seed, n_jobs=-1)
    if name == "rf":
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(random_state=seed, n_jobs=-1)
    raise ValueError("model 은 xgb 또는 rf 여야 합니다.")


def _score(y, p) -> dict:
    return {"rmse": float(np.sqrt(mean_squared_error(y, p))), "mae": float(mean_absolute_error(y, p)),
            "r2": float(r2_score(y, p))}


# ---------------------------------------------------------------- 실행
def run_grid(by_rule: dict, model_name="xgb", n_splits=5, n_repeats=3, seed=42, out_dir=DEFAULT_OUT_DIR, save=True):
    n = len(next(iter(by_rule.values())))
    assert all(len(v) == n for v in by_rule.values()), "판정 기준별 행 수가 같아야 합니다."

    base = list(NORMAL_RANGES)
    feature_sets = {"base11": base, "base11+derived": base + list(DERIVED_FEATURES)}
    splits = list(RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed).split(np.arange(n)))
    print(f"대상 {n}행, 모델 {model_name}, 폴드 {len(splits)}개, 조합 {len(RULES) * len(IMPUTERS)}개")

    rows = []
    for fold, (tr_i, va_i) in enumerate(splits):
        for rule, tv in by_rule.items():
            tr, va = tv.iloc[tr_i], tv.iloc[va_i]
            for imp in IMPUTERS:
                tr_prep, va_prep = prepare(imp, tr, tr), prepare(imp, tr, va)
                for fs_name, cols in feature_sets.items():
                    x_tr = pd.get_dummies(tr_prep[cols])
                    x_va = pd.get_dummies(va_prep[cols]).reindex(columns=x_tr.columns, fill_value=0)
                    for target in TARGET_COLUMNS:
                        model = make_model(model_name, seed).fit(x_tr, tr[target])
                        rows.append({"fold": fold, "rule": rule, "imputer": imp, "feature_set": fs_name,
                                     "target": target, **_score(va[target], model.predict(x_va))})
        print(f"  폴드 {fold + 1}/{len(splits)} 완료")

    folds = pd.DataFrame(rows)
    summary = (folds.groupby(["target", "feature_set", "rule", "imputer"])
               .agg(rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"), mae_mean=("mae", "mean"),
                    r2_mean=("r2", "mean"), n_folds=("fold", "count"))
               .reset_index().round(4))
    paired = paired_differences(folds)

    if save:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        folds.to_csv(out_dir / "imputation_grid_folds.csv", index=False, encoding="utf-8-sig")
        summary.to_csv(out_dir / "imputation_grid_summary.csv", index=False, encoding="utf-8-sig")
        paired.to_csv(out_dir / "imputation_grid_paired.csv", index=False, encoding="utf-8-sig")
        print(f"\n결과 저장: {out_dir}/imputation_grid_summary.csv, imputation_grid_paired.csv, imputation_grid_folds.csv")

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print("\n=== 조합별 RMSE (평균 ± 표준편차) ===")
        print(summary.to_string(index=False))
        print("\n=== 같은 폴드끼리 짝 비교 (diff = 앞 - 뒤 RMSE, 음수면 앞이 더 좋음) ===")
        print(paired.to_string(index=False))
    return folds, summary, paired


def paired_differences(folds: pd.DataFrame) -> pd.DataFrame:
    """(1) 같은 대치에서 mine - team (판정 효과), (2) 같은 판정에서 iterative - median_reg (대치 효과)."""

    key = ["fold", "target", "feature_set"]
    out = []

    def add(kind, label, a, b, group_cols):
        merged = a.merge(b, on=key + group_cols, suffixes=("_a", "_b"))
        merged["diff"] = merged["rmse_a"] - merged["rmse_b"]
        for gk, g in merged.groupby(["target", "feature_set"] + group_cols):
            d = g["diff"]
            out.append({"effect": kind, "comparison": label, "target": gk[0], "feature_set": gk[1],
                        "given": "/".join(str(x) for x in gk[2:]),
                        "diff_mean": round(float(d.mean()), 4), "diff_std": round(float(d.std()), 4),
                        "a_better_folds": int((d < 0).sum()), "n_folds": int(len(d))})

    f = folds[key + ["rule", "imputer", "rmse"]]
    for imp in IMPUTERS:
        s = f[f.imputer == imp]
        add("판정", "mine - team", s[s.rule == "mine"].drop(columns=["rule"]), s[s.rule == "team"].drop(columns=["rule"]),
            ["imputer"])
    for rule in RULES:
        s = f[f.rule == rule]
        add("대치", "iterative - median_reg", s[s.imputer == "iterative"].drop(columns=["imputer"]),
            s[s.imputer == "median_reg"].drop(columns=["imputer"]), ["rule"])
    return pd.DataFrame(out).sort_values(["effect", "target", "feature_set", "given"]).reset_index(drop=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["xgb", "rf"], default="xgb")
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    run_grid(load_by_rule(DEFAULT_INPUT), args.model, args.splits, args.repeats)
