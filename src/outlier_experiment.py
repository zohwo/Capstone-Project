"""이상치 기준 비교 실험 (기존 파일은 수정하지 않고 import만 한다).

실행: python -m src.outlier_experiment            (내 기준만)
      python -m src.outlier_experiment --with-team  (팀 기준도 함께 비교)

같은 팀 코드(정제 → 시간순 분할 → Train 기준 대치 → 파생변수)에서 이상치 기준만 바꿔 실행한다.
- team : 정상범위 밖 값 전부 NaN (팀 코드 replace_outliers_with_nan)
- mine : 명백한 이상치(센티널 등)는 NaN, 경미한 범위 초과는 '원료등급별 IQR 펜스 밖'일 때만 NaN
         (IQR 펜스는 Train 구간에서만 계산해 Validation/Test 정보가 섞이지 않게 함)
저장 파일 (data/processed/, data/ 는 git에 포함되지 않음)
- outlier_comparison.csv : 기준별·모델별 성능표
- outlier_cells.csv      : 이상치로 NaN 처리된 칸 목록 (배치ID, 컬럼, 원래 값, 정상범위, 처리 유형)
- cleaned_team.csv / cleaned_mine.csv : 기준별로 이상치 NaN → 날짜 보간 → Train 기준 대치 → 파생변수까지 마친 데이터
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import COMPOSITION_COLUMNS, DERIVED_FEATURES, NORMAL_RANGES, PROCESS_COLUMNS, TARGET_COLUMNS
from src.features import engineer_features
from src.imputation import impute_missing_from_reference, interpolate_analysis_date, replace_outliers_with_nan
from src.preprocessing import clean_schema, load_raw, resolve_duplicate_batches
from src.splitting import temporal_train_val_test_split
from src.validation import _prepare_split
from src.xgb_experiment import DEFAULT_INPUT, _features, _metrics, make_model

OUTPUT = Path("data/processed/outlier_comparison.csv")
NUM_COLS = list(NORMAL_RANGES.keys())
SENTINELS = [999, -999, 9999, 99999]
IQR_K = 1.5


def obvious_mask(df: pd.DataFrame) -> pd.DataFrame:
    """물리적으로 불가능하거나 입력 오류가 확실한 값(센티널, 음수, 상한 10배 이상 등)."""

    m = pd.DataFrame(False, index=df.index, columns=NUM_COLS)
    for c in NUM_COLS:
        s, (lo, hi) = df[c], NORMAL_RANGES[c]
        m[c] |= s.isin(SENTINELS) & ((s < lo) | (s > hi))  # 정상범위 안의 999는 실제 값일 수 있어 제외
        m[c] |= s >= hi * 10
        if c.endswith("_pct"):
            m[c] |= (s < 0) | (s > 100)
        if c == "침출온도_C":
            m[c] |= s < 0
        if c in ("침출시간_hr", "원료투입량_kg", "황산투입량_L"):
            m[c] |= s <= 0
    return m


def classify_mine(df: pd.DataFrame, train_index: pd.Index, k: float = IQR_K):
    """내 기준의 판정 마스크를 돌려준다: (명백한 이상치, 경미 초과 + IQR 밖).
    IQR 펜스는 Train 행에서만 계산한다 (명백한 이상치는 제외하고 계산)."""

    obv = obvious_mask(df)
    oor = pd.DataFrame(
        {c: (df[c] < NORMAL_RANGES[c][0]) | (df[c] > NORMAL_RANGES[c][1]) for c in NUM_COLS}
    )
    mild = oor & ~obv

    base_vals = df[NUM_COLS].mask(obv)
    train_vals, train_grade = base_vals.loc[train_index], df.loc[train_index, "원료등급"]

    mild_drop = pd.DataFrame(False, index=df.index, columns=NUM_COLS)
    for c in NUM_COLS:
        q1 = train_vals[c].groupby(train_grade).quantile(0.25)
        q3 = train_vals[c].groupby(train_grade).quantile(0.75)
        fence_lo = df["원료등급"].map(q1 - k * (q3 - q1)).astype(float)
        fence_hi = df["원료등급"].map(q3 + k * (q3 - q1)).astype(float)
        beyond = (df[c] < fence_lo) | (df[c] > fence_hi)
        mild_drop[c] = mild[c] & beyond
    return obv, mild_drop


def replace_outliers_mine(df: pd.DataFrame, train_index: pd.Index, k: float = IQR_K) -> pd.DataFrame:
    """내 기준: 명백한 이상치 + (경미한 범위 초과 & 원료등급별 IQR 펜스 밖)을 NaN으로 바꾼다."""

    out = df.copy()
    obv, mild_drop = classify_mine(out, train_index, k)
    to_nan = obv | mild_drop
    for c in NUM_COLS:
        out.loc[to_nan[c], c] = np.nan
    return out


def changed_cells(before: pd.DataFrame, after: pd.DataFrame, rule: str, kinds: dict, split: pd.Series) -> pd.DataFrame:
    """이상치 처리로 새로 NaN이 된 칸의 목록 (원래 빈 칸은 제외)."""

    rows = []
    for c in NUM_COLS:
        mask = before[c].notna() & after[c].isna()
        for i in before.index[mask]:
            lo, hi = NORMAL_RANGES[c]
            rows.append({
                "rule": rule, "배치ID": before.loc[i, "배치ID"], "원료등급": before.loc[i, "원료등급"],
                "split": split.get(i), "컬럼": c, "원래값": before.loc[i, c],
                "정상범위": f"{lo}~{hi}", "처리유형": kinds[c].get(i, "범위 밖"),
            })
    return pd.DataFrame(rows)


def build_cleaned(df_nan: pd.DataFrame, before: pd.DataFrame, train_i, val_i, test_i) -> pd.DataFrame:
    """이상치 NaN → 날짜 보간 → Train 기준 대치 → 파생변수 (팀 pipeline.run 과 같은 순서)."""

    d = interpolate_analysis_date(df_nan)
    d["split"] = "train"
    d.loc[val_i, "split"] = "validation"
    d.loc[test_i, "split"] = "test"
    for c in NUM_COLS:  # 이상치로 NaN이 된 칸 표시
        d[c + "_outlier"] = (before[c].notna() & df_nan[c].isna()).astype(int)
    imputed = impute_missing_from_reference(d.loc[train_i], d)
    return engineer_features(imputed)


def _evaluate(df, train_i, val_i, test_i, model_names, rule) -> list:
    train_df, val_df, test_df = df.loc[train_i], df.loc[val_i], df.loc[test_i]
    base = COMPOSITION_COLUMNS + PROCESS_COLUMNS
    feature_sets = {"base": base, "base_plus_derived": base + DERIVED_FEATURES}

    train_prep = _prepare_split(train_df, train_df)
    val_prep = _prepare_split(train_df, val_df)
    train_val_df = pd.concat([train_df, val_df])
    tv_prep = _prepare_split(train_val_df, train_val_df)
    test_prep = _prepare_split(train_val_df, test_df)

    rows = []
    for model_name in model_names:
        for target in TARGET_COLUMNS:
            val_scores = {}
            for case, cols in feature_sets.items():
                x_tr = _features(train_prep, cols)
                x_val = _features(val_prep, cols, x_tr.columns)
                model = make_model(model_name).fit(x_tr, train_df[target])
                val_scores[case] = _metrics(val_df[target], model.predict(x_val))

            best = min(val_scores, key=lambda c: val_scores[c]["rmse"])
            cols = feature_sets[best]
            x_tv = _features(tv_prep, cols)
            x_te = _features(test_prep, cols, x_tv.columns)
            model = make_model(model_name).fit(x_tv, train_val_df[target])
            test_scores = _metrics(test_df[target], model.predict(x_te))

            for case, m in val_scores.items():
                rows.append({"rule": rule, "model": model_name, "target": target, "feature_set": case,
                             "stage": "validation", "selected": case == best, **m})
            rows.append({"rule": rule, "model": model_name, "target": target, "feature_set": best,
                         "stage": "test", "selected": True, **test_scores})
    return rows


def save_csv(df: pd.DataFrame, path) -> Path:
    """CSV 저장. 엑셀 등에서 파일이 열려 있어 쓸 수 없으면 시각을 붙인 새 이름으로 저장한다."""

    from datetime import datetime

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_csv(path, index=False, encoding="utf-8-sig")
        return path
    except PermissionError:
        alt = path.with_name(f"{path.stem}_{datetime.now():%Y%m%d_%H%M%S}{path.suffix}")
        df.to_csv(alt, index=False, encoding="utf-8-sig")
        print(f"[알림] {path.name} 이(가) 열려 있어 저장할 수 없습니다 (엑셀 등을 닫으세요). 대신 {alt.name} 로 저장했습니다.")
        return alt


def run_experiment(input_path=DEFAULT_INPUT, output_path=OUTPUT, model_names=("rf", "xgb"), rules=("mine",)) -> pd.DataFrame:
    df0 = clean_schema(load_raw(input_path))
    df0 = resolve_duplicate_batches(df0)

    # 시간순 분할은 배치ID 순번만 쓰므로 이상치 처리 전에 인덱스를 정해도 팀 파이프라인과 같다.
    train_i, val_i, test_i = temporal_train_val_test_split(df0)
    print(f"분할: train {len(train_i)} / validation {len(val_i)} / test {len(test_i)}")

    split_label = pd.Series("train", index=df0.index)
    split_label.loc[val_i] = "validation"
    split_label.loc[test_i] = "test"

    obv, mild_drop = classify_mine(df0, train_i)
    kinds = {
        "mine": {c: {**{i: "명백한 이상치" for i in df0.index[obv[c]]},
                     **{i: "경미 초과 + IQR 밖" for i in df0.index[mild_drop[c]]}} for c in NUM_COLS},
        "team": {c: {} for c in NUM_COLS},
    }
    makers = {
        "mine": lambda: replace_outliers_mine(df0, train_i),
        "team": lambda: replace_outliers_with_nan(df0),
    }
    variants = {r: makers[r]() for r in rules}
    out_dir = Path(output_path).parent

    cells = pd.concat(
        [changed_cells(df0, variants[r], r, kinds[r], split_label) for r in rules], ignore_index=True
    )
    save_csv(cells, out_dir / "outlier_cells.csv")
    for rule, d in variants.items():
        save_csv(build_cleaned(d, df0, train_i, val_i, test_i), out_dir / f"cleaned_{rule}.csv")

    print("\n이상치로 새로 NaN이 된 칸 수 (원래 빈 칸 제외)")
    print(cells.groupby(["rule", "처리유형"]).size().to_string())

    print("\n컬럼별 NaN 개수 (원래 결측 포함)")
    counts = pd.DataFrame({"raw": df0[NUM_COLS].isna().sum()})
    for r in rules:
        counts[r] = variants[r][NUM_COLS].isna().sum()
    print(counts.to_string())
    print("합계  " + "  ".join(f"{c} {counts[c].sum()}" for c in counts.columns))

    rows = []
    for rule, d in variants.items():
        d = interpolate_analysis_date(d)
        rows += _evaluate(d, train_i, val_i, test_i, model_names, rule)

    result = pd.DataFrame(rows)
    show = result[(result.stage == "test")][["rule", "model", "target", "feature_set", "r2", "rmse", "mae"]]
    print("\n[Test] 성능 (Validation RMSE로 선택한 조합)")
    print(show.round(4).to_string(index=False))
    val = result[(result.stage == "validation") & (result.feature_set == "base_plus_derived")]
    print("\n[Validation] base_plus_derived")
    print(val[["rule", "model", "target", "r2", "rmse", "mae"]].round(4).to_string(index=False))

    saved = save_csv(result, output_path)
    print(f"\n저장 위치: {out_dir}")
    print(f"  {saved.name}, outlier_cells.csv, " + ", ".join(f"cleaned_{r}.csv" for r in rules))
    return result


if __name__ == "__main__":
    import sys

    run_experiment(rules=("mine", "team") if "--with-team" in sys.argv else ("mine",))
