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
