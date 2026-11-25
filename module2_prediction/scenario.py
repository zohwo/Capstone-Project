import numpy as np
import pandas as pd

from module2_prediction.config import (
    COMPOSITION_COLUMNS,
    FEATURE_COLUMNS,
    PROCESS_COLUMN_RANGES,
    PROCESS_COLUMNS,
)


def partial_dependence(
    model, base_row: pd.Series, feature: str, values, feature_columns: list = FEATURE_COLUMNS
) -> pd.DataFrame:
    """base_row를 고정한 채 feature 값만 바꿔가며 예측값이 어떻게 변하는지 확인

    반환값: columns=[feature, "prediction"]인 DataFrame, len(values)행
    """
    values = list(values)

    rows = []
    for value in values:
        row = base_row.copy()
        row[feature] = value
        rows.append(row[feature_columns])

    X = pd.DataFrame(rows)
    predictions = model.predict(X)

    return pd.DataFrame({feature: values, "prediction": predictions})


def predict_process_scenario(
    model, base_row: pd.Series, feature: str, values, feature_columns: list = FEATURE_COLUMNS
) -> pd.DataFrame:
    """공정조건(PROCESS_COLUMNS) 또는 황산투입량_L 값을 바꿔가며 예측값이 어떻게 변하는지 확인

    - partial_dependence와 달리 feature 변경 후 engineer_features로 파생변수 전체 재계산
    - 반환값: columns=[feature, "prediction"]인 DataFrame, len(values)행
    """
    from module1_preprocessing.config import NORMAL_RANGES
    from module1_preprocessing.features import engineer_features
    allowed = PROCESS_COLUMNS + ["황산투입량_L"]
    if feature not in allowed:
        raise ValueError(f"feature는 {allowed} 중 하나여야 함")

    for col in PROCESS_COLUMNS:
        low, high = PROCESS_COLUMN_RANGES[col]
        base_value = base_row[col]
        if not np.isfinite(base_value) or not (low <= base_value <= high):
            raise ValueError(
                f"base_row의 '{col}' 값이 허용 범위({low}~{high}) 밖입니다: {base_value}"
            )

    low, high = NORMAL_RANGES[feature]
    values = list(values)
    invalid = [v for v in values if not np.isfinite(v) or not (low <= v <= high)]
    if invalid:
        raise ValueError(
            f"'{feature}' 값은 {low}~{high} 범위의 유한한 값이어야 합니다 — 허용 범위 밖: {invalid}"
        )

    rows = []
    for value in values:
        row = base_row.copy()
        row[feature] = value
        row["분석일자"] = pd.NaT
        frame = pd.DataFrame([row])
        frame["분석일자"] = pd.to_datetime(frame["분석일자"])
        rows.append(engineer_features(frame).iloc[0][feature_columns])

    X = pd.DataFrame(rows)
    predictions = model.predict(X)

    return pd.DataFrame({feature: list(values), "prediction": predictions})


def build_blend_features(
    composition_a: dict | pd.Series,
    composition_b: dict | pd.Series,
    ratio_a: float,
    *,
    total_ore_kg: float,
    acid_l: float,
    temperature_c: float,
    time_hr: float,
) -> pd.DataFrame:
    """동일 질량 기준의 실제 분석값(wt%) 혼합, 이번 배치 조건 적용

    - ratio_a: A의 질량분율
    - 등급, 과거 운전조건은 입력 파라미터에 없음
    """
    from module1_preprocessing.config import NORMAL_RANGES
    from module1_preprocessing.features import engineer_features

    if not np.isfinite(ratio_a) or not 0 <= ratio_a <= 1:
        raise ValueError("ratio_a는 유한한 0~1 질량분율이어야 합니다.")
    compositions = []
    for name, source in (("A", composition_a), ("C", composition_b)):
        missing = set(COMPOSITION_COLUMNS) - set(source.keys())
        if missing:
            raise ValueError(f"{name} 원료 분석값이 누락되었습니다: {sorted(missing)}")
        try:
            values = pd.Series({c: float(source[c]) for c in COMPOSITION_COLUMNS})
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} 원료 분석값은 숫자여야 합니다.") from exc
        if not np.isfinite(values).all() or not values.between(0, 100).all() or values.sum() > 100 + 1e-8:
            raise ValueError(f"{name} 원료 성분은 유한한 0~100 wt%이며 합계는 100 이하여야 합니다.")
        compositions.append(values)
    batch = {
        "원료투입량_kg": total_ore_kg, "황산투입량_L": acid_l,
        "침출온도_C": temperature_c, "침출시간_hr": time_hr,
    }
    for col, value in batch.items():
        low, high = PROCESS_COLUMN_RANGES.get(col, NORMAL_RANGES[col])
        if not np.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{col}은 {low}~{high} 범위의 유한한 값이어야 합니다.")
    if total_ore_kg <= 0 or acid_l <= 0:
        raise ValueError("원료투입량_kg과 황산투입량_L은 0보다 커야 합니다.")
    blend = (compositions[0] * ratio_a + compositions[1] * (1 - ratio_a)).to_dict()
    blend.update(batch)
    blend["분석일자"] = pd.NaT  # 날짜: 모델 피처 제외
    return engineer_features(pd.DataFrame([blend]))


def predict_blend_scenario(
    model,
    composition_a: dict | pd.Series,
    composition_b: dict | pd.Series,
    ratio_a: float,
    *,
    total_ore_kg: float,
    acid_l: float,
    temperature_c: float,
    time_hr: float,
    feature_columns: list = FEATURE_COLUMNS,
    blend_grade: str | None = None,
) -> float:
    """composition_a, composition_b를 ratio_a 비율로 혼합한 가상 배치의 회수율 예측

    원료등급이 feature_columns에 포함되면 blend_grade 필수(결측 시 ValueError)
    """
    from module2_prediction.modeling import validate_selected_features

    frame = build_blend_features(
        composition_a, composition_b, ratio_a, total_ore_kg=total_ore_kg,
        acid_l=acid_l, temperature_c=temperature_c, time_hr=time_hr,
    )
    if "원료등급" in feature_columns:
        if blend_grade is None:
            raise ValueError("원료등급을 사용하는 모델에는 혼합 배치의 blend_grade 입력이 필요함")
        frame["원료등급"] = blend_grade
    validate_selected_features(frame, feature_columns)
    return float(model.predict(frame[feature_columns])[0])
