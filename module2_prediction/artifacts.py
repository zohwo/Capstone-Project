"""학습 모델과 고정 개발 데이터 대치 기준 함께 저장하고 재사용"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from module1_preprocessing.transformation import RAW_INPUT_COLUMNS, prepare_split
from module1_preprocessing.config import BASE_FEATURES, COMPOSITION_COLUMNS, NORMAL_RANGES
from module2_prediction.modeling import validate_selected_features


def save_bundle(path, model, reference, features, target):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = dict(schema_version=1, model=model, target=target, features=list(features),
                  reference=reference[RAW_INPUT_COLUMNS].copy(), primary_metric='rmse')
    joblib.dump(bundle, path)
    return bundle


def load_bundle(path):
    """프로젝트에서 생성한 신뢰할 수 있는 모델 파일 로드"""
    bundle = joblib.load(path)
    if bundle.get('schema_version') != 1:
        raise ValueError('지원하지 않는 모델 저장 버전임')
    return bundle


def _validate_prediction_inputs(frame, *, allow_missing):
    """조성, 운전값 범위 위반 검사, 무한대 값 검사 — 대치 전 거부"""
    missing = set(BASE_FEATURES) - set(frame.columns)
    if missing:
        raise ValueError(f"예측 입력 컬럼이 없습니다: {sorted(missing)}")
    out = frame.copy()
    for col in BASE_FEATURES:
        values = pd.to_numeric(out[col], errors="coerce")
        low, high = (0, 100) if col in COMPOSITION_COLUMNS else NORMAL_RANGES[col]
        invalid = out[col].notna() & (values.isna() | ~np.isfinite(values) | ~values.between(low, high))

        if not allow_missing or col == "원료투입량_kg":
            invalid |= values.isna()
        if invalid.any():
            rows = out.index[invalid.fillna(False)].tolist()
            raise ValueError(f"{col}: {low}~{high} 범위의 유한한 숫자가 필요합니다. 행: {rows}")
        out[col] = values
    return out


def predict_bundle(bundle, frame):
    """고정 학습 참조로 대치하고 파생변수를 다시 계산하여 예측

    - 원료등급 필요 여부는 행 단위로 판단 — 조성 결측 있는 행, 등급이 모델 입력 피처인 경우 전체 행
    - 컬럼 자체 결측과 값은 있지만 공백, 결측인 행을 모두 검사
    """
    frame = _validate_prediction_inputs(frame, allow_missing=True)
    if '분석일자' not in frame:
        frame['분석일자'] = pd.NaT

    grade_required = frame[COMPOSITION_COLUMNS].isna().any(axis=1)
    if "원료등급" in bundle["features"]:
        grade_required[:] = True

    if "원료등급" not in frame:
        frame["원료등급"] = pd.NA

    grade = frame["원료등급"].astype("string").str.strip()
    invalid_grade = grade_required & ~grade.isin(["A", "B", "C"])

    if invalid_grade.any():
        rows = frame.index[invalid_grade].tolist()
        raise ValueError(f"조성 대치 또는 모델 입력에 유효한 원료등급(A/B/C)이 필요합니다. 행: {rows}")

    frame["원료등급"] = grade

    prepared = prepare_split(bundle['reference'], frame)
    prepared = _validate_prediction_inputs(prepared, allow_missing=False)

    composition_sum = prepared[COMPOSITION_COLUMNS].sum(axis=1)
    over_100 = composition_sum > 100 + 1e-8
    if over_100.any():
        rows = prepared.index[over_100].tolist()
        raise ValueError(f"동일 질량 기준의 성분 합계가 100%를 초과합니다. 행: {rows}")

    validate_selected_features(prepared, bundle['features'])
    return bundle['model'].predict(prepared[bundle['features']])
