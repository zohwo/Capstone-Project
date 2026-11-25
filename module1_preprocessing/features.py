import numpy as np
import pandas as pd

def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """양수 분모에 대해서만 나눗셈 수행"""
    safe_denominator = denominator.where(denominator > 0, np.nan)
    return numerator / safe_denominator


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """전처리된 배치 데이터에 모델 후보 파생변수 6종 및 탐색용 지표 추가"""
    df = df.copy()

    # 원료 1 kg당 투입한 황산 용액 부피(L/kg)
    df["acid_to_ore_ratio"] = _safe_divide(
        df["황산투입량_L"], df["원료투입량_kg"]
    )

    # 침출온도(°C)와 침출시간(h)의 경험적 상호작용항
    df["leach_intensity"] = df["침출온도_C"] * df["침출시간_hr"]

    # 측정한 Cu·Fe·Al의 합계(wt%), 원료 조성을 요약하는 지표
    df["total_impurity_pct"] = (
        df["Cu_불순물_pct"] + df["Fe_불순물_pct"] + df["Al_불순물_pct"]
    )

    # Ni·Co·Cu·Fe·Al 합계 중 Ni+Co의 질량 비중(0~1), Li와 Mn은 계산에서 제외
    target_metals = df["Ni_함량_pct"] + df["Co_함량_pct"]
    df["target_purity_ratio"] = _safe_divide(
        target_metals, target_metals + df["total_impurity_pct"]
    )

    # 분석일자의 월(1~12), 추천 시점의 가용성이 확인되기 전까지 탐색용으로 사용
    df["analysis_month"] = df["분석일자"].dt.month

    # 개별 불순물/Ni 질량비(무차원), Ni 함량이 낮아져도 비율이 커질 수 있는 실험 후보
    df["Fe_Ni_ratio"] = _safe_divide(df["Fe_불순물_pct"], df["Ni_함량_pct"])
    df["Cu_Ni_ratio"] = _safe_divide(df["Cu_불순물_pct"], df["Ni_함량_pct"])
    df["Al_Ni_ratio"] = _safe_divide(df["Al_불순물_pct"], df["Ni_함량_pct"])

    # 투입 황산 용액 1 L당 원료에 포함된 Cu·Fe·Al 질량(g/L)
    # ×10은 wt%를 g/kg으로 환산, 분자는 용해 전 원료 기준
    df["feed_impurity_g_per_l"] = _safe_divide(
        10 * df["total_impurity_pct"], df["acid_to_ore_ratio"]
    )
    target_mass_kg = df["원료투입량_kg"] * target_metals / 100
    df["acid_per_target_L_kg"] = _safe_divide(df["황산투입량_L"], target_mass_kg)
    df["acid_time"] = df["acid_to_ore_ratio"] * df["침출시간_hr"]

    return df
