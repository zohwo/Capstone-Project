import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """침출 공정 도메인 지식 기반 파생변수 5종을 추가한다.

    - acid_to_ore_ratio: 산-광석비 (침출 화학양론 지표)
    - leach_intensity: 침출강도 = 온도 x 시간 (반응속도론적 진행도 근사)
    - total_impurity_pct: 총불순물 (경쟁 반응 유발 요인)
    - target_purity_ratio: 목표금속(Ni+Co) 순도비
    - analysis_month: 계절성/설비 드리프트 탐색용
    """
    df = df.copy()

    df["acid_to_ore_ratio"] = df["황산투입량_L"] / df["원료투입량_kg"]
    df["leach_intensity"] = df["침출온도_C"] * df["침출시간_hr"]
    df["total_impurity_pct"] = (
        df["Cu_불순물_pct"] + df["Fe_불순물_pct"] + df["Al_불순물_pct"]
    )

    target_metals = df["Ni_함량_pct"] + df["Co_함량_pct"]
    df["target_purity_ratio"] = target_metals / (target_metals + df["total_impurity_pct"])

    df["analysis_month"] = df["분석일자"].dt.month

    return df
