"""
Module 1의 데이터사전 범위, 프로젝트 보존 예외 및 입력 컬럼 설정
"""

NORMAL_RANGES = {
    "Ni_함량_pct": (8, 30),
    "Co_함량_pct": (2, 15),
    "Li_함량_pct": (2, 6),
    "Mn_함량_pct": (5, 16),
    "Cu_불순물_pct": (0.3, 6),
    "Fe_불순물_pct": (0.2, 6),
    "Al_불순물_pct": (0.5, 7),
    "원료투입량_kg": (80, 200),
    "황산투입량_L": (400, 1500),
    "침출온도_C": (40, 95),
    "침출시간_hr": (1, 8),
}

REFERENCE_EXCEPTION_RANGES = {
    "Ni_함량_pct": ((5.9, 8), (30, 38.2)),
    "Co_함량_pct": (),
    "Li_함량_pct": ((1.81, 2), (6, 6.76)),
    "Mn_함량_pct": (),
    "Cu_불순물_pct": ((0.05, 0.3), (6, 6.08)),
    "Fe_불순물_pct": ((0.02, 0.2),),
    "Al_불순물_pct": ((0.13, 0.5), (7, 7.46)),
}

COMPOSITION_COLUMNS = [
    "Ni_함량_pct",
    "Co_함량_pct",
    "Li_함량_pct",
    "Mn_함량_pct",
    "Cu_불순물_pct",
    "Fe_불순물_pct",
    "Al_불순물_pct",
]

PROCESS_COLUMNS = ["침출온도_C", "침출시간_hr"]

AMOUNT_COLUMNS = ["원료투입량_kg", "황산투입량_L"]
BASE_FEATURES = COMPOSITION_COLUMNS + PROCESS_COLUMNS + AMOUNT_COLUMNS

TARGET_COLUMNS = ["Ni_회수율_pct", "Co_회수율_pct"]

DERIVED_FEATURES = [
    "acid_to_ore_ratio",
    "leach_intensity",
    "total_impurity_pct",
    "feed_impurity_g_per_l",
]

RAW_SHEET_NAME = "원료분석_원본데이터"
