"""원료분석 데이터의 정상 범위 및 컬럼 그룹 정의 (데이터사전 기준, 단일 소스)."""

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

TARGET_COLUMNS = ["Ni_회수율_pct", "Co_회수율_pct"]

RAW_SHEET_NAME = "원료분석_원본데이터"
