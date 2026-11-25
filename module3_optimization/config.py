"""목표 회수율과 프로젝트 탐색 조건 정의함"""
from module1_preprocessing.config import NORMAL_RANGES

CONTROL_COLUMNS = ['황산투입량_L', '침출온도_C', '침출시간_hr']
BOUNDS = [NORMAL_RANGES[c] for c in CONTROL_COLUMNS]
TARGETS = {'Ni_회수율_pct': 95.0, 'Co_회수율_pct': 90.0}
STEPS = [1.0, 0.1, 0.1]
WEIGHTS = [1.0, 1.0, 1.0]
SEEDS = [7, 42, 83]
