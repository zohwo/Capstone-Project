from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell(
    "# 모듈 1: 블랙파우더 성분 데이터 전처리 - EDA 및 검증\n\n"
    "`src/pipeline.py`의 각 단계를 순서대로 호출해 원본 데이터를 정제하고, "
    "전/후 비교 시각화와 파생변수·원료등급 기여도(ablation test)를 확인한다."
))

cells.append(nbf.v4.new_code_cell(
    "import sys\n"
    "sys.path.append('..')\n\n"
    "import pandas as pd\n"
    "from pathlib import Path\n\n"
    "from src.preprocessing import load_raw, clean_schema\n"
    "from src.imputation import flag_outliers, impute_missing\n"
    "from src.features import engineer_features\n"
    "from src.validation import (\n"
    "    check_ranges,\n"
    "    plot_missing_before_after,\n"
    "    plot_distributions,\n"
    "    plot_correlation_heatmap,\n"
    "    run_ablation_test,\n"
    ")\n"
    "from src.config import COMPOSITION_COLUMNS, PROCESS_COLUMNS, TARGET_COLUMNS\n\n"
    "RAW_PATH = Path('../data/raw/영풍그룹_금속회수율최적화_RawMaterialData_학생배포용.xlsx')\n"
    "FIGURES_DIR = Path('../reports/figures')"
))

cells.append(nbf.v4.new_markdown_cell("## 1. 원본 로드 및 정제"))

cells.append(nbf.v4.new_code_cell(
    "raw_df = load_raw(RAW_PATH)\n"
    "clean_df = clean_schema(raw_df)\n"
    "outlier_flagged_df = flag_outliers(clean_df)\n"
    "imputed_df = impute_missing(outlier_flagged_df)\n"
    "final_df = engineer_features(imputed_df)\n"
    "final_df.head()"
))

cells.append(nbf.v4.new_markdown_cell("## 2. 결측치 처리 전/후 비교"))

cells.append(nbf.v4.new_code_cell(
    "plot_missing_before_after(outlier_flagged_df, imputed_df, FIGURES_DIR)"
))

cells.append(nbf.v4.new_markdown_cell("## 3. 이상치 제거 전/후 분포"))

cells.append(nbf.v4.new_code_cell(
    "plot_distributions(raw_df, ['Ni_함량_pct', '황산투입량_L', '침출온도_C'], FIGURES_DIR / 'before')\n"
    "plot_distributions(final_df, ['Ni_함량_pct', '황산투입량_L', '침출온도_C'], FIGURES_DIR / 'after')"
))

cells.append(nbf.v4.new_markdown_cell("## 4. 상관관계 히트맵"))

cells.append(nbf.v4.new_code_cell(
    "feature_columns = COMPOSITION_COLUMNS + PROCESS_COLUMNS + [\n"
    "    'acid_to_ore_ratio', 'leach_intensity', 'total_impurity_pct', 'target_purity_ratio'\n"
    "]\n"
    "plot_correlation_heatmap(final_df, feature_columns, TARGET_COLUMNS, FIGURES_DIR)"
))

cells.append(nbf.v4.new_markdown_cell("## 5. 정합성 검증"))

cells.append(nbf.v4.new_code_cell(
    "violations = check_ranges(final_df)\n"
    "print('정상범위 위반:', violations if violations else '없음')"
))

cells.append(nbf.v4.new_markdown_cell("## 6. 파생변수 및 원료등급 기여도 검증 (Ablation Test)"))

cells.append(nbf.v4.new_code_cell(
    "base_features = COMPOSITION_COLUMNS + PROCESS_COLUMNS\n"
    "derived_features = ['acid_to_ore_ratio', 'leach_intensity', 'total_impurity_pct', 'target_purity_ratio']\n\n"
    "feature_sets = {\n"
    "    'base_only': base_features,\n"
    "    'base_plus_grade': base_features + ['원료등급'],\n"
    "    'base_plus_derived': base_features + derived_features,\n"
    "    'all_features': base_features + ['원료등급'] + derived_features,\n"
    "}\n\n"
    "results = run_ablation_test(final_df, feature_sets, TARGET_COLUMNS)\n"
    "pd.DataFrame(results).T"
))

cells.append(nbf.v4.new_markdown_cell(
    "## 7. 심층 인터뷰 답변\n\n"
    "**1) 실제 공정 데이터 품질 문제를 어떻게 식별·처리했는가?**\n\n"
    "데이터사전에 정의된 정상 범위를 벗어나는 값(`Ni_함량_pct=999.0`, `침출온도_C=-5.0`, "
    "`황산투입량_L=15000`)을 물리적으로 불가능한 이상치로 식별해 `flag_outliers`에서 NaN으로 "
    "치환했다. 이후 남은 랜덤 결측(조성 6~9%, 공정조건)은 변수 성격에 따라 원료등급별 중앙값, "
    "회귀 기반, 배치ID 순서 기반 보간으로 구분 처리했다 (설계 문서 §4).\n\n"
    "**2) 특성공학 과정에서 도메인 지식을 어떻게 활용했는가?**\n\n"
    "산-광석비, 침출강도(온도×시간), 총불순물, 목표금속 순도비는 모두 습식제련 침출 공정의 "
    "화학양론·반응속도론적 지식에서 도출했다 (설계 문서 §6). 위 ablation test로 파생변수·원료등급 "
    "포함 시 R²/RMSE가 실제로 개선되는지 정량 확인했다.\n\n"
    "**3) 데이터 불균형 문제가 있었다면 어떻게 해결했는가?**\n\n"
    "`원료등급`(A/B/C) 분포를 아래 셀에서 확인했다."
))

cells.append(nbf.v4.new_code_cell(
    "final_df['원료등급'].value_counts(normalize=True) * 100"
))

nb["cells"] = cells

Path("notebooks").mkdir(exist_ok=True)
with open("notebooks/01_eda_and_validation.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("노트북 생성 완료: notebooks/01_eda_and_validation.ipynb")
