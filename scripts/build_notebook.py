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
    "    'acid_to_ore_ratio', 'leach_intensity', 'total_impurity_pct', 'target_purity_ratio',\n"
    "    'Fe_Ni_ratio', 'Cu_Ni_ratio', 'Al_Ni_ratio'\n"
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
    "# analysis_month는 회수율 예측 후보 변수가 아니라 계절성/설비 드리프트 모니터링용 캘린더 "
    "파생변수이므로, 특성 기여도 비교(히트맵·ablation test) 대상에서 제외한다.\n"
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
    "### 6.1 개별 불순물 비율(Fe/Ni, Cu/Ni, Al/Ni) 추가 검증\n\n"
    "`total_impurity_pct`는 Cu/Fe/Al 불순물을 동일 가중치로 합산해, 어느 불순물이 "
    "실제로 회수율 저하를 유발하는지 정보를 가린다. 개별 비율로 분리했을 때 "
    "`base_plus_derived` 대비 실제로 성능이 더 좋아지는지 검증한다."
))

cells.append(nbf.v4.new_code_cell(
    "impurity_ratio_features = ['Fe_Ni_ratio', 'Cu_Ni_ratio', 'Al_Ni_ratio']\n\n"
    "ratio_feature_sets = {\n"
    "    'base_plus_derived': base_features + derived_features,\n"
    "    'base_plus_derived_plus_ratios': base_features + derived_features + impurity_ratio_features,\n"
    "}\n\n"
    "ratio_results = run_ablation_test(final_df, ratio_feature_sets, TARGET_COLUMNS)\n"
    "pd.DataFrame(ratio_results).T"
))

cells.append(nbf.v4.new_markdown_cell(
    "**결론**: 실측 결과 `base_plus_derived`(Ni R²=0.8107±0.0307, Co R²=0.7925±0.0220) 대비 "
    "`base_plus_derived_plus_ratios`(Ni R²=0.8098±0.0329, Co R²=0.7874±0.0207)는 개선되지 "
    "않고 오히려 소폭 하락했다. 차이(Ni -0.0009, Co -0.0051)는 fold별 표준편차(r2_std, "
    "약 0.02~0.03)보다 훨씬 작아 **통계적으로 노이즈 수준과 구분이 안 되는 차이**다 — "
    "즉 \"떨어졌다\"고 말하기도 어려울 만큼, 교차검증 fold를 어떻게 나누느냐에 따라 흔들릴 "
    "수 있는 범위 안에 있다. `base_features`에 이미 `Cu/Fe/Al_불순물_pct`와 `Ni_함량_pct` "
    "원본 컬럼, 그리고 `total_impurity_pct` 파생변수가 포함되어 있어, Fe/Ni·Cu/Ni·Al/Ni "
    "비율은 모델이 이미 접근 가능한 정보의 재조합(다중공선성)에 가까워 새로운 신호를 "
    "추가하지 못한 것으로 보인다. 따라서 이 세 비율 변수는 `engineer_features`에는 "
    "유지하되(해석용 참고 지표로서의 가치는 있을 수 있음), 최종 모델 입력 특성 세트에는 "
    "포함하지 않기로 결론짓는다 — \"도메인상 그럴듯해 보인다\"는 이유만으로 파생변수를 "
    "채택하면 안 되고, ablation test로 (평균뿐 아니라 fold 간 변동폭까지 고려해) 실측 "
    "검증해야 한다는 것을 보여주는 사례다."
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
    "전체 424행 기준 `원료등급`(A/B/C) 분포는 B 38.9%, A 35.1%, C 25.9%로(아래 셀 실행 결과), "
    "가장 적은 클래스(C)도 약 26%를 차지해 특정 클래스가 극단적으로 희소한 심각한 불균형은 "
    "아니라고 판단했다. 설계 문서 §9에서는 구현 단계의 실제 분포를 보고 불균형 여부와 "
    "계층적 샘플링(stratified split) 적용 여부를 결정하기로 했는데, 실측 결과 클래스 간 "
    "비율 차이가 26~39% 수준으로 완만해 SMOTE·계층적 샘플링·클래스 가중치 등 별도의 "
    "불균형 보정은 적용하지 않기로 결론지었다. 다만 위 ablation test 결과를 보면 `base_only`(Ni "
    "R²=0.580±0.041, Co R²=0.601±0.042) 대비 `base_plus_grade`(Ni R²=0.581±0.041, Co "
    "R²=0.602±0.040)는 R²가 약 0.001 수준만 상승했는데, 이는 fold별 표준편차(r2_std≈0.04)보다 "
    "훨씬 작아 통계적으로 유의미한 개선이라고 볼 수 없다 — `원료등급`을 모델 입력 특성으로 그대로 "
    "추가하는 것 자체는 회귀 성능에 영향을 주지 않는다. 그럼에도 `원료등급`은 Task 5의 등급별 "
    "중앙값 보간(grade-based median "
    "imputation)에서 그룹핑 키로 쓰이므로 컬럼 자체는 계속 유지한다. 반면 산-광석비·침출강도·"
    "총불순물·목표금속 순도비 등 파생변수는 `base_plus_derived`에서 R²를 Ni 0.580→0.811, "
    "Co 0.601→0.793 수준으로 크게 끌어올려, 현재 데이터에서 회귀 성능에 실질적으로 기여하는 "
    "것은 `원료등급`이 아니라 파생변수 쪽임을 ablation test가 보여준다."
))

cells.append(nbf.v4.new_code_cell(
    "final_df['원료등급'].value_counts(normalize=True) * 100"
))

nb["cells"] = cells

Path("notebooks").mkdir(exist_ok=True)
with open("notebooks/01_eda_and_validation.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("노트북 생성 완료: notebooks/01_eda_and_validation.ipynb")
