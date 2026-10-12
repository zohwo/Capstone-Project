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
    "    plot_ablation_results,\n"
    "    plot_grade_distribution,\n"
    ")\n"
    "from src.config import COMPOSITION_COLUMNS, PROCESS_COLUMNS, TARGET_COLUMNS\n\n"
    "RAW_PATH = Path('../data/raw/영풍그룹_금속회수율최적화_RawMaterialData_학생배포용.xlsx')\n"
    "FIGURES_DIR = Path('../reports/figures')"
))

cells.append(nbf.v4.new_markdown_cell("## 1. 원본 로드 및 정제"))

cells.append(nbf.v4.new_code_cell(
    "raw_df = load_raw(RAW_PATH)\n"
    "clean_df = clean_schema(raw_df)\n"
    "clean_df.shape"
))

cells.append(nbf.v4.new_markdown_cell(
    "### 1.1 완전 중복 배치 제거\n\n"
    "`배치ID` 기준 중복 4건(8행)은 분석일자·원료등급·회수율까지 완전히 동일한 복사본으로 "
    "확인됐다 — 서로 다른 두 배치에 ID가 우연히 겹친 게 아니라 같은 레코드가 두 번 입력된 "
    "것이다. `clean_schema`가 남긴 `is_duplicate_batch` 플래그를 근거로 첫 occurrence만 "
    "남기고 제거한다 — 그대로 두면 교차검증에서 동일 데이터 포인트가 train/test 양쪽에 걸쳐 "
    "들어가 성능 지표가 부풀려질 수 있다. 남은 행의 `is_duplicate_batch=True`는 \"원래 중복이 "
    "있었고 하나는 제거됐다\"는 감사 기록으로 유지된다 (`src/pipeline.py`의 `run()`과 동일한 "
    "순서)."
))

cells.append(nbf.v4.new_code_cell(
    "clean_df = clean_df.drop_duplicates(subset='배치ID', keep='first')\n"
    "print(f'중복 제거 후: {len(clean_df)}행')"
))

cells.append(nbf.v4.new_code_cell(
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

cells.append(nbf.v4.new_markdown_cell(
    "조성 변수 중 목표금속(Ni_함량_pct, Co_함량_pct, Li_함량_pct)은 회수율과 양의 "
    "상관(+0.30~+0.49), 불순물(Cu/Fe/Al_불순물_pct)과 Mn_함량_pct는 음의 상관(-0.26~-0.57)을 "
    "보여 — 원료 안에 목표금속이 많을수록, 불순물이 많을수록 회수율에 미치는 방향이 §6 "
    "특성공학에서 가정한 도메인 지식(경쟁 반응 가설)과 일치한다. 공정조건(침출온도_C, "
    "침출시간_hr)도 양의 상관(+0.27~+0.48)을 보여 반응속도론적 기대와 부합한다.\n\n"
    "특히 파생변수(acid_to_ore_ratio, leach_intensity, total_impurity_pct, "
    "target_purity_ratio)는 대체로 그 구성요소인 원본 변수 개별 상관보다 더 강하다 — 예: "
    "leach_intensity(Ni 0.51)는 구성요소인 침출온도_C(0.48)·침출시간_hr(0.36) 개별 상관보다 "
    "높고, total_impurity_pct(Co -0.60)도 Cu/Fe/Al_불순물_pct 개별 상관(-0.53~-0.57)보다 더 "
    "강하다. 이는 §6 ablation test에서 파생변수가 R²를 크게 끌어올린 결과(Ni 0.55→0.81)와 "
    "방향이 일치한다.\n\n"
    "반면 Fe_Ni_ratio·Cu_Ni_ratio·Al_Ni_ratio도 -0.41~-0.54 수준의 꽤 강한 단순 상관을 "
    "보이는데, §6.1 ablation test에서는 모델 R²에 전혀 기여하지 못했다(§6.1 결론 참고) — "
    "변수 하나와 타겟 사이의 단순 상관과, 다른 변수들과 함께 썼을 때의 모델 기여도는 "
    "서로 다른 질문이라는 걸 보여주는 사례다."
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
    "results = run_ablation_test(outlier_flagged_df, feature_sets, TARGET_COLUMNS)\n"
    "pd.DataFrame(results).T"
))

cells.append(nbf.v4.new_code_cell(
    "plot_ablation_results(results, FIGURES_DIR)"
))

cells.append(nbf.v4.new_markdown_cell(
    "### 6.1 개별 불순물 비율(Fe/Ni, Cu/Ni, Al/Ni) 추가 검증\n\n"
    "`total_impurity_pct`는 Cu/Fe/Al 불순물을 동일 가중치로 합산해, 어느 불순물이 "
    "실제로 회수율 저하를 유발하는지 정보를 가린다. 위 §6 ablation과 동일한 4-케이스 구조로, "
    "비율 변수가 (a) 단독으로도 신호가 있는지, (b) 이미 검증된 최선 조합(`base_plus_derived`) "
    "위에 추가로 기여하는지를 함께 검증한다."
))

cells.append(nbf.v4.new_code_cell(
    "impurity_ratio_features = ['Fe_Ni_ratio', 'Cu_Ni_ratio', 'Al_Ni_ratio']\n\n"
    "ratio_feature_sets = {\n"
    "    'base_only': base_features,\n"
    "    'base_plus_ratios': base_features + impurity_ratio_features,\n"
    "    'base_plus_derived': base_features + derived_features,\n"
    "    'base_plus_derived_plus_ratios': base_features + derived_features + impurity_ratio_features,\n"
    "}\n\n"
    "ratio_results = run_ablation_test(outlier_flagged_df, ratio_feature_sets, TARGET_COLUMNS)\n"
    "pd.DataFrame(ratio_results).T"
))

cells.append(nbf.v4.new_code_cell(
    "plot_ablation_results(ratio_results, FIGURES_DIR / 'ratios')"
))

cells.append(nbf.v4.new_markdown_cell(
    "**결론**: 4개 케이스 실측 결과는 다음과 같다.\n\n"
    "- `base_only` → `base_plus_ratios`: Ni R² 0.5504±0.0474 → 0.5439±0.0454 (차이 -0.0065), "
    "Co R² 0.5886±0.0550 → 0.5810±0.0572 (차이 -0.0076) — **비율 변수는 단독으로도 "
    "아무 신호가 없다.**\n"
    "- `base_plus_derived` → `base_plus_derived_plus_ratios`: Ni R² 0.8080±0.0217 → "
    "0.8006±0.0219 (차이 -0.0074), Co R² 0.7935±0.0143 → 0.7903±0.0169 (차이 -0.0032) — "
    "이미 검증된 최선 조합 위에 추가해도 역시 변화 없음.\n\n"
    "두 비교 모두 차이가 fold별 표준편차(r2_std, 약 0.01~0.06)보다 훨씬 작아 "
    "**통계적으로 노이즈 수준과 구분이 안 되는 차이**다. 상관관계 히트맵에서는 "
    "Fe_Ni_ratio·Cu_Ni_ratio·Al_Ni_ratio가 타겟과 -0.4~-0.5 수준의(결코 약하지 않은) "
    "단순 상관을 보였는데도 모델 R²에는 전혀 기여하지 못했다는 점이 흥미롭다 — "
    "RandomForest는 이미 `Cu/Fe/Al_불순물_pct`와 `Ni_함량_pct` 원본 컬럼에 직접 접근할 "
    "수 있어서, 비율을 미리 계산해주지 않아도 트리 분기로 비슷한 경계를 스스로 학습할 "
    "수 있기 때문으로 보인다(이는 비율 항이 명시적으로 필요한 선형모델과의 차이점이다). "
    "따라서 이 세 비율 변수는 `engineer_features`에는 유지하되(해석용 참고 지표로서의 "
    "가치는 있을 수 있음), 최종 모델 입력 특성 세트에는 포함하지 않기로 결론짓는다 — "
    "\"단순 상관계수가 높아 보인다\"는 것과 \"모델에 실제로 기여한다\"는 것은 다른 "
    "질문이며, 후자는 반드시 ablation test로 (평균뿐 아니라 fold 간 변동폭까지 고려해) "
    "검증해야 한다는 것을 보여주는 사례다."
))

cells.append(nbf.v4.new_markdown_cell(
    "### 6.2 결측치 대치와 교차검증의 Data Leakage 방지\n\n"
    "올바른 순서는 `원본 → train/test(또는 fold) 분할 → 분할 이후에 결측치 대치`다 — "
    "대치 통계(중앙값, 회귀계수)를 train 쪽으로만 계산해서 test 쪽에 적용해야 test가 "
    "진짜 처음 보는 데이터로 남는다. 초기 구현은 이 순서를 지키지 않고 전체 420행 기준 "
    "통계로 먼저 전부 대치한 뒤 5-fold로 나눴는데, 이러면 test fold의 정보가 그 fold "
    "자신의 대치값 계산에 섞여 들어가는 data leakage다.\n\n"
    "영향이 작다고 해서 구조적 문제가 없어지는 건 아니므로, `run_ablation_test`를 "
    "다시 작성해 fold마다 train 구간의 통계로만 결측치를 대치하고(`_fold_safe_prepare`), "
    "그 통계를 test 구간에 적용한 뒤 파생변수를 계산하도록 고쳤다 — 지금 이 노트북이 "
    "호출하는 `run_ablation_test`는 결측치가 남아있는 `outlier_flagged_df`(대치 전)를 "
    "받아 내부에서 fold-safe하게 처리한다.\n\n"
    "고치기 전(전체 통계로 먼저 대치)과 고친 후(fold별 train 통계만 사용)를 비교하면:\n\n"
    "| 케이스 | 고치기 전 R² | 고친 후 R² | 차이 |\n"
    "|---|---|---|---|\n"
    "| base_only, Ni | 0.5495 | 0.5504 | +0.0009 |\n"
    "| base_only, Co | 0.5900 | 0.5886 | -0.0014 |\n"
    "| base_plus_derived, Ni | 0.8071 | 0.8080 | +0.0009 |\n"
    "| base_plus_derived, Co | 0.7927 | 0.7935 | +0.0008 |\n\n"
    "네 경우 모두 차이가 0.0014 이하로 fold별 R² 표준편차(0.02~0.06)보다 훨씬 작다. "
    "중앙값은 test fold(전체의 20%)를 빼고 계산해도 거의 변하지 않는 robust한 통계치이고, "
    "회귀로 대치하는 `황산투입량_L` 결측도 2건뿐이라 영향이 미미했던 것으로 보인다. "
    "이 프로젝트에서는 결론(파생변수는 유의미, 원료등급·불순물비율은 무의미)이 숫자상 "
    "크게 바뀌지 않았지만, 데이터가 더 작거나 결측 비율이 더 높았다면 달라졌을 수 있으므로 "
    "구조 자체를 fold-safe하게 고쳐두는 쪽을 택했다."
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
    "전체 420행(중복 배치 제거 후) 기준 `원료등급`(A/B/C) 분포는 B 38.6%, A 35.2%, C "
    "26.2%로(아래 셀 실행 결과), 가장 적은 클래스(C)도 약 26%를 차지해 특정 클래스가 "
    "극단적으로 희소한 심각한 불균형은 아니라고 판단했다. 설계 문서 §9에서는 구현 단계의 "
    "실제 분포를 보고 불균형 여부와 계층적 샘플링(stratified split) 적용 여부를 결정하기로 "
    "했는데, 실측 결과 클래스 간 비율 차이가 26~39% 수준으로 완만해 SMOTE·계층적 "
    "샘플링·클래스 가중치 등 별도의 불균형 보정은 적용하지 않기로 결론지었다. 다만 위 "
    "ablation test 결과(fold-safe 대치 적용, §6.2 참고)를 보면 `base_only`(Ni "
    "R²=0.5504±0.0474, Co R²=0.5886±0.0550) 대비 `base_plus_grade`(Ni R²=0.5496±0.0478, "
    "Co R²=0.5883±0.0533)는 둘 다 거의 그대로(차이 -0.0008, -0.0003)인데, 이는 fold별 "
    "표준편차(r2_std≈0.04~0.06)보다 훨씬 작아 통계적으로 유의미한 변화라고 볼 수 없다 — "
    "`원료등급`을 모델 입력 특성으로 그대로 추가하는 것 자체는 회귀 성능에 영향을 주지 "
    "않는다. 그럼에도 `원료등급`은 Task 5의 등급별 중앙값 보간(grade-based median "
    "imputation)에서 그룹핑 키로 쓰이므로 컬럼 자체는 계속 유지한다. 반면 산-광석비·침출강도·"
    "총불순물·목표금속 순도비 등 파생변수는 `base_plus_derived`에서 R²를 Ni 0.5504→0.8080, "
    "Co 0.5886→0.7935 수준으로 크게 끌어올려, 현재 데이터에서 회귀 성능에 실질적으로 기여하는 "
    "것은 `원료등급`이 아니라 파생변수 쪽임을 ablation test가 보여준다."
))

cells.append(nbf.v4.new_code_cell(
    "final_df['원료등급'].value_counts(normalize=True) * 100"
))

cells.append(nbf.v4.new_code_cell(
    "plot_grade_distribution(final_df, FIGURES_DIR)"
))

nb["cells"] = cells

Path("notebooks").mkdir(exist_ok=True)
with open("notebooks/01_eda_and_validation.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("노트북 생성 완료: notebooks/01_eda_and_validation.ipynb")
