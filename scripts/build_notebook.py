from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell(
    "# 모듈 1: 블랙파우더 성분 데이터 전처리 - EDA 및 검증\n\n"
    "- `src/pipeline.py`의 각 단계를 순서대로 실행함\n"
    "- 전/후 비교 시각화, 파생변수·원료등급 기여도(ablation test) 확인함"
))

cells.append(nbf.v4.new_code_cell(
    "import sys\n"
    "sys.path.append('..')\n\n"
    "import pandas as pd\n"
    "from pathlib import Path\n\n"
    "from src.preprocessing import load_raw, clean_schema, resolve_duplicate_batches\n"
    "from src.imputation import replace_outliers_with_nan, impute_missing_from_reference, interpolate_analysis_date\n"
    "from src.features import engineer_features\n"
    "from src.splitting import temporal_train_val_test_split\n"
    "from src.validation import (\n"
    "    check_ranges,\n"
    "    check_missing,\n"
    "    plot_missing_before_after,\n"
    "    plot_distributions,\n"
    "    plot_correlation_heatmap,\n"
    "    run_ablation_test,\n"
    "    plot_ablation_results,\n"
    "    plot_grade_distribution,\n"
    "    evaluate_final_model,\n"
    ")\n"
    "from src.config import COMPOSITION_COLUMNS, PROCESS_COLUMNS, TARGET_COLUMNS\n\n"
    "RAW_PATH = Path('../data/raw/영풍그룹_금속회수율최적화_RawMaterialData_학생배포용.xlsx')\n"
    "FIGURES_DIR = Path('../reports/figures')"
))

cells.append(nbf.v4.new_markdown_cell("## 1. 원본 로드 및 정제"))

cells.append(nbf.v4.new_code_cell(
    "raw_df = load_raw(RAW_PATH)\n"
    "raw_df.shape"
))

cells.append(nbf.v4.new_markdown_cell(
    "### 1.0 정제 전 정상범위 위반 확인\n\n"
    "- 정제 전 위반 건수를 먼저 측정함\n"
    "- §5(정제 후)와 직접 대비함"
))

cells.append(nbf.v4.new_code_cell(
    "raw_violations = check_ranges(raw_df)\n"
    "print('raw 데이터 정상범위 위반:', raw_violations if raw_violations else '없음')"
))

cells.append(nbf.v4.new_code_cell(
    "clean_df = clean_schema(raw_df)\n"
    "clean_df.shape"
))

cells.append(nbf.v4.new_markdown_cell(
    "### 1.1 완전 중복 배치 제거\n\n"
    "- `배치ID` 기준 중복 4건(8행) 확인함\n"
    "- 전체 컬럼까지 완전히 동일한 행만 자동 제거함(`resolve_duplicate_batches`) — "
    "배치ID만 같고 다른 값이 있는 충돌은 지우지 않고 `is_conflicting_duplicate`로 "
    "표시해 사람이 확인하게 함 (방치 시 동일 데이터가 train/test 양쪽에 걸쳐 들어가 "
    "성능 지표가 부풀려짐)\n"
    "- 남은 행의 `is_duplicate_batch=True`는 감사 기록으로 유지함 (`pipeline.py`의 "
    "`run()`과 동일한 순서)"
))

cells.append(nbf.v4.new_code_cell(
    "clean_df = resolve_duplicate_batches(clean_df)\n"
    "print(f'중복 제거 후: {len(clean_df)}행, 충돌(수동 확인 필요): "
    "{clean_df[\"is_conflicting_duplicate\"].sum()}건')"
))

cells.append(nbf.v4.new_code_cell(
    "outlier_flagged_df = replace_outliers_with_nan(clean_df)\n"
    "outlier_flagged_df.shape"
))

cells.append(nbf.v4.new_code_cell(
    "missing = check_missing(outlier_flagged_df)\n"
    "print('결측치 확인:', missing if missing else '없음')"
))

cells.append(nbf.v4.new_markdown_cell(
    "### 1.1.5 결측 패턴 진단 (체계적 편향 여부 확인)\n\n"
    "결측을 그냥 채우기 전에, 결측이 특정 조건(원료등급)에 쏠려있거나 여러 컬럼에서 "
    "동시에 발생하는 체계적 패턴인지 확인한다 — 체계적 패턴이 있다면 단순 중앙값 "
    "대치가 편향을 만들 수 있기 때문이다."
))

cells.append(nbf.v4.new_code_cell(
    "from scipy.stats import chi2_contingency\n\n"
    "missing_cols = [\n"
    "    c for c in outlier_flagged_df.columns\n"
    "    if outlier_flagged_df[c].isna().any() and c != '분석일자'\n"
    "]\n\n"
    "print('컬럼별 결측 비율:')\n"
    "for col in missing_cols:\n"
    "    print(f'  {col}: {outlier_flagged_df[col].isna().mean() * 100:.1f}%')"
))

cells.append(nbf.v4.new_code_cell(
    "print('카이제곱 독립성 검정 (원료등급 vs 결측여부):')\n"
    "for col in missing_cols:\n"
    "    table = pd.crosstab(outlier_flagged_df['원료등급'], outlier_flagged_df[col].isna())\n"
    "    chi2, p, dof, expected = chi2_contingency(table)\n"
    "    print(f'  {col}: chi2={chi2:.3f}, p={p:.3f}')"
))

cells.append(nbf.v4.new_code_cell(
    "missing_flags = outlier_flagged_df[missing_cols].isna().astype(int)\n"
    "missing_flags.corr().round(2)"
))

cells.append(nbf.v4.new_markdown_cell(
    "**결과**: 결측 비율은 컬럼별로 6.9~10.7% 수준이다. 카이제곱 검정 결과 10개 "
    "컬럼 중 8개는 p>0.05로 원료등급과 결측 여부가 독립적이지만, `Mn_함량_pct`"
    "(p=0.043)와 `Cu_불순물_pct`(p=0.006) 두 컬럼은 원료등급에 따라 결측 "
    "발생률이 통계적으로 다르다 — 완전히 무작위(MCAR)는 아니라는 뜻이다. 다만 "
    "이 두 컬럼도 조성 변수라 이미 원료등급별 그룹 중앙값으로 대치하고 있어서"
    "(`impute_missing_from_reference`), 등급에 따른 결측 편향 자체가 대치 "
    "방식에 반영되어 있다. 결측 플래그 간 상관계수는 전부 -0.09~0.10 사이로 "
    "0에 가까워, 여러 컬럼이 동시에 결측되는 패턴은 없다."
))

cells.append(nbf.v4.new_markdown_cell(
    "### 1.2 분석일자 보간\n\n"
    "- 분석일자는 모델 피처로 안 쓰는 위치 기반 로직(배치ID 순서로 보간)임 — "
    "train/val/test 구분 불필요함\n"
    "- 분할(§1.3) 전에 전체 데이터로 한 번에 처리함\n"
    "- `pre_impute_df`는 보간 이후, 그룹별 결측치 대치 이전 시점 스냅샷임"
))

cells.append(nbf.v4.new_code_cell(
    "interpolated_df = interpolate_analysis_date(outlier_flagged_df)\n"
    "pre_impute_df = interpolated_df.copy()\n"
    "print('분석일자 결측:', interpolated_df['분석일자'].isna().sum())"
))

cells.append(nbf.v4.new_markdown_cell(
    "### 1.3 시간순 Train/Validation/Test 분할\n\n"
    "- 배치는 2025년 1~10월 순차 데이터임 — 무작위 분할 금지\n"
    "- 배치ID 순번(분석일자 근사) 기준 앞 70% Train, 다음 15% Validation, 마지막 15% "
    "Test로 고정 분할함 (`pipeline.py`의 `run()`과 동일)\n"
    "- 여기서 계산한 `train_i`는 §1.4의 결측치 대치 기준으로 재사용함. "
    "§6/§6.1의 feature 선택은 `train_i`/`val_i`만 사용함 — `test_i`는 feature "
    "선택에 끼워 넣지 않고 최종 평가용으로 따로 남겨둔다 (§6.2 참고)"
))

cells.append(nbf.v4.new_code_cell(
    "train_i, val_i, test_i = temporal_train_val_test_split(interpolated_df)\n"
    "print(f'Train: {len(train_i)}행 / Validation: {len(val_i)}행 / Test: {len(test_i)}행')"
))

cells.append(nbf.v4.new_markdown_cell(
    "### 1.4 Train 통계 기반 결측치 대치 + 파생변수 생성\n\n"
    "- Validation·Test 결측치도 Train 통계(그룹별/전체 중앙값, 회귀계수)로만 채움\n"
    "- Validation·Test 자신의 값은 대치 계산에 쓰지 않음 (`impute_missing_from_reference`)"
))

cells.append(nbf.v4.new_code_cell(
    "train_df = interpolated_df.loc[train_i]\n"
    "imputed_df = impute_missing_from_reference(train_df, interpolated_df)\n"
    "final_df = engineer_features(imputed_df)\n"
    "final_df.head()"
))

cells.append(nbf.v4.new_markdown_cell(
    "## 2. 결측치 처리 전/후 비교\n\n"
    "- \"전\" 기준은 `outlier_flagged_df` (분석일자 보간도 적용 전, 진짜 결측 그대로인 "
    "시점)임 — `pre_impute_df`는 분석일자가 이미 채워진 상태라 보간 효과가 안 보임"
))

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
    "- 목표금속(Ni/Co/Li_함량_pct)은 회수율과 양의 상관(+0.30~+0.49), 불순물"
    "(Cu/Fe/Al_불순물_pct)·Mn_함량_pct는 음의 상관(-0.26~-0.57)임 — §6 도메인 가설과 "
    "방향 일치함\n"
    "- 공정조건(침출온도_C, 침출시간_hr)도 양의 상관(+0.27~+0.48)임\n"
    "- 파생변수(acid_to_ore_ratio, leach_intensity, total_impurity_pct, "
    "target_purity_ratio)는 구성요소 원본 변수보다 상관이 더 강함 — 예: "
    "leach_intensity(Ni 0.51) > 침출온도_C(0.48)·침출시간_hr(0.36)\n"
    "- Fe_Ni_ratio·Cu_Ni_ratio·Al_Ni_ratio는 단순 상관은 -0.41~-0.54로 강하지만, §6.1 "
    "ablation test에서는 모델 R²에 기여 못함 — 단순 상관과 모델 기여도는 다른 질문임"
))

cells.append(nbf.v4.new_markdown_cell(
    "## 5. 정합성 검증\n\n"
    "- §1.0 raw 위반 건수와 비교해 정제 효과 확인함\n"
    "- `pipeline.py`의 `run()`은 위반 발견 시 `assert`로 즉시 중단함 (남아있다면 "
    "대치·파생변수 로직 버그라는 뜻)\n"
    "- 이 노트북은 탐색용이라 중단 없이 결과만 출력함"
))

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
    "results = run_ablation_test(\n"
    "    pre_impute_df.loc[train_i], pre_impute_df.loc[val_i],\n"
    "    feature_sets, TARGET_COLUMNS\n"
    ")\n"
    "pd.DataFrame(results).T"
))

cells.append(nbf.v4.new_code_cell(
    "plot_ablation_results(results, FIGURES_DIR)"
))

cells.append(nbf.v4.new_markdown_cell(
    "### 6.1 개별 불순물 비율(Fe/Ni, Cu/Ni, Al/Ni) 추가 검증\n\n"
    "- `total_impurity_pct`는 Cu/Fe/Al을 동일 가중치로 합산해 개별 불순물 정보가 가려짐\n"
    "- §6과 동일한 4-케이스 구조로 (a) 비율 변수 단독 신호 여부, (b) `base_plus_derived` "
    "위 추가 기여 여부를 검증함"
))

cells.append(nbf.v4.new_code_cell(
    "impurity_ratio_features = ['Fe_Ni_ratio', 'Cu_Ni_ratio', 'Al_Ni_ratio']\n\n"
    "ratio_feature_sets = {\n"
    "    'base_only': base_features,\n"
    "    'base_plus_ratios': base_features + impurity_ratio_features,\n"
    "    'base_plus_derived': base_features + derived_features,\n"
    "    'base_plus_derived_plus_ratios': base_features + derived_features + impurity_ratio_features,\n"
    "}\n\n"
    "ratio_results = run_ablation_test(\n"
    "    pre_impute_df.loc[train_i], pre_impute_df.loc[val_i],\n"
    "    ratio_feature_sets, TARGET_COLUMNS\n"
    ")\n"
    "pd.DataFrame(ratio_results).T"
))

cells.append(nbf.v4.new_code_cell(
    "plot_ablation_results(ratio_results, FIGURES_DIR / 'ratios')"
))

cells.append(nbf.v4.new_markdown_cell(
    "**결론** (Validation 기준 — feature 선택은 Validation만 사용하고 Test는 "
    "건드리지 않음, §6.2 참고)\n\n"
    "- `base_only`→`base_plus_ratios`: Validation R² Ni 0.4836→0.4695(-0.0141), "
    "Co 0.5344→0.5150(-0.0194) — 두 타겟 모두 하락함\n"
    "- `base_plus_derived`→`base_plus_derived_plus_ratios`: Validation R² Ni "
    "0.7505→0.7428(-0.0078), Co 0.7168→0.7148(-0.0020) — 파생변수가 이미 있는 "
    "상태에서도 추가로 하락함\n"
    "- 비교: `base_only`→`base_plus_derived`(파생변수 4종 추가)는 Ni +0.2669, "
    "Co +0.1825로 큰 폭 상승함 — 비율 변수(두 경우 모두 하락)와 방향이 다름\n"
    "- 단순 상관(-0.4~-0.5)은 강하지만 Validation 기준 모델 기여는 오히려 음수임 — "
    "RandomForest가 원본 불순물·Ni_함량 컬럼으로 이미 유사한 경계를 학습하기 "
    "때문으로 판단함\n"
    "- 결론: 비율 변수 3종은 `engineer_features`에는 유지(해석용), 모델 입력 "
    "특성에는 미포함함 (Validation 기준 두 비교 모두 일관되게 하락)"
))

cells.append(nbf.v4.new_markdown_cell(
    "### 6.2 Data Leakage를 피하기 위한 평가 설계\n\n"
    "- Test가 \"처음 보는 데이터\"로 남도록 세 가지를 지킴\n\n"
    "**1) 시간순 고정 분할 (무작위 섞기 없음)**\n"
    "- 배치는 2025년 1~10월 순차 데이터임 — 미래 배치로 과거를 맞히는 상황 방지함\n"
    "- 배치ID 순번 기준 앞 70% Train, 다음 15% Validation, 마지막 15% Test로 고정 "
    "분할함(`temporal_train_val_test_split`, 294/63/63행)\n\n"
    "**2) 결측치 대치 통계는 Train에서만 계산**\n"
    "- Validation·Test 결측치도 Train 통계(중앙값·회귀계수)로만 채움"
    "(`impute_missing_from_reference`)\n"
    "- 초기 구현(전체 420행 기준 대치 후 무작위 5-fold)과 비교 시 차이는 0.0014 이하로 "
    "작았으나, 구조적 문제로 판단해 수정함\n\n"
    "**3) Feature 선택에는 Test를 보지 않음**\n"
    "- `run_ablation_test`는 Train/Validation만 받는다 — §6/§6.1의 모든 feature "
    "비교·선택 판단은 Validation 성능만으로 내린다\n"
    "- Validation 결과를 보고 feature를 고르는 순간 Test도 같이 봐버리면, Test가 "
    "더 이상 \"모델이 한 번도 안 본\" 최종평가셋이 아니게 된다 — Test는 분할 "
    "이후 이 노트북 어디에서도 feature 선택에 쓰이지 않고 남아있음"
))

cells.append(nbf.v4.new_markdown_cell(
    "### 6.3 최종 feature set의 Test 평가 (1회)\n\n"
    "§6.1에서 Validation만으로 확정한 `base_plus_derived`(조성+공정+파생변수 4종, "
    "비율 변수 미포함)로 최종 모델을 만든다 — Train+Validation을 합쳐 다시 학습하고 "
    "(`evaluate_final_model`, `pipeline.py`의 `run()`과 동일한 함수), 지금까지 "
    "한 번도 feature 선택에 쓰이지 않은 `test_i`를 여기서 처음, 딱 한 번 사용한다."
))

cells.append(nbf.v4.new_code_cell(
    "final_feature_set = base_features + derived_features\n\n"
    "final_test_results = evaluate_final_model(\n"
    "    pre_impute_df.loc[train_i], pre_impute_df.loc[val_i], pre_impute_df.loc[test_i],\n"
    "    final_feature_set, TARGET_COLUMNS\n"
    ")\n"
    "pd.DataFrame(final_test_results).T"
))

cells.append(nbf.v4.new_markdown_cell(
    "## 7. 심층 인터뷰 답변\n\n"
    "**1) 실제 공정 데이터 품질 문제를 어떻게 식별·처리했는가?**\n\n"
    "- 데이터사전 정상범위 벗어난 값(`Ni_함량_pct=999.0`, `침출온도_C=-5.0`, "
    "`황산투입량_L=15000`)을 물리적 불가능 이상치로 식별, `replace_outliers_with_nan`에서 NaN 치환함\n"
    "- 남은 결측(6.9~10.7%)은 §1.1.5에서 카이제곱 독립성 검정, 결측 동시발생 "
    "상관으로 패턴을 확인함 — 결측 플래그 간 상관은 전부 0에 가까워 여러 컬럼이 "
    "동시에 결측되는 패턴은 없음. 다만 `Mn_함량_pct`·`Cu_불순물_pct`는 원료등급과 "
    "결측 여부가 통계적으로 독립이 아님(p<0.05) — 완전 무작위(MCAR)는 아니지만, "
    "두 컬럼 다 조성 변수라 이미 등급별 그룹 중앙값으로 대치하고 있어 이 편향이 "
    "대치 방식에 반영되어 있음\n"
    "- 변수 성격별로 원료등급별 중앙값, 회귀 기반, 배치ID 순서 기반 보간으로 구분 "
    "처리함 (설계 문서 §4)\n\n"
    "**2) 특성공학 과정에서 도메인 지식을 어떻게 활용했는가?**\n\n"
    "- 산-광석비, 침출강도(온도×시간), 총불순물, 목표금속 순도비는 습식제련 침출 "
    "공정의 화학양론·반응속도론적 지식에서 도출함 (설계 문서 §6)\n"
    "- ablation test로 파생변수·원료등급 포함 시 R²/RMSE 개선 여부를 정량 확인함\n\n"
    "**3) 데이터 불균형 문제가 있었다면 어떻게 해결했는가?**\n\n"
    "- 전체 420행 기준 `원료등급` 분포: B 38.6%, A 35.2%, C 26.2% — 최소 클래스도 "
    "26%로 심각한 불균형 아님\n"
    "- 클래스 간 비율 차이 26~39% 수준이라 SMOTE·계층적 샘플링·클래스 가중치는 "
    "미적용함\n"
    "- ablation test 결과(Validation 기준): `base_only`→`base_plus_grade` Ni "
    "0.4836→0.4618(-0.0218), Co 0.5344→0.5321(-0.0022) — 두 타겟 모두 하락, "
    "`원료등급`의 안정적 성능 기여 없음\n"
    "- 다만 `원료등급`은 등급별 중앙값 보간의 그룹핑 키로 쓰여 컬럼은 유지함\n"
    "- 파생변수는 `base_only`→`base_plus_derived`에서 Ni 0.4836→0.7505(+0.2669), "
    "Co 0.5344→0.7168(+0.1825)로 뚜렷하게 상승 — 실질적 성능 기여는 원료등급이 "
    "아니라 파생변수 쪽임"
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
