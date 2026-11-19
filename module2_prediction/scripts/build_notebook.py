from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell(
    "# 모듈 2: 회수율 예측 모델 학습, 평가\n\n"
    "- 모듈 1의 대치 전 데이터, 선택 결과를 읽고 전처리 기능을 재사용\n"
    "- 타깃별 선택 피처를 유지하고 CV 평균 RMSE로 모델, 하이퍼파라미터 선택"
))

cells.append(nbf.v4.new_code_cell(
    "import sys\n"
    "sys.path.append('..')\n\n"
    "import pandas as pd\n"
    "import numpy as np\n"
    "import json\n"
    "from pathlib import Path\n\n"
    "from module2_prediction.config import FEATURE_COLUMNS, TARGET_COLUMNS\n"
    "from module2_prediction.data_loading import load_modeling_inputs, split_by_column\n"
    "from module1_preprocessing.transformation import prepare_split\n"
    "from module2_prediction.modeling import compare_models\n"
    "from module2_prediction.tuning import tune_models_cv\n"
    "from module2_prediction.evaluation import evaluate_final_model\n"
    "from module2_prediction.pipeline import save_selection_report\n"
    "from module2_prediction.artifacts import load_bundle\n"
    "from module2_prediction.interpretation import compute_shap_values\n"
    "from module2_prediction.scenario import predict_process_scenario, predict_blend_scenario\n"
    "from sklearn.inspection import permutation_importance\n\n"
    "import matplotlib\n"
    "import matplotlib.pyplot as plt\n"
    "get_ipython().run_line_magic('matplotlib', 'inline')\n"
    "matplotlib.rcParams['font.family'] = ['Noto Sans CJK JP', 'DejaVu Sans']\n"
    "matplotlib.rcParams['axes.unicode_minus'] = False\n\n"
    "DATA_PATH = Path('../data/processed/pre_impute_data.csv')\n"
    "FIGURES_DIR = Path('../module2_prediction/reports/figures')\n"
    "FIGURES_DIR.mkdir(parents=True, exist_ok=True)\n"
    "MODEL_DIR = Path('../module2_prediction/reports/final')"
))

cells.append(nbf.v4.new_markdown_cell("## 1. 데이터 로드"))

cells.append(nbf.v4.new_code_cell(
    "df, selection = load_modeling_inputs(DATA_PATH)\n"
    "features_by_target = {t: entry['features'] for t, entry in selection['targets'].items()}\n"
    "train_df, val_df, test_df = split_by_column(df)\n"
    "cv_pool = pd.concat([train_df, val_df], ignore_index=True)\n"
    "print(f'Train: {len(train_df)}행 / Validation: {len(val_df)}행 / Test: {len(test_df)}행')\n"
    "print(f'CV 풀(Train+Validation): {len(cv_pool)}행')"
))

cells.append(nbf.v4.new_markdown_cell(
    "## 2. 모델 비교 (기본값, 단일 Validation 분할)\n\n"
    "- RandomForest, XGBoost, LightGBM, ElasticNet, SVR을 동일한 feature set, 기본 하이퍼파라미터로 비교\n"
    "- Test는 안 봄, 참고용 베이스라인\n"
    "- 실제 모델, 하이퍼파라미터 선정은 §3의 CV 결과로 함 — 단일 분할은 분할 운에 따라 순위가 흔들릴 수 있음"
))

cells.append(nbf.v4.new_code_cell(
    "default_results = compare_models(train_df, val_df, feature_columns=features_by_target)\n"
    "pd.DataFrame(default_results).T"
))

cells.append(nbf.v4.new_markdown_cell(
    "## 3. 하이퍼파라미터 튜닝 (Stratified 5-Fold CV)\n\n"
    "- `config.MODEL_HYPERPARAMETER_GRIDS`의 각 조합을 CV 풀(Train+Validation)에서 "
    "Stratified 5-Fold CV로 평가해 모델, 타겟별 최적값을 고름\n"
    "- 단일 Validation 분할(63행)은 분할 운에 따라 순위가 흔들릴 수 있음(모듈1에서 "
    "실제로 확인됨) — 각 fold의 학습 부분만으로 대치하고 CV 평균 RMSE로 선택함. CV는 선택용 점수이며 일반화 성능은 Test로 보고함\n"
    "- Test는 CV 풀에 포함하지 않음"
))

cells.append(nbf.v4.new_code_cell(
    "cv = selection['cv']\n"
    "tuned = tune_models_cv(cv_pool, feature_columns=features_by_target, n_splits=cv['n_splits'], random_state=cv['random_state'], stratify_column=cv['stratify_column'])\n"
    "pd.DataFrame({key: v['metrics'] for key, v in tuned.items()}).T"
))

cells.append(nbf.v4.new_markdown_cell("## 4. 최종 모델 선정 (타겟별 CV 평균 RMSE 최소 모델)"))

cells.append(nbf.v4.new_code_cell(
    "best_per_target = {\n"
    "    target: min(\n"
    "        (key for key in tuned if key[1] == target),\n"
    "        key=lambda key: tuned[key]['metrics']['val_rmse']\n"
    "    )\n"
    "    for target in TARGET_COLUMNS\n"
    "}\n"
    "for target, (model_name, _) in best_per_target.items():\n"
    "    p = tuned[(model_name, target)]\n"
    "    print(f'{target}: {model_name}, params={p[\"params\"]}, val_rmse={p[\"metrics\"][\"val_rmse\"]:.4f}')"
))

cells.append(nbf.v4.new_markdown_cell("""### 선정 근거와 과적합 진단
- 모듈 1의 split을 그대로 읽고 Train+Validation 내부에서만 CV 수행
- 동일 fold의 평균 검증 RMSE 최소 모델을 타깃별 선정, MAE, R²는 보조 지표
- 학습 RMSE: 해당 fold 모델의 학습 데이터 적합 오차. 검증 RMSE와의 차이(rmse_gap)는 과적합 진단용 — 오차 차이가 작다는 이유로 모델을 재선정하지 않음. 단위는 %p
- CV RMSE는 모델, 하이퍼파라미터 선택에 사용한 점수이므로 독립 평가 점수로 해석하지 않음
- 현재 피처 비교 결과는 참고 기록이며 튜닝에는 고정 피처 사용
- CV 표준편차는 분할별 성능 변동이며, 개별 배치의 예측구간, 포함률은 미구현
"""))
cells.append(nbf.v4.new_code_cell("""cv_table = save_selection_report(tuned, best_per_target, selection, MODEL_DIR)
display(cv_table[['model', 'target', 'train_rmse', 'train_rmse_std',
                  'val_rmse', 'val_rmse_std', 'rmse_gap', 'val_mae', 'val_r2']])
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, target in zip(axes, TARGET_COLUMNS):
    table = cv_table[cv_table.target == target].set_index('model')
    x = np.arange(len(table))
    ax.bar(x - .2, table.train_rmse, .4, yerr=table.train_rmse_std, label='학습', capsize=3)
    ax.bar(x + .2, table.val_rmse, .4, yerr=table.val_rmse_std, label='검증', capsize=3)
    ax.set_xticks(x, table.index, rotation=35)
    ax.set(title=target, ylabel='RMSE (%p)')
    ax.legend()
fig.tight_layout()
fig.savefig(FIGURES_DIR / 'train_validation_rmse.png')
plt.show()
plt.close(fig)
"""))

cells.append(nbf.v4.new_markdown_cell("## 5. 최종 Test 평가 (선정 완료 후 Train+Validation 재학습)"))

cells.append(nbf.v4.new_code_cell(
    "final_results = {}\n"
    "for target, (model_name, _) in best_per_target.items():\n"
    "    params = tuned[(model_name, target)]['params']\n"
    "    result = evaluate_final_model(\n"
    "        train_df, val_df, test_df,\n"
    "        model_name=model_name, model_params=params,\n"
    "        target_columns=[target], feature_columns=features_by_target[target], output_dir=MODEL_DIR,\n"
    "    )\n"
    "    final_results[target] = dict(model=model_name, params=params, primary_metric='rmse', features=features_by_target[target], **tuned[(model_name, target)]['metrics'], **result[target])\n\n"
    "(MODEL_DIR / 'results.json').write_text(json.dumps(final_results, ensure_ascii=False, indent=2), encoding='utf-8')\n"
    "pd.DataFrame(final_results).T"
))

cells.append(nbf.v4.new_markdown_cell("""## 6. Test 성능 심화 분석 (등급별 오차, 단순 기준선 비교)
- 등급별 오차: `{target}_predictions.csv`에 원료등급을 붙여 A/B/C별 RMSE, MAE 비교
  - 전체 평균 하나만 보면 특정 등급에서 유독 못 맞추는지 가려질 수 있음
  - 등급 단독 성능이 좋아도 7:3 혼합 배치의 정확도가 검증되는 것은 아님(§9)
- 단순 기준선: 원료, 공정 정보를 전혀 쓰지 않고 Train+Validation 평균만 예측하는
  `DummyRegressor`와 비교해, 실제 모델이 입력 정보로 얼마나 개선됐는지 확인
- 평균 부호 오차(%p) = 평균(예측값 - 실제값) — 양수면 평균적으로 과대예측, 음수면 과소예측.
  개별 배치의 과대, 과소는 섞여 있을 수 있으며, 이 값은 그 합산 방향만 나타냄
"""))
cells.append(nbf.v4.new_code_cell("""from sklearn.dummy import DummyRegressor

grade_error_rows = []
baseline_rows = []
for target in TARGET_COLUMNS:
    predictions = pd.read_csv(MODEL_DIR / f'{target}_predictions.csv')
    predictions = predictions.merge(
        test_df[['배치ID', '원료등급']].rename(columns={'배치ID': 'batch_id'}),
        on='batch_id', validate='one_to_one',
    )
    if set(predictions['batch_id']) != set(test_df['배치ID']):
        raise ValueError(f'{target}: 예측 CSV와 Test 배치ID 집합이 다름')

    error = predictions['prediction'] - predictions['actual']

    for grade, group in predictions.groupby('원료등급'):
        g_error = group['prediction'] - group['actual']
        grade_error_rows.append(dict(
            target=target, grade=grade, n=len(group),
            rmse=float(np.sqrt((g_error ** 2).mean())),
            mae=float(g_error.abs().mean()),
        ))

    baseline = DummyRegressor(strategy='mean').fit(cv_pool[[]], cv_pool[target])
    baseline_preds = baseline.predict(test_df[[]])
    baseline_rmse = float(np.sqrt(((baseline_preds - test_df[target]) ** 2).mean()))
    model_rmse = final_results[target]['test_rmse']
    baseline_rows.append(dict(
        target=target, baseline_rmse=baseline_rmse, model_rmse=model_rmse,
        rmse_reduction_pct=100 * (1 - model_rmse / baseline_rmse),
        mean_signed_error_pctp=float(error.mean()),
    ))

grade_error_table = pd.DataFrame(grade_error_rows)
grade_error_table.to_csv(MODEL_DIR / 'test_error_by_grade.csv', index=False)
display(grade_error_table)

baseline_table = pd.DataFrame(baseline_rows)
baseline_table.to_csv(MODEL_DIR / 'baseline_comparison.csv', index=False)
display(baseline_table)
"""))

cells.append(nbf.v4.new_markdown_cell("""## 7. SHAP 중요도와 permutation importance 비교
- 최종 평가에서 저장한 동일 모델을 불러옴
- 개발 데이터에서 고정 추출한 최대 100행을 배경으로 사용, Test 입력의 예측을 설명 — Test 정답은 SHAP 계산에 미사용
- 평균 절대 SHAP 값은 전체 중요도(%p), beeswarm은 배치별 기여 방향과 분포를 나타냄
- Test에서 SHAP 중요도와 permutation importance를 비교해 최종 모델의 피처 의존성을 살펴봄
- permutation importance: 같은 Test 입력에서 피처 하나를 섞었을 때 평균 RMSE가 얼마나 달라지는지(%p)만 봄
  - 0에 가깝거나 음수인 값 → 이 Test에서 성능 악화가 관찰되지 않았다는 뜻
  - 원변수와 파생변수가 겹치는 경우, 그 값만으로 피처를 빼도 된다고 판단하기는 어려움
- 이 결과로 피처나 모델을 재선택하지 않음
"""))
cells.append(nbf.v4.new_code_cell("""train_val_df = prepare_split(cv_pool, cv_pool)
test_prepared = prepare_split(cv_pool, test_df)
background_positions = np.random.default_rng(42).choice(len(train_val_df), size=min(100, len(train_val_df)), replace=False)
pd.DataFrame({'batch_id': cv_pool.iloc[background_positions]['배치ID'].to_numpy()}).to_csv(
    MODEL_DIR / 'shap_background_batches.csv', index=False)
interpretation_models = {}
shap_by_target = {}
importance_by_target = {}
perm_importance_by_target = {}
for target in TARGET_COLUMNS:
    bundle = load_bundle(MODEL_DIR / f'{target}.joblib')
    model = bundle['model']
    columns = bundle['features']
    interpretation_models[target] = model
    X = test_prepared[columns]
    background = train_val_df.iloc[background_positions][columns]
    values = compute_shap_values(model, X, background=background)
    shap_by_target[target] = values
    importance_by_target[target] = pd.Series(np.abs(values).mean(axis=0), index=columns)
    pd.DataFrame(values, columns=columns, index=test_df['배치ID']).to_csv(MODEL_DIR / f'{target}_shap_values.csv')

    result = permutation_importance(
        model, X, test_prepared[target],
        scoring='neg_root_mean_squared_error', n_repeats=10, random_state=42,
    )
    perm_importance_by_target[target] = pd.Series(result.importances_mean, index=X.columns)
importance_table = pd.DataFrame(importance_by_target)
importance_table.to_csv(MODEL_DIR / 'shap_importance.csv', index_label='feature')
perm_importance_table = pd.DataFrame(perm_importance_by_target)
perm_importance_table.to_csv(MODEL_DIR / 'permutation_importance.csv', index_label='feature')
combined_importance = pd.concat(
    {'SHAP_절대평균': importance_table, '퍼뮤테이션_RMSE증가_pctp': perm_importance_table},
    axis=1,
)
display(combined_importance.round(4))
"""))
cells.append(nbf.v4.new_code_cell("""import shap
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
for ax, target in zip(axes, TARGET_COLUMNS):
    importance_table[target].sort_values().plot.barh(ax=ax)
    ax.set(title=target, xlabel='평균 절대 SHAP (%p)')
fig.tight_layout()
fig.savefig(FIGURES_DIR / 'shap_importance.png')
plt.show()
plt.close(fig)
for target in TARGET_COLUMNS:
    plt.figure()
    shap.summary_plot(shap_by_target[target], test_prepared[features_by_target[target]], show=False)
    plt.title(f'{target}: Test 예측의 SHAP 기여')
    plt.savefig(FIGURES_DIR / f'shap_summary_{target}.png', bbox_inches='tight')
    plt.show()
    plt.close()
"""))
cells.append(nbf.v4.new_markdown_cell("""### 핵심 인자 해석과 한계
- 아래 표: 타깃별 상위 3개 변수의 중요도 + 낮은 값/높은 값 집단의 평균 SHAP
  - 두 집단은 각 변수의 Test 25, 75 분위수로 구분, 집단 간 차이는 관측된 연관성
  - 다른 조건을 통제한 효과나 단조성을 증명하지 않으므로 beeswarm과 함께 해석

공정 관점에서 검토할 가설:
- 침출온도, 침출시간, 침출강도: 반응 진행 조건을 요약. 온도 상승의 기여 방향이 양수인지 확인하되,
  온도×시간과 중복된 정보를 개별 효과로 더하지 않음
- 산-광석비, 황산투입량: 원료 대비 산 용액 공급 조건과 관련. 산 농도가 입력에 없으므로
  용액 부피를 유효 산 몰수나 산 소비량으로 해석하지 않음
- 총불순물, 개별 불순물, 산용액당불순물량: 원료 조성과 산 공급의 관계를 나타냄.
  경쟁 반응이나 산 소비와 연결될 가능성은 검토 가설이며 실제 원인은 실험으로 확인 필요

- 이 설명은 도메인 검토를 위한 가설 — 문헌, 운전 기록, 혼합 실험 확인 전에는 도메인 지식과의
  일치나 새로운 발견을 확정하지 않음
- 상관된 원변수, 파생변수를 독립적으로 변화시키는 SHAP 계산에는 현실에 없는 조합이 포함될 수 있음
"""))
cells.append(nbf.v4.new_code_cell("""rows = []
for target in TARGET_COLUMNS:
    columns = features_by_target[target]
    for feature in importance_table[target].nlargest(3).index:
        x = test_prepared[feature]
        contribution = shap_by_target[target][:, columns.index(feature)]
        low, high = x.quantile([.25, .75])
        low_mean = float(contribution[x <= low].mean())
        high_mean = float(contribution[x >= high].mean())
        rows.append(dict(target=target, feature=feature,
            mean_abs_shap=importance_table.loc[feature, target],
            low_threshold=low, high_threshold=high,
            low_group_mean_shap=low_mean, high_group_mean_shap=high_mean,
            high_minus_low_shap=high_mean-low_mean))
domain_summary = pd.DataFrame(rows)
domain_summary.to_csv(MODEL_DIR / 'shap_top_factors.csv', index=False)
display(domain_summary.round(4))
"""))

cells.append(nbf.v4.new_markdown_cell(
    "## 8. 공정조건별 변화 분석\n\n"
    "- 침출온도_C, 침출시간_hr을 각각 바꿔가며 Ni_회수율, Co_회수율 예측값 변화 확인\n"
    "  (나머지 변수는 고정, 2타깃×2변수=4개 조합 전부 확인)\n"
    "- 공정조건(PROCESS_COLUMNS) 변경이므로 leach_intensity(=온도×시간)를 함께 재계산하는\n"
    "  `predict_process_scenario` 사용\n"
    "- `partial_dependence`는 파생변수를 그대로 둬서, 값만 바꾸면 학습 데이터의 파생식과\n"
    "  모순되는 입력이 되므로 여기서는 쓰지 않음"
))

cells.append(nbf.v4.new_code_cell(
    "target = TARGET_COLUMNS[0]\n"
    "model = interpretation_models[target]\n"
    "base_row = train_val_df.iloc[0]\n\n"
    "pd_result = predict_process_scenario(model, base_row, feature='침출온도_C', values=list(range(40, 96, 5)), feature_columns=features_by_target[target])\n"
    "pd_result.plot(x='침출온도_C', y='prediction', marker='o', title=f'{target} vs 침출온도_C')"
))

cells.append(nbf.v4.new_code_cell(
    "target = TARGET_COLUMNS[0]\n"
    "model = interpretation_models[target]\n"
    "base_row = train_val_df.iloc[0]\n\n"
    "pd_result = predict_process_scenario(model, base_row, feature='침출시간_hr', values=list(range(1, 9)), feature_columns=features_by_target[target])\n"
    "pd_result.plot(x='침출시간_hr', y='prediction', marker='o', title=f'{target} vs 침출시간_hr')"
))

cells.append(nbf.v4.new_code_cell(
    "target = TARGET_COLUMNS[1]\n"
    "model = interpretation_models[target]\n"
    "base_row = train_val_df.iloc[0]\n\n"
    "pd_result = predict_process_scenario(model, base_row, feature='침출온도_C', values=list(range(40, 96, 5)), feature_columns=features_by_target[target])\n"
    "pd_result.plot(x='침출온도_C', y='prediction', marker='o', title=f'{target} vs 침출온도_C')"
))

cells.append(nbf.v4.new_code_cell(
    "target = TARGET_COLUMNS[1]\n"
    "model = interpretation_models[target]\n"
    "base_row = train_val_df.iloc[0]\n\n"
    "pd_result = predict_process_scenario(model, base_row, feature='침출시간_hr', values=list(range(1, 9)), feature_columns=features_by_target[target])\n"
    "pd_result.plot(x='침출시간_hr', y='prediction', marker='o', title=f'{target} vs 침출시간_hr')"
))

cells.append(nbf.v4.new_markdown_cell(
    "## 9. 7:3 혼합 시나리오 예측\n\n"
    "- 실제 성분 분석값을 동일 질량 기준으로 7:3 혼합, 이번 배치의 운전조건 적용\n"
    "- 아래 값은 사용법 설명용 가상 입력 — 오늘의 측정값이 아님, 실제 사용 시 두 원료 분석값과 운전조건을 교체\n"
    "- 혼합 배치 정확도는 실험으로 검증 필요"
))

cells.append(nbf.v4.new_code_cell(
    "target = TARGET_COLUMNS[0]\n"
    "model = interpretation_models[target]\n\n"
    "# 설명용 가상 분석값(wt%, 동일 질량 기준). 오늘 입고된 원료의 측정값으로 교체하세요.\n"
    "composition_a = {'Ni_함량_pct': 20, 'Co_함량_pct': 8, 'Li_함량_pct': 4, 'Mn_함량_pct': 10, 'Cu_불순물_pct': 1, 'Fe_불순물_pct': 1, 'Al_불순물_pct': 1}\n"
    "composition_c = {'Ni_함량_pct': 10, 'Co_함량_pct': 4, 'Li_함량_pct': 3, 'Mn_함량_pct': 10, 'Cu_불순물_pct': 4, 'Fe_불순물_pct': 4, 'Al_불순물_pct': 5}\n"
    "blend_prediction = predict_blend_scenario(model, composition_a, composition_c, ratio_a=0.7, total_ore_kg=100, acid_l=600, temperature_c=70, time_hr=4, feature_columns=features_by_target[target])\n"
    "print(f'가상 입력 예시 A:C = 7:3 혼합 예상 {target}: {blend_prediction:.2f}%')"
))

cells.append(nbf.v4.new_markdown_cell("## 10. 예측값 비교 및 황산량 시나리오\n\n- 저장한 최종 모델의 평가 결과와 공정조건별 예측 확인함\n- 모델 해석은 인과관계를 의미하지 않음\n- 현재 공식 실행에서 Test를 선택에 쓰는 경로는 발견되지 않았고, 과거 피처 선택에 현재 Test 배치가 쓰였는지는 미확인. 미래 배치는 시간순 미평가이며, 7:3 혼합은 실측 미검증이다."))
cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, target in zip(axes, TARGET_COLUMNS):
    predictions = pd.read_csv(MODEL_DIR / f'{target}_predictions.csv')
    ax.scatter(predictions['actual'], predictions['prediction'], alpha=0.7)
    ax.plot([0, 100], [0, 100], '--', color='gray')
    ax.set(xlim=(0, 100), ylim=(0, 100), xlabel='실제 회수율 (%)', ylabel='예측 회수율 (%)', title=target)
fig.tight_layout()
fig.savefig(FIGURES_DIR / 'actual_vs_predicted.png')
plt.show()
for target in TARGET_COLUMNS:
    display(predict_process_scenario(interpretation_models[target], base_row,
        feature='황산투입량_L', values=[400, 600, 800, 1000, 1200, 1500],
        feature_columns=features_by_target[target]))
"""))
nb["cells"] = cells

Path("notebooks").mkdir(exist_ok=True)
with open("notebooks/02_model_training.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("노트북 생성 완료: notebooks/02_model_training.ipynb")
