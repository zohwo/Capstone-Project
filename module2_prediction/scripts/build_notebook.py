from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell(
    "# 모듈 2: 회수율 예측 모델 학습·평가\n\n"
    "- 모듈 1의 산출물(`cleaned_features.csv`)만 읽어서 시작, 모듈 1 코드는 import하지 않음\n"
    "- feature는 모듈 1에서 이미 확정됨(`base_plus_derived`) — 모듈 2는 모델 종류·하이퍼파라미터만 탐색"
))

cells.append(nbf.v4.new_code_cell(
    "import sys\n"
    "sys.path.append('..')\n\n"
    "import pandas as pd\n"
    "from pathlib import Path\n\n"
    "from module2_prediction.config import FEATURE_COLUMNS, TARGET_COLUMNS\n"
    "from module2_prediction.data_loading import load_cleaned_features, split_by_column\n"
    "from module2_prediction.modeling import compare_models, MODEL_FACTORIES\n"
    "from module2_prediction.tuning import tune_models\n"
    "from module2_prediction.evaluation import evaluate_final_model\n"
    "from module2_prediction.interpretation import get_feature_importance, compute_shap_values\n"
    "from module2_prediction.scenario import partial_dependence, predict_blend_scenario\n\n"
    "CLEANED_DATA_PATH = Path('../data/processed/cleaned_features.csv')\n"
    "FIGURES_DIR = Path('../module2_prediction/reports/figures')\n"
    "FIGURES_DIR.mkdir(parents=True, exist_ok=True)"
))

cells.append(nbf.v4.new_markdown_cell("## 1. 데이터 로드"))

cells.append(nbf.v4.new_code_cell(
    "df = load_cleaned_features(CLEANED_DATA_PATH)\n"
    "train_df, val_df, test_df = split_by_column(df)\n"
    "print(f'Train: {len(train_df)}행 / Validation: {len(val_df)}행 / Test: {len(test_df)}행')"
))

cells.append(nbf.v4.new_markdown_cell(
    "## 2. 모델 비교 (기본값, Validation 기준)\n\n"
    "RandomForest, XGBoost, LightGBM을 동일한 feature set으로 비교한다. Test는 안 본다."
))

cells.append(nbf.v4.new_code_cell(
    "default_results = compare_models(train_df, val_df)\n"
    "pd.DataFrame(default_results).T"
))

cells.append(nbf.v4.new_markdown_cell(
    "## 3. 하이퍼파라미터 튜닝 (Validation 기준)\n\n"
    "`config.MODEL_HYPERPARAMETER_GRIDS`의 각 조합을 Validation으로 평가해 모델·타겟별 최적값을 고른다."
))

cells.append(nbf.v4.new_code_cell(
    "tuned = tune_models(train_df, val_df)\n"
    "pd.DataFrame({key: v['metrics'] for key, v in tuned.items()}).T"
))

cells.append(nbf.v4.new_markdown_cell("## 4. 최종 모델 선정 (타겟별 Validation R² 최고 모델)"))

cells.append(nbf.v4.new_code_cell(
    "best_per_target = {\n"
    "    target: max(\n"
    "        (key for key in tuned if key[1] == target),\n"
    "        key=lambda key: tuned[key]['metrics']['val_r2']\n"
    "    )\n"
    "    for target in TARGET_COLUMNS\n"
    "}\n"
    "for target, (model_name, _) in best_per_target.items():\n"
    "    p = tuned[(model_name, target)]\n"
    "    print(f'{target}: {model_name}, params={p[\"params\"]}, val_r2={p[\"metrics\"][\"val_r2\"]:.4f}')"
))

cells.append(nbf.v4.new_markdown_cell("## 5. 최종 Test 평가 (1회, Train+Validation 재학습)"))

cells.append(nbf.v4.new_code_cell(
    "final_results = {}\n"
    "for target, (model_name, _) in best_per_target.items():\n"
    "    params = tuned[(model_name, target)]['params']\n"
    "    result = evaluate_final_model(\n"
    "        train_df, val_df, test_df,\n"
    "        model_name=model_name, model_params=params,\n"
    "        target_columns=[target],\n"
    "    )\n"
    "    final_results[target] = result[target]\n\n"
    "pd.DataFrame(final_results).T"
))

cells.append(nbf.v4.new_markdown_cell(
    "## 6. Feature Importance / SHAP\n\n"
    "최종 선정 모델을 Train+Validation으로 다시 학습해 해석용으로 사용한다."
))

cells.append(nbf.v4.new_code_cell(
    "train_val_df = pd.concat([train_df, val_df], ignore_index=True)\n\n"
    "interpretation_models = {}\n"
    "for target, (model_name, _) in best_per_target.items():\n"
    "    params = tuned[(model_name, target)]['params']\n"
    "    model = MODEL_FACTORIES[model_name](**params)\n"
    "    model.fit(train_val_df[FEATURE_COLUMNS], train_val_df[target])\n"
    "    interpretation_models[target] = model\n\n"
    "    importance = get_feature_importance(model)\n"
    "    print(f'--- {target} ({model_name}) Feature Importance ---')\n"
    "    print(importance)"
))

cells.append(nbf.v4.new_code_cell(
    "import shap\n"
    "import matplotlib.pyplot as plt\n\n"
    "target = TARGET_COLUMNS[0]\n"
    "model = interpretation_models[target]\n"
    "shap_values = compute_shap_values(model, train_val_df[FEATURE_COLUMNS])\n"
    "shap.summary_plot(shap_values, train_val_df[FEATURE_COLUMNS], show=False)\n"
    "plt.savefig(FIGURES_DIR / 'shap_summary.png', bbox_inches='tight')\n"
    "plt.close()"
))

cells.append(nbf.v4.new_markdown_cell(
    "## 7. 공정조건별 변화 분석 (Partial Dependence)\n\n"
    "침출온도_C를 바꿔가며 예측값이 어떻게 변하는지 확인한다 (나머지 변수는 고정)."
))

cells.append(nbf.v4.new_code_cell(
    "target = TARGET_COLUMNS[0]\n"
    "model = interpretation_models[target]\n"
    "base_row = train_val_df.iloc[0]\n\n"
    "pd_result = partial_dependence(model, base_row, feature='침출온도_C', values=list(range(40, 96, 5)))\n"
    "pd_result.plot(x='침출온도_C', y='prediction', marker='o', title=f'{target} vs 침출온도_C')"
))

cells.append(nbf.v4.new_markdown_cell(
    "## 8. 7:3 혼합 시나리오 예측\n\n"
    "A급 원료 70% + C급 원료 30%를 혼합했을 때의 예상 Ni 회수율을 예측한다."
))

cells.append(nbf.v4.new_code_cell(
    "target = TARGET_COLUMNS[0]\n"
    "model = interpretation_models[target]\n\n"
    "blend_prediction = predict_blend_scenario(model, train_val_df, grade_a='A', grade_b='C', ratio_a=0.7)\n"
    "print(f'A:C = 7:3 혼합 예상 {target}: {blend_prediction:.2f}%')"
))

nb["cells"] = cells

Path("notebooks").mkdir(exist_ok=True)
with open("notebooks/02_model_training.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("노트북 생성 완료: notebooks/02_model_training.ipynb")
