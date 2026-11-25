# 실행 가이드

프로젝트 개요·데이터 정책·방법론은 [README](../README.md) 참고. 이 문서는 실행 절차만 다룸.

아래 터미널 명령은 모두 저장소 루트(`capstone/`)에서 실행함.

## 1. 환경 설치

**Python 3.12.7 권장.** 다른 작업에 쓰는 conda 환경(예: PyTorch용)과 섞이면 패키지 버전 충돌로 저장된 모델(`.joblib`)을 못 불러오는 문제가 생기므로, 이 프로젝트 전용 가상환경을 따로 두는 걸 권장함.

```bash
conda create -n capstone python=3.12.7 -y
conda activate capstone
pip install -r requirements.txt
```

**필수 라이브러리** (`requirements.txt`):

| 분류 | 패키지 |
|---|---|
| 데이터 처리 | `pandas`, `numpy`, `scipy` |
| 모델링 | `scikit-learn`, `xgboost`, `lightgbm`, `shap` |
| 시각화 | `matplotlib`, `seaborn`, `missingno` |
| 입출력 | `openpyxl`(xlsx), `xlrd`(xls) |
| 실행·검증 | `pytest`, `jupyter`, `nbformat`, `nbconvert` |

노트북을 Jupyter에서 실행할 거면 이 환경을 커널로 등록해둠(이름이 겹치지 않게 지정):

```bash
pip install ipykernel
python -m ipykernel install --user --name capstone --display-name "Python (capstone)"
```

### 학습 후 생성된 모델을 재사용할 때의 고정 환경

`requirements.txt`는 개발용 최소 버전 목록으로 유지함. **기본은 학습한 환경에서 모델을 그대로 재사용하는 것임.** `requirements-submission.txt` 고정 환경은 그 파일에 기록된 패키지 버전(Python 3.12.7 / Linux x86_64 기준)으로 생성된 모델에 한정된 선택 절차임 — 모델 로딩·Module 3 후보 예측에 필요한 패키지와 전이 의존성을 고정함. `requirements.txt`로 새로 학습하면 설치되는 패키지 버전이 달라질 수 있어, 그렇게 만든 모델이 이 고정 환경에서 항상 재사용된다고 보장할 수 없음. 노트북·SHAP 도구의 전체 개발 환경을 고정한 파일도 아님.

```bash
# Python 3.12.7을 활성화한 상태에서 프로젝트 루트에서 실행
python -m venv /tmp/capstone-submission
/tmp/capstone-submission/bin/python -m pip install -r requirements-submission.txt
/tmp/capstone-submission/bin/python -m pip check
```

저장된 모델을 다시 불러와 예측이 일치하는지는 Module 2 재실행 시 자동으로 확인함(아래 "3. Module 2 학습" 참고). 교차환경 수치 비교는 이 저장소의 기본 실행·검증 절차에 포함하지 않음.

## 2. 제공 데이터 확인

저장소를 새로 받으면 원본 데이터와 학습된 모델은 없지만, `data/processed/pre_impute_data.csv`·`feature_selection.json`(Module 1 산출물)은 이미 포함되어 있음. 그래서 기본 경로는 Module 1 없이 Module 2부터 실행함:

- `pre_impute_data.csv`의 SHA256이 `feature_selection.json`에 기록된 `data_sha256`과 일치해야 함(둘 중 하나만 바뀌면 Module 2가 거부함) — Module 2 실행 시 자동으로 검사하지만, 직접 확인하려면:

```bash
sha256sum data/processed/pre_impute_data.csv
python3 -c "import json; print(json.load(open('data/processed/feature_selection.json'))['data_sha256'])"
```

  두 출력이 같아야 함 — 저장소를 그대로 받았다면 이미 일치함
- `cleaned_features.csv`(품질 확인용)와 학습된 모델(`.joblib`)은 포함되지 않음 — 필요하면 아래 Module 1·2를 실행해 생성함

### 원본 데이터를 가진 경우 (선택) — Module 1부터 실행

원본 데이터는 과제 규정상 리포지토리에 포함·배포하지 않음. 원본 엑셀 파일(시트명 `원료분석_원본데이터`)을 직접 가진 경우에만 Module 1부터 다시 실행함:

```bash
python -m module1_preprocessing.pipeline          # 원본 → data/processed
```

Module 1 기본 명령은 배포용 XLSX 경로를 사용함 — 다른 파일은 아래 `run()` 예시로 지정함:

```python
from module1_preprocessing.pipeline import run

run("/path/to/input.xls", "data/processed/cleaned_features.csv")
```

이미 저장된 대치 전 데이터에서 고정 split을 유지하면서 피처 선택만 갱신할 수 있음(Test 평가 없음):

```python
import pandas as pd
from module1_preprocessing.pipeline import select_and_save_features

df = pd.read_csv("data/processed/pre_impute_data.csv", float_precision="round_trip")
select_and_save_features(df, "data/processed/cleaned_features.csv")
```

이 함수는 선택 결과와 대치 전 CSV를 저장하며 품질 확인용 `cleaned_features.csv`는 변경하지 않음.

Module 1 검증: `python -m pytest module1_preprocessing/tests -q` — 원본 없이 합성 입력으로 실행함.

## 3. Module 2 학습

```bash
python -m module2_prediction.pipeline
```

- `module2_prediction/reports/final/`에 CV 비교표·선정 근거·평가 결과·타깃별 모델 저장함
- 모델 파일에 개발 데이터의 고정 대치 참조 데이터와 입력 순서 포함함(사전 계산한 통계가 아니라, 예측 시점에 대치 기준을 다시 계산할 참조용 원본 데이터임)
- `artifacts.load_bundle()`과 `artifacts.predict_bundle()`로 새 입력 예측 가능함
- 저장한 모델을 다시 불러와 예측 일치 검증함
- `notebooks/02_model_training.ipynb`에서 개발 데이터의 고정 배경 표본으로 Test 예측의 SHAP를 계산함. SHAP 배경 배치 목록·배치별 SHAP·중요도·상위 인자 요약은 `reports/final/`, 그림은 `reports/figures/`에 저장함. CLI는 모델·평가만 생성하며 SHAP는 노트북 실행 필요

**오늘 입고된 원료의 혼합 예측** — `predict_blend_scenario`는 두 원료의 실제 분석값(wt%)과 이번 배치의 공정조건을 받음. 두 분석값과 투입 질량은 동일한 건량/습량 기준이어야 함. `ratio_a=0.7`은 질량 기준 7:3임.

```python
from module2_prediction.artifacts import load_bundle
from module2_prediction.scenario import predict_blend_scenario

bundle = load_bundle("module2_prediction/reports/final/Ni_회수율_pct.joblib")

# composition_a, composition_b: COMPOSITION_COLUMNS 7개 성분(wt%)을 담은 dict 또는 Series. 실제 분석값으로 교체함.
composition_a = {"Ni_함량_pct": 20.0, "Co_함량_pct": 8.0, "Li_함량_pct": 4.0, "Mn_함량_pct": 10.0,
                  "Cu_불순물_pct": 1.0, "Fe_불순물_pct": 1.0, "Al_불순물_pct": 1.0}
composition_b = {"Ni_함량_pct": 15.0, "Co_함량_pct": 6.0, "Li_함량_pct": 3.0, "Mn_함량_pct": 8.0,
                  "Cu_불순물_pct": 1.5, "Fe_불순물_pct": 1.5, "Al_불순물_pct": 1.5}

prediction = predict_blend_scenario(
    bundle["model"], composition_a, composition_b, ratio_a=0.7,
    total_ore_kg=100, acid_l=600, temperature_c=70, time_hr=4,
    feature_columns=bundle["features"],
)
```

조성만 혼합비로 가중평균함. 운전조건은 입력한 값으로 유지하고 파생변수는 다시 계산함. 예시 공정조건은 실제 운전값으로 교체해야 함. 혼합 배치에 대한 정확도는 실제 실험으로 검증해야 함.

Module 2 검증: `python -m pytest module2_prediction/tests -q`

## 4. Module 3 추천

```bash
python -m module3_optimization.pipeline --input module3_optimization/examples/blend_7_3.json --output-dir module3_optimization/reports/blend_7_3
```

- Module 2 저장 모델이 먼저 필요함 — 모델 재학습 없이 추천에 사용함
- 입력 JSON의 `feed`에 조성 7종(wt%)과 `원료투입량_kg` 입력함
- `current_controls`는 비교할 현재 황산량·온도·시간이며 선택 사항임
- `targets` 생략 시 Ni 95%, Co 90% 적용함
- `weights`는 황산량·온도·시간 순서의 비음수 가중치이며 기본값은 동일 가중치임
- 황산 400–1,500 L, 온도 40–95°C, 시간 1–8 h에서 탐색함
- `recommendation.json`, `candidates.csv`, `search_comparison.csv`, `recovery_surface.png` 저장함
- `predicted_feasible`은 예측상 목표 충족, `target_not_found`는 탐색 예산 내 미발견임
- examples의 A·C·7:3은 가상 조성 예제이며 실제 관측 결과로 보고하지 않음

**Test 배치 실행 예시** — 결측 없는 Test 배치 전체를 데모 입력으로 돌려 추천·재예측을 확인함(Test는 입력값으로만 사용하며 모델·설정 재선택에는 쓰지 않음):

```bash
python -m module3_optimization.scripts.test_batch_examples
```

`module3_optimization/reports/test_batches/`에 배치별 입력·추천 결과와 `summary.csv`(등급·실측값·추천값 비교), `metadata.json`(선정 기준·한계) 저장함.

아래 두 가지는 Module 3 CLI 기본 실행 경로·pytest에는 필요하지 않은 추가 분석임. 단 `notebooks/03_process_optimization.ipynb`를 끝까지 실행하려면 필요함(아래 "5. 노트북 전체 실행 준비" 참고).

**등급별 대표 조성과 혼합 비교:**

```bash
python -m module3_optimization.scripts.compare_blends
```

`module3_optimization/reports/representative_blends/`에 조성·추천 결과·비교 그림 저장함.

**B급 원료 투입량·Co 최대화 비교:**

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m module3_optimization.scripts.b_grade_capacity
```

`module3_optimization/reports/b_grade_capacity/`에 결과·후보·처리량별 예상 회수 금속량 저장함.

Module 3 검증: `python -m pytest module3_optimization/tests -q`

### 전체 파이프라인 한 번에 실행

`run_pipeline.py`가 세 모듈의 `run()`을 순서대로 호출하는 단일 진입점임. `--feed-request`를 지정하면 Module 3 추천까지 이어서 실행함. Module 3은 저장된 모델을 재사용하므로 지정하지 않으면 Module 2까지만 실행함.

```bash
python run_pipeline.py --raw-input /path/to/원본파일.xlsx --feed-request module3_optimization/examples/blend_7_3.json
```

저장소 루트에서 `python -m pytest -q`로 세 모듈의 테스트를 실행할 수 있음.

## 5. 노트북 전체 실행 준비

```bash
jupyter notebook notebooks/01_eda_and_validation.ipynb
```

기존 노트북을 열어 위에서부터 실행함 — 생성 스크립트를 먼저 실행할 필요 없음.

- Module 1은 원본 입력 경로와 Excel 시트명을 확인한 뒤 전체 셀을 순서대로 실행함
- Module 2는 Module 1 산출물, Module 3는 Module 2 저장 모델이 필요함
- `01_eda_and_validation.ipynb`는 원본 입력 기준 노트북임. `01_eda_and_validation_processed.ipynb`에는 기존 정제 데이터부터 실행한 표·그림·CV 결과가 저장되어 있음 — 원본 정제 단계의 재실행 결과는 아님. 정제 데이터 기반 노트북은 `python -m module1_preprocessing.scripts.build_notebook --processed`로 생성 가능함

`notebooks/03_process_optimization.ipynb`를 셀 오류 없이 끝까지 실행하려면 아래 산출물이 먼저 있어야 함(순서대로 실행):

```bash
python -m module2_prediction.pipeline
python -m module3_optimization.pipeline --input module3_optimization/examples/a_example.json --output-dir module3_optimization/reports/a_example
python -m module3_optimization.pipeline --input module3_optimization/examples/c_example.json --output-dir module3_optimization/reports/c_example
python -m module3_optimization.pipeline --input module3_optimization/examples/blend_7_3.json --output-dir module3_optimization/reports/blend_7_3
python -m module3_optimization.scripts.test_batch_examples
python -m module3_optimization.scripts.compare_blends
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m module3_optimization.scripts.b_grade_capacity
```

마지막 두 개(`compare_blends`, `b_grade_capacity`)는 Module 3 CLI 기본 경로·pytest에는 필요 없는 추가 분석이지만, 노트북 03의 후반 셀이 그 결과(`representative_blends/`, `b_grade_capacity/`)를 직접 읽으므로 노트북을 전부 실행하려면 필요함.

노트북 구조를 생성 스크립트에서 변경한 경우에만 아래 명령으로 재생성함(재생성 시 기존 출력은 제거되므로 다시 실행·저장해야 함):

```bash
python -m module1_preprocessing.scripts.build_notebook
python -m module2_prediction.scripts.build_notebook
python -m module3_optimization.scripts.build_notebook
```

Jupyter 실행 시 "python3"처럼 이름이 겹치는 커널이 여러 개 있으면 매번 다른 환경이 선택될 수 있음 — `jupyter kernelspec list`로 확인하고, 이 프로젝트용으로 등록한 커널(`capstone`)을 노트북에서 직접 선택함.

## 6. 결과 파일 위치

| 경로 | 생성 명령 | 내용 |
|---|---|---|
| `data/processed/pre_impute_data.csv`, `feature_selection.json` | `module1_preprocessing.pipeline` | 모델링 입력, 선택 피처·CV 설정 (저장소에 기본 포함) |
| `data/processed/cleaned_features.csv` | `module1_preprocessing.pipeline` | 품질 확인용 대치·파생변수 결과 |
| `module2_prediction/reports/final/{Ni,Co}_회수율_pct.joblib` 외 CV·평가 CSV | `module2_prediction.pipeline` | 타깃별 모델, CV 비교표, Test 평가 결과 |
| `module2_prediction/reports/figures/` | `notebooks/02_model_training.ipynb` | SHAP 그림 (CLI는 생성하지 않음) |
| `module3_optimization/reports/{blend_7_3,a_example,c_example}/` | `module3_optimization.pipeline --input ... --output-dir ...` | 추천 결과(`recommendation.json`), 후보(`candidates.csv`), 탐색 비교(`search_comparison.csv`), 회수율 표면(`recovery_surface.png`) |
| `module3_optimization/reports/test_batches/` | `module3_optimization.scripts.test_batch_examples` | Test 배치별 입력·추천, `summary.csv`, `metadata.json` |
| `module3_optimization/reports/representative_blends/` | `module3_optimization.scripts.compare_blends` | 등급별 대표 조성·혼합 비교 결과 |
| `module3_optimization/reports/b_grade_capacity/` | `module3_optimization.scripts.b_grade_capacity` | B급 투입량별 Co 최대화 비교 결과 |
