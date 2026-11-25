# 블랙파우더 성분 기반 금속 회수율 예측

영풍그룹 블랙파우더 원료의 성분 분석값·공정조건으로 Ni·Co 침출 회수율을 예측하고, 예측상 목표 회수율을 만족하는 운전조건을 추천하는 프로젝트. 3개 모듈이 순서대로 이어짐.

| 모듈 | 역할 |
|---|---|
| **Module 1 — 전처리** | 기본 정제·중복 처리, 대치·파생변수 생성, 품질 검증·시각화, 개발 데이터 CV 기반 피처 선택 |
| **Module 2 — 예측** | Module 1의 전처리를 fold별로 호출하여 모델·하이퍼파라미터 선택, 개발 데이터 전체로 전처리·모델 재학습, Test 최종 평가 |
| **Module 3 — 최적화** | Module 2가 저장한 모델을 재사용해 목표 회수율을 만족하는 운전조건(황산량·온도·시간)을 탐색·추천 |

- **모델 선정 지표:** Ni/Co 각각 fold별 RMSE의 평균 최소화, fold 표준편차 함께 기록함. MAE·R²는 보조 지표이며 RMSE·MAE 단위는 회수율 %p임
- Module 1 선택 파일의 MAE는 기존 특징 검토 기록이며 Module 2 모델 선정은 RMSE 기준임

## 환경 설정

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

### 제출 모델 재현을 위한 고정 환경

`requirements.txt`는 개발용 최소 버전 목록으로 유지함. 제출된 `.joblib`을 재사용할 때는 **Python 3.12.7 / Linux x86_64**와 `requirements-submission.txt`를 사용함. 이 파일은 현재 번들의 생성 환경 기록과 재로드에 성공한 환경을 기준으로, 모델 로딩·Module 3 후보 예측에 필요한 패키지와 전이 의존성을 고정함. 노트북·SHAP 도구의 전체 개발 환경을 고정한 파일은 아님. 생성 환경 기록은 `module2_prediction/reports/final/environment.json`에 있음.

```bash
# Python 3.12.7을 활성화한 상태에서 프로젝트 루트에서 실행
python -m venv /tmp/capstone-submission
/tmp/capstone-submission/bin/python -m pip install -r requirements-submission.txt
/tmp/capstone-submission/bin/python -m pip check
```

저장된 모델을 다시 불러와 예측이 일치하는지는 Module 2 재실행 시 자동으로 확인함(아래 "모듈별 실행 방법" 참고). 교차환경 수치 비교는 과제 범위에서 별도로 요구하지 않아 수행하지 않음.

## 데이터셋 설명과 유의사항

### 원본 데이터

- 위치: `data` 폴더의 `raw` 하위(엑셀 파일, 시트명 `원료분석_원본데이터`). 과제 규정상 원본 데이터는 리포지토리에 포함·배포하지 않음(`data` 전체가 `.gitignore` 대상임) — 저장소만 새로 받으면 원본과 `data/processed` 산출물이 모두 없으므로, 원본 파일을 가진 사람이 Module 1부터 실행해야 함
- Module 1은 사용자가 제공한 원본을 직접 읽어 전처리함. Module 2·3은 Module 1이 만든 `data/processed` 산출물만 사용하며 원본에 다시 접근하지 않음
- 입력 형식은 CSV·XLSX·XLS 지원함 — XLSX는 `openpyxl`, XLS는 `xlrd` 사용함
- 1행 = 1배치. 현재 `pre_impute_data.csv` 기준 420개 배치, 층화 분할로 Train 294 / Validation 63 / Test 63으로 고정함

### 주요 컬럼

| 구분 | 컬럼 |
|---|---|
| 조성(wt%) | `Ni_함량_pct`, `Co_함량_pct`, `Li_함량_pct`, `Mn_함량_pct`, `Cu_불순물_pct`, `Fe_불순물_pct`, `Al_불순물_pct` |
| 공정조건 | `침출온도_C`, `침출시간_hr`, `원료투입량_kg`, `황산투입량_L` |
| 범주 | `원료등급`(A/B/C) — 대치·층화 분할에 사용, 최종 모델 입력에서는 제외함 |
| 타깃 | `Ni_회수율_pct`, `Co_회수율_pct` |
| 파생변수 | `acid_to_ore_ratio`, `leach_intensity`, `total_impurity_pct`, `feed_impurity_g_per_l` |

- 정상범위와 문헌 기반 보존 예외는 `module1_preprocessing/config.py`의 `NORMAL_RANGES`/`REFERENCE_EXCEPTION_RANGES`에 정의함. 보존 예외의 문헌 보고값·원료 종류·Table 위치·적용 범위는 [보존 예외 출처](docs/reference_ranges.md)에 정리함
- 회수율 정답은 대치하거나 자르지 않음. 결측·무한대·0~100% 범위 위반은 배치 식별자와 함께 오류로 보고함
- Excel 날짜 객체는 보존함. 날짜·월은 모델 입력에 사용하지 않음

### Module 1 산출물 (`data/processed`)

| 파일 | 용도 |
|---|---|
| `pre_impute_data.csv` | 모델링 입력. 결측치를 보존한 원본 입력 컬럼과 고정 `split` 포함 |
| `feature_selection.json` | 타깃별 피처·선택 MAE·CV 설정·입력 파일 SHA256 |
| `cleaned_features.csv` | Train 기준 대치·파생변수 생성 결과. 품질 확인용이며 CV 입력으로 사용하지 않음 |

모듈 2는 CSV와 선택 결과의 해시를 확인하여 서로 다른 실행의 산출물 혼용을 차단함. 새 데이터를 사용하면 모듈 1부터 다시 실행함.

이상치 처리 이력은 새로 생성하는 `pre_impute_data.csv`와 `cleaned_features.csv`에 함께 저장함.
- `{컬럼}_outlier_reason`: 원래 결측(`original_missing`), 변환하지 않음(`not_replaced`), 보존 범위 이탈(`outside_accepted_range`), 무한대(`non_finite`)를 구분함
- `{컬럼}_outlier_original_value`: 이상치 처리로 결측으로 바꾼 값만 보존함. 이후 대치해도 이력은 유지되며 모델 입력에는 사용하지 않음
- 기존 정제 파일에서 이미 제거된 원값은 복원할 수 없으므로, 이력은 원본 입력부터 다시 정제할 때 기록됨

### 피처 구성

- `base`: 조성 7개 + 침출온도·시간 + 원료투입량·황산투입량(11개)
- `base_plus_derived`: 원변수 + 파생변수 4종으로 총 15개임
- `base_plus_grade`: 원변수 + 원료등급으로 총 12개임
- `all_features`: 원변수 + 파생변수 4종 + 원료등급으로 총 16개임
- **최종 입력은 Ni·Co 모두 원변수 11종 + 파생변수 4종으로 통일함**
- 공통 4종은 3개 시드(42·17·83)의 반복 5-fold CV와 입력 구성의 단순성을 고려해 채택함
- 실행 시 저장되는 CV 점수는 시드 42의 재평가 결과이며, 채택 근거인 반복 CV 평균과 구분함
- 원료등급 포함 조합은 fold별 학습 데이터 기준 원-핫 인코딩 후 비교함
- 피처 선택 산출물 버전은 2임. 이전 버전은 현재 후보 구성으로 다시 선택해야 함

### 데이터 흐름과 Test 누수 방지

1. 사전에 정한 규칙으로 스키마·중복 배치를 정리하고, 정상범위와 문헌 기반 보존 예외 밖의 값을 결측으로 변환함
2. 원료등급별 무작위 층화 분할로 Train/Validation/Test를 고정함
3. 모듈 1은 Train+Validation에서 fold별 대치 후 네 조합(원변수 / 원변수+파생변수 / 원변수+원료등급 / 원변수+파생변수+원료등급)을 동일한 LightGBM 5-fold CV로 비교함. 최저 오차 조합은 참고용으로 기록하고, 반복 CV 비교 후 채택한 원변수+파생변수 공통 구성을 저장함. **Test는 비교·선택에서 제외함**
4. 모듈 2는 대치 전 데이터를 읽고 같은 선택 피처로 모델을 비교함. 각 fold의 학습 부분으로만 대치 기준을 계산하고, 대치 → 파생변수 → 모델 학습 순서로 실행함. ElasticNet/SVR 표준화도 fold 안에서 학습함
5. 모든 선택을 확정한 후 Train+Validation 전체로 대치 기준과 모델을 다시 학습하고 Test에 적용함. **모듈 1은 Test 성능을 계산하지 않음**

유의사항:
- CV RMSE는 모델·하이퍼파라미터 선택에 사용한 점수이므로 독립 평가 점수로 해석하지 않음. 현재 실행의 피처 비교 결과는 참고 기록이며 튜닝에는 고정 피처를 사용함
- Test 결과·SHAP를 본 후 피처나 모델을 재선택하면 독립 평가가 아니게 됨
- 무작위 층화 평가는 동일 모집단의 배치를 가정함. 미래 배치 성능을 주장하려면 시간순 평가를 추가해야 함
- 날짜 보간은 품질 확인용임. 날짜·월은 모델 선택 후보에서 제외함

## 모듈별 실행 방법과 예시

### 빠른 시작

저장소를 새로 받으면 원본 데이터·전처리 산출물·학습된 모델이 전혀 없는 상태임(위 환경 설정·데이터셋 설명 참고). 원본 파일을 가진 경우 Module 1부터 전체를 실행함:

```bash
python -m module1_preprocessing.pipeline          # 원본 → data/processed
python -m module2_prediction.pipeline             # data/processed → 모델 저장
python -m module3_optimization.pipeline --input module3_optimization/examples/blend_7_3.json --output-dir module3_optimization/reports/blend_7_3   # 저장된 모델 → 추천
python -m pytest -q
```

- 프로젝트 루트에서 실행함
- Module 1 기본 명령은 배포용 XLSX 경로를 사용함 — 다른 파일은 아래 `run()` 예시로 지정함
- 원본 없이 `data/processed`(위 "Module 1 산출물" 참고)만 받았다면 Module 1은 건너뛰고 Module 2부터 시작해도 됨. 단 `pre_impute_data.csv`의 SHA256이 `feature_selection.json`에 기록된 `data_sha256`과 일치해야 함(둘 중 하나만 바뀌면 Module 2가 거부함)
- 학습된 모델 번들(`.joblib`)까지 이미 받았고 추천만 해보려는 거면 Module 2도 다시 실행할 필요 없이 **Module 3부터** 바로 실행함 — Module 2를 다시 실행하면 모델을 재학습해서 기존 저장 모델을 덮어씀(재학습과 기존 모델 재사용은 다른 작업임)

### Module 1 — 전처리

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

### Module 2 — 예측 모델 저장과 재사용

```bash
python -m module2_prediction.pipeline
```

- `module2_prediction/reports/final/`에 CV 비교표·선정 근거·평가 결과·타깃별 모델 저장함
- 모델 파일에 개발 데이터의 고정 대치 참조 데이터와 입력 순서 포함함(사전 계산한 통계가 아니라, 예측 시점에 대치 기준을 다시 계산할 참조용 원본 데이터임)
- `artifacts.load_bundle()`과 `artifacts.predict_bundle()`로 새 입력 예측 가능함
- 저장한 모델을 다시 불러와 예측 일치 검증함
- Module 1의 split 유지. Train+Validation 내부 CV에서 평균 검증 RMSE가 최소인 모델을 타깃별 선정하며 MAE·R²는 보조 지표임
- `cv_comparison.csv`는 모델별 최적 후보의 학습/검증 RMSE 평균·표준편차와 차이를, `cv_fold_scores.csv`는 해당 후보의 fold별 오차를 저장함. 학습 오차는 해당 fold 모델의 적합 오차이며 선정 지표가 아님
- CV RMSE는 모델·하이퍼파라미터 선택에 사용한 점수이므로 독립 평가 점수로 해석하지 않음. Test 결과는 피처·모델 재선정에 사용하지 않음
- 현재 공식 실행에서 Test를 선택에 쓰는 경로는 발견되지 않았고, 과거 피처 선택에 현재 Test 배치가 쓰였는지는 미확인. 미래 배치는 시간순 미평가이며, 7:3 혼합은 실측 미검증임
- `notebooks/02_model_training.ipynb`에서 개발 데이터의 고정 배경 표본으로 Test 예측의 SHAP를 계산함. 평균 절대 SHAP 중요도·beeswarm·상위 인자별 기여 비교를 저장하고 공정 해석의 가설과 한계를 설명함
- SHAP 배경 배치 목록·배치별 SHAP·중요도·상위 인자 요약은 `reports/final/`, 그림은 `reports/figures/`에 저장함. CLI는 모델·평가만 생성하며 SHAP는 노트북 실행 필요
- CV 표준편차는 분할별 성능 변동임. 개별 배치 예측구간과 포함률 평가는 이번 범위에서 미구현

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

조성만 혼합비로 가중평균함. 운전조건은 입력한 값으로 유지하고 파생변수는 다시 계산함. 예시 공정조건은 실제 운전값으로 교체해야 함. 혼합 배치에 대한 정확도는 실제 실험으로 검증함.

Module 2 검증: `python -m pytest module2_prediction/tests -q`

### Module 3 — 공정 레시피 추천

```bash
python -m module3_optimization.pipeline --input module3_optimization/examples/blend_7_3.json --output-dir module3_optimization/reports/blend_7_3
```

- Module 2 저장 모델이 먼저 필요함 — 모델 재학습 없이 추천에 사용함
- 입력 JSON의 `feed`에 조성 7종(wt%)과 `원료투입량_kg` 입력함
- `current_controls`는 비교할 현재 황산량·온도·시간이며 선택 사항임
- `targets` 생략 시 Ni 95%, Co 90% 적용함
- `weights`는 황산량·온도·시간 순서의 비음수 가중치이며 기본값은 동일 가중치임
- 황산 400–1,500 L, 온도 40–95°C, 시간 1–8 h에서 탐색함
- 목표 충족 후보 중 정규화 운전 부담이 낮은 후보 선택함 — 실제 원가 최적화가 아님
- DE·실수형 유전 알고리즘(GA)을 각각 3개 seed로 실행하고 비교용 격자를 평가하며 반올림 후 목표 충족을 다시 검사함
- GA는 개체 36개·60세대, 토너먼트 선택·산술 교차·돌연변이·상위 2개체 보존 적용함
- DE와 GA의 최대 평가 예산은 seed당 2,196개이며 DE 조기 종료 시 실제 평가 수는 줄어듦
- 반올림 단위 1 L·0.1°C·0.1 h는 프로젝트 시연 가정임
- `recommendation.json`, `candidates.csv`, `search_comparison.csv`, `recovery_surface.png` 저장함
- `predicted_feasible`은 예측상 목표 충족, `target_not_found`는 탐색 예산 내 미발견임
- DE 종료 상태와 목표 충족 여부는 별개이며 전역 최적성·실제 공정 달성을 보장하지 않음
- `outside_training_feature_ranges`는 모델 입력별 학습 범위 이탈 목록이며 조합 분포 전체를 검증한 것은 아님
- examples의 A·C·7:3은 가상 조성 예제이며 실제 관측 결과로 보고하지 않음
- `notebooks/03_process_optimization.ipynb`에서 저장 결과 재검증 및 그림 확인 가능함

**Test 배치 실행 예시** — 결측 없는 Test 배치 전체를 데모 입력으로 돌려 추천·재예측을 확인함(Test는 입력값으로만 사용하며 모델·설정 재선택에는 쓰지 않음):

```bash
python -m module3_optimization.scripts.test_batch_examples
```

`module3_optimization/reports/test_batches/`에 배치별 입력·추천 결과와 `summary.csv`(등급·실측값·추천값 비교), `metadata.json`(선정 기준·한계) 저장함.

아래 두 가지는 기본 실행 경로에 필요하지 않은 추가 분석임.

**등급별 대표 조성과 혼합 비교:**

```bash
python -m module3_optimization.scripts.compare_blends
```

- 대치 전 개발 데이터의 등급별 관측 성분 중앙값으로 가상 대표 조성 구성함
- Test와 회수율은 대표 조성 산출에 사용하지 않음
- 원료 140 kg 고정, A·B·C 단독 및 A:B·A:C·B:C 질량비 3:7·5:5·7:3 총 12종 비교함
- 각 조합에서 DE·GA·격자로 운전조건을 별도 탐색함
- `module3_optimization/reports/representative_blends/`에 조성·추천 결과·비교 그림 저장함
- 실측 혼합 실험이 아니므로 실제 회수율·등급 전체의 달성 가능성을 보장하지 않음

**B급 원료 투입량·Co 최대화 비교:**

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m module3_optimization.scripts.b_grade_capacity
```

- 기존 B급 대표 조성·최종 SVR을 유지하고 Ni 95% 이상 제약 아래 Co 회수율 최대화함
- 140 kg·데이터사전 범위에서 목적함수 변경 효과를 비교함
- 원료 81·100·120·140·160·180·199 kg은 모델 입력변수별 학습 범위 안에서 비교함
- 학습 범위와 반올림 단위를 반영한 운전조건은 황산 402~1,495 L·온도 40.1~94.9°C·시간 1.1~7.9 h임
- 기존 추천 조건도 범위 안에 있으면 탐색 후보에 포함함
- `module3_optimization/reports/b_grade_capacity/`에 결과·후보·처리량별 예상 회수 금속량 저장함
- 기본 최적화 정책과 최종 모델은 변경하지 않으며 Test는 사용하지 않음

Module 3 검증: `python -m pytest module3_optimization/tests -q`

### 노트북으로 실행

```bash
jupyter notebook notebooks/01_eda_and_validation.ipynb
```

기존 노트북을 열어 위에서부터 실행함 — 생성 스크립트를 먼저 실행할 필요 없음.

- Module 1은 원본 입력 경로와 Excel 시트명을 확인한 뒤 전체 셀을 순서대로 실행함
- Module 2는 Module 1 산출물, Module 3는 Module 2 저장 모델이 필요함
- CLI는 전처리·선택 결과를 저장하고, Module 1 노트북은 단계별 설명·EDA 그림 생성을 함께 수행함
- Module 1 노트북은 실행 시 주요 그림을 셀 출력에 표시하고 이미지 파일도 저장함
- 원본부터 시작하는 `01_eda_and_validation.ipynb`는 원본 읽기 제한으로 전체 실행 결과 미저장 상태임
- `01_eda_and_validation_processed.ipynb`에는 기존 정제 데이터부터 실행한 표·그림·CV 결과 저장됨 — 원본 정제 단계의 재실행 결과는 아님
- 정제 데이터 기반 노트북은 `python -m module1_preprocessing.scripts.build_notebook --processed`로 생성 가능함
- 생성 스크립트는 해당 노트북의 기존 출력을 지우므로 재생성 후 다시 실행·저장해야 함
- 두 노트북 모두 특징 비교 CV를 한 번 실행하고 저장 단계에서 같은 결과 재사용함
- 공정조건별 예측 회수율 변화 그래프는 Module 2 영향도 분석에서 다룸

노트북 구조를 생성 스크립트에서 변경한 경우에만 아래 명령으로 재생성함:

```bash
python -m module1_preprocessing.scripts.build_notebook
python -m module2_prediction.scripts.build_notebook
python -m module3_optimization.scripts.build_notebook
```

- `notebooks/01_eda_and_validation.ipynb`: 품질 확인·탐색적 ablation·공식 피처 선택과 산출물 저장. Test 평가 없음
- `notebooks/02_model_training.ipynb`: RMSE 기준 모델 선택·Test 평가·중요도/SHAP·공정/혼합 시나리오
- CLI와 노트북은 같은 공용 함수를 사용함. 노트북 재생성 시 이전 출력은 제거됨

Jupyter 실행 시 "python3"처럼 이름이 겹치는 커널이 여러 개 있으면 매번 다른 환경이 선택될 수 있음(위 환경 설정 참고) — `jupyter kernelspec list`로 확인하고, 이 프로젝트용으로 등록한 커널(`capstone`)을 노트북에서 직접 선택함.

## 전체 파이프라인 한 번에 실행

`run_pipeline.py`가 세 모듈의 `run()`을 순서대로 호출하는 단일 진입점임. `--feed-request`를 지정하면 Module 3 추천까지 이어서 실행함. Module 3은 저장된 모델을 재사용하므로 지정하지 않으면 Module 2까지만 실행함.

```bash
python run_pipeline.py --raw-input /path/to/원본파일.xlsx --feed-request module3_optimization/examples/blend_7_3.json
```

Module 1 → Module 2 → Module 3로 이어지는 각 경계에서 무엇을 확인하고 무엇을 기록하는지 정리함.

**Module 1 (전처리·피처 선택)**
- 확인: 충돌 중복 배치ID·0 이하 투입량·정제 후 잔존 결측·범위 위반은 원인을 확인할 수 있는 오류 메시지와 함께 중단함
- 확인: 피처 평가 시점과 저장 시점의 데이터 일관성(해시 대조) — 다르면 재평가를 요구함
- 기록: `pre_impute_data.csv`(모델링 입력), `feature_selection.json`(선택 피처·CV 설정·입력 SHA256)

**Module 1 → Module 2**
- 확인: Module 2는 입력 산출물(`pre_impute_data.csv`, `feature_selection.json`)의 해시를 대조해 서로 다른 실행 결과가 섞이지 않았는지 확인함
- 기록: `cv_comparison.csv`·`cv_fold_scores.csv`(모델 선정 근거), 타깃별 `.joblib`

**Module 2 (모델 저장과 재사용)**
- 확인: 저장한 모델을 다시 불러와 Test 예측이 저장 전과 일치하는지 비교함
- 기록(`python -m module2_prediction.pipeline` 실행 시 자동 저장): 모델 번들에 피처 목록·참조 데이터·타깃 포함
- 기록(별도 수동 기록, 자동 생성 아님): `environment.json`(생성 환경 스냅샷). 모델을 다시 학습하면 다시 기록해야 최신 모델과 일치함

**Module 2 → Module 3**
- 확인: Module 3은 로드한 모델의 `target` 메타데이터가 기대한 타깃과 일치하는지 확인함
- 기록: 사용한 모델의 SHA256을 매 추천 결과(`recommendation.json`)에 남겨 추적 가능하게 함(실행을 자동으로 막지는 않음)

**Module 3 (추천)**
- 확인: 입력(조성·운전조건)이 정의된 범위 안인지, 반올림한 운전조건을 재예측한 값이 저장하려는 예측값과 일치하는지 확인함
- 확인: `outcome()`은 목표·예측의 키가 맞지 않거나 목표값이 유효하지 않으면 예외를 내고, 그 외에는 `(valid, feasible, margins)`를 반환함. `optimize()`가 `feasible` 여부로 판정 결과 문자열(`predicted_feasible`/`target_not_found`)을 만듦 — 입력 오류(예외)와 판정 결과(반환값)를 구분함
- 기록: `recommendation.json`(추천 조건·예측·목표 대비 여유·사용 모델 해시), `candidates.csv`(반올림 후 탐색 출처별 중복을 제거하고 재평가한 후보 — 탐색 과정에서 평가한 연속값 전체가 아님)

저장소 루트에서 `python -m pytest -q`로 세 모듈의 테스트를 실행할 수 있음. 이 문서 작성 시점 기준 260개가 통과했으나, 코드가 바뀌면 달라질 수 있으므로 직접 실행해 확인하는 게 정확함.

## 코드 구조와 작성 원칙

- 각 모듈은 설정을 `config.py`에 두고, 전처리·모델링·검증·실행 로직을 역할별 파일로 분리함. 재현·분석 스크립트는 `scripts/`, 동작을 검증하는 테스트는 `tests/`에 둠
- 파이프라인 경계에서 입력 범위와 산출물의 일관성을 검사함. 잘못된 입력이나 서로 맞지 않는 산출물에는 원인을 확인할 수 있는 오류 메시지를 제공함
- 주요 함수에는 역할을 짧게 설명하고, 실험 결과와 메타데이터에는 해석 범위와 검증 한계를 기록함
- 저장소 루트에서 `python -m pytest -q`로 세 모듈의 테스트를 실행할 수 있음
