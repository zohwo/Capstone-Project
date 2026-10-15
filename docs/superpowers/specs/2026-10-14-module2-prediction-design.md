# 모듈 2: 회수율 예측 모델 — 레포 재구조화 + 모듈 2 설계

- 작성일: 2026-10-14
- 배경: 과제 최종 요구 레포 구조(`data/`, `module1_preprocessing/`, `module2_prediction/`, `module3_optimization/`, `notebooks/`, `docs/`)에 맞춰, 지금까지 `src/`에 있던 모듈 1 코드를 `module1_preprocessing/`으로 옮기고, 그 패턴으로 `module2_prediction/`을 새로 만든다.

## 1. 목표

지금까지 노트북에서만 탐색적으로 진행한 모델 학습(RandomForest/XGBoost/LightGBM 비교)을, 모듈 1과 같은 "재사용 가능한 `*.py` 모듈 + 이를 호출하는 노트북" 구조로 정식화한다. 범위:

1. 모델 학습
2. 하이퍼파라미터 튜닝
3. Validation 기반 최종 모델(모델 종류 + 하이퍼파라미터) 선정
4. Test 평가 (1회)
5. Feature Importance / SHAP 분석
6. 공정 조건별 변화 분석 (partial dependence — 특정 변수를 바꿔가며 예측값 변화 확인)
7. 7:3 혼합 시나리오(A급+C급) 회수율 예측

## 2. 모듈 간 경계

- **모듈 2는 모듈 1의 코드를 import하지 않는다.** `data/processed/cleaned_features.csv`(모듈 1의 산출물, `split` 컬럼으로 train/val/test 구분됨)만 읽어서 시작 — 완전히 독립된 모듈
- **모듈 2는 feature 선택을 다시 하지 않는다.** 모듈 1이 ablation test로 이미 확정한 feature set(`base_plus_derived` = 조성 7종 + 공정조건 2종 + 파생변수 4종)을 그대로 받아, **모델 종류·하이퍼파라미터만** 탐색한다. (RF/XGBoost/LightGBM 모두 트리 기반 앙상블이라 feature를 다루는 방식이 유사해, feature 재검증이 불필요하다고 판단)

## 3. 레포 재구조화 (모듈 1 마이그레이션)

| 기존 | 변경 후 |
|---|---|
| `src/*.py` | `module1_preprocessing/*.py` |
| `tests/*.py` | `module1_preprocessing/tests/*.py` |
| `scripts/build_notebook.py` | `module1_preprocessing/scripts/build_notebook.py` |
| `reports/figures/` | `module1_preprocessing/reports/figures/` |

- 모든 `from src.X import Y` → `from module1_preprocessing.X import Y`로 변경 (소스 코드, 테스트, 노트북 생성 스크립트 전체)
- `pytest.ini`의 `testpaths`를 양쪽 모듈의 `tests/`를 찾도록 수정
- `.gitignore`의 `reports/figures/` 경로를 모듈별 경로로 수정
- `notebooks/`는 과제 명세대로 **top-level 유지** (모듈 폴더 안에 안 넣음) — 각 모듈의 `scripts/build_notebook.py`는 결과물을 공유 `notebooks/`에 씀

## 4. `module2_prediction/` 구조

```
module2_prediction/
  config.py            # TARGET_COLUMNS, FEATURE_COLUMNS(모듈1의 base_plus_derived 재정의), 모델 하이퍼파라미터 탐색 범위
  data_loading.py      # cleaned_features.csv 로드, split 컬럼 기준 train/val/test DataFrame 분리
  modeling.py          # RF/XGBoost/LightGBM 모델 후보 정의 + Validation 기준 학습·비교 (타겟별 독립 모델)
  tuning.py            # 모델별 하이퍼파라미터 탐색 (Validation 기준, Test는 안 봄)
  evaluation.py        # 최종 선정 모델로 Train+Validation 재학습 → Test 1회 평가
  interpretation.py    # Feature Importance, SHAP 분석
  scenario.py          # 공정조건별 변화 분석(partial dependence), 7:3 혼합 시나리오 예측
  pipeline.py          # 엔트리포인트: python -m module2_prediction.pipeline
  tests/
  scripts/
    build_notebook.py  # notebooks/02_model_training.ipynb 생성
  reports/figures/
```

### 4.1 핵심 설계 원칙 (모듈 1과 동일하게 유지)

- 모델 종류 선택도 **Validation 성능만으로** 결정 — Test는 모델 선정에 쓰지 않음
- 최종 평가는 선정된 모델을 **Train+Validation으로 재학습**한 뒤 Test에 **딱 한 번** 적용 (모듈 1의 `evaluate_final_model`과 동일한 패턴)
- 타겟(Ni_회수율_pct, Co_회수율_pct)마다 완전히 독립된 모델을 학습 — 교수님 피드백("모델 1/모델 2 분리") 반영

### 4.2 아직 정의 필요 (구현 계획 단계에서 구체화)

- "7:3 혼합 시나리오"의 feature 값 구성 방법 — 예: A등급·C등급 그룹별 중앙값(또는 평균)을 7:3 가중평균해서 가상의 입력 행을 만드는 방식(안)
- 하이퍼파라미터 탐색 범위(그리드 크기) — 데이터가 작아(Train 294행) 과도한 탐색은 지양

## 5. 범위 밖

- 모듈 3(공정 최적화)은 이번 설계에 포함하지 않음
- 모델 서빙/API, pickle 저장은 포함하지 않음 — 평가·해석까지만
