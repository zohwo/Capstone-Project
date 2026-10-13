# 모듈 1: 블랙파우더 성분 데이터 전처리 파이프라인 설계

- 작성일: 2026-09-28 / 최종 갱신: 2026-10-13 (실제 구현 기준으로 전면 재작성)
- 대상 모듈: 모듈 1 (데이터 가공)
- 원본 데이터: `data/raw/영풍그룹_금속회수율최적화_RawMaterialData_학생배포용.xlsx`
  - 시트 `원료분석_원본데이터`: 424행 16컬럼 (배치 단위 원료 조성/공정조건/회수율 실측값)
  - 시트 `데이터사전`: 16개 컬럼에 대한 설명·단위·정상 범위 정의 (검증 기준으로 사용)

## 1. 배경 및 목표

랩실에서 배치(BP-2025-XXXX)별로 원료(블랙파우더) 성분을 분석한 원본 데이터는 다음 문제를 가진다:

- `분석일자` 컬럼에 6종 이상의 날짜 표기 형식이 혼재 (`2025-01-02`, `2025/01/02`, `03.01.2025`, `20250103`, `Jan 03, 2025`, `01/03/25`)하고 21건 결측
- `원료등급` 컬럼에 표기 불일치 (`A`, `A급`, `a`, `" A "`, `B급`, `b` 등 변형이 실질 3개 등급 A/B/C를 가리킴)
- 조성/불순물 컬럼(Ni/Co/Li/Mn_함량, Cu/Fe/Al_불순물) 7종에 각 6.9~10.7%의 결측
- 공정조건 컬럼(침출온도_C, 침출시간_hr)에 결측 및 물리적으로 불가능한 값(온도 -5.0℃) 포함
- 명백한 입력/센서 오류로 보이는 이상치: `Ni_함량_pct=999.0`(정상범위 8~30), `황산투입량_L=15000`(정상범위 400~1500)
- `배치ID` 420 unique / 424행 → 4건 중복. 전부 분석일자·원료등급·회수율까지 완전히 동일한 복사본으로 확인됨

목표: 이 원본 데이터를 머신러닝(니켈/코발트 회수율 예측) 입력으로 쓸 수 있도록, **재실행 가능하고 형식이 달라도 수동 개입 없이 동일 품질로 정제되는** 파이프라인을 구축하고, 회수율에 영향을 미치는 파생변수를 도메인 지식에 근거해 추가한 뒤, **data leakage 없이** 정제 결과를 정량적으로 검증한다.

배치는 2025년 1~10월에 걸쳐 순차적으로 들어온 시계열성 데이터이므로, 데이터 분할·결측치 대치·성능 평가 전 과정에서 "미래 배치 정보가 과거 배치 처리에 섞이지 않도록" 하는 것이 설계의 핵심 제약이다.

## 2. 산출물 형태

두 가지를 모두 만든다:

1. **재사용 가능한 파이프라인 모듈** (`src/` 이하 파이썬 코드) — 새로운 원본 파일이 들어와도 동일 로직으로 정제·저장
2. **분석 노트북** (`notebooks/01_eda_and_validation.ipynb`) — 파이프라인 함수를 호출해 EDA, 전/후 비교 시각화, ablation test 결과, 심층 인터뷰 3문항에 대한 답변을 정리

## 3. 파이프라인 아키텍처

```
load_raw
  → check_ranges(raw)                         # 정제 전 위반 건수 기록 (전/후 대비용)
  → clean_schema                              # 날짜 표준화·등급 정규화·중복 플래그
  → resolve_duplicate_batches                 # 완전 중복만 제거, 충돌은 플래그 후 보존
  → replace_outliers_with_nan                 # 정상범위 이탈 → NaN
  → check_missing                             # 결측 현황 기록
  → interpolate_analysis_date                 # 분석일자만 분할 전 전체 데이터로 보간
  → temporal_train_val_test_split             # 시간순 70/15/15 고정 분할 (1회만 계산)
  → impute_missing_from_reference(train, 전체) # Train 통계로 전체 결측 대치 (leak-safe)
  → engineer_features                         # 파생변수 8종
  → check_missing / check_ranges (assert)     # 정제 후 위반 시 즉시 중단
  → run_ablation_test(train, val)             # Validation만으로 feature 선택
  → evaluate_final_model(train+val, test)     # 선택된 feature로 Test 1회 최종 평가
  → CSV 저장
```

모듈 구성 (`src/` 아래):

| 파일 | 책임 |
|---|---|
| `config.py` | `NORMAL_RANGES`, `COMPOSITION_COLUMNS`, `PROCESS_COLUMNS`, `TARGET_COLUMNS`, `DERIVED_FEATURES`, `RAW_SHEET_NAME` 등 공유 상수 |
| `preprocessing.py` | `load_raw(path)`: xlsx/csv 확장자 분기 로딩. `clean_schema(df)`: 날짜 파싱·표준화, 범주형(`원료등급`) 정규화, 중복 배치ID 플래그(`is_duplicate_batch`). `resolve_duplicate_batches(df)`: 전체 컬럼까지 완전 일치하는 중복만 제거, 값이 다른 충돌은 `is_conflicting_duplicate`로 표시해 보존 |
| `splitting.py` | `temporal_train_val_test_split(df, train_frac, val_frac)`: 배치ID 순번 기준 시간순 고정 분할. fraction이 양수가 아니거나 합이 1 이상이면 `ValueError` |
| `imputation.py` | `replace_outliers_with_nan(df)`: 정상범위 밖 값을 NaN으로 치환. `interpolate_analysis_date(df)`: 배치ID 순서 기반 분석일자 보간. `impute_missing_from_reference(reference_df, target_df)`: reference 통계(그룹/전체 중앙값, 회귀계수)로만 target의 결측을 채우는 leak-safe 대치 |
| `features.py` | `engineer_features(df)`: §6 파생변수 8종 생성 |
| `validation.py` | `check_ranges`, `check_missing`: 정합성 검증. `plot_missing_before_after`, `plot_distributions`, `plot_correlation_heatmap`, `plot_ablation_results`, `plot_grade_distribution`: 시각화. `run_ablation_test(train_df, val_df, ...)`: Validation만으로 feature 조합 비교. `evaluate_final_model(train_df, val_df, test_df, ...)`: 선택된 feature로 Train+Validation 재학습 후 Test 1회 평가 |
| `pipeline.py` | 위 함수들을 순서대로 호출하는 엔트리포인트. `python -m src.pipeline`로 실행하면 `data/processed/cleaned_features.csv` 생성 |

각 함수는 DataFrame을 입력받아 DataFrame(또는 분할 인덱스/딕셔너리)을 반환하는 순수 함수로 작성하여, 노트북과 파이프라인 스크립트 양쪽에서 동일하게 재사용한다.

## 4. 결측치·이상치 처리 규칙

| 컬럼군 | 처리 | 근거 |
|---|---|---|
| `Ni_함량_pct`(999.0), `침출온도_C`(<0), `황산투입량_L`(>1500 대비 10배 이상) 등 데이터사전 범위 이탈 | `replace_outliers_with_nan`에서 NaN으로 치환 | 물리적으로 불가능한 값은 센서/입력 오류로 판단 |
| 조성 7종 (Ni/Co/Li/Mn_함량, Cu/Fe/Al_불순물) | `impute_missing_from_reference`에서 **reference_df(Train) 기준** 원료등급(A/B/C) 그룹별 중앙값 대치, 해당 그룹 통계가 없으면 전체 중앙값으로 대체 | 원료등급별 조성 분포가 다르므로 전체 중앙값 대치는 편향 유발. 평균이 아닌 중앙값은 잔존 이상치에 강건 |
| `침출온도_C`, `침출시간_hr` | reference_df(Train)의 전체 중앙값 대치 | 배치마다 독립적으로 운전조건이 설정되어 시계열 보간의 물리적 근거가 없음 |
| `황산투입량_L` (결측 2건) | reference_df(Train)로 학습한 `원료투입량_kg → 황산투입량_L` 선형회귀로 대치. `원료투입량_kg`도 같이 결측인 행은 예측 대상에서 제외(해당 행은 NaN으로 남음) | 황산 투입량은 광석 질량에 비례하는 화학양론적 관계(산-광석비)를 가짐 |
| `분석일자` (21건 결측) | 배치ID 순번 기준 정렬 후 시간 보간. **Train/Validation/Test 분할 전**, 전체 데이터에 한 번만 적용 | 배치ID가 순차 발번되어 생산·분석 순서를 근사. 이 보간은 train/val/test 구분이 의미 없는 위치 기반 로직이라 분할 이전에 처리해도 leak-safety에 영향 없음 |
| `원료등급` 표기 불일치 | 공백 제거 + 대문자화 + `급` 접미사 제거 → `A/B/C` 표준화 | 그룹별 대치가 제대로 동작하려면 선행 필수 |
| `배치ID` 중복 4건(8행) | `resolve_duplicate_batches`에서 전체 컬럼까지 완전 일치하는 경우만 첫 occurrence를 남기고 제거, 남은 행은 `is_duplicate_batch=True`로 감사 기록 유지. 값이 다른데 배치ID만 겹치는 "충돌" 중복은 자동 제거하지 않고 `is_conflicting_duplicate=True`로 표시해 사람이 확인 | 같은 ID로 서로 다른 레코드가 들어온 경우까지 자동으로 지우면 데이터 품질 문제를 놓칠 수 있음 |
| 결측치 패턴 진단 | 카이제곱 독립성 검정(원료등급 vs 결측여부), 결측 플래그 간 상관계수로 체계적 편향 여부 확인 (노트북 §1.1.5) | 결측이 무작위가 아니라면(MAR) 단순 중앙값 대치가 편향을 만들 수 있음. 실측 결과 `Mn_함량_pct`·`Cu_불순물_pct`는 원료등급과 결측 여부가 독립이 아님(p<0.05)이 확인됐으나, 두 컬럼 모두 이미 등급별 그룹 중앙값으로 대치되어 이 편향이 대치 방식에 반영됨 |

> **data leakage 방지**: 결측치 대치 통계(중앙값·회귀계수)는 항상 `reference_df`(Train 구간, 또는 최종 평가 시 Train+Validation)로만 계산해 `target_df`(Validation/Test)에 적용한다. Validation/Test 자신의 값은 대치 계산에 전혀 쓰이지 않는다.

> **날짜 파싱 주의사항**: `pd.to_datetime(..., format="mixed")`로 일괄 처리하면 `"03.01.2025"` 같은 점(`.`) 구분 표기를 월-일(MM.DD) 순으로 오인해 3월 1일로 잘못 파싱한다(이 데이터의 점 표기는 일-월-년(DD.MM.YYYY) 규칙). 따라서 정규식으로 6개 형식을 먼저 식별한 뒤 형식별로 명시적 파싱한다.

## 5. 데이터 분할 (Train/Validation/Test)

`temporal_train_val_test_split`이 배치ID 순번(분석일자 순서의 근사)으로 정렬한 뒤 앞 70%를 Train(294행), 다음 15%를 Validation(63행), 마지막 15%를 Test(63행)로 **고정 분할**한다. 무작위 섞기를 쓰지 않는 이유는 배치가 2025년 1~10월에 걸친 시계열 데이터라, 미래 배치로 학습해 과거 배치를 맞히는 비현실적인 평가를 막기 위함이다.

이 분할은 `pipeline.py`에서 **단 한 번만** 계산되어 결측치 대치(§4)와 feature 선택(§6.1), 최종 평가(§6.2) 전체에서 재사용된다 — 여러 곳에서 각자 분할을 계산하면 서로 어긋날 위험이 있기 때문이다.

## 6. 특성공학 (파생변수)

| 파생변수 | 계산식 | 도메인 근거 |
|---|---|---|
| 산-광석비 (`acid_to_ore_ratio`) | `황산투입량_L / 원료투입량_kg` | 침출 공정 핵심 화학양론 지표 |
| 침출강도 (`leach_intensity`) | `침출온도_C × 침출시간_hr` | 반응속도론적으로 온도·시간이 결합해 반응 진행도를 결정 |
| 총불순물 (`total_impurity_pct`) | `Cu_불순물_pct + Fe_불순물_pct + Al_불순물_pct` | 불순물이 목표금속과 경쟁 반응을 일으켜 회수율 저하 요인 |
| 목표금속 순도비 (`target_purity_ratio`) | `(Ni+Co) / (Ni+Co+총불순물)` | 원료 내 회수 대상 금속 대비 방해 성분의 상대적 비중 |
| 분석월 (`analysis_month`) | `분석일자`에서 월 추출 | 계절성/설비 드리프트 모니터링용 — 모델 피처로는 사용하지 않음 |
| 개별 불순물 비율 (`Fe_Ni_ratio`, `Cu_Ni_ratio`, `Al_Ni_ratio`) | 각 불순물_pct / `Ni_함량_pct` | `total_impurity_pct`가 가리는 "어느 불순물이 실제 원인인지"를 분리해서 보기 위함 |

### 6.1 Feature 선택 (Validation만 사용)

`run_ablation_test(train_df, val_df, feature_sets, target_columns)`가 **Validation 성능만으로** feature 조합을 비교한다. Test는 이 단계에서 전혀 보지 않는다 — Validation 결과로 feature를 고르면서 Test까지 같이 보면, Test가 더 이상 "모델이 한 번도 안 본" 최종평가셋이 아니게 되기 때문이다.

실측 결과(Validation R²):
- `base_only`(조성+공정) → `base_plus_derived`(+파생변수 4종): Ni 0.4836→0.7505(+0.27), Co 0.5344→0.7168(+0.18) — 뚜렷한 개선
- `base_only` → `base_plus_grade`(+원료등급): Ni 0.4836→0.4618(-0.02), Co 0.5344→0.5321(-0.002) — 두 타겟 모두 하락, 안정적 기여 없음
- `base_plus_derived` → `base_plus_derived_plus_ratios`(+개별 불순물 비율 3종): Ni/Co 둘 다 소폭 하락 — 단순 상관(-0.4~-0.5)은 강하지만 모델 기여는 없음(RandomForest가 원본 불순물·Ni_함량 컬럼으로 이미 유사한 경계를 학습하기 때문으로 판단)

최종 선택: `base_plus_derived`(조성+공정+파생변수 4종). `원료등급`은 모델 입력 피처에서는 제외하지만, 그룹별 중앙값 대치의 키로는 계속 사용하므로 컬럼 자체는 유지. 개별 불순물 비율 3종은 `engineer_features`에는 유지(해석용)하되 모델 입력에는 미포함.

### 6.2 최종 모델 평가 (Test, 1회)

`evaluate_final_model(train_df, val_df, test_df, final_features, target_columns)`가 선택된 feature(`base_plus_derived`)로 **Train+Validation을 합쳐 재학습**한 뒤 Test에 딱 한 번 적용한다. 실측 결과: Ni test_r2=0.767, Co test_r2=0.794.

## 7. 데이터 검증 (정합성 + 시각화)

- **정합성 체크**: `check_ranges`로 정제 후 모든 수치 컬럼이 정상 범위 내인지, `check_missing`으로 결측이 없는지 확인 — 위반 시 `assert`로 즉시 중단(경고만 출력하지 않음). `flag_outliers`가 범위 밖 값을 이미 다 걸러낸 뒤이므로, 여기서 위반이 발견되면 대치·파생변수 로직의 버그라는 뜻
- **정제 전/후 대비**: `pipeline.py`는 `load_raw` 직후에도 `check_ranges`를 한 번 호출해 raw 상태의 위반 건수를 출력, 정제 후 0건과 직접 대비
- **결측치 처리 전/후 비교**: `plot_missing_before_after` — 컬럼별 결측 비율 막대그래프
- **분포 확인**: `plot_distributions` — 이상치 제거 전/후 박스플롯
- **상관관계 히트맵**: `plot_correlation_heatmap` — 원본 변수 + 파생변수 vs Ni/Co_회수율_pct
- **결측 패턴 진단**: 카이제곱 독립성 검정 + 결측 동시발생 상관 (§4 참고)
- **클래스 분포 확인**: `plot_grade_distribution` — 원료등급(A/B/C) 빈도 (B 38.6%, A 35.2%, C 26.2% — 심각한 불균형 아님)
- 모든 플롯은 `reports/figures/`에 PNG로 저장, 노트북에도 인라인 표시

## 8. 산출 파일 구조

```
src/
  config.py          # 공유 상수 (NORMAL_RANGES, COMPOSITION_COLUMNS 등)
  preprocessing.py   # load_raw, clean_schema, resolve_duplicate_batches
  splitting.py       # temporal_train_val_test_split
  imputation.py      # replace_outliers_with_nan, interpolate_analysis_date, impute_missing_from_reference
  features.py        # engineer_features
  validation.py      # check_ranges, check_missing, plot_*, run_ablation_test, evaluate_final_model
  pipeline.py        # 엔트리포인트: python -m src.pipeline
scripts/
  build_notebook.py  # notebooks/01_eda_and_validation.ipynb 생성 스크립트
tests/
  test_*.py          # src/ 각 모듈에 대응하는 pytest 테스트
notebooks/
  01_eda_and_validation.ipynb
data/processed/
  cleaned_features.csv   # 파이프라인 최종 산출물
reports/figures/
  *.png                  # 시각화 결과
```

## 9. 심층 인터뷰 답변 매핑

| 질문 | 설계상 근거 위치 |
|---|---|
| 실제 공정 데이터 품질 문제를 어떻게 식별·처리했는가 | §1 문제 목록, §4 표, 결측 패턴 진단(카이제곱·상관) |
| 특성공학에서 도메인 지식을 어떻게 활용했는가 | §6 표, §6.1 ablation test 결과 |
| 데이터 불균형 문제가 있었다면 어떻게 해결했는가 | §7 클래스 분포 확인, §6.1의 `원료등급` ablation 결과 |

## 10. 범위 밖 (Out of scope)

- 모듈 2(회수율 예측 모델링), 모듈 3(공정 최적화)은 이번 설계에 포함하지 않음
- 딥러닝 기반 결측치 대치(MICE, KNN-imputer 등)는 사용하지 않음 — 데이터 규모(420행)와 도메인 해석 가능성을 고려해 통계적/회귀 기반 방법으로 충분하다고 판단
- 시계열 교차검증(rolling/TimeSeriesSplit)은 적용하지 않음 — Train 294행 기준으로 추가 분할하면 윈도우가 너무 작아져 노이즈가 커질 수 있어, 단일 시간순 분할 + Validation/Test 분리로 충분하다고 판단
