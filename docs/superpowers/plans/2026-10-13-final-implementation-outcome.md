# 모듈 1 전처리 파이프라인 — 최종 구현 결과 및 계획 대비 변경사항

- 원본 계획: `2026-09-28-blackpowder-preprocessing.md` (Task 1~13, 2026-09-28 작성, **원본 그대로 보존**)
- 이 문서: 2026-10-13 작성. 원본 계획과 실제 최종 구현 사이의 차이, 그 차이가 왜 생겼는지를 정리한다 — "어떤 어려움을 겪었는가" 작성 시 참고용

## 1. 이 문서의 목적

원본 계획(`2026-09-28-...md`)은 프로젝트 시작 시점의 Task별 설계다. 이후 구현·검증 과정에서 여러 구조적 문제(주로 data leakage)가 발견되어 계획이 상당 부분 바뀌었다. 원본 문서는 수정하지 않고 "처음에 이렇게 계획했다"는 기록으로 남겨두고, 이 문서에서 **무엇이, 왜 바뀌었는지**를 정리한다.

## 2. 계획 대비 주요 변경점

| 항목 | 원본 계획 (Task 1~13) | 최종 구현 | 변경 이유 |
|---|---|---|---|
| 이상치 처리 함수명 | `flag_outliers` | `replace_outliers_with_nan` | 함수가 실제로 하는 일(NaN 치환)과 이름이 더 정확히 맞도록 변경 |
| 결측치 대치 함수 | `impute_missing(df)` — 전체 데이터 기준 한 번에 대치 | `impute_missing_from_reference(reference_df, target_df)` — reference와 target을 분리 | 전체 데이터로 먼저 대치하면 그 통계에 Validation/Test 값이 섞여 들어가는 data leakage가 생김. reference(Train)와 target(Validation/Test)을 분리해 leak-safe하게 재설계 |
| 중복 배치 처리 | `is_duplicate_batch` 플래그만 달고 삭제는 안 함 | `resolve_duplicate_batches`: 완전 중복은 제거, 값이 다른 충돌은 `is_conflicting_duplicate`로 별도 표시 | 처음엔 노트북에서만 중복을 제거하고 CSV 산출물엔 그대로 남겨둬서, 이 CSV를 쓰는 후속 분석에 중복 데이터가 그대로 섞여 들어가는 문제가 있었음. pipeline.py 자체에서 처리하도록 수정. 이후 "배치ID만 같고 값이 다른 경우"까지 고려해 완전중복/충돌중복을 구분하는 구조로 한 번 더 발전 |
| 검증/평가 방식 | K-fold 교차검증으로 R²/RMSE 비교 | 시간순 Train(70%)/Validation(15%)/Test(15%) **고정 분할** | 배치가 2025년 1~10월 순차 데이터라는 프로젝트 요구사항("시계열 데이터는 시간 순서를 고려")에 따라 무작위 K-fold를 폐기하고 시간순 고정분할로 전환 |
| ablation test 구조 | Case A/B 두 그룹을 K-fold로 비교 | `run_ablation_test(train_df, val_df, feature_sets, target_columns)` — **Validation만** 사용 | 처음엔 Validation과 Test를 같이 보고 "두 구간에서 방향이 일치하면 진짜 효과"라고 판단했는데, 이러면 feature 선택에 Test 결과를 이미 사용한 셈이 되어 Test가 더 이상 순수한 최종평가셋이 아니게 됨. Validation만으로 feature를 고르고, Test는 완전히 분리하는 구조로 재설계 |
| 최종 성능 평가 | 명시적 단계 없음 (ablation test 결과가 곧 최종 결과로 혼용됨) | `evaluate_final_model(train_df, val_df, test_df, ...)` — 선택된 feature로 Train+Validation 재학습 후 Test에 **딱 한 번** 적용 | "Train→학습, Validation→feature 선택, Test→최종 1회 평가"라는 3단계 구조를 명확히 분리하기 위해 신설 |
| 파생변수 개수 | 5종 (산-광석비, 침출강도, 총불순물, 순도비, 분석월) | 8종 (+ `Fe_Ni_ratio`, `Cu_Ni_ratio`, `Al_Ni_ratio`) | `total_impurity_pct`가 Cu/Fe/Al을 동일 가중치로 합산해 "어느 불순물이 실제 원인인지"를 가린다는 문제를 보완하기 위해 개별 비율 3종 추가. ablation test 결과 모델 성능 기여는 없는 것으로 확인되어 해석용으로만 유지, 최종 모델 입력 feature에는 미포함 |
| 모듈 구조 | `src/`에 5개 파일 | `src/`에 7개 파일 (`splitting.py`, `config.py` 추가) | Train/Val/Test 분할 로직과 공유 상수를 재사용 가능한 별도 모듈로 분리 |
| 날짜 보간 위치 | 명시 안 됨 | 분할(Train/Val/Test) **이전**, 전체 데이터에 한 번만 적용 | 분석일자 보간은 train/val/test 구분이 의미 없는 위치 기반 로직이라는 게 뒤늦게 명확해짐 — 애초에 분할과 무관하게 처리해도 leak-safety에 영향 없음 |

## 3. 진행 중 겪은 주요 이슈

### 3.1 날짜 파싱 버그
`pd.to_datetime(..., format="mixed")`로 일괄 처리하면 `"03.01.2025"` 같은 점(`.`) 구분 표기를 월-일(MM.DD) 순으로 오인해 3월 1일로 잘못 파싱했다. 실제로는 이 데이터의 점 표기가 일-월-년(DD.MM.YYYY) 규칙이라, 배치ID 순서와 비교했을 때 날짜가 역행하는 걸 보고 발견했다. 정규식으로 6개 형식을 먼저 식별한 뒤 형식별로 명시적 파싱하도록 수정했다.

### 3.2 전처리 단계의 data leakage (가장 핵심적인 반복 이슈)
초기 구현은 전체 420행 기준 통계로 결측치를 먼저 대치한 뒤 K-fold로 나눠서 평가했다. 영향을 실측해보니 R² 차이는 0.003 이하로 작았지만, "영향이 작다고 구조적 문제가 없어지는 건 아니다"라는 판단 아래 Train 기준 통계만 사용하는 구조로 전면 수정했다. 이 문제는 이후 다음 지점들에서 반복적으로 재발·재검토됐다:
- ablation test 내부의 결측치 대치
- 중복 배치가 train/test에 걸쳐 나뉘는 경우 (완전중복을 분할 전에 제거해 원천 차단)
- feature 선택에 Test 결과를 같이 보는 것 (§2의 ablation test 구조 변경)

### 3.3 코드 품질 이슈 (반복적으로 발견)
- 리팩터링 과정에서 실수로 중복 제거 코드가 삭제됐다가 테스트 실패로 발견·복구됨
- 사용하지 않는 함수(`impute_acid_volume`, `impute_missing`)가 리팩터링 후에도 삭제되지 않고 남아있던 것을 발견해 제거
- `run_ablation_test`가 매 feature 조합마다 결측치 대치·파생변수 생성을 중복 계산하던 비효율을 발견해 루프 밖으로 이동
- 함수 이름 변경(`flag_outliers`→`replace_outliers_with_nan`) 후 일부 호출부(테스트, 노트북 생성 스크립트)가 갱신되지 않아 전체 테스트가 깨진 것을 발견·수정

### 3.4 검증되지 않은 주장이 노트북 텍스트에 포함된 문제
심층 인터뷰 답변 초안에 "결측이 카이제곱 독립성 검정(p>0.05)으로 체계적 패턴이 없음을 확인했다"는 문장이 있었는데, 실제로 이 검정을 실행하는 코드가 어디에도 없었다. 실제로 돌려본 결과 10개 컬럼 중 8개는 독립(p>0.05)이었지만 `Mn_함량_pct`(p=0.043), `Cu_불순물_pct`(p=0.006) 두 컬럼은 원료등급과 결측 여부가 통계적으로 독립이 아니었다 — 원래 주장이 실제로는 틀렸던 것. 노트북에 실제 검정 코드를 추가하고 결과를 정정했다.

### 3.5 설계 문서와 구현의 불일치
설계 문서(`docs/superpowers/specs/`)가 프로젝트 초기(09-28) 작성 이후 갱신되지 않아, 위 변경사항(함수명, 파생변수 개수, K-fold→시간순 분할 등)이 전혀 반영되지 않은 상태로 방치되어 있었다. 2026-10-13에 현재 구현 기준으로 전면 재작성했다.

## 4. 최종 검증 현황

- 테스트: `src/` 7개 모듈에 대응하는 pytest 46개, 전부 통과
- 파이프라인(`python -m src.pipeline`): raw 420행 → 정제 후 결측 0건, 범위 위반 0건 → ablation test(Validation) → 최종 feature 선택(`base_plus_derived`) → Test 1회 평가(Ni R²=0.767, Co R²=0.794)까지 전체 실행 확인
- 노트북(`notebooks/01_eda_and_validation.ipynb`): `scripts/build_notebook.py`로 생성, `jupyter nbconvert --execute`로 끝까지 에러 없이 실행 확인
