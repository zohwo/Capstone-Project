# 모듈 1: 블랙파우더 성분 데이터 전처리 파이프라인

금속회수율최적화 프로젝트 — 배치(BP-2025-XXXX)별 블랙파우더(폐배터리 분쇄물) 성분 데이터를 정제하고, 
니켈/코발트 회수율 예측에 쓸 수 있는 형태로 가공하는 파이프라인

## 무엇을 하는가

1. **정제**: 날짜 형식 통일, 원료등급 표기 정규화, 완전 중복 배치 제거, 정상범위 이탈 값을 결측으로 치환
2. **분할**: 배치ID 순번(시간 순서 근사) 기준 Train(70%)/Validation(15%)/Test(15%) 고정 분할 — 무작위 분할 없음
3. **결측치 대치**: Train 통계(그룹별/전체 중앙값, 회귀계수)만 사용해 Validation/Test에 적용 — data leakage 방지
4. **특성공학**: 침출 공정 도메인 지식 기반 파생변수 8종 생성
5. **feature 선택**: Validation 성능만으로 비교(ablation test) — Test는 이 단계에서 보지 않음
6. **최종 평가**: 선택된 feature로 Train+Validation 재학습 후 Test에 딱 한 번 적용

자세한 설계 근거는 [`docs/superpowers/specs/2026-09-28-blackpowder-preprocessing-design.md`](docs/superpowers/specs/2026-09-28-blackpowder-preprocessing-design.md) 참고.

## 프로젝트 구조

```
src/
  config.py          # 공유 상수 (정상범위, 컬럼 목록 등)
  preprocessing.py   # 로딩, 스키마 정제, 중복 처리
  splitting.py        # 시간순 Train/Validation/Test 분할
  imputation.py       # 이상치 치환, 날짜 보간, leak-safe 결측치 대치
  features.py         # 파생변수 생성
  validation.py       # 검증, 시각화, ablation test, 최종 모델 평가
  pipeline.py          # 엔트리포인트
scripts/
  build_notebook.py    # EDA/검증 노트북 생성 스크립트
tests/                 # pytest 테스트
notebooks/
  01_eda_and_validation.ipynb
data/                  # 원본 데이터 (비공개, git에 포함 안 됨)
reports/figures/        # 시각화 산출물 (재생성 가능, git에 포함 안 됨)
```

## 설치

```bash
pip install -r requirements.txt
```

## 실행

```bash
# 파이프라인 실행 → data/processed/cleaned_features.csv 생성
python -m src.pipeline

# EDA/검증 노트북 재생성
python scripts/build_notebook.py

# 노트북 직접 실행(검증용)
jupyter nbconvert --to notebook --execute notebooks/01_eda_and_validation.ipynb

# 테스트
pytest
```

## 주요 결과

- 원본 424행 → 완전 중복 4건 제거 후 420행
- 최종 선택 feature: 조성(7종) + 공정조건(2종) + 파생변수(4종) — 원료등급과 개별 불순물 비율은 모델 성능에 안정적으로 기여하지 않아 제외(ablation test로 검증)
- Test 최종 성능: Ni_회수율_pct R²=0.767, Co_회수율_pct R²=0.794
