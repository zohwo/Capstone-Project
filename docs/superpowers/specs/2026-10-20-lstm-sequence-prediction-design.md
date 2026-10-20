# LSTM 시퀀스 예측 설계

## 배경 및 목적

`module2_prediction`은 지금까지 배치를 독립된 행으로 보고 RandomForest/XGBoost/LightGBM을 비교해왔다(`modeling.py`). 여기에 과거 N개 배치의 흐름으로 다음 배치의 회수율(Ni/Co)을 예측하는 **진짜 시퀀스 기반 LSTM**을 추가한다.

모듈 2의 기존 원칙을 그대로 유지한다: 모듈 1 코드를 import하지 않음, feature set은 모듈 1이 정한 13개(`FEATURE_COLUMNS`)로 고정, Validation으로 모델/하이퍼파라미터를 선택하고 Test는 최종 평가에 1회만 사용, 타겟(Ni/Co)별로 독립된 모델을 학습.

## 핵심 설계 원칙

> 모든 시퀀스는 target 시점보다 과거에 실제 관측 가능한 정보만 사용하며, split은 target row 기준으로 부여한다. 과거 context는 split 경계를 넘을 수 있으나 미래 정보는 절대 포함하지 않는다.

이 원칙은 tuning 단계뿐 아니라 **final(Train+Val 재학습 → Test 평가) 단계에서도 동일하게 적용**된다. `build_sequences()`는 어느 단계에서 호출되든 항상 Train+Val+Test 전체 시간축을 순회해야 하며, 특정 split만 미리 잘라서 넘기면 안 된다 (그러면 그 split 초반 target들의 context가 부족해지는 문제가 재발한다).

## 파일 구조

```
module2_prediction/
├── modeling.py / tuning.py / evaluation.py   (기존, 변경 없음)
├── sequence_modeling.py                       (신규, 독립 파일)
└── tests/test_sequence_modeling.py            (신규)
```

`sequence_modeling.py`는 모듈 1을 import하지 않고, `modeling.py`/`tuning.py`/`evaluation.py`도 import하지 않는다 (입력 shape가 flat DataFrame이 아닌 3D 시퀀스 텐서라 인터페이스가 근본적으로 다름). 평가 지표는 `sklearn.metrics`를 직접 호출한다 (기존 파일들도 자체 helper 없이 동일하게 사용하므로 재사용할 대상이 없음).

## 함수 구성

### `build_sequences(df, feature_columns, target_column, window_size, include_past_target) -> (X, y, split_labels, target_indices)`

- 전체 시간축(Train+Val+Test 전부)을 순회하며, 각 target row마다 **직전 window_size개 행**을 window로 구성
- **split 할당은 target row의 `split` 값 기준** — 윈도우가 걸치는 범위가 아님
- 현재 target은 절대 X에 포함하지 않음
- `include_past_target=True`면 과거 window_size개 행의 target 값도 함께 입력에 포함 (LSTM-X+y)
- scaling은 책임지지 않음 — 이미 transform된 데이터를 받는다고 가정
- `target_indices`를 반환해 디버깅/테스트에서 어느 행이 어느 target에 대응하는지 추적 가능

**전제**: `include_past_target=True`일 때 과거 target 값은 실제로 완료되어 측정·기록된 배치의 회수율이라고 가정한다. 이 가정은 이 데이터셋에 대해 별도로 검증되지 않았다 (실제 공정에서는 분석 결과 통보까지 시간차가 있을 수 있음) — 캡스톤 프로젝트 범위에서 합리적인 모델링 가정으로 채택한다.

### Scaling — `fit_feature_scaler`, `fit_history_target_scaler`, `transform_sequence_inputs`

| 대상 | tuning 단계 | final 단계 |
|---|---|---|
| `fit_feature_scaler` (13개 feature) | Train만 fit | Train+Val로 재fit |
| `fit_history_target_scaler` (과거 y, X+y 전용) | Train target만 fit | Train+Val target으로 재fit |
| 현재 y (예측 대상) | 스케일링 안 함, 원 단위(%) 유지 | 동일 |

과거 y는 LSTM의 다른 입력 차원과 스케일을 맞춰야 하므로 스케일링하고, 현재 y는 예측 대상이라 원 단위로 두는 것이 해석과 지표 계산에 자연스럽다. 두 결정은 서로 모순되지 않는다. 현재 데이터 범위(0~100%)와 구현 단순성을 고려해 타겟 스케일링은 하지 않으며, 학습 중 수렴이 불안정해지는 것이 확인되면 추가를 재검토한다.

### `LSTMRegressor` (`torch.nn.Module`)

- LSTM + Linear head, `input_size` = 13(X) 또는 14(X+y)
- Ni/Co 타겟별로 독립된 모델
- `num_layers=1`일 때 `dropout=0.0`을 강제한다 (`dropout = value if num_layers > 1 else 0.0`) — PyTorch의 `nn.LSTM`은 단일 레이어에서 dropout이 적용되지 않고 경고만 발생시키므로, 지금 그리드에 dropout 파라미터가 없더라도 향후 추가 시를 위한 가드로 포함

### `train_lstm(model, train_seq, val_seq, max_epochs, patience) -> best_state_dict, best_epoch`

- Adam optimizer + MSELoss
- Validation loss 기준 early stopping
- **`best_state_dict`**: validation loss가 최저였던 시점의 가중치 (patience 동안 더 학습한 뒤 멈춘 시점의 가중치가 아님)
- **`best_epoch`**: 최종 재학습에 사용할 **총 epoch 수**로 정의한다 (0-based 인덱스가 아님). 즉 val loss가 최저였던 epoch의 0-based 인덱스가 `i`라면 `best_epoch = i + 1`. 최종 학습 루프는 `for epoch in range(best_epoch)`로 그대로 사용 가능해야 한다.
- `best_epoch`은 **(target, model variant, config) 조합마다 개별적으로** 저장한다. Ni/Co, LSTM-X/LSTM-X+y, 서로 다른 하이퍼파라미터 조합이 같은 epoch에서 최적일 이유가 없다.

### 재현성

- `random`, `numpy`, `torch`, `torch.cuda`의 시드를 고정한다
- `torchcu`(GPU) 환경만 사용하므로 `torch.backends.cudnn.deterministic = True`, `torch.backends.cudnn.benchmark = False`를 추가로 설정한다 (cuDNN 커널은 시드 고정만으로는 완전히 결정적이지 않을 수 있음)

### `tune_sequence_models(df, feature_columns, target_columns, ...) -> dict`

- Grid: `window_size ∈ {5, 10, 20} × hidden_size ∈ {16, 32} × num_layers ∈ {1, 2} × include_past_target ∈ {False, True}` (24 조합 × 2 타겟)
- 선택 기준: **Validation R²** (기존 `tune_models()`와 통일, `module2_prediction/tuning.py:26` 참고)
- Test는 전혀 접근하지 않음
- 반환값에 선택된 config와 `best_epoch`을 (target, variant) 키로 포함

### `evaluate_final_sequence_model(...) -> dict`

1. Train+Val로 `fit_feature_scaler`/`fit_history_target_scaler` 재fit
2. `build_sequences()`를 **Train+Val+Test 전체 df**에 대해 재호출 (Test 초반 window가 Train+Val context를 포함할 수 있도록)
3. 선택된 config로 모델 초기화, **`best_epoch`만큼 고정 학습** (early stopping 재실행 없음)
4. Test 타겟에 대해서만 예측·평가 — `{target: {test_r2, test_rmse, test_mae}}` 반환 (기존 `evaluate_final_model`과 동일 출력 포맷)

### 결과 비교 프레임

최종 보고에서는 단순 성능 순위가 아니라 3개 층위로 구분한다:
- **Tabular baseline**: RandomForest / XGBoost / LightGBM
- **Sequence model**: LSTM-X
- **Autoregressive sequence model**: LSTM-X+y (입력 정보가 더 많다는 점을 명시 — LSTM-X와의 비교는 "과거 회수율 정보 추가 효과"에 대한 ablation으로 해석)

## 반드시 막아야 하는 버그

1. 현재 target이 X+y 입력에 섞이는 것
2. Validation/Test 데이터가 scaler fit에 들어가는 것
3. split별로 먼저 잘라서 Validation/Test 초반 window의 context가 사라지는 것 (tuning 단계와 final 단계 양쪽에서 모두 발생 가능)

## 테스트 (`module2_prediction/tests/test_sequence_modeling.py`)

1. window 크기 정확성 — N=5면 정확히 직전 5개 행만 포함되는지
2. target 기준 split 할당 — split_label이 target row의 split과 일치하는지
3. **경계 window context 직접 검증** — Validation/Test 첫 타겟의 window가 실제로 이전 split의 행 인덱스를 포함하는지 (split_label만 보는 테스트와는 별도로, window의 실제 인덱스 범위를 확인)
4. 미래 정보 미포함 — target t의 window에 t 또는 t 이후 행이 없는지
5. scaler leakage — Validation/Test 값을 바꿔도 Train(또는 Train+Val) 기준 scaler의 mean/std가 변하지 않는지
6. X+y 현재 타겟 미포함 (최우선 테스트) — off-by-one으로 현재 target이 입력에 섞이지 않는지
7. output shape — LSTM-X는 `(samples, N, 13)`, LSTM-X+y는 `(samples, N, 14)`
8. smoke test — 작은 synthetic 데이터로 fit → predict 성공, NaN/Inf 없음

## 의존성 및 환경

`requirements.txt`에 `torch` 추가. **`torchcu` 환경에만 설치** (`base` 환경은 건드리지 않음).
