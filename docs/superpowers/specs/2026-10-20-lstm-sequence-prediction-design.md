# LSTM 시퀀스 예측 설계

## 1. 배경 및 목적

`module2_prediction`에서는 지금까지 각 배치를 독립된 행으로 간주하여 RandomForest, XGBoost, LightGBM 기반 회수율 예측을 수행해왔다.

여기에 과거 N개 배치의 시간적 흐름을 입력으로 사용하여 다음 배치의 Ni/Co 회수율을 예측하는 시퀀스 기반 LSTM 모델을 추가한다.

기존 모듈 2의 원칙은 그대로 유지한다.

- 모듈 1 import 없음
- feature set 13개 고정
- Ni/Co 타겟별 독립 모델
- Validation 성능으로 모델 및 하이퍼파라미터 선택
- Test는 최종 평가에만 사용
- 기존 `modeling.py`, `tuning.py`, `evaluation.py`는 변경하지 않음

## 2. 핵심 설계 원칙

모든 시퀀스는 target 시점보다 과거에 실제 관측 가능한 정보만 사용하며, split은 target row를 기준으로 부여한다. 과거 context는 split 경계를 넘을 수 있으나 미래 정보는 절대 포함하지 않는다.

이 원칙은 tuning 단계와 final evaluation 단계 모두 동일하게 적용한다.

따라서 `build_sequences()`는 Train/Validation/Test를 각각 잘라서 시퀀스를 생성하지 않고, 항상 전체 시간축을 유지한 상태에서 직전 N개 행을 window로 구성한다.

예를 들어 Validation 첫 번째 target을 예측할 때 직전 N개 배치 중 일부가 Train에 속해 있더라도, 모두 target보다 과거에 관측된 정보이므로 사용할 수 있다.

동일하게 Test 첫 target의 입력 window에도 Train/Validation의 마지막 배치들이 포함될 수 있다.

## 3. 파일 구조

```
module2_prediction/
├── modeling.py
├── tuning.py
├── evaluation.py
├── sequence_modeling.py
└── tests/
    └── test_sequence_modeling.py
```

`sequence_modeling.py`는 LSTM 전용 preprocessing, sequence 생성, scaling, 모델 정의, 학습, 튜닝, 최종 평가를 담당한다.

기존 `modeling.py`, `tuning.py`, `evaluation.py`와 import 관계를 만들지 않고 독립적으로 구성한다.

평가지표는 `sklearn.metrics`의 R², RMSE, MAE를 직접 사용한다.

## 4. Sequence 구성

### `build_sequences()`

각 target row `t`에 대해 다음과 같이 직전 N개 배치를 입력으로 구성한다.

```
t-N, ..., t-2, t-1 → t 예측
```

현재 target row `t`의 정보는 입력 sequence에 포함하지 않는다.

**기본 동작**

- 전체 DataFrame을 시간순으로 순회
- 각 target row마다 직전 `window_size=N`개의 행을 window로 사용
- split은 sequence 자체가 아니라 target row의 split을 기준으로 결정
- 현재 target 값은 절대 입력 X에 포함하지 않음
- 미래 행도 절대 포함하지 않음

**반환값**

```
X
y
split_labels
target_indices
```

`target_indices`는 각 sequence가 원래 DataFrame의 어느 target row를 예측하는지 확인할 수 있도록 반환하며, leakage 및 경계 window 디버깅에 사용한다.

## 5. 입력 유형

두 가지 LSTM 변형을 비교한다.

### 5.1 LSTM-X

과거 N개 배치의 13개 공정 feature만 사용한다.

```
Input shape
(samples, window_size, 13)
```

예:

```
과거 10개 배치의 13개 feature
        ↓
      LSTM
        ↓
다음 배치 Ni 또는 Co 회수율
```

### 5.2 LSTM-X+y

과거 N개 배치의 13개 feature와 각 배치의 실제 과거 회수율을 함께 사용한다.

```
Input shape
(samples, window_size, 14)
```

Ni 모델에는 과거 Ni 회수율만 사용하고, Co 모델에는 과거 Co 회수율만 사용한다.

현재 예측 대상의 회수율은 절대 입력에 포함하지 않는다.

**과거 y 사용에 대한 운영 가정**

LSTM-X+y는 다음과 같은 rolling one-step-ahead prediction 환경을 가정한다.

```
배치 t-1 완료
→ 실제 회수율 측정
→ 해당 회수율을 알고 있는 상태에서 배치 t 예측
```

따라서 Test 구간에서도 이전 Test 배치의 실제 회수율이 이미 측정 완료되었다는 조건에서는 다음 Test 배치의 입력 context로 사용할 수 있다.

단, 이 데이터셋에서 실제 공정상 이전 배치 회수율이 다음 배치 예측 시점 전에 항상 확보되는지는 확인되지 않았으므로, 이를 운영 가정(deployment assumption)으로 명시한다.

즉 LSTM-X+y는 LSTM-X보다 더 강한 정보 가정을 가진 모델이다.

## 6. Scaling 정책

LSTM은 입력 feature의 scale에 민감하므로 StandardScaler를 사용한다.

Sequence 생성과 scaling의 책임은 분리한다.

```
fit scaler
↓
전체 row transform
↓
build_sequences()
```

`build_sequences()` 내부에서는 scaling을 수행하지 않는다.

### 6.1 Feature scaler

13개 공정 feature에 적용한다.

**Tuning 단계**

```
Scaler.fit(Train rows only)
↓
Train / Validation / Test transform
```

Validation/Test 값은 scaler 통계 계산에 절대 포함하지 않는다.

**Final 단계**

하이퍼파라미터 선택이 완료된 후에는 Train+Validation을 최종 학습 데이터로 사용하므로,

```
Scaler.fit(Train + Validation)
↓
Train + Validation + Test transform
```

을 수행한다.

Test는 transform만 수행하며 scaler fit에는 참여하지 않는다.

### 6.2 History-target scaler

`LSTM-X+y`에서 입력으로 들어가는 과거 회수율 전용 scaler를 별도로 둔다.

공정 feature와 회수율은 scale과 단위가 다르므로 feature scaler와 공유하지 않는다.

**Tuning**

```
history target scaler.fit(Train target only)
```

**Final**

```
history target scaler.fit(Train + Validation target)
```

Ni 모델과 Co 모델은 각각 자기 target의 scaler를 사용한다.

### 6.3 현재 예측 target

현재 예측 대상인 y는 별도 scaling을 하지 않는다.

즉 모델은 실제 회수율 단위(%)를 직접 예측한다.

```
MSE loss → 실제 회수율 단위
RMSE → 실제 %
MAE → 실제 %
R² → 원 단위 기준
```

회수율 자체가 일반적으로 0~100 수준이므로 현재 프로젝트 규모에서는 target scaling 없이도 충분하다고 판단한다.

향후 실제 학습에서 loss convergence 문제가 확인되는 경우에만 target scaling 도입을 재검토한다.

## 7. `transform_sequence_inputs()`

Scaling 적용을 담당한다.

기본 역할:

- 13개 feature transform
- `include_past_target=True`이면 과거 target도 별도 scaler로 transform
- 현재 target y는 transform하지 않음
- 원래 row 순서와 split 정보 유지

이후 transform된 전체 시간축을 `build_sequences()`로 전달한다.

## 8. LSTM 모델

### `LSTMRegressor`

구조는 단순한 LSTM + Linear regression head로 시작한다.

```
Input sequence
      ↓
     LSTM
      ↓
마지막 hidden state
      ↓
Linear(1)
      ↓
회수율 예측
```

**Input size**

```
LSTM-X   → 13
LSTM-X+y → 14
```

Ni와 Co는 각각 독립적인 모델로 학습한다.

**`num_layers`와 dropout**

PyTorch LSTM은 `num_layers=1`일 때 내부 dropout이 적용되지 않는다.

따라서 향후 dropout parameter가 추가되더라도 다음 가드를 둔다.

```
num_layers == 1
→ dropout = 0.0
```

현재 tuning grid에는 dropout 자체는 포함하지 않는다.

## 9. 학습

### `train_lstm()`

**Optimizer**

```
Adam
```

**Loss**

```
MSELoss
```

Tuning 단계에서는 Validation loss 기반 early stopping을 사용한다.

학습 중 다음 두 값을 저장한다.

**`best_state_dict`**

Validation loss가 가장 낮았던 epoch의 모델 weight.

Early stopping patience가 끝난 마지막 모델 weight가 아니라 반드시 최적 validation 시점 weight를 복원한다.

**`best_epoch`**

Validation loss가 가장 낮았던 시점까지의 총 학습 epoch 수로 정의한다.

예를 들어 0-based epoch index 17에서 validation loss가 최소였다면:

```
best_epoch = 18
```

로 저장한다.

이렇게 정의하여 final training에서 발생할 수 있는 off-by-one 오류를 방지한다.

`best_state_dict`와 `best_epoch`은 각

```
(target, variant, config)
```

조합별로 독립적으로 관리한다.

## 10. 재현성

LSTM은 초기 weight와 mini-batch 학습에 따라 결과가 달라질 수 있으므로 seed를 고정한다.

다음 seed를 모두 설정한다.

```
random
numpy
torch
torch.cuda
```

GPU 환경에서도 가능한 수준의 재현성을 확보하기 위해:

```
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

를 설정한다.

단, PyTorch/CUDA 연산 특성상 플랫폼이나 라이브러리 버전까지 다른 환경에서 bit-level 완전 동일성을 항상 보장하는 것은 별개의 문제다.

## 11. Hyperparameter tuning

### `tune_sequence_models()`

다음 grid를 탐색한다.

```
window_size:
5, 10, 20

hidden_size:
16, 32

num_layers:
1, 2

include_past_target:
False, True
```

총:

```
3 × 2 × 2 × 2
= 24 configurations
```

Ni/Co 각각 별도의 모델로 수행한다.

**모델 선택 기준**

기존 `tune_models()`와 동일하게:

```
Validation R² 최대
```

를 기준으로 최적 configuration을 선택한다.

함께 기록할 지표:

```
val_r2
val_rmse
val_mae
best_epoch
```

**Test 사용 원칙**

Tuning 과정에서 Test는 다음 항목에 절대 사용하지 않는다.

- scaler fit
- model training
- early stopping
- hyperparameter selection
- best epoch 결정
- configuration 간 성능 비교

다만 전체 시간축의 row 구조를 유지하기 위해 `build_sequences()`가 Test row까지 포함한 DataFrame을 순회할 수는 있다.

이 경우에도 Test target이나 Test 성능은 tuning decision에 사용하지 않는다.

## 12. Final evaluation

### `evaluate_final_sequence_model()`

Validation으로 최적 configuration이 확정된 후 최종 모델을 생성한다.

**Step 1. Train + Validation 결합**

```
Final training data
= Train + Validation
```

Test는 포함하지 않는다.

**Step 2. scaler 재학습**

```
feature scaler.fit(Train + Validation)

LSTM-X+y인 경우:
history target scaler.fit(Train + Validation target)
```

**Step 3. 전체 시간축 transform**

```
Train + Validation + Test
```

전체 row에 대해 앞서 fit한 scaler로 transform한다.

Test는 scaler fit에는 전혀 사용되지 않는다.

**Step 4. 전체 시간축에서 sequence 재생성**

전체 DataFrame을 이용하여 sequence를 다시 생성한다.

이렇게 하는 이유는 Test 초반 target의 과거 context를 유지하기 위해서다.

예:

```
Train | Validation | Test
                 ↑
          Test 첫 target의
          과거 context
```

Train/Validation의 마지막 배치들이 Test 첫 target의 입력에 포함될 수 있다.

Test 후반의 target은 이전 Test row의 feature를 과거 context로 사용할 수도 있다.

LSTM-X+y에서는 rolling one-step-ahead 가정이 충족되는 경우 이전 Test 배치의 실제 회수율 역시 이후 Test target의 history input으로 사용할 수 있다.

**Step 5. Final model 재학습**

Tuning 단계에서 선택된:

```
window_size
hidden_size
num_layers
include_past_target
best_epoch
```

를 그대로 사용한다.

Train+Validation sequence로:

```
best_epoch
```

만큼 정확히 학습한다.

Final 단계에서는 early stopping을 다시 수행하지 않는다.

Test를 early stopping에 사용하는 것도 금지한다.

**Step 6. Test 평가**

모든 선택 및 최종 학습이 끝난 뒤 Test를 1회 평가한다.

반환 지표:

```
test_r2
test_rmse
test_mae
```

Test 결과를 본 뒤 하이퍼파라미터나 window size를 다시 변경하지 않는다.

## 13. 결과 비교 프레임

모든 모델을 단순히 한 줄로 순위를 매기기보다 입력 정보와 모델 성격에 따라 세 층위로 구분한다.

**1) Tabular baseline**

```
RandomForest
XGBoost
LightGBM
```

각 배치를 독립된 행으로 처리한다.

**2) Sequence model**

```
LSTM-X
```

과거 N개 배치의 공정 feature 흐름을 이용한다.

과거 회수율은 사용하지 않는다.

**3) Autoregressive sequence model**

```
LSTM-X+y
```

과거 공정 feature + 실제 과거 회수율을 입력으로 사용한다.

LSTM-X보다 입력 정보가 많으며, 직전 배치의 실제 회수율이 다음 배치 예측 전에 확보된다는 추가적인 운영 가정이 필요하다.

따라서 결과 해석 시:

```
LSTM-X vs Tabular models
→ 시계열 구조 자체의 효과

LSTM-X vs LSTM-X+y
→ 과거 실제 회수율 정보 추가 효과
```

로 구분해서 해석한다.

## 14. 반드시 방지해야 하는 핵심 버그

**① 현재 target leakage**

LSTM-X+y에서 target row의 현재 회수율이 입력 sequence에 포함되면 안 된다.

```
허용:
y(t-N) ... y(t-1)

금지:
y(t)
```

가장 우선적으로 테스트한다.

**② scaler leakage**

Tuning에서는 Validation/Test가 scaler fit에 포함되면 안 된다.

```
Tuning:
Train only

Final:
Train + Validation only
```

Test는 어떠한 경우에도 scaler 통계 계산에 사용하지 않는다.

**③ split 선분리**

다음 방식은 금지한다.

```
Train 분리 → sequence
Validation 분리 → sequence
Test 분리 → sequence
```

이렇게 하면 Validation/Test 시작 부분의 정상적인 과거 context가 사라진다.

반드시:

```
전체 시간축 유지
↓
sequence 생성
↓
target row 기준 split assignment
```

순서를 지킨다.

Tuning과 Final 모두 동일하다.

## 15. 테스트

파일:

```
module2_prediction/tests/test_sequence_modeling.py
```

**Test 1. Window 크기 정확성**

`window_size=5`이면 target마다 정확히 직전 5개 row만 입력되는지 확인한다.

**Test 2. Target 기준 split 할당**

Sequence 내부에 어느 split의 row가 들어갔는지가 아니라:

```
target row의 split
```

이 최종 sequence split label이 되는지 확인한다.

**Test 3a. Unit — 경계 context**

`build_sequences()` 자체에서 Validation 첫 target의 window가 실제 Train 마지막 row들을 포함하는지 확인한다.

동일하게 Test 첫 target의 window가 Validation 마지막 row들을 포함할 수 있는지 확인한다.

**Test 3b. Integration — Final 경계 context**

`evaluate_final_sequence_model()`의 실제 실행 경로에서도 전체 df로 sequence가 생성되는지 검증한다.

즉 Final 단계에서 Test 첫 target의 window에 Train+Validation의 마지막 context가 실제로 유지되는지 확인한다.

**Test 4. 미래 정보 미포함**

Target index가 `t`라면 입력 window의 모든 index가 반드시:

```
< t
```

인지 검증한다.

`t` 또는 `t+1` 이상의 row가 들어오면 실패시킨다.

**Test 5. Scaler leakage**

Validation/Test 값을 극단적으로 변경해도 tuning 단계에서 Train으로 fit한 scaler의:

```
mean
scale
```

이 변하지 않는지 확인한다.

Final 단계에서도 Test 값 변경이 Train+Validation scaler 통계에 영향을 주지 않는지 확인한다.

**Test 6. X+y 현재 target 미포함**

가장 중요한 leakage 테스트.

Synthetic data에서 target 값을 row index와 쉽게 구분 가능한 값으로 설정한 후:

```
현재 target y(t)
```

가 해당 sequence 입력에 존재하지 않는지 명시적으로 검증한다.

허용되는 것은:

```
y(t-N) ... y(t-1)
```

뿐이다.

**Test 7. Output shape**

LSTM-X:

```
(samples, N, 13)
```

LSTM-X+y:

```
(samples, N, 14)
```

인지 확인한다.

**Test 8. Smoke test**

작은 synthetic dataset으로:

```
sequence 생성
→ scaler
→ model 생성
→ fit
→ predict
```

전체 흐름이 실행되는지 확인한다.

예측 결과에:

```
NaN
Inf
```

가 존재하지 않아야 한다.

## 16. 최종 데이터 흐름

**Tuning**

```
전체 시간순 DataFrame
        ↓
Train만으로 feature scaler fit
(+ X+y이면 Train target으로 history-y scaler fit)
        ↓
전체 DataFrame transform
        ↓
전체 시간축에서 sequence 생성
        ↓
target row 기준 Train / Validation / Test label
        ↓
Train sequence → 학습
Validation sequence → early stopping 및 config 선택
Test → 선택 과정에 사용하지 않음
        ↓
best config + best_epoch 저장
```

**Final**

```
전체 시간순 DataFrame
        ↓
Train + Validation으로 scaler refit
        ↓
전체 DataFrame transform
        ↓
전체 시간축에서 sequence 재생성
        ↓
Train + Validation target sequence → final training
        ↓
best_epoch 고정 학습
early stopping 없음
        ↓
Test target sequence → 최종 1회 평가
        ↓
R² / RMSE / MAE
```

## 17. 최종 모델 후보

```
Tabular
├─ RandomForest
├─ XGBoost
└─ LightGBM

Sequence
└─ LSTM-X

Autoregressive Sequence
└─ LSTM-X+y
```

최종 비교에서는 단순 성능 순위뿐 아니라:

- 시계열 정보가 실제 성능 개선에 기여했는가
- 과거 실제 회수율 추가가 어느 정도 효과를 냈는가
- 그 성능 향상이 추가적인 운영 가정을 감수할 만큼 의미 있는가

를 함께 해석한다.

## 18. 의존성 및 환경

`requirements.txt`에 `torch`를 추가한다.

설치는 **`torchcu` 환경에만** 수행한다 (`base` 환경은 건드리지 않음).
