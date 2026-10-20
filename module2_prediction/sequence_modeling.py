import copy
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from module2_prediction.config import FEATURE_COLUMNS, TARGET_COLUMNS


def build_sequences(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    window_size: int,
    include_past_target: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str], list[int]]:
    """전체 시간순 df에서 각 target row마다 직전 window_size개 행을 window로 구성한다.

    split은 window가 걸치는 범위가 아니라 target row 자신의 `split` 값으로 정해진다.
    window_size보다 앞선 행이 없는 target(맨 앞 window_size개 행)은 생성하지 않는다.

    include_past_target=True면 f"{target_column}_history" 컬럼(transform_sequence_inputs()가
    생성하는, 스케일된 과거 target)을 각 timestep 입력의 마지막 차원에 추가한다. 과거 배치의
    실제 회수율이 다음 배치 예측 전에 이미 확보되어 있다는 운영 가정 하에서만 유효하다
    (설계 문서 섹션 5.2 참고, 이 데이터셋에 대해 별도로 검증되지 않은 가정).
    """
    history_column = f"{target_column}_history"
    if include_past_target and history_column not in df.columns:
        raise ValueError(
            f"include_past_target=True면 '{history_column}' 컬럼이 필요합니다 — "
            "transform_sequence_inputs()로 먼저 생성하세요."
        )

    X_list: list[np.ndarray] = []
    y_list: list[float] = []
    split_labels: list[str] = []
    target_indices: list[int] = []

    for i in range(window_size, len(df)):
        window = df.iloc[i - window_size : i]
        feats = window[feature_columns].to_numpy(dtype=float)

        if include_past_target:
            history = window[[history_column]].to_numpy(dtype=float)
            feats = np.concatenate([feats, history], axis=1)

        X_list.append(feats)
        y_list.append(float(df.iloc[i][target_column]))
        split_labels.append(df.iloc[i]["split"])
        target_indices.append(i)

    n_features = len(feature_columns) + (1 if include_past_target else 0)
    X = np.stack(X_list) if X_list else np.empty((0, window_size, n_features))
    y = np.array(y_list, dtype=float)

    return X, y, split_labels, target_indices


def fit_feature_scaler(df: pd.DataFrame, feature_columns: list[str]) -> StandardScaler:
    """feature_columns에 대해서만 StandardScaler를 fit한다. 호출자가 넘긴 행만 사용한다
    (tuning 단계에서는 Train만, final 단계에서는 Train+Validation을 넘겨야 한다)."""
    scaler = StandardScaler()
    scaler.fit(df[feature_columns])
    return scaler


def fit_history_target_scaler(df: pd.DataFrame, target_column: str) -> StandardScaler:
    """LSTM-X+y에서 과거 입력(history)으로 쓰는 target 값 전용 scaler다. 현재 예측
    대상 y에도 별도의 target_scaler가 쓰이지만(tune_sequence_models/
    evaluate_final_sequence_model이 직접 관리하는 다른 StandardScaler 인스턴스),
    이 함수가 반환하는 scaler는 그것과 다른 scaler로, history input에만 쓴다."""
    scaler = StandardScaler()
    scaler.fit(df[[target_column]])
    return scaler


def transform_sequence_inputs(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    feature_scaler: StandardScaler,
    history_target_scaler: StandardScaler | None = None,
) -> pd.DataFrame:
    """feature_columns는 feature_scaler로 transform하고, history_target_scaler가 있으면
    f"{target_column}_history" 컬럼(과거 입력용 scaled target)을 추가한다.

    target_column 자체(현재 예측 대상)는 원 단위(raw)로 그대로 둔다.
    """
    result = df.copy()
    result[feature_columns] = feature_scaler.transform(df[feature_columns])

    if history_target_scaler is not None:
        result[f"{target_column}_history"] = history_target_scaler.transform(df[[target_column]])

    return result


def set_seed(seed: int = 42) -> None:
    """random/numpy/torch/cuda 시드를 고정하고 cuDNN을 deterministic 모드로 설정한다.

    cuDNN 커널 특성상 플랫폼/라이브러리 버전까지 다른 환경에서 bit-level 완전 동일성을
    항상 보장하지는 않는다.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class LSTMRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(x)
        last_hidden = hidden[-1]
        return self.head(last_hidden).squeeze(-1)


def train_lstm(
    model: "LSTMRegressor",
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    max_epochs: int = 500,
    patience: int = 30,
    lr: float = 1e-3,
    batch_size: int = 32,
    seed: int = 42,
) -> tuple[dict, int]:
    """Validation loss 기준 early stopping으로 학습한다 (mini-batch).

    y_train/y_val은 호출자가 이미 스케일링해서 넘긴다고 가정한다 — 이 함수는 scaler를
    전혀 모른다 (X와 동일한 원칙). Validation은 배치로 나누지 않고 한 번에 forward한다.

    반환값: (best_state_dict, best_epoch).
    best_state_dict는 validation loss가 가장 낮았던 시점의 가중치다 (patience가 끝난
    시점의 마지막 가중치가 아니다). best_epoch은 그 시점까지의 총 epoch 수로 정의한다
    (0-based 인덱스가 아니라 인덱스 + 1).
    """
    set_seed(seed)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    best_val_loss = float("inf")
    best_state_dict = copy.deepcopy(model.state_dict())
    best_epoch_index = 0
    epochs_without_improvement = 0

    for epoch_index in range(max_epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = loss_fn(preds, y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_t)
            val_loss = loss_fn(val_preds, y_val_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = copy.deepcopy(model.state_dict())
            best_epoch_index = epoch_index
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    best_epoch = best_epoch_index + 1
    return best_state_dict, best_epoch


SEQUENCE_HYPERPARAMETER_GRID = {
    "window_size": [5, 10, 20],
    "hidden_size": [16, 32],
    "num_layers": [1, 2],
    "include_past_target": [False, True],
}


def _iter_sequence_configs(grid: dict):
    for window_size in grid["window_size"]:
        for hidden_size in grid["hidden_size"]:
            for num_layers in grid["num_layers"]:
                for include_past_target in grid["include_past_target"]:
                    yield {
                        "window_size": window_size,
                        "hidden_size": hidden_size,
                        "num_layers": num_layers,
                        "include_past_target": include_past_target,
                    }


_VARIANT_NAMES = {False: "LSTM-X", True: "LSTM-X+y"}


def tune_sequence_models(
    df: pd.DataFrame,
    feature_columns: list[str] = FEATURE_COLUMNS,
    target_columns: list[str] = TARGET_COLUMNS,
    grid: dict = SEQUENCE_HYPERPARAMETER_GRID,
    max_epochs: int = 500,
    patience: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
) -> dict:
    """grid의 각 조합을 Validation R²로 평가해 타겟별 최적 config를 고른다.

    LSTM-X(include_past_target=False)와 LSTM-X+y(include_past_target=True)는
    서로 독립적으로 최적화된다 — 한쪽이 다른 쪽보다 val_r2가 낮다는 이유로
    버려지지 않고, 둘 다 각자의 최적 config로 반환된다.

    현재 target(y)은 Train만으로 fit한 StandardScaler로 스케일링해서 학습하고(LSTM이
    85% 같은 큰 값을 그대로 맞추려다 수렴하지 못하는 문제를 막기 위함), 예측값만
    inverse_transform해서 원 단위로 되돌린 뒤 metric을 계산한다 — true y는 항상
    build_sequences()가 준 원본(스케일 안 한) 값을 그대로 쓴다.

    Test는 scaler fit, 학습, early stopping, config 선택 중 어디에도 쓰지 않는다
    (df에 Test 행이 포함돼 있어도 전체 시간축을 유지하기 위한 것일 뿐, 선택 과정에는 전혀 반영되지 않는다).

    반환값: {(variant, target): {"config", "best_epoch", "batch_size", "lr",
    "val_r2", "val_rmse", "val_mae"}}. variant는 "LSTM-X" 또는 "LSTM-X+y".
    """
    train_df = df[df["split"] == "train"].reset_index(drop=True)

    best: dict = {}
    for target in target_columns:
        for config in _iter_sequence_configs(grid):
            window_size = config["window_size"]
            include_past_target = config["include_past_target"]

            feature_scaler = fit_feature_scaler(train_df, feature_columns)
            history_target_scaler = (
                fit_history_target_scaler(train_df, target) if include_past_target else None
            )

            transformed = transform_sequence_inputs(
                df, feature_columns, target, feature_scaler, history_target_scaler
            )

            X, y, split_labels, _ = build_sequences(
                transformed, feature_columns, target, window_size, include_past_target
            )
            split_labels_arr = np.array(split_labels)

            X_train = X[split_labels_arr == "train"]
            y_train = y[split_labels_arr == "train"]
            X_val = X[split_labels_arr == "validation"]
            y_val = y[split_labels_arr == "validation"]

            target_scaler = StandardScaler()
            target_scaler.fit(y_train.reshape(-1, 1))
            y_train_scaled = target_scaler.transform(y_train.reshape(-1, 1)).ravel()
            y_val_scaled = target_scaler.transform(y_val.reshape(-1, 1)).ravel()

            input_size = len(feature_columns) + (1 if include_past_target else 0)
            set_seed()
            model = LSTMRegressor(
                input_size=input_size,
                hidden_size=config["hidden_size"],
                num_layers=config["num_layers"],
            )

            best_state_dict, best_epoch = train_lstm(
                model, X_train, y_train_scaled, X_val, y_val_scaled,
                max_epochs=max_epochs, patience=patience, batch_size=batch_size, lr=lr,
            )
            model.load_state_dict(best_state_dict)

            model.eval()
            with torch.no_grad():
                val_preds_scaled = model(torch.tensor(X_val, dtype=torch.float32)).numpy()
            val_preds = target_scaler.inverse_transform(val_preds_scaled.reshape(-1, 1)).ravel()

            metrics = {
                "val_r2": float(r2_score(y_val, val_preds)),
                "val_rmse": float(np.sqrt(mean_squared_error(y_val, val_preds))),
                "val_mae": float(mean_absolute_error(y_val, val_preds)),
            }

            entry = {
                "config": config,
                "best_epoch": best_epoch,
                "batch_size": batch_size,
                "lr": lr,
                **metrics,
            }

            variant = _VARIANT_NAMES[include_past_target]
            key = (variant, target)
            if key not in best or metrics["val_r2"] > best[key]["val_r2"]:
                best[key] = entry

    return best


def _build_final_sequences(
    df: pd.DataFrame,
    target: str,
    config: dict,
    feature_columns: list[str] = FEATURE_COLUMNS,
) -> dict:
    """evaluate_final_sequence_model()이 쓰는 Step 1~4: Train+Validation으로 scaler를
    재fit하고, 전체 df를 transform한 뒤 전체 시간축에서 시퀀스를 재생성한다.

    Test 초반 target의 window가 Train+Validation 마지막 행들을 포함할 수 있도록
    df를 split별로 먼저 자르지 않는다.
    """
    window_size = config["window_size"]
    include_past_target = config["include_past_target"]

    train_val_df = df[df["split"].isin(["train", "validation"])].reset_index(drop=True)

    feature_scaler = fit_feature_scaler(train_val_df, feature_columns)
    history_target_scaler = (
        fit_history_target_scaler(train_val_df, target) if include_past_target else None
    )

    transformed = transform_sequence_inputs(
        df, feature_columns, target, feature_scaler, history_target_scaler
    )

    X, y, split_labels, target_indices = build_sequences(
        transformed, feature_columns, target, window_size, include_past_target
    )
    split_labels_arr = np.array(split_labels)
    target_indices_arr = np.array(target_indices)

    train_val_mask = np.isin(split_labels_arr, ["train", "validation"])
    test_mask = split_labels_arr == "test"

    return {
        "X_train_val": X[train_val_mask],
        "y_train_val": y[train_val_mask],
        "X_test": X[test_mask],
        "y_test": y[test_mask],
        "test_target_indices": target_indices_arr[test_mask],
    }


def evaluate_final_sequence_model(
    df: pd.DataFrame,
    target: str,
    config: dict,
    best_epoch: int,
    feature_columns: list[str] = FEATURE_COLUMNS,
    batch_size: int = 32,
    lr: float = 1e-3,
) -> dict:
    """Train+Validation으로 scaler를 재fit하고, 선택된 config로 best_epoch만큼 고정
    학습한 뒤(early stopping 없음) Test를 1회 평가한다.

    현재 target(y)도 Train+Validation으로 fit한 StandardScaler로 스케일링해서 학습하고,
    Test 예측값만 inverse_transform해서 원 단위로 되돌린 뒤 metric을 계산한다 — true y는
    항상 _build_final_sequences()가 준 원본(스케일 안 한) 값을 그대로 쓴다. tuning
    단계(tune_sequence_models)와 동일한 mini-batch 방식(batch_size, lr)으로 학습해야
    best_epoch의 의미가 두 단계에서 일관된다.
    """
    sequences = _build_final_sequences(df, target, config, feature_columns)

    include_past_target = config["include_past_target"]
    input_size = len(feature_columns) + (1 if include_past_target else 0)
    set_seed()
    model = LSTMRegressor(
        input_size=input_size,
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
    )

    target_scaler = StandardScaler()
    target_scaler.fit(sequences["y_train_val"].reshape(-1, 1))
    y_train_val_scaled = target_scaler.transform(sequences["y_train_val"].reshape(-1, 1)).ravel()

    X_train_val_t = torch.tensor(sequences["X_train_val"], dtype=torch.float32)
    y_train_val_t = torch.tensor(y_train_val_scaled, dtype=torch.float32)
    train_dataset = TensorDataset(X_train_val_t, y_train_val_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    model.train()
    for _ in range(best_epoch):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = loss_fn(preds, y_batch)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        test_preds_scaled = model(torch.tensor(sequences["X_test"], dtype=torch.float32)).numpy()
    test_preds = target_scaler.inverse_transform(test_preds_scaled.reshape(-1, 1)).ravel()

    y_test = sequences["y_test"]
    return {
        "test_r2": float(r2_score(y_test, test_preds)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, test_preds))),
        "test_mae": float(mean_absolute_error(y_test, test_preds)),
    }
