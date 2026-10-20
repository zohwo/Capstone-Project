import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


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
    """LSTM-X+y에서 과거 입력으로 쓰는 target 값 전용 scaler. 현재 예측 대상 y는
    별도로 스케일링하지 않으므로 이 scaler는 history input에만 쓴다."""
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
