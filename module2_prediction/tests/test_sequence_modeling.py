import numpy as np
import pandas as pd
import pytest
import torch

from module2_prediction.sequence_modeling import (
    LSTMRegressor,
    _build_final_sequences,
    build_sequences,
    evaluate_final_sequence_model,
    fit_feature_scaler,
    fit_history_target_scaler,
    set_seed,
    train_lstm,
    transform_sequence_inputs,
    tune_sequence_models,
)


def _make_fixture(n_train=12, n_val=6, n_test=6, seed=0):
    """시간순 합성 데이터: feature 2개(f1, f2), target(y), 행 순서 추적용 idx, split 컬럼."""
    rng = np.random.default_rng(seed)
    n = n_train + n_val + n_test
    df = pd.DataFrame({
        "idx": np.arange(n),
        "f1": rng.normal(0, 1, n),
        "f2": rng.normal(0, 1, n),
        "y": rng.normal(50, 5, n),
        "split": ["train"] * n_train + ["validation"] * n_val + ["test"] * n_test,
    })
    return df


def test_build_sequences_window_contains_exactly_preceding_n_rows():
    df = _make_fixture()
    window_size = 5

    X, y, split_labels, target_indices = build_sequences(
        df, ["idx"], "y", window_size=window_size
    )

    sample_position = target_indices.index(10)
    expected_window = df.loc[5:9, "idx"].to_numpy().reshape(window_size, 1)

    np.testing.assert_array_equal(X[sample_position], expected_window)


def test_build_sequences_assigns_split_by_target_row_not_window():
    df = _make_fixture(n_train=12, n_val=6, n_test=6)
    window_size = 5

    _, _, split_labels, target_indices = build_sequences(df, ["f1", "f2"], "y", window_size=window_size)

    for position, target_index in enumerate(target_indices):
        assert split_labels[position] == df.loc[target_index, "split"]


def test_build_sequences_validation_first_target_window_spans_into_train():
    df = _make_fixture(n_train=12, n_val=6, n_test=6)
    window_size = 5

    X, _, split_labels, target_indices = build_sequences(df, ["idx"], "y", window_size=window_size)

    first_val_position = split_labels.index("validation")
    first_val_target_index = target_indices[first_val_position]

    assert first_val_target_index == 12
    # window = idx 7..11, 전부 train 구간(0~11)에 속함
    window_values = X[first_val_position, :, 0]
    np.testing.assert_array_equal(window_values, np.arange(7, 12))
    for idx_value in window_values:
        assert df.loc[int(idx_value), "split"] == "train"


def test_build_sequences_test_first_target_window_spans_into_validation():
    df = _make_fixture(n_train=12, n_val=6, n_test=6)
    window_size = 5

    X, _, split_labels, target_indices = build_sequences(df, ["idx"], "y", window_size=window_size)

    first_test_position = split_labels.index("test")
    first_test_target_index = target_indices[first_test_position]

    assert first_test_target_index == 18
    window_values = X[first_test_position, :, 0]
    np.testing.assert_array_equal(window_values, np.arange(13, 18))
    for idx_value in window_values:
        assert df.loc[int(idx_value), "split"] == "validation"


def test_build_sequences_never_includes_target_row_or_future_rows():
    df = _make_fixture()
    window_size = 4

    X, _, _, target_indices = build_sequences(df, ["idx"], "y", window_size=window_size)

    for position, target_index in enumerate(target_indices):
        window_values = X[position, :, 0]
        assert all(idx_value < target_index for idx_value in window_values)


def test_build_sequences_excludes_current_target_when_using_past_target():
    df = _make_fixture(n_train=12, n_val=6, n_test=6)
    df["y"] = df["idx"] * 1000.0
    df["y_history"] = df["y"]
    window_size = 4

    X, y, _, target_indices = build_sequences(
        df, ["f1", "f2"], "y", window_size=window_size, include_past_target=True
    )

    for position, target_index in enumerate(target_indices):
        current_target_value = target_index * 1000.0
        history_values = X[position, :, -1]
        assert current_target_value not in history_values
        expected_history = np.arange(target_index - window_size, target_index) * 1000.0
        np.testing.assert_array_equal(history_values, expected_history)


def test_build_sequences_requires_history_column_when_include_past_target_true():
    df = _make_fixture()

    with pytest.raises(ValueError):
        build_sequences(df, ["f1", "f2"], "y", window_size=3, include_past_target=True)


def test_build_sequences_output_shape_for_x_and_xy_variants():
    df = _make_fixture()
    df["y_history"] = df["y"]
    window_size = 5

    X_only, y_only, _, _ = build_sequences(df, ["f1", "f2"], "y", window_size=window_size)
    X_xy, y_xy, _, _ = build_sequences(df, ["f1", "f2"], "y", window_size=window_size, include_past_target=True)

    assert X_only.shape[1:] == (window_size, 2)
    assert X_xy.shape[1:] == (window_size, 3)
    assert X_only.shape[0] == X_xy.shape[0] == len(y_only) == len(y_xy)


def test_fit_feature_scaler_matches_given_rows_statistics():
    df = _make_fixture()
    train_df = df[df["split"] == "train"]

    scaler = fit_feature_scaler(train_df, ["f1", "f2"])

    expected_mean = train_df[["f1", "f2"]].mean().to_numpy()
    np.testing.assert_allclose(scaler.mean_, expected_mean)


def test_fit_feature_scaler_unaffected_by_rows_not_passed_in():
    df = _make_fixture()
    train_df = df[df["split"] == "train"].copy()
    scaler_before = fit_feature_scaler(train_df, ["f1", "f2"])

    df.loc[df["split"] != "train", ["f1", "f2"]] = 1e6
    scaler_after = fit_feature_scaler(train_df, ["f1", "f2"])

    np.testing.assert_allclose(scaler_before.mean_, scaler_after.mean_)


def test_fit_history_target_scaler_matches_given_rows_statistics():
    df = _make_fixture()
    train_df = df[df["split"] == "train"]

    scaler = fit_history_target_scaler(train_df, "y")

    assert scaler.mean_[0] == pytest.approx(train_df["y"].mean())


def test_transform_sequence_inputs_scales_features_and_adds_history_column():
    df = _make_fixture()
    train_df = df[df["split"] == "train"]
    feature_scaler = fit_feature_scaler(train_df, ["f1", "f2"])
    history_scaler = fit_history_target_scaler(train_df, "y")

    transformed = transform_sequence_inputs(df, ["f1", "f2"], "y", feature_scaler, history_scaler)

    assert "y_history" in transformed.columns
    pd.testing.assert_series_equal(transformed["y"], df["y"])
    assert not transformed["f1"].equals(df["f1"])


def test_transform_sequence_inputs_without_history_scaler_omits_history_column():
    df = _make_fixture()
    train_df = df[df["split"] == "train"]
    feature_scaler = fit_feature_scaler(train_df, ["f1", "f2"])

    transformed = transform_sequence_inputs(df, ["f1", "f2"], "y", feature_scaler, history_target_scaler=None)

    assert "y_history" not in transformed.columns


def test_lstm_regressor_forward_pass_shape():
    model = LSTMRegressor(input_size=3, hidden_size=8, num_layers=1)
    x = torch.rand(4, 5, 3)

    output = model(x)

    assert output.shape == (4,)


def test_lstm_regressor_forces_zero_dropout_when_single_layer():
    model = LSTMRegressor(input_size=3, hidden_size=8, num_layers=1, dropout=0.5)

    assert model.lstm.dropout == 0.0


def test_lstm_regressor_keeps_dropout_when_multiple_layers():
    model = LSTMRegressor(input_size=3, hidden_size=8, num_layers=2, dropout=0.3)

    assert model.lstm.dropout == 0.3


def test_set_seed_makes_weight_initialization_reproducible():
    set_seed(123)
    model_a = LSTMRegressor(input_size=3, hidden_size=8, num_layers=1)

    set_seed(123)
    model_b = LSTMRegressor(input_size=3, hidden_size=8, num_layers=1)

    for p_a, p_b in zip(model_a.parameters(), model_b.parameters()):
        assert torch.equal(p_a, p_b)


def _val_loss(model, X_val, y_val) -> float:
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_val, dtype=torch.float32))
        return torch.nn.functional.mse_loss(
            preds, torch.tensor(y_val, dtype=torch.float32)
        ).item()


def test_train_lstm_returns_best_epoch_within_bounds():
    rng = np.random.default_rng(1)
    X_train = rng.normal(0, 1, (16, 4, 3)).astype(np.float32)
    y_train = (X_train[:, -1, 0] * 2.0).astype(np.float32)
    X_val = rng.normal(0, 1, (8, 4, 3)).astype(np.float32)
    y_val = (X_val[:, -1, 0] * 2.0).astype(np.float32)

    model = LSTMRegressor(input_size=3, hidden_size=4, num_layers=1)
    best_state_dict, best_epoch = train_lstm(
        model, X_train, y_train, X_val, y_val, max_epochs=10, patience=3
    )

    assert 1 <= best_epoch <= 10
    assert set(best_state_dict.keys()) == set(model.state_dict().keys())


def test_train_lstm_restores_best_validation_weights_not_final_epoch_weights():
    rng = np.random.default_rng(2)
    X_train = rng.normal(0, 1, (16, 4, 3)).astype(np.float32)
    y_train = (X_train[:, -1, 0] * 2.0 + rng.normal(0, 0.1, 16)).astype(np.float32)
    X_val = rng.normal(0, 1, (8, 4, 3)).astype(np.float32)
    y_val = (X_val[:, -1, 0] * 2.0 + rng.normal(0, 0.1, 8)).astype(np.float32)

    model = LSTMRegressor(input_size=3, hidden_size=4, num_layers=1)
    # patience >= max_epochs → early stopping이 절대 발동하지 않고 max_epochs를 전부 돈다.
    # 이러면 best_val_loss는 전체 궤적의 최솟값이라는 정의상 final epoch의 loss보다
    # 작거나 같을 수밖에 없다 (수치적 우연이 아니라 알고리즘 자체의 성질).
    best_state_dict, best_epoch = train_lstm(
        model, X_train, y_train, X_val, y_val, max_epochs=40, patience=40
    )

    final_val_loss = _val_loss(model, X_val, y_val)

    restored_model = LSTMRegressor(input_size=3, hidden_size=4, num_layers=1)
    restored_model.load_state_dict(best_state_dict)
    restored_val_loss = _val_loss(restored_model, X_val, y_val)

    assert restored_val_loss <= final_val_loss + 1e-6


def _make_tuning_fixture(n_train=20, n_val=8, n_test=8, seed=3):
    rng = np.random.default_rng(seed)
    n = n_train + n_val + n_test
    df = pd.DataFrame({
        "f1": rng.normal(0, 1, n),
        "f2": rng.normal(0, 1, n),
        "split": ["train"] * n_train + ["validation"] * n_val + ["test"] * n_test,
    })
    df["y"] = df["f1"] * 3 + rng.normal(0, 0.5, n)
    return df


def test_tune_sequence_models_optimizes_lstm_x_and_lstm_xy_independently():
    df = _make_tuning_fixture()
    tiny_grid = {
        "window_size": [3],
        "hidden_size": [4],
        "num_layers": [1],
        "include_past_target": [False, True],
    }

    best = tune_sequence_models(
        df, feature_columns=["f1", "f2"], target_columns=["y"],
        grid=tiny_grid, max_epochs=3, patience=1,
    )

    assert set(best.keys()) == {("LSTM-X", "y"), ("LSTM-X+y", "y")}
    assert best[("LSTM-X", "y")]["config"]["include_past_target"] is False
    assert best[("LSTM-X+y", "y")]["config"]["include_past_target"] is True
    for entry in best.values():
        for key in ["config", "best_epoch", "batch_size", "lr", "val_r2", "val_rmse", "val_mae"]:
            assert key in entry


def test_tune_sequence_models_never_uses_test_target():
    df = _make_tuning_fixture()
    df.loc[df["split"] == "test", "y"] = np.nan
    tiny_grid = {
        "window_size": [3],
        "hidden_size": [4],
        "num_layers": [1],
        "include_past_target": [False],
    }

    best = tune_sequence_models(
        df, feature_columns=["f1", "f2"], target_columns=["y"],
        grid=tiny_grid, max_epochs=3, patience=1,
    )

    assert ("LSTM-X", "y") in best
    assert not np.isnan(best[("LSTM-X", "y")]["val_r2"])


def test_final_stage_scalers_unaffected_by_test_row_mutation():
    df = _make_tuning_fixture()
    train_val_df = df[df["split"].isin(["train", "validation"])]

    feature_scaler_before = fit_feature_scaler(train_val_df, ["f1", "f2"])
    history_scaler_before = fit_history_target_scaler(train_val_df, "y")

    df = df.copy()
    df.loc[df["split"] == "test", ["f1", "f2", "y"]] = 1e6

    train_val_df_after = df[df["split"].isin(["train", "validation"])]
    feature_scaler_after = fit_feature_scaler(train_val_df_after, ["f1", "f2"])
    history_scaler_after = fit_history_target_scaler(train_val_df_after, "y")

    np.testing.assert_allclose(feature_scaler_before.mean_, feature_scaler_after.mean_)
    np.testing.assert_allclose(history_scaler_before.mean_, history_scaler_after.mean_)


def test_build_final_sequences_first_test_window_spans_train_val_boundary():
    df = _make_tuning_fixture(n_train=10, n_val=5, n_test=5)
    config = {"window_size": 4, "hidden_size": 4, "num_layers": 1, "include_past_target": False}

    sequences = _build_final_sequences(df, "y", config, feature_columns=["f1", "f2"])

    n_train_val = 15
    first_test_target_index = int(sequences["test_target_indices"][0])

    # Test의 첫 target은 train_val이 끝나는 지점에서 시작한다
    assert first_test_target_index == n_train_val
    # window = [first_test_target_index - window_size, first_test_target_index)이
    # 전부 train_val 구간(인덱스 < n_train_val) 안에 있어야 한다 — final 단계에서도
    # split을 먼저 자르지 않고 전체 df로 시퀀스를 재생성했다는 증거다
    assert first_test_target_index - config["window_size"] < n_train_val


def test_evaluate_final_sequence_model_returns_test_metrics():
    df = _make_tuning_fixture(n_train=20, n_val=8, n_test=8)
    config = {"window_size": 3, "hidden_size": 4, "num_layers": 1, "include_past_target": False}

    metrics = evaluate_final_sequence_model(
        df, "y", config, best_epoch=3, feature_columns=["f1", "f2"]
    )

    for key in ["test_r2", "test_rmse", "test_mae"]:
        assert key in metrics


def test_evaluate_final_sequence_model_predictions_have_no_nan_or_inf():
    df = _make_tuning_fixture(n_train=20, n_val=8, n_test=8)
    config = {"window_size": 3, "hidden_size": 4, "num_layers": 1, "include_past_target": True}

    metrics = evaluate_final_sequence_model(
        df, "y", config, best_epoch=3, feature_columns=["f1", "f2"]
    )

    for value in metrics.values():
        assert np.isfinite(value)


def test_train_lstm_batch_size_controls_steps_per_epoch():
    rng = np.random.default_rng(6)
    X_train = rng.normal(0, 1, (10, 3, 2)).astype(np.float32)
    y_train = rng.normal(0, 1, 10).astype(np.float32)
    X_val = rng.normal(0, 1, (4, 3, 2)).astype(np.float32)
    y_val = rng.normal(0, 1, 4).astype(np.float32)

    model = LSTMRegressor(input_size=2, hidden_size=4, num_layers=1)

    step_calls = []
    original_step = torch.optim.Adam.step

    def counting_step(self, *args, **kwargs):
        step_calls.append(1)
        return original_step(self, *args, **kwargs)

    torch.optim.Adam.step = counting_step
    try:
        train_lstm(model, X_train, y_train, X_val, y_val, max_epochs=1, patience=1, batch_size=4)
    finally:
        torch.optim.Adam.step = original_step

    # 10개 샘플, batch_size=4 → ceil(10/4)=3 step이 1 epoch 안에서 발생해야 한다
    assert len(step_calls) == 3


def test_train_lstm_learns_input_dependent_variation_not_constant_output():
    rng = np.random.default_rng(5)
    n = 64
    X = rng.normal(0, 1, (n, 4, 2)).astype(np.float32)
    # 주의: target을 0 근처로 유지한다(오프셋을 더하지 않음). 이 테스트는 mini-batch
    # 하나만의 효과를 분리해서 보려는 것이고, target이 0에서 멀리 떨어져 있으면(예:
    # +50) Adam이 input-dependent weight보다 bias(평균 맞추기)부터 흡수해버려서
    # mini-batch를 추가해도 입력 의존적 분산이 쉽게 안 생긴다 — 그건 Task 2/3의
    # target scaling이 해결할 별개의 원인이라 여기서 같이 테스트하면 안 된다.
    y = (X[:, -1, 0] * 3.0).astype(np.float32)

    X_train, y_train = X[:48], y[:48]
    X_val, y_val = X[48:], y[48:]

    model = LSTMRegressor(input_size=2, hidden_size=8, num_layers=1)
    best_state_dict, best_epoch = train_lstm(
        model, X_train, y_train, X_val, y_val,
        max_epochs=100, patience=100, batch_size=8,
    )

    restored = LSTMRegressor(input_size=2, hidden_size=8, num_layers=1)
    restored.load_state_dict(best_state_dict)
    restored.eval()
    with torch.no_grad():
        val_preds = restored(torch.tensor(X_val, dtype=torch.float32)).numpy()

    # 예전 full-batch(epoch당 1 step)였다면 이 정도 epoch 수로는 거의 상수(std≈0)에
    # 머물렀다 — mini-batch(batch_size=8, 6 step/epoch)라면 input에 따라 실제로
    # 달라지는 예측을 내야 한다. target이 0 근처라 bias 쪽에 큰 gradient가 쏠리지
    # 않으므로, 이 변동성은 mini-batch 자체의 효과로 귀속할 수 있다.
    assert np.std(val_preds) > 0.5


def test_tune_sequence_models_metrics_in_original_target_scale():
    rng = np.random.default_rng(7)
    n = 80
    df = pd.DataFrame({
        "f1": rng.normal(0, 1, n),
        "f2": rng.normal(0, 1, n),
        "split": ["train"] * 50 + ["validation"] * 15 + ["test"] * 15,
    })
    # 실제 회수율(~85%) 스케일을 흉내낸, 0에서 멀리 떨어진 target
    df["y"] = 85.0 + df["f1"] * 3.0 + rng.normal(0, 0.5, n)

    tiny_grid = {
        "window_size": [3],
        "hidden_size": [8],
        "num_layers": [1],
        "include_past_target": [False],
    }

    best = tune_sequence_models(
        df, feature_columns=["f1", "f2"], target_columns=["y"],
        grid=tiny_grid, max_epochs=100, patience=20, batch_size=8,
    )

    entry = best[("LSTM-X", "y")]
    # target scaling + mini-batch가 제대로 적용됐다면 val_rmse가 target 변동폭(표준편차
    # 몇 단위) 근처여야 한다. inverse_transform을 빼먹었거나 scaled 공간에서 true y와
    # 비교하면(=이번에 고치는 버그 그 자체) val_rmse가 ~85 가까운 값이 되고 val_r2가
    # 극단적으로 음수가 된다 — 지금 우리가 실제 데이터에서 본 val_r2=-177 같은 증상.
    assert entry["val_rmse"] < 20.0
    assert entry["val_r2"] > -5.0


def test_evaluate_final_sequence_model_uses_mini_batches():
    df = _make_tuning_fixture(n_train=20, n_val=8, n_test=8)
    config = {"window_size": 3, "hidden_size": 4, "num_layers": 1, "include_past_target": False}

    step_calls = []
    original_step = torch.optim.Adam.step

    def counting_step(self, *args, **kwargs):
        step_calls.append(1)
        return original_step(self, *args, **kwargs)

    torch.optim.Adam.step = counting_step
    try:
        evaluate_final_sequence_model(
            df, "y", config, best_epoch=1, feature_columns=["f1", "f2"], batch_size=8
        )
    finally:
        torch.optim.Adam.step = original_step

    # train_val 샘플 수 = (20+8) - window_size(3) = 25. batch_size=8 → ceil(25/8)=4
    # step이 best_epoch=1 안에서 발생해야 한다.
    assert len(step_calls) == 4


def test_evaluate_final_sequence_model_metrics_in_original_target_scale():
    rng = np.random.default_rng(8)
    n = 80
    df = pd.DataFrame({
        "f1": rng.normal(0, 1, n),
        "f2": rng.normal(0, 1, n),
        "split": ["train"] * 50 + ["validation"] * 15 + ["test"] * 15,
    })
    df["y"] = 85.0 + df["f1"] * 3.0 + rng.normal(0, 0.5, n)

    config = {"window_size": 3, "hidden_size": 8, "num_layers": 1, "include_past_target": False}

    metrics = evaluate_final_sequence_model(
        df, "y", config, best_epoch=50, feature_columns=["f1", "f2"], batch_size=8
    )

    assert metrics["test_rmse"] < 20.0
    assert metrics["test_r2"] > -5.0


def test_tune_then_evaluate_final_handoff_uses_consistent_batch_size_and_lr():
    df = _make_tuning_fixture(n_train=24, n_val=10, n_test=10)
    tiny_grid = {
        "window_size": [3],
        "hidden_size": [8],
        "num_layers": [1],
        "include_past_target": [False],
    }

    best = tune_sequence_models(
        df, feature_columns=["f1", "f2"], target_columns=["y"],
        grid=tiny_grid, max_epochs=5, patience=5, batch_size=8, lr=5e-4,
    )
    entry = best[("LSTM-X", "y")]
    assert entry["batch_size"] == 8
    assert entry["lr"] == 5e-4

    metrics = evaluate_final_sequence_model(
        df, "y", entry["config"], entry["best_epoch"],
        feature_columns=["f1", "f2"], batch_size=entry["batch_size"], lr=entry["lr"],
    )

    for value in metrics.values():
        assert np.isfinite(value)
