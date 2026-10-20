import numpy as np
import pandas as pd
import pytest
import torch

from module2_prediction.sequence_modeling import (
    LSTMRegressor,
    build_sequences,
    fit_feature_scaler,
    fit_history_target_scaler,
    set_seed,
    transform_sequence_inputs,
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
