import pandas as pd
import pytest

from module1_preprocessing.splitting import stratified_train_val_test_split


def _make_graded_df(n_per_grade=60):
    grades = ["A"] * n_per_grade + ["B"] * n_per_grade + ["C"] * n_per_grade
    return pd.DataFrame({
        "배치ID": [f"BP-2025-{i:04d}" for i in range(len(grades))],
        "원료등급": grades,
    })


def test_stratified_split_sizes_match_requested_fractions():
    df = _make_graded_df(n_per_grade=60)  # 180행

    train_i, val_i, test_i = stratified_train_val_test_split(df, train_frac=0.7, val_frac=0.15)

    # stratify 특성상 클래스별 반올림으로 ±1 오차가 날 수 있다
    assert len(train_i) == pytest.approx(126, abs=1)
    assert len(val_i) == pytest.approx(27, abs=1)
    assert len(test_i) == pytest.approx(27, abs=1)
    assert len(train_i) + len(val_i) + len(test_i) == len(df)


def test_stratified_split_preserves_grade_ratio_in_each_split():
    df = _make_graded_df(n_per_grade=60)

    train_i, val_i, test_i = stratified_train_val_test_split(df, train_frac=0.7, val_frac=0.15)

    for idx in (train_i, val_i, test_i):
        counts = df.loc[idx, "원료등급"].value_counts(normalize=True)
        for grade in ("A", "B", "C"):
            assert counts[grade] == pytest.approx(1 / 3, abs=0.05)


def test_stratified_split_is_disjoint_and_covers_all_rows():
    df = _make_graded_df(n_per_grade=60)

    train_i, val_i, test_i = stratified_train_val_test_split(df, train_frac=0.7, val_frac=0.15)

    assert set(train_i) & set(val_i) == set()
    assert set(val_i) & set(test_i) == set()
    assert set(train_i) & set(test_i) == set()
    assert set(train_i) | set(val_i) | set(test_i) == set(df.index)


def test_stratified_split_is_deterministic_for_fixed_random_state():
    df = _make_graded_df(n_per_grade=60)

    train_i1, val_i1, test_i1 = stratified_train_val_test_split(df, random_state=42)
    train_i2, val_i2, test_i2 = stratified_train_val_test_split(df, random_state=42)

    assert list(train_i1) == list(train_i2)
    assert list(val_i1) == list(val_i2)
    assert list(test_i1) == list(test_i2)


def test_stratified_split_rejects_fractions_summing_to_one_or_more():
    df = _make_graded_df(n_per_grade=10)

    with pytest.raises(ValueError):
        stratified_train_val_test_split(df, train_frac=0.7, val_frac=0.3)


def test_stratified_split_rejects_non_positive_fractions():
    df = _make_graded_df(n_per_grade=10)

    with pytest.raises(ValueError):
        stratified_train_val_test_split(df, train_frac=0.0, val_frac=0.5)


def test_test_is_reserved_before_development_split():
    from sklearn.model_selection import train_test_split
    df = _make_graded_df(n_per_grade=140)
    train_i, val_i, test_i = stratified_train_val_test_split(df)
    development_i, expected_test = train_test_split(
        df.index, test_size=63, stratify=df['원료등급'], random_state=42,
    )
    assert (len(train_i), len(val_i), len(test_i)) == (294, 63, 63)
    assert list(test_i) == list(expected_test)
    assert set(train_i) | set(val_i) == set(development_i)


def test_split_preserves_unequal_grade_distribution():
    df = pd.DataFrame({'원료등급': ['A'] * 210 + ['B'] * 126 + ['C'] * 84})
    for indices in stratified_train_val_test_split(df):
        counts = df.loc[indices, '원료등급'].value_counts()
        for grade, ratio in {'A': .5, 'B': .3, 'C': .2}.items():
            assert abs(counts[grade] - len(indices) * ratio) <= 1


def test_custom_fractions_and_nonconsecutive_indices():
    df = _make_graded_df(n_per_grade=100)
    df.index = pd.Index(range(5, 605, 2))
    parts = stratified_train_val_test_split(df, train_frac=.6, val_frac=.2)
    assert [len(part) for part in parts] == [180, 60, 60]
    assert set().union(*map(set, parts)) == set(df.index)


def test_split_rejects_zero_rounded_size():
    with pytest.raises(ValueError, match='최소 1개'):
        stratified_train_val_test_split(_make_graded_df(1))
