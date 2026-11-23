import pandas as pd

from module3_optimization.config import TARGETS
from module3_optimization.scripts.margin_analysis import (
    co_threshold_comparison, feasibility_counts, margin_sweep,
)

NI, CO = TARGETS
TEST_TARGETS = {NI: 95.0, CO: 90.0}  # 운영 TARGETS 값과 분리된 테스트 전용 목표값


def synthetic_wide():
    """기존 통과 1건·완화로 되찾는 1건·완화로 추가되는 실측 미달 1건·무관 1건으로 구성 (TEST_TARGETS=95/90 기준)"""
    rows = [
        dict(배치ID='baseline_pass', actual_ni=96.0, actual_co=91.0, predicted_ni=96.0, predicted_co=91.0),
        dict(배치ID='recaptured', actual_ni=96.0, actual_co=91.0, predicted_ni=96.0, predicted_co=89.6),
        dict(배치ID='false_after_relax', actual_ni=96.0, actual_co=85.0, predicted_ni=96.0, predicted_co=89.6),
        dict(배치ID='clean_reject', actual_ni=80.0, actual_co=70.0, predicted_ni=80.0, predicted_co=70.0),
    ]
    df = pd.DataFrame(rows).set_index('배치ID')
    actual = df[['actual_ni', 'actual_co']].rename(columns={'actual_ni': NI, 'actual_co': CO})
    predicted = df[['predicted_ni', 'predicted_co']].rename(columns={'predicted_ni': NI, 'predicted_co': CO})
    return actual, predicted


def test_feasibility_counts_matches_hand_calculation():
    actual, predicted = synthetic_wide()
    counts = feasibility_counts(actual, predicted, targets=TEST_TARGETS)
    assert counts == dict(total_batches=4, predicted_feasible_batches=1,
                           actual_missed_among_predicted_feasible=0,
                           miss_rate_among_predicted_feasible=0.0)


def test_margin_sweep_zero_margin_matches_feasibility_counts():
    actual, predicted = synthetic_wide()
    counts = feasibility_counts(actual, predicted, targets=TEST_TARGETS)
    row = margin_sweep(actual, predicted, targets=TEST_TARGETS, candidates=[0.0]).iloc[0]
    assert row['passed_batches'] == counts['predicted_feasible_batches']
    assert row['actual_missed_among_passed'] == counts['actual_missed_among_predicted_feasible']


def test_co_threshold_comparison_recaptures_and_adds_false_candidate():
    actual, predicted = synthetic_wide()
    table = co_threshold_comparison(actual, predicted, [90.0, 89.5], targets=TEST_TARGETS)
    baseline, relaxed = table.iloc[0], table.iloc[1]
    assert (baseline['passed_batches'], baseline['actual_achieved_among_passed'],
            baseline['actual_missed_among_passed']) == (1, 1, 0)
    assert (relaxed['passed_batches'], relaxed['actual_achieved_among_passed'],
            relaxed['actual_missed_among_passed']) == (3, 2, 1)
