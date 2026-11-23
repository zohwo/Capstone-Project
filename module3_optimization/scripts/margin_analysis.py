"""OOF 오차로 목표 근처 배치 수와 여유폭 후보별 통과·미달을 확인함 — margin 확정 아님, Test 미사용"""
import json
from pathlib import Path

import pandas as pd

from module3_optimization.config import TARGETS

OOF_PATH = Path('module2_prediction/reports/oof_margin/oof_errors.csv')
OUTPUT_DIR = Path('module3_optimization/reports/margin_analysis')
MARGIN_CANDIDATES_PP = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]


def load_wide(oof_path=OOF_PATH):
    """배치ID 기준으로 실측·예측을 타깃별 열로 재구성"""
    oof = pd.read_csv(oof_path)
    if set(oof['target']) != set(TARGETS):
        raise ValueError('OOF 타깃이 module3 TARGETS와 다름')
    actual = oof.pivot(index='배치ID', columns='target', values='actual')
    predicted = oof.pivot(index='배치ID', columns='target', values='prediction')
    return actual, predicted


def feasibility_counts(actual, predicted, targets=TARGETS):
    """여유 없이(margin=0) 예측상 두 목표 모두 충족한 배치 중 실제 미달 건수와 분모"""
    ni, co = targets
    predicted_feasible = (predicted[ni] >= targets[ni]) & (predicted[co] >= targets[co])
    actual_missed = (actual[ni] < targets[ni]) | (actual[co] < targets[co])
    denominator = int(predicted_feasible.sum())
    numerator = int((predicted_feasible & actual_missed).sum())
    return dict(total_batches=len(predicted), predicted_feasible_batches=denominator,
                actual_missed_among_predicted_feasible=numerator,
                miss_rate_among_predicted_feasible=(numerator / denominator) if denominator else None)


def margin_sweep(actual, predicted, targets=TARGETS, candidates=MARGIN_CANDIDATES_PP):
    """여유폭(두 금속에 동일 적용) 후보별 통과 건수와 그중 실제 미달 건수를 같은 표로 비교"""
    ni, co = targets
    actual_missed = (actual[ni] < targets[ni]) | (actual[co] < targets[co])
    rows = []
    for margin in candidates:
        passed = (predicted[ni] >= targets[ni] + margin) & (predicted[co] >= targets[co] + margin)
        rows.append(dict(margin_pp=margin, passed_batches=int(passed.sum()),
                          actual_missed_among_passed=int((passed & actual_missed).sum())))
    return pd.DataFrame(rows)


def co_threshold_comparison(actual, predicted, co_thresholds, targets=TARGETS):
    """Ni 예측 95% 이상을 고정한 채 Co 예측 임계값만 바꿔 통과·실측달성·실측미달 건수를 비교"""
    ni, co = targets
    actual_feasible = (actual[ni] >= targets[ni]) & (actual[co] >= targets[co])
    rows = []
    for threshold in co_thresholds:
        passed = (predicted[ni] >= targets[ni]) & (predicted[co] >= threshold)
        rows.append(dict(co_threshold_pct=threshold, passed_batches=int(passed.sum()),
                          actual_achieved_among_passed=int((passed & actual_feasible).sum()),
                          actual_missed_among_passed=int((passed & ~actual_feasible).sum())))
    return pd.DataFrame(rows)


def main():
    actual, predicted = load_wide()
    counts = feasibility_counts(actual, predicted)
    sweep = margin_sweep(actual, predicted)

    ni, co = TARGETS
    actual_feasible = (actual[ni] >= TARGETS[ni]) & (actual[co] >= TARGETS[co])
    baseline_passed = (predicted[ni] >= TARGETS[ni]) & (predicted[co] >= TARGETS[co])
    missed_at_baseline = actual_feasible & ~baseline_passed
    boundary_co_threshold = float(predicted.loc[missed_at_baseline, co].min())
    near_target_co_threshold = 89.5
    co_comparison = co_threshold_comparison(actual, predicted,
        [TARGETS[co], boundary_co_threshold, near_target_co_threshold])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(OUTPUT_DIR / 'margin_sweep.csv', index=False)
    co_comparison.to_csv(OUTPUT_DIR / 'co_threshold_comparison.csv', index=False)
    metadata = dict(targets=TARGETS, oof_source=str(OOF_PATH), development_batches=len(predicted),
                    zero_margin_counts=counts, margin_candidates_pp=MARGIN_CANDIDATES_PP,
                    co_process_target_pct=TARGETS[co],
                    co_near_target_candidate_threshold_pct=near_target_co_threshold,
                    co_threshold_comparison=co_comparison.to_dict('records'),
                    policy_decision='Ni 예측 기준을 95%로 고정하고 Co 예측 기준을 90.0%에서 89.5%로 완화한 결과, '
                        '개발 데이터 OOF에서 실측 동시 달성 배치 2건을 추가로 포착했고 실측 미달 배치 1건이 '
                        '추가로 포함되었다. 89.5%는 목표 근접 추천을 위한 잠정 기준이며, Co 공정 목표는 90%이다. '
                        '새 배치에서의 성능은 아직 검증하지 않았다.',
                    test_used=False,
                    limitations=['개발 데이터 기준 탐색 결과이며 실제 추천 배치의 목표 달성률을 보장하지 않음',
                                 '표본이 적은 margin 구간은 추정이 불안정할 수 있음',
                                 'Ni·Co에 동일한 margin을 적용한 결과이며 금속별 최적 margin을 찾는 과정은 아님',
                                 '89.5%는 이번 OOF 표본에서 도출한 잠정 기준이며 새 배치에서의 일반화는 확인되지 않음'])
    (OUTPUT_DIR / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(counts, ensure_ascii=False, indent=2))
    print(sweep.to_string(index=False))
    print(co_comparison.to_string(index=False))


if __name__ == '__main__':
    main()
