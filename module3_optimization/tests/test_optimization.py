import numpy as np
import pandas as pd
import pytest
from module3_optimization import optimization as opt
from module3_optimization.config import TARGETS, CONTROL_COLUMNS
from module3_optimization.validation import validate_feed, validate_controls, outcome

FEED = dict(zip(['Ni_함량_pct','Co_함량_pct','Li_함량_pct','Mn_함량_pct','Cu_불순물_pct','Fe_불순물_pct','Al_불순물_pct','원료투입량_kg'],[20,8,4,10,1,1,1,140]))


def bundles():
    ref=opt.candidate_frame(FEED, [[400,40,1],[1500,95,8]])
    return {t:dict(reference=ref,features=CONTROL_COLUMNS,target=t) for t in TARGETS}


def test_feasible_recommendation_and_rounding(monkeypatch):
    monkeypatch.setattr(opt,'prepare_split',lambda ref,frame:frame)
    def predict(bundle,frame):
        offset=0 if bundle['target'].startswith('Ni') else -5
        return 80 + (frame['황산투입량_L'].to_numpy()-400)/50 + offset
    monkeypatch.setattr(opt,'predict_bundle',predict)
    result,candidates=opt.optimize(bundles(),FEED,seeds=[7],maxiter=5,grid_points=5)
    assert result['status']=='predicted_feasible'
    assert {'ga_7', 'de_7', 'grid'} == set(candidates.source)
    assert candidates.loc[candidates.source == 'ga_7', 'feasible'].any()
    assert result['controls']['황산투입량_L']>=1150
    assert all(m>=0 for m in result['margins_pct_point'].values())
    assert not candidates.loc[~candidates.valid, "feasible"].any()
    for c,step in result['rounding_steps'].items():
        assert result['controls'][c]/step==pytest.approx(round(result['controls'][c]/step))
    again,_=opt.optimize(bundles(),FEED,seeds=[7],maxiter=5,grid_points=5)
    assert result==again


def test_target_not_found_is_not_reported_as_feasible(monkeypatch):
    monkeypatch.setattr(opt,'prepare_split',lambda ref,frame:frame)
    monkeypatch.setattr(opt,'predict_bundle',lambda bundle,frame:np.full(len(frame),80.0))
    result,_=opt.optimize(bundles(),FEED,seeds=[7],maxiter=2,grid_points=2)
    assert result['status']=='target_not_found'
    assert result['margins_pct_point']=={'Ni_회수율_pct':-15.0,'Co_회수율_pct':-10.0}


def test_invalid_model_predictions_rejected(monkeypatch):
    monkeypatch.setattr(opt,'predict_bundle',lambda bundle,frame:np.full(len(frame),101.0))
    with pytest.raises(ValueError,match='유효한'):
        opt.optimize(bundles(),FEED,seeds=[7],maxiter=1,grid_points=2)


@pytest.mark.parametrize('bad',[0,np.nan,999])
def test_invalid_ore_rejected(bad):
    with pytest.raises(ValueError):validate_feed(dict(FEED,원료투입량_kg=bad))


def test_missing_composition_and_invalid_controls_rejected():
    with pytest.raises(ValueError):validate_feed({})
    with pytest.raises(ValueError):validate_controls([1501,70,4])


def test_upper_preservation_limits_accepted():
    feed=dict(FEED,Ni_함량_pct=38.2,Co_함량_pct=15,Li_함량_pct=6.76,Mn_함량_pct=16,Cu_불순물_pct=6.08,Fe_불순물_pct=6,Al_불순물_pct=7.46)
    # 정상·예외 범위의 합은 100 미만이므로 범위 검증과 합계 검증 모두 유지함
    validate_feed(feed)


def test_candidate_frame_keeps_feed_fixed():
    frame=opt.candidate_frame(FEED,[[400,40,1],[1500,95,8]])
    for c,value in FEED.items():
        assert frame[c].tolist()==[value,value]
    assert frame['황산투입량_L'].tolist()==[400,1500]


@pytest.mark.parametrize('weights',[[0,0,0],[-1,1,1],[1,np.nan,1]])
def test_invalid_objective_weights_rejected(weights):
    with pytest.raises(ValueError,match='가중치'):
        opt.optimize(bundles(),FEED,weights=weights)


@pytest.mark.parametrize('bad_controls', [5.0, np.empty((0, 3)), np.full((2, 4, 3), [700, 60, 4], dtype=float)])
def test_validate_controls_rejects_bad_shapes(bad_controls):
    with pytest.raises(ValueError):
        validate_controls(bad_controls)


@pytest.mark.parametrize('predictions,targets', [
    ({}, {}),
    ({'Ni_회수율_pct': 96.0}, {'Ni_회수율_pct': 95.0, 'Co_회수율_pct': 90.0}),
    ({'Ni_회수율_pct': 96.0, 'Co_회수율_pct': 92.0, 'extra': 1.0}, {'Ni_회수율_pct': 95.0, 'Co_회수율_pct': 90.0}),
    ({'Ni_회수율_pct': 96.0, 'Co_회수율_pct': 92.0}, {'Ni_회수율_pct': float('nan'), 'Co_회수율_pct': 90.0}),
])
def test_outcome_rejects_invalid_inputs(predictions, targets):
    with pytest.raises(ValueError):
        outcome(predictions, targets)


def test_genetic_search_bounds_elitism_and_reproducibility():
    from module3_optimization.genetic import genetic_search
    evaluated = []
    def objective(x):
        assert ((x >= [-2, 5]) & (x <= [3, 9])).all()
        evaluated.append(x.copy())
        return ((x - [0, 7]) ** 2).sum(axis=1)
    first = genetic_search(objective, [(-2, 3), (5, 9)], 42, generations=20)
    second = genetic_search(objective, [(-2, 3), (5, 9)], 42, generations=20)
    assert first == second
    assert first['evaluations'] == 36 * 21
    assert np.all(np.diff(first['best_score_history']) <= 0)
    assert first['best_score_history'][-1] < 0.01
