import numpy as np
import pytest

from module3_optimization.scripts import b_grade_capacity as experiment
from module3_optimization.tests.test_optimization import FEED, bundles


def test_ni_constraint_has_priority_over_higher_co():
    scores, valid = experiment.co_objective([[94.99, 99], [95, 85], [96, 91]], [True] * 3)
    assert valid.all()
    assert scores[2] < scores[1] < scores[0]


def test_invalid_or_unsupported_predictions_excluded():
    scores, valid = experiment.co_objective([[96, 101], [np.nan, 92], [96, 92]], [True, True, False])
    assert not valid.any()
    assert (scores == 1e9).all()


def test_known_conditions_included_and_rounding_rechecked(monkeypatch):
    monkeypatch.setattr(experiment, 'prepare_split', lambda reference, frame: frame)
    def prediction(bundles, feed, controls):
        controls = np.atleast_2d(controls)
        co = 92 - ((controls[:, 0] - 900) / 100) ** 2
        return np.column_stack([np.full(len(controls), 96), co])
    monkeypatch.setattr(experiment, 'predict_candidates', prediction)
    result, candidates = experiment.maximize_co(bundles(), FEED, known_controls=[900, 70, 4],
                                               seeds=[7], maxiter=1, grid_points=2)
    assert result['predictions']['Co_회수율_pct'] == 92
    assert result['status'] == 'predicted_feasible'
    assert 'known' in set(candidates.source)


@pytest.mark.parametrize('bad_maxiter', [-1, 0])
def test_maxiter_below_one_rejected(bad_maxiter):
    with pytest.raises(ValueError, match='maxiter'):
        experiment.maximize_co(bundles(), FEED, maxiter=bad_maxiter)


def test_co_objective_follows_dynamic_ni_target(monkeypatch):
    monkeypatch.setattr(experiment, 'TARGETS', {'Ni_회수율_pct': 94.0, 'Co_회수율_pct': 90.0})
    scores, valid = experiment.co_objective([[94.5, 95.0], [96.0, 70.0]], [True, True])
    assert valid.all()
    assert scores[0] == pytest.approx(0.05)
    assert scores[1] == pytest.approx(0.30)


def test_maximize_co_final_selection_follows_dynamic_ni_target(monkeypatch):
    monkeypatch.setattr(experiment, 'TARGETS', {'Ni_회수율_pct': 94.0, 'Co_회수율_pct': 90.0})
    monkeypatch.setattr(experiment, 'prepare_split', lambda reference, frame: frame)
    def prediction(bundles, feed, controls):
        controls = np.atleast_2d(controls)
        acid = controls[:, 0]
        ni = np.where(acid < 950, 94.5, 96.0)
        co = np.where(acid < 950, 95.0, 70.0)
        return np.column_stack([ni, co])
    monkeypatch.setattr(experiment, 'predict_candidates', prediction)
    result, candidates = experiment.maximize_co(bundles(), FEED, seeds=[7], maxiter=1, grid_points=2)
    assert result['predictions']['Co_회수율_pct'] == pytest.approx(95.0)


def test_training_ranges_restrict_candidates(monkeypatch):
    monkeypatch.setattr(experiment, 'prepare_split', lambda reference, frame: frame)
    restricted = bundles()
    for b in restricted.values():
        b['reference']['황산투입량_L'] = [400, 1000]
    monkeypatch.setattr(experiment, 'predict_candidates', lambda bundles, feed, controls:
        np.column_stack([np.full(len(controls), 96), 80 + np.asarray(controls)[:, 0] / 100]))
    result, candidates = experiment.maximize_co(restricted, FEED, enforce_training_ranges=True,
        known_controls=[1000, 70, 4], seeds=[7], maxiter=1, grid_points=2)
    assert result['controls']['황산투입량_L'] == 1000
    assert not candidates.loc[candidates['황산투입량_L'] > 1000, 'valid'].any()
