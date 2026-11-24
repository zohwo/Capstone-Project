from pathlib import Path

import run_pipeline as orchestrator


def test_runs_module1_then_module2_with_derived_pre_impute_path(monkeypatch):
    calls = []
    monkeypatch.setattr(orchestrator, 'run_module1', lambda *a, **k: calls.append(('module1', a, k)))
    monkeypatch.setattr(orchestrator, 'run_module2', lambda *a, **k: calls.append(('module2', a, k)) or {'ok': True})
    monkeypatch.setattr(orchestrator, 'run_module3', lambda *a, **k: calls.append(('module3', a, k)))

    result = orchestrator.run_pipeline('raw.xlsx', cleaned_output_path='data/processed/cleaned_features.csv',
                                        model_output_dir='models_dir')

    assert [name for name, _, _ in calls] == ['module1', 'module2']
    assert calls[0][1] == ('raw.xlsx', 'data/processed/cleaned_features.csv')
    assert calls[1][2] == dict(data_path=Path('data/processed/pre_impute_data.csv'), output_dir='models_dir')
    assert result == dict(module2_results={'ok': True}, module3_recommendation=None)


def test_runs_module3_only_when_feed_request_given(monkeypatch):
    calls = []
    monkeypatch.setattr(orchestrator, 'run_module1', lambda *a, **k: calls.append(('module1', a, k)))
    monkeypatch.setattr(orchestrator, 'run_module2', lambda *a, **k: calls.append(('module2', a, k)) or {'ok': True})
    monkeypatch.setattr(orchestrator, 'run_module3', lambda *a, **k: calls.append(('module3', a, k)) or {'status': 'predicted_feasible'})

    result = orchestrator.run_pipeline('raw.xlsx', feed_request_path='request.json',
                                        model_output_dir='models_dir', recommendation_output_dir='rec_dir')

    assert [name for name, _, _ in calls] == ['module1', 'module2', 'module3']
    assert calls[2][1] == ('request.json',)
    assert calls[2][2] == dict(model_dir='models_dir', output_dir='rec_dir')
    assert result['module3_recommendation'] == {'status': 'predicted_feasible'}
