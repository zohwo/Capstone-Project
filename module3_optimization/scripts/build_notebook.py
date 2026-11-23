"""Module 3 실행 결과와 검증 근거를 보여주는 노트북 생성함"""
from pathlib import Path
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.cells = [
    nbf.v4.new_markdown_cell('''# Module 3 공정 레시피 추천

- 저장된 Ni·Co SVR 모델을 재학습 없이 사용함
- 원료 조성과 원료투입량 고정 후 황산량·온도·시간 탐색함
- 목표는 Ni 95%, Co 90%이며 실제 달성 보장이 아닌 예측 기준임
- 운전 부담은 세 변수의 범위 정규화 후 동일 가중 평균이며 실제 비용이 아님
- Differential Evolution·유전 알고리즘 각각 3개 seed와 9×9×9 격자 탐색 비교함
- GA는 개체 36개·60세대, 토너먼트 선택·산술 교차·돌연변이·상위 2개체 보존 적용함
- 황산 1 L, 온도 0.1°C, 시간 0.1 h는 시연용 반올림 단위이며 설비 해상도는 아님
- 반올림 후 다시 예측하여 목표 충족 여부 판정함
- 미발견은 탐색 예산 내 결과이며 달성 불가능을 증명하지 않음

프로젝트 루트에서 결과 재생성 가능함
```bash
python -m module3_optimization.pipeline --input module3_optimization/examples/blend_7_3.json
```
아래는 reports의 A·C·7:3 예제 실행 결과를 읽어 재검증함
'''),
    nbf.v4.new_code_cell('''import sys
from pathlib import Path
sys.path.insert(0, str(Path('..').resolve()))
import json
import numpy as np
import pandas as pd
from IPython.display import display, Image
from module2_prediction.artifacts import load_bundle
from module3_optimization.optimization import predict_candidates
from module3_optimization.validation import validate_controls, outcome
from module3_optimization.config import TARGETS, CONTROL_COLUMNS
root = Path('..')
reports = root / 'module3_optimization/reports'
bundles = {t: load_bundle(root / 'module2_prediction/reports/final' / f'{t}.joblib') for t in TARGETS}
'''),
    nbf.v4.new_markdown_cell('''## 1. 입력과 추천 결과

- A·C는 가상 분석값으로 만든 예제이며 실측 배치가 아님
- 7:3은 두 조성을 질량 기준으로 혼합한 예제임
- 동일 원료의 현재 조건과 추천 조건을 비교함
'''),
    nbf.v4.new_code_cell('''rows = []
results = {}
for name in ['a_example', 'c_example', 'blend_7_3']:
    result = json.loads((reports / name / 'recommendation.json').read_text())
    results[name] = result
    controls = [result['controls'][c] for c in CONTROL_COLUMNS]
    validate_controls(controls)
    predictions = predict_candidates(bundles, result['feed'], [controls])[0]
    np.testing.assert_allclose(predictions, [result['predictions'][t] for t in TARGETS], atol=1e-10)
    valid, feasible, margins = outcome(dict(zip(TARGETS, predictions)), result['targets'])
    assert valid and feasible == (result['status'] == 'predicted_feasible')
    row = dict(scenario=name, status=result['status'], **result['controls'], **result['predictions'], burden=result['burden'])
    current = result.get('current')
    if current is None:
        row['current_comparison'] = '현재 조건 입력 없음'
    elif current['predictions'] is None:
        row['current_comparison'] = current['comparison_excluded_reason']
    else:
        row.update({f'current_{t}': v for t, v in current['predictions'].items()})
        row.update({f'change_{t}': v for t, v in result['predicted_change_pct_point'].items()})
    rows.append(row)
pd.DataFrame(rows)
'''),
    nbf.v4.new_markdown_cell('## 2. DE·GA seed별 결과와 격자 기준선\n\n- 알고리즘 종료 여부와 추천 목표 충족 여부를 구분함\n- 전체 후보 중 유효한 반올림 후보를 비교함'),
    nbf.v4.new_code_cell('''for name, result in results.items():
    print(name)
    display(pd.read_csv(reports / name / 'search_comparison.csv'))
    display(pd.DataFrame(result['runs']))
    print('목표 대비 여유 (%p):', result['margins_pct_point'])
    print('학습 변수 범위 밖:', result['outside_training_feature_ranges'])
'''),
    nbf.v4.new_markdown_cell('## 3. 조건별 예측 회수율\n\n- 추천 시간에 고정한 황산량·온도 단면임\n- 흰색 선은 목표 회수율이며 범위 안에 없으면 표시하지 않음\n- 별 표시는 추천 후보이며 미발견 사례에서는 목표에 가장 가까운 후보임'),
    nbf.v4.new_code_cell('''for name in results:
    print(name)
    display(Image(filename=str(reports / name / 'recovery_surface.png')))
'''),
    nbf.v4.new_markdown_cell('''## 4. 한계

- 실험실·현장 검증 전이므로 실제 회수율·경제적 효과를 보장하지 않음
- 학습 변수별 범위 검사는 조합 전체의 학습 분포 포함 여부를 보장하지 않음
- 모델 예측을 0~100으로 잘라 목표 충족시키지 않으며 범위 밖 예측은 제외함
- CV·Test 오차를 신뢰구간으로 간주하지 않음
- 모델 재학습·Test 재선택은 수행하지 않음
'''),
]
nb.cells[-1:-1] = [
    nbf.v4.new_markdown_cell("""## 4. 등급별 대표 조성과 혼합 비교

- 대치 전 개발 데이터 357행에서 등급별 관측 성분 중앙값 산출함
- 대표 조성은 실측 단일 배치가 아니며 Test는 산출에 사용하지 않음
- 원료 140 kg 고정 후 단독 3종·질량비 3:7, 5:5, 7:3 혼합 9종 비교함
- 각 시나리오의 운전조건을 따로 최적화하므로 동일 운전조건에서의 조성 효과 비교는 아님
- 목표 미발견은 해당 탐색 결과이며 실제 달성 불가능을 뜻하지 않음"""),
    nbf.v4.new_code_cell("""comparison_dir = reports / 'representative_blends'
display(pd.read_csv(comparison_dir / 'representative_compositions.csv'))
display(pd.read_csv(comparison_dir / 'summary.csv'))
display(Image(filename=str(comparison_dir / 'blend_comparison.png')))"""),
]
nb.cells[-1].source = nb.cells[-1].source.replace('## 4. 한계', '## 5. 한계')
nb.cells[-1:-1] = [
    nbf.v4.new_markdown_cell("""## 5. B급 원료 투입량 비교

- 기존 목적함수와 별도로 Ni 95% 이상 제약 아래 Co 예측 회수율 최대화함
- 140 kg은 기존 데이터사전 범위로 먼저 비교함
- 원료량별 비교는 학습 관측 범위 안의 81·100·120·140·160·180·199 kg 7개 지점을 사용함
- 운전조건 및 파생변수를 포함한 모델 입력별 학습 최솟값·최댓값 안으로 제한함
- 각 투입량에서 공정조건을 별도로 탐색하고 반올림 후 재예측함
- 변수별 범위 준수가 조합 전체의 학습 지원이나 실제 달성을 보장하지 않음
- 처리량 감소와 배치당 예상 회수 금속량을 함께 비교하며 실제 비용·시간당 생산성은 평가하지 않음"""),
    nbf.v4.new_code_cell("""capacity_dir = reports / 'b_grade_capacity'
display(pd.read_csv(capacity_dir / 'summary.csv'))
display(Image(filename=str(capacity_dir / 'ore_sensitivity.png')))"""),
]
nb.cells[-1].source = nb.cells[-1].source.replace('## 5. 한계', '## 6. 한계')
Path('notebooks').mkdir(exist_ok=True)
nbf.write(nb, 'notebooks/03_process_optimization.ipynb')
