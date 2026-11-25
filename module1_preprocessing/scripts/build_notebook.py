from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell(r"""# Module 1 — 블랙파우더 성분 데이터 전처리

- 목적: 분석 데이터의 형식을 통일하고 결측·이상값을 처리하여 모델 입력 준비함
- 흐름: 데이터 정제 → 데이터 분할 → 초기 EDA → 결측치 대치 → 특성 공학 → 데이터 검증·처리 후 EDA → 특징 조합 비교 → 산출물 저장
- 공용 모듈을 호출하여 CLI와 같은 전처리 기준 적용함
- Test는 특징·모델 선택에 사용하지 않음
- 그림과 표는 셀 실행 시 표시되며 제출 전 전체 실행 후 저장 필요함

## 1. 실행 준비 및 데이터 구성

- 프로젝트 루트 또는 notebooks 폴더에서 실행 가능함
- CSV·XLSX·XLS 지원하며 Excel 기본 시트명은 `원료분석_원본데이터`임
- 조성 7종·공정 입력 4종을 원변수로 사용하며 회수율 2종은 정답으로 구분함"""))

cells.append(nbf.v4.new_code_cell(r"""import sys
from pathlib import Path

ROOT = Path.cwd().resolve()
if not (ROOT / 'module1_preprocessing').is_dir():
    ROOT = ROOT.parent
if not (ROOT / 'module1_preprocessing').is_dir():
    raise RuntimeError('프로젝트 루트 또는 notebooks 폴더에서 실행 필요함')
sys.path.insert(0, str(ROOT))

from IPython.display import Image, display
import pandas as pd
from pathlib import Path

from module1_preprocessing.preprocessing import load_raw, clean_schema, resolve_duplicate_batches
from module1_preprocessing.imputation import replace_outliers_with_nan, impute_missing_from_reference, interpolate_analysis_date
from module1_preprocessing.transformation import validate_targets, validate_model_features
from module1_preprocessing.features import engineer_features
from module1_preprocessing.splitting import stratified_train_val_test_split
from module1_preprocessing.validation import (
    check_ranges,
    format_range_violations,
    check_composition_sum,
    check_missing,
    plot_missing_before_after,
    plot_distributions,
    plot_correlation_heatmap,
    plot_ablation_results,
    plot_grade_distribution,
)
from module1_preprocessing.config import (
    COMPOSITION_COLUMNS, PROCESS_COLUMNS, TARGET_COLUMNS, NORMAL_RANGES, REFERENCE_EXCEPTION_RANGES,
    DERIVED_FEATURES, BASE_FEATURES,
)

RAW_PATH = (ROOT / 'data/raw/영풍그룹_금속회수율최적화_RawMaterialData_학생배포용.xlsx')
FIGURES_DIR = ROOT / 'module1_preprocessing/reports/figures'"""))

cells.append(nbf.v4.new_markdown_cell(r"""## 2. 데이터 정제"""))

cells.append(nbf.v4.new_code_cell(r"""raw_df = load_raw(RAW_PATH)
raw_df.shape"""))

cells.append(nbf.v4.new_markdown_cell(r"""### 2.1 입력 형식 및 회수율 검증

- 날짜·원료등급·숫자 형식을 통일하고 필수 컬럼 확인함
- 이상값 처리 전 정상범위 이탈 현황 확인함
- 회수율은 대치하지 않으며 결측·무한대·0~100% 범위 이탈 시 중단함"""))

cells.append(nbf.v4.new_code_cell(r"""clean_df = clean_schema(raw_df)
raw_violations = check_ranges(clean_df)
print('원본 데이터 정상범위 이탈 현황:')
print(format_range_violations(clean_df, raw_violations))
validate_targets(clean_df, TARGET_COLUMNS)"""))

cells.append(nbf.v4.new_markdown_cell(r"""### 2.2 중복 배치 처리

- 같은 배치ID에서 모든 값이 같으면 첫 행만 유지함
- 값이 다르면 충돌 플래그를 부여하고 수동 검토를 위해 중단함
- 분할 전에 중복을 처리하여 같은 배치가 여러 세트에 포함되는 것을 방지함"""))

cells.append(nbf.v4.new_code_cell(r"""clean_df = resolve_duplicate_batches(clean_df)
print(f'중복 제거 후: {len(clean_df)}행, 충돌(수동 확인 필요): {clean_df["is_conflicting_duplicate"].sum()}건')
conflicting = clean_df.loc[clean_df['is_conflicting_duplicate'], '배치ID'].unique().tolist()
if conflicting:
    raise ValueError(f'충돌 중복 배치ID 수동 검토 필요: {conflicting}')"""))

cells.append(nbf.v4.new_markdown_cell(r"""### 2.3 이상값 처리

- 데이터사전의 정상범위 내 값은 원본 유지함
- 정상범위를 벗어나더라도 문헌의 실측값을 근거로 설정한 보존 범위에 해당하면 원본을 유지하고 범위 이탈 플래그 부여함
- 정상범위와 문헌 기반 보존 범위를 모두 벗어난 값은 결측으로 변환한 뒤 대치함"""))

cells.append(nbf.v4.new_markdown_cell(r"""**문헌 기반 보존 기준**

- 아래 표는 config의 실제 적용 범위임
- 문헌 보고값을 참고한 프로젝트 보존 기준이며 보편적인 정상범위를 의미하지 않음
- 출처 요약은 마지막 표, 상세 근거는 `docs/reference_ranges.md`에 기재함"""))

cells.append(nbf.v4.new_code_cell(r"""display(pd.DataFrame([
    {'변수': col, '데이터사전 정상범위': NORMAL_RANGES[col],
     '문헌 기반 보존 예외': REFERENCE_EXCEPTION_RANGES.get(col, '없음')}
    for col in COMPOSITION_COLUMNS
]))"""))

cells.append(nbf.v4.new_code_cell(r"""outlier_flagged_df = replace_outliers_with_nan(clean_df)
flag_cols = [c for c in outlier_flagged_df.columns if c.endswith('_dict_range_flag')]
print('보존 기준 적용 후 shape:', outlier_flagged_df.shape)
print('데이터사전 범위 이탈이지만 보존된 값(컬럼별 플래그 True 개수):')
for c in flag_cols:
    n = outlier_flagged_df[c].sum()
    if n > 0:
        print(f'  {c}: {n}개')
composition_before = check_composition_sum(outlier_flagged_df).add_prefix('pre_impute_')
outlier_flagged_df = outlier_flagged_df.join(composition_before)"""))

cells.append(nbf.v4.new_code_cell(r"""missing = check_missing(outlier_flagged_df, columns=['분석일자'] + BASE_FEATURES + TARGET_COLUMNS)
print('결측치 확인:', missing if missing else '없음')"""))

cells.append(nbf.v4.new_markdown_cell(r"""### 2.4 분석일자 보간

- 배치ID의 순차 발번이 생산·분석 순서를 근사한다고 가정하여 순서 기반 보간 적용함
- 배치ID로 정렬한 뒤 행 간격 기준으로 선형 보간하고, 양 끝의 결측은 가장 가까운 관측 날짜로 채움"""))

cells.append(nbf.v4.new_code_cell(r"""pre_impute_df = outlier_flagged_df.copy()
interpolated_df = interpolate_analysis_date(outlier_flagged_df)
print('분석일자 결측:', interpolated_df['분석일자'].isna().sum())"""))

cells.append(nbf.v4.new_markdown_cell(r"""## 3. 데이터 분할 및 초기 EDA

- 원료 조성과 운전조건을 입력받아 회수율을 예측하는 모델의 성능을 평가하기 위해, 수집 데이터와 유사한 조건의 미사용 배치를 평가 대상으로 설정함
- 원료등급별 구성비를 유지하도록 무작위 층화 분할을 적용하여 전체 420개 배치를 Train 294개, Validation 63개, Test 63개로 구분함
- 결측치 대치 기준에 검증·테스트 데이터의 정보가 유입되지 않도록 수치형 결측치 대치 전에 분할함
- Test를 먼저 분리한 뒤 개발 데이터에서 Train·Validation 분리함"""))

cells.append(nbf.v4.new_code_cell(r"""train_i, val_i, test_i = stratified_train_val_test_split(interpolated_df)
interpolated_df['split'] = 'train'
interpolated_df.loc[val_i, 'split'] = 'validation'
interpolated_df.loc[test_i, 'split'] = 'test'
pre_impute_df['split'] = interpolated_df['split']
print(f'Train: {len(train_i)}행 / Validation: {len(val_i)}행 / Test: {len(test_i)}행')"""))

cells.append(nbf.v4.new_code_cell(r"""# 분할 전후 원료등급 구성비 비교함
overall_ratio = (interpolated_df['원료등급'].value_counts(normalize=True) * 100).sort_index()
split_ratios = pd.DataFrame({
    '전체': overall_ratio,
    'Train': (interpolated_df.loc[train_i, '원료등급'].value_counts(normalize=True) * 100).sort_index(),
    'Validation': (interpolated_df.loc[val_i, '원료등급'].value_counts(normalize=True) * 100).sort_index(),
    'Test': (interpolated_df.loc[test_i, '원료등급'].value_counts(normalize=True) * 100).sort_index(),
})
split_ratios.round(1)"""))

cells.append(nbf.v4.new_markdown_cell(r"""### 3.1 배치ID 기준 인접 배치의 회수율 관계

- 배치ID가 처리 순서를 근사한다고 가정하고, 개발 데이터에서 순번 차이가 1인 쌍의 회수율 관계를 확인함
- 산점도에 Pearson 상관계수와 유효 쌍 수를 표시함"""))

cells.append(nbf.v4.new_code_cell(r"""import matplotlib.pyplot as plt

lag_pool = pre_impute_df.loc[pre_impute_df['split'].isin(['train', 'validation'])].copy()
lag_pool['배치순번'] = pd.to_numeric(lag_pool['배치ID'].str.extract(r'(\d+)$')[0], errors='raise')
if lag_pool['배치순번'].duplicated().any():
    raise ValueError('중복된 배치 숫자 순번 확인 필요함')
lag_pool = lag_pool.sort_values('배치순번')
previous = lag_pool.shift(1)
adjacent = lag_pool['배치순번'].diff().eq(1)
lag_rows = []
for target in TARGET_COLUMNS:
    valid = adjacent & lag_pool[target].notna() & previous[target].notna()
    x, y = previous.loc[valid, target], lag_pool.loc[valid, target]
    lag_rows.append({'target': target, 'pairs': len(x), 'lag1_correlation': x.corr(y)})
lag_summary = pd.DataFrame(lag_rows)
fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
for ax, target, metal in zip(axes, TARGET_COLUMNS, ['Ni', 'Co']):
    valid = adjacent & lag_pool[target].notna() & previous[target].notna()
    x, y = previous.loc[valid, target], lag_pool.loc[valid, target]
    r = x.corr(y)
    ax.scatter(x, y, s=24, alpha=.65, color='steelblue')
    ax.set(title=f'{metal} — r={r:.3f}, n={len(x)}',
           xlabel='직전 배치 회수율 (%)', ylabel='현재 배치 회수율 (%)',
           xlim=(0, 100), ylim=(0, 100))
    ax.set_aspect('equal')
    ax.grid(alpha=.2)
fig.suptitle('인접 배치 회수율 — Lag 1')
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
lag_path = FIGURES_DIR / 'lag1_recovery.png'
fig.savefig(lag_path, dpi=150)
plt.close(fig)
lag_summary.to_csv(FIGURES_DIR / 'lag1_recovery.csv', index=False)
display(Image(filename=str(lag_path)))
print('배치ID가 연속한 개발 데이터 쌍 기준')
display(lag_summary)"""))

cells.append(nbf.v4.new_markdown_cell(r"""**현재 개발 데이터의 분석 결과**

- Ni·Co의 인접 쌍 Pearson 상관계수는 각각 0.060·0.082로 낮게 나타났으며, 이 결과는 배치 독립성을 입증하는 근거로 사용하지 않음
- 두 타깃 모두 유효 쌍 수는 301쌍이며 입력 데이터 변경 시 재확인 필요함"""))

cells.append(nbf.v4.new_markdown_cell(r"""### 3.2 대치 전 원변수 분포

- 개발 데이터의 대치 전 관측값으로 분포의 치우침과 극단값 영향을 확인함
- 아래 히스토그램은 이상값 처리 후 남은 관측값 기준임"""))

cells.append(nbf.v4.new_code_cell(r"""from module1_preprocessing.validation import plot_numeric_histograms

histogram_df = pre_impute_df.loc[pre_impute_df['split'].isin(['train', 'validation'])]
display(Image(filename=str(plot_numeric_histograms(
    histogram_df, BASE_FEATURES, FIGURES_DIR
))))"""))

cells.append(nbf.v4.new_markdown_cell(r"""## 4. 결측치 대치

- Train에서 대치 기준을 계산하고 Validation·Test에 동일하게 적용함
- 교차검증에서는 각 fold의 학습 데이터에서 대치 기준을 다시 계산함

| 대상 | 방법 | 선정 근거 |
|---|---|---|
| 조성 7종 | Train 원료등급별 중앙값 | 등급별 성분 차이 반영 및 극단값 영향 완화 |
| 침출 온도·시간 | Train 전체 중앙값 | 배치 간 조건의 연속성을 가정하기 어려움 |
| 황산 투입량 | Train 원료 투입량 기반 선형회귀 | 원료 처리량과 황산 투입량 간 관계 반영 |

- 등급별 중앙값을 구할 수 없는 경우 Train 전체 중앙값 사용함
- 아래 그림은 이상값 처리 후부터 날짜 보간·수치형 대치 완료까지의 결측률 변화임"""))

cells.append(nbf.v4.new_code_cell(r"""train_df = interpolated_df.loc[train_i].copy()
imputed_df = impute_missing_from_reference(train_df, interpolated_df)"""))

cells.append(nbf.v4.new_code_cell(r"""missing_compare_cols = COMPOSITION_COLUMNS + PROCESS_COLUMNS + [
    '원료투입량_kg', '황산투입량_L', '분석일자'
]
display(Image(filename=str(plot_missing_before_after(
    outlier_flagged_df, imputed_df, FIGURES_DIR, missing_compare_cols
))))"""))

cells.append(nbf.v4.new_markdown_cell(r"""## 5. 특성 공학

- 원료 조성과 운전조건의 관계를 표현하는 파생변수 4종을 생성함

| 변수 | 계산식 | 의미 |
|---|---|---|
| `acid_to_ore_ratio` | 황산 투입량 ÷ 원료 투입량 | 원료 질량당 산 투입 부피 L/kg |
| `leach_intensity` | 침출 온도 × 침출 시간 | 온도·시간 결합 지표 |
| `total_impurity_pct` | Cu + Fe + Al | 측정한 세 불순물의 합계 wt% |
| `feed_impurity_g_per_l` | 10 × 불순물 합계 ÷ 산 투입비 | 산 투입 부피당 원료 내 불순물 질량 g/L |

- 파생변수의 예측 기여도는 7절의 특징 조합 비교에서 확인함"""))

cells.append(nbf.v4.new_code_cell(r"""for col in ['원료투입량_kg', '황산투입량_L']:
    bad_batches = imputed_df.loc[imputed_df[col] <= 0, '배치ID'].tolist()
    if bad_batches:
        raise ValueError(f'{col}이 0 이하인 배치로 파생변수 계산 불가: {bad_batches}')
final_df = engineer_features(imputed_df)
final_df = final_df.join(check_composition_sum(final_df))
final_df.loc[train_i].head()"""))

cells.append(nbf.v4.new_markdown_cell(r"""## 6. 데이터 검증 및 처리 후 EDA

- 모델 입력과 회수율의 잔여 결측·무한대 및 회수율의 0–100% 범위 확인함
- 원변수가 정상범위 또는 문헌 기반 보존 범위를 충족하는지 점검함
- 동일 시료·동일 질량 기준의 조성 7종 합계가 100%를 초과하는 배치를 검토 대상으로 표시함"""))

cells.append(nbf.v4.new_code_cell(r"""for stage, column in [('대치 전', 'pre_impute_composition_sum_review_flag'),
                      ('대치 후', 'composition_sum_review_flag')]:
    batches = final_df.loc[final_df[column].fillna(False), '배치ID'].tolist()
    print(f'{stage} 성분 합계 100% 초과 검토 대상: {batches or "없음"} — 자동 보정 없음')

model_features = BASE_FEATURES + DERIVED_FEATURES
remaining_missing = check_missing(final_df, columns=model_features + TARGET_COLUMNS)
if remaining_missing:
    raise ValueError(f'정제 후 잔여 결측 발견: {remaining_missing}')
validate_model_features(final_df, model_features)
validate_targets(final_df, TARGET_COLUMNS)

range_violations = check_ranges(final_df, reference_exception_ranges=REFERENCE_EXCEPTION_RANGES)
if range_violations:
    raise ValueError('정제 후 보존 기준 위반:\n' + format_range_violations(final_df, range_violations))
print('모델 입력·회수율·보존 기준 검증 통과함')
print('정상범위 이탈 현황 — 보존 예외 포함:')
print(format_range_violations(final_df, check_ranges(final_df)))"""))

cells.append(nbf.v4.new_markdown_cell(r"""### 6.1 처리 전후 분포 및 상관관계

- 개발 데이터의 처리 전후 분포 변화를 확인함
- 처리 후 원변수·파생변수와 회수율 간 상관관계를 탐색함"""))

cells.append(nbf.v4.new_code_cell(r"""trainval_i = train_i.append(val_i)
for frame, folder in [(clean_df.loc[trainval_i], 'before'),
                      (final_df.loc[trainval_i], 'after')]:
    display(Image(filename=str(plot_distributions(
        frame, ['Ni_함량_pct', '황산투입량_L', '침출온도_C'], FIGURES_DIR / folder
    ))))
display(Image(filename=str(plot_grade_distribution(final_df.loc[trainval_i], FIGURES_DIR))))"""))

cells.append(nbf.v4.new_code_cell(r"""feature_columns = BASE_FEATURES + DERIVED_FEATURES
# 그래프에서만 한글 라벨 사용함
derived_display_labels = {
    'acid_to_ore_ratio': '산-광석비',
    'leach_intensity': '침출강도',
    'total_impurity_pct': '총불순물비율',
    'feed_impurity_g_per_l': '산용액당불순물량',
    'acid_per_target_L_kg': '목표금속당산용액량',
    'acid_time': '산투입비시간곱',
    'target_purity_ratio': '목표금속순도비',
    'Fe_Ni_ratio': 'Fe/Ni비',
    'Cu_Ni_ratio': 'Cu/Ni비',
    'Al_Ni_ratio': 'Al/Ni비',
}
trainval_i = train_i.append(val_i)
display(Image(filename=str(plot_correlation_heatmap(
    final_df.loc[trainval_i], feature_columns, TARGET_COLUMNS, FIGURES_DIR,
    display_labels=derived_display_labels,
))))"""))

cells.append(nbf.v4.new_markdown_cell(r"""**해석 기준**

- 상관관계는 연관성 탐색용이며 인과관계나 예측 기여도를 뜻하지 않음
- 예측 기여도는 동일한 CV 조건의 ablation test로 별도 확인함"""))

cells.append(nbf.v4.new_markdown_cell(r"""### 6.2 월별 변화 탐색 및 날짜·월의 사용 범위

- 대치 전 개발 데이터에서 실제 분석일자가 있는 배치만 사용함
- 월별 회수율·원료등급 구성비·입력변수의 평균 변화를 확인함
- 입력변수는 단위를 맞춰 비교하도록 z-score로 표시함
- 분석일자·월은 변화 탐색에 활용하며, 반복되는 계절 효과가 확인되지 않은 현재 모델에서는 입력으로 사용하지 않음"""))

cells.append(nbf.v4.new_code_cell(r"""import matplotlib.pyplot as plt
import seaborn as sns

monthly_pool = pre_impute_df.loc[pre_impute_df['split'].isin(['train', 'validation'])].copy()
observed_dates = pd.to_datetime(monthly_pool['분석일자'], errors='coerce')
print(f'개발 데이터 {len(monthly_pool)}행 중 실제 분석일자 확인 {observed_dates.notna().sum()}행 사용함')
print(f'분석일자 결측 {observed_dates.isna().sum()}행 제외함 — 보간 날짜 미사용')
monthly_pool = monthly_pool.loc[observed_dates.notna()].copy()
monthly_pool['분석월'] = observed_dates.loc[monthly_pool.index].dt.to_period('M').astype(str)
if monthly_pool.empty:
    print('월별 분석에 사용할 관측 날짜 없음')
else:
    months = sorted(monthly_pool['분석월'].unique())
    monthly_summary = monthly_pool.groupby('분석월')[TARGET_COLUMNS].agg(['count', 'mean', 'std'])
    grade_ratio = pd.crosstab(monthly_pool['분석월'], monthly_pool['원료등급'], normalize='index').reindex(months) * 100
    observed_inputs = monthly_pool[BASE_FEATURES]
    scales = observed_inputs.std().replace(0, float('nan'))
    input_z = (observed_inputs - observed_inputs.mean()) / scales
    monthly_z = input_z.groupby(monthly_pool['분석월']).mean().reindex(months)
    labels = {'Ni_함량_pct': 'Ni', 'Co_함량_pct': 'Co', 'Li_함량_pct': 'Li', 'Mn_함량_pct': 'Mn',
              'Cu_불순물_pct': 'Cu', 'Fe_불순물_pct': 'Fe', 'Al_불순물_pct': 'Al',
              '침출온도_C': '온도', '침출시간_hr': '시간', '원료투입량_kg': '원료량', '황산투입량_L': '황산량'}
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for ax, target, metal in zip(axes[0], TARGET_COLUMNS, ['Ni', 'Co']):
        stats = monthly_summary[target].reindex(months)
        x = range(len(months))
        ax.errorbar(x, stats['mean'], yerr=stats['std'], marker='o', capsize=3)
        ax.set_xticks(list(x), [m[5:] for m in months])
        ax.set(title=f'{metal} 회수율 — 평균 ± 표준편차', xlabel='분석월', ylabel='회수율 (%)', ylim=(0, 100))
        ax.grid(axis='y', alpha=.25)
    grade_ratio.plot.bar(stacked=True, ax=axes[1, 0], rot=0)
    axes[1, 0].set_xticklabels([m[5:] for m in months])
    axes[1, 0].set(title='원료등급 구성비', xlabel='분석월', ylabel='비율 (%)', ylim=(0, 100))
    axes[1, 0].legend(title='원료등급', ncols=3)
    sns.heatmap(monthly_z.rename(columns=labels).T, ax=axes[1, 1], cmap='RdBu_r', center=0,
                xticklabels=[m[5:] for m in months], cbar_kws={'label': '평균 z-score'})
    axes[1, 1].set(title='입력변수의 월별 평균 변화', xlabel='분석월', ylabel='')
    fig.suptitle(f'월별 데이터 변화 ({months[0]}–{months[-1]})')
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    monthly_path = FIGURES_DIR / 'monthly_changes.png'
    fig.savefig(monthly_path, dpi=150)
    plt.close(fig)
    monthly_summary.to_csv(FIGURES_DIR / 'monthly_summary.csv')
    grade_ratio.to_csv(FIGURES_DIR / 'monthly_grade_ratio.csv')
    monthly_z.to_csv(FIGURES_DIR / 'monthly_input_zscore.csv')
    display(Image(filename=str(monthly_path)))
    display(monthly_summary.round(2))
    for target in TARGET_COLUMNS:
        means = monthly_summary[target]['mean'].dropna()
        if len(means):
            print(f'{target}: 월평균 {means.min():.2f}–{means.max():.2f}%, 최대 차이 {means.max() - means.min():.2f}%p')
    print('월별 평균 차이는 탐색 결과이며 드리프트 유무·배치 독립성의 확정 근거는 아님')"""))

cells.append(nbf.v4.new_markdown_cell(r"""**현재 개발 데이터의 월별 분석 결과**

- 분석일자가 있는 340행 사용했으며 결측 17행 제외함
- Ni 월평균은 82.93–87.05%, Co 월평균은 76.17–80.26%로 나타남
- 월평균의 최대 차이는 Ni 4.12%p, Co 4.09%p이며 원료등급 구성과 입력조건도 월별로 달랐음
- 회수율이 일정하게 증가하거나 감소하는 패턴은 뚜렷하지 않지만 월별 차이가 없다고 단정할 수 없음
- 원료·운전조건의 영향을 분리하지 않은 탐색 결과이므로 월 변수를 추가해야 한다는 근거로 바로 사용하지 않음
- 위 수치는 현재 저장된 개발 데이터 기준이며 입력 변경 시 재확인 필요함"""))

cells.append(nbf.v4.new_markdown_cell(r"""## 7. 특징 조합 비교 및 공통 구성 채택

- 대치 전 개발 데이터에서 원료등급 기준 5-fold CV 수행함
- 각 fold의 학습 데이터에서만 대치·등급 인코딩 기준을 계산함
- 같은 LightGBM·분할 조건에서 네 조합을 비교함

| 조합 | 입력 |
|---|---|
| 원변수 | 11종 |
| 원변수 + 파생변수 | 11종 + 파생 4종 |
| 원변수 + 원료등급 | 11종 + 등급 |
| 원변수 + 파생변수 + 원료등급 | 11종 + 파생 4종 + 등급 |

- 현재 실행은 네 조합의 재평가이며 최저 오차 조합을 자동 채택하지 않음
- 기존 비교에서 승인한 원변수 11종 + 파생 4종을 Ni·Co 공통 구성으로 유지함
- 최종 모델의 종류·튜닝·평가는 Module 2에서 다룸"""))

cells.append(nbf.v4.new_code_cell(r"""from module1_preprocessing.pipeline import evaluate_feature_selection

selection, lgbm_results = evaluate_feature_selection(pre_impute_df)
display(pd.DataFrame(lgbm_results).T)
display(Image(filename=str(plot_ablation_results(lgbm_results, FIGURES_DIR))))"""))

cells.append(nbf.v4.new_markdown_cell(r"""**결과 해석**

- MAE·RMSE 단위는 %p이며 낮을수록 우수함
- 반복 비교 결과와 입력 구성의 단순성을 고려한 기존 공통 15종을 유지함 현재 CV는 재평가 결과임
- [기존 특징 비교](../module1_preprocessing/reports/derived_subset_selection/subset_summary.csv) · [SVR 추가 비교](../module2_prediction/reports/svr_feature_comparison/nested_summary.csv)
- Test의 과거 사용 이력은 최종 성능 해석 시 고려해야 함"""))

cells.append(nbf.v4.new_markdown_cell(r"""## 8. 산출물 저장 및 Module 2 연결

- CV 결과를 재사용하여 승인된 공통 입력 15종과 평가 결과 저장함
- 대치 전 데이터·분할 정보·입력 해시를 함께 저장하여 Module 2에서 학습 구간별 대치 가능하도록 함
- 대치·파생변수 생성 결과는 품질 확인용으로 별도 저장함"""))

cells.append(nbf.v4.new_code_cell(r"""from module1_preprocessing.pipeline import save_feature_selection

output_path = ROOT / 'data/processed/cleaned_features.csv'
selection = save_feature_selection(pre_impute_df, selection, output_path)
final_df.to_csv(output_path, index=False)
quality_df = final_df
pd.DataFrame(selection['targets']).T"""))

cells.append(nbf.v4.new_markdown_cell(r"""**산출물 확인**

- Ni·Co 공통 입력 15종 저장 확인함 최종 모델 평가는 Module 2에서 수행함"""))

cells.append(nbf.v4.new_markdown_cell(r"""## 문헌 기반 보존 범위 출처

| 변수 | 정상범위 밖 보존 경계 wt% | 근거 |
|---|---|---|
| Ni | 하한 5.9 / 상한 38.2 | [Table 3 NMC](https://doi.org/10.1016/j.ceramint.2024.08.220) / [Table 2](https://doi.org/10.1021/acssuschemeng.5c10344) |
| Li | 하한 1.81 / 상한 6.76 | [OREAS AC18.10665](https://www.oreas.com/crm/ac18-10665/) / [Table 1](https://doi.org/10.7844/kirr.2024.33.1.58) |
| Cu | 하한 0.05 / 상한 6.08 | [Table A1](https://publications.rwth-aachen.de/record/1024070/files/1024070.pdf) / [Table 1](https://doi.org/10.7844/kirr.2024.33.1.58) |
| Fe | 하한 0.02 | [§3.1](https://doi.org/10.1016/j.wasman.2025.115136) |
| Al | 하한 0.13 / 상한 7.46 | [Table 1](https://doi.org/10.1016/j.ceja.2024.100648) / [OREAS AC18.10664](https://www.oreas.com/crm/ac18-10664/) |

- Co·Mn은 예외 없음 [Mn 15.95 wt% 사례](https://doi.org/10.1039/D4SU00427B)는 정상범위 안이므로 상한 확대에 미사용함
- 문헌 보고값과 정상범위를 연결한 프로젝트 보존 기준이며 보편적인 정상범위는 아님
- 원료 종류·보고값·서지정보는 [문헌 근거 정리](../docs/reference_ranges.md)에 기재함"""))

# 원본 접근 없이 기존 정제 데이터부터 실행하는 별도 노트북 생성 옵션
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--processed', action='store_true', help='기존 정제 데이터부터 시작하는 별도 노트북 생성함')
args = parser.parse_args()
if args.processed:
    start = next(i for i, cell in enumerate(cells) if cell.source.startswith('## 3. 데이터 분할'))
    setup = cells[1].source
    setup = '\n'.join(line for line in setup.splitlines() if not line.startswith('RAW_PATH ='))
    setup = setup.replace("ROOT / 'module1_preprocessing/reports/figures'",
                          "ROOT / 'module1_preprocessing/reports/processed_replay/figures'")
    intro = nbf.v4.new_markdown_cell("""# Module 1 — 기존 정제 데이터 기반 실행 결과

- 원본 읽기 금지 조건에 따라 `data/processed/pre_impute_data.csv`부터 실행함
- 입력 형식 정제·중복 제거·이상값의 결측 변환은 이미 완료된 입력임 — 해당 단계를 이번에 재실행한 결과가 아님
- 기존 분할과 입력 해시를 확인하고 날짜 보간·EDA·수치형 대치·특성 공학·검증·특징 비교를 실행함
- 결과는 `module1_preprocessing/reports/processed_replay/`에 별도 저장함
- 원본부터 실행하는 제출용 전체 파이프라인은 `01_eda_and_validation.ipynb`에 유지함
""")
    load = nbf.v4.new_code_cell("""from module2_prediction.data_loading import load_modeling_inputs

pre_impute_df, stored_selection = load_modeling_inputs(ROOT / 'data/processed/pre_impute_data.csv')
outlier_flagged_df = pre_impute_df.copy()
interpolated_df = interpolate_analysis_date(outlier_flagged_df)
train_i = interpolated_df.index[interpolated_df['split'].eq('train')]
val_i = interpolated_df.index[interpolated_df['split'].eq('validation')]
test_i = interpolated_df.index[interpolated_df['split'].eq('test')]
print('기존 정제 데이터·분할 해시 확인 완료함')
print(f'Train {len(train_i)}행 / Validation {len(val_i)}행 / Test {len(test_i)}행')
print('분석일자 보간 후 결측:', interpolated_df['분석일자'].isna().sum())
""")
    replay = []
    for cell in cells[start:]:
        if cell.source.startswith('train_i, val_i, test_i = stratified_train_val_test_split'):
            continue
        if cell.source.startswith('## 3. 데이터 분할'):
            cell.source = '## 3. 기존 데이터 분할 확인 및 초기 EDA\n\n- 저장된 고정 분할을 그대로 사용하며 이번 실행에서 다시 분할하지 않음'
        if cell.source.startswith('### 6.1 처리 전후 분포'):
            cell.source = '### 6.1 처리 후 분포 및 상관관계\n\n- 원본 접근 없이 대치·특성 공학 이후 개발 데이터의 분포와 상관관계 확인함'
        if 'for frame, folder in [(clean_df.loc[trainval_i]' in cell.source:
            cell.source = cell.source.replace("[(clean_df.loc[trainval_i], 'before'),\n                      (final_df.loc[trainval_i], 'after')]", "[(final_df.loc[trainval_i], 'after')]")
        cell.source = cell.source.replace("ROOT / 'data/processed/cleaned_features.csv'",
                                          "ROOT / 'module1_preprocessing/reports/processed_replay/cleaned_features.csv'")
        replay.append(cell)
    cells = [intro, nbf.v4.new_code_cell(setup), load] + replay

nb["cells"] = cells
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata["language_info"] = {"name": "python"}

name = "01_eda_and_validation_processed.ipynb" if args.processed else "01_eda_and_validation.ipynb"
output_path = Path(__file__).resolve().parents[2] / "notebooks" / name
output_path.parent.mkdir(exist_ok=True)
nbf.write(nb, output_path)
print(f"노트북 생성 완료: {output_path}")
