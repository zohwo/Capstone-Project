import numpy as np
import pandas as pd

from module1_preprocessing.config import COMPOSITION_COLUMNS
from module3_optimization.scripts.compare_blends import representative_feeds, scenarios


def test_representatives_ignore_test_and_missing_observations():
    rows = []
    for grade in ['A', 'B', 'C']:
        for split, ni in [('train', 10), ('validation', 20), ('train', np.nan), ('test', 999)]:
            row = dict(zip(COMPOSITION_COLUMNS, [ni, 5, 3, 8, 1, 1, 1]))
            rows.append(dict(row, split=split, 원료등급=grade))
    feeds, development = representative_feeds(pd.DataFrame(rows))
    assert len(development) == 9
    assert all(feed['Ni_함량_pct'] == 15 for feed in feeds.values())


def test_blend_mass_balance_and_fixed_total():
    feeds = {grade: dict(zip(COMPOSITION_COLUMNS, [ni, 5, 3, 8, 1, 1, 1]), 원료투입량_kg=140)
             for grade, ni in [('A', 25), ('B', 20), ('C', 10)]}
    cases = scenarios(feeds)
    assert len(cases) == 12
    assert cases['AC_7_3']['Ni_함량_pct'] == 20.5
    assert all(feed['원료투입량_kg'] == 140 for feed in cases.values())
