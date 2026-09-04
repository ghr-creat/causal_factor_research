"""Unit tests for robustness checks."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from causal_factor_research.config import RANDOM_SEED
from causal_factor_research.robustness import (
    _sensitivity_bounds,
    _shuffle_factor_cross_section,
    placebo_test,
    time_permutation_test,
)


def test_sensitivity_bounds_monotonic():
    rng = np.random.default_rng(RANDOM_SEED)
    paired_diff = rng.normal(0.1, 1.0, 50)
    prev_p_max = None
    for gamma in [1.0, 1.2, 1.5, 2.0]:
        p_min, p_max = _sensitivity_bounds(paired_diff, gamma)
        assert 0 <= p_min <= p_max <= 1
        if prev_p_max is not None:
            assert p_max >= prev_p_max
        prev_p_max = p_max


def test_shuffle_factor_cross_section(robustness_panel):
    df = robustness_panel
    shuffled = _shuffle_factor_cross_section(df, "EP", seed=42)
    # Means per date should be preserved, but stock-level ordering destroyed
    orig_means = df.groupby("trade_date")["EP"].mean().values
    shuf_means = shuffled.groupby("trade_date")["EP"].mean().values
    np.testing.assert_array_almost_equal(orig_means, shuf_means)
    # Correlation between original and shuffled per date should be near zero on average
    corrs = []
    for d in df["trade_date"].unique():
        o = df[df["trade_date"] == d]["EP"].values
        s = shuffled[shuffled["trade_date"] == d]["EP"].values
        corrs.append(np.corrcoef(o, s)[0, 1])
    assert np.mean(corrs) < 0.5


def test_placebo_test(robustness_panel):
    df = robustness_panel
    res = placebo_test(df, n_seeds=3)
    assert not res.empty
    # Noise placebo should not be significant for most methods
    noise_p = res[(res["placebo_type"] == "noise") & (res["method"] == "DML")]["pvalue"].iloc[0]
    assert noise_p > 0.05 or np.isnan(noise_p)


def test_time_permutation_test(robustness_panel):
    df = robustness_panel
    res = time_permutation_test(df, factors=["EP"])
    assert not res.empty
    assert res.iloc[0]["factor"] == "EP"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
