"""Basic sanity tests for the causal factor research package."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from causal_factor_research.config import FACTORS, RANDOM_SEED, TRAIN_END, TRAIN_START
from causal_factor_research.utils import (
    compute_ic,
    compute_ic_summary,
    load_features,
    set_random_seeds,
    split_train_test,
)


def test_load_features(df_full):
    assert "label" in df_full.columns
    assert all(f in df_full.columns for f in FACTORS)
    # Rebalance dates start after 60 days of history for 60-day momentum
    assert df_full["trade_date"].min().year >= 2018
    assert df_full["trade_date"].max().year >= 2023


def test_train_test_split(df_full):
    train, test = split_train_test(df_full)
    assert train["trade_date"].max() < test["trade_date"].min()
    assert len(train) > 0
    assert len(test) > 0


def test_ic_functions(df_full, sample_factor):
    ic = compute_ic(df_full, sample_factor)
    assert not ic.empty
    summary = compute_ic_summary(df_full, sample_factor)
    assert "mean_ic" in summary
    assert "tstat" in summary


def test_industry_dummies(df_full):
    ind_cols = [c for c in df_full.columns if c.startswith("ind_")]
    assert len(ind_cols) > 0, "Industry/board dummies should be present"


def test_random_seeds_reproducibility():
    set_random_seeds(RANDOM_SEED)
    a = np.random.rand(5)
    set_random_seeds(RANDOM_SEED)
    b = np.random.rand(5)
    np.testing.assert_array_equal(a, b)


def test_compute_ic_synthetic(synthetic_ic_data):
    df = synthetic_ic_data
    ic = compute_ic(df, "EP")
    assert not ic.empty
    # Synthetic relationship is positive on average
    assert ic["ic"].mean() > 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
