"""Unit tests for factor comparison module."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from causal_factor_research.config import FACTORS, RANDOM_SEED
from causal_factor_research.factor_comparison import (
    classify_factors,
    factor_crowding_analysis,
    factor_decay_analysis,
)


def make_causal_results(causal_factors, non_causal_factors):
    """Build a minimal causal_results dataframe for classification tests."""
    records = []
    for f in causal_factors:
        for method in ["DML", "IV", "FrontDoor", "CausalForest"]:
            records.append({
                "factor": f,
                "method": method,
                "ate": 0.05,
                "pvalue": 0.01,
                "n": 100,
                "dml_valid": True,
                "iv_valid": True,
                "fd_valid": True,
                "cf_valid": True,
            })
    for f in non_causal_factors:
        records.append({
            "factor": f,
            "method": "DML",
            "ate": 0.001,
            "pvalue": 0.5,
            "n": 100,
            "dml_valid": True,
            "iv_valid": True,
            "fd_valid": True,
            "cf_valid": True,
        })
    return pd.DataFrame(records)


def make_panel(n_dates=12, n_stocks=30, factor_name="EP"):
    """Create a panel with a stable factor-label relationship."""
    rng = np.random.default_rng(RANDOM_SEED)
    dates = pd.date_range("2020-01-01", periods=n_dates, freq="B")
    records = []
    for d in dates:
        for s in range(n_stocks):
            factor = rng.normal(0, 1)
            label = 0.05 * factor + rng.normal(0, 0.3)
            row = {"trade_date": d, "ts_code": f"s{s:03d}", factor_name: factor, "label": label}
            for f in FACTORS:
                if f not in row:
                    row[f] = rng.normal(0, 1)
            records.append(row)
    return pd.DataFrame(records)


def test_classify_factors():
    train = make_panel(n_dates=12, n_stocks=30, factor_name="EP")
    test = make_panel(n_dates=12, n_stocks=30, factor_name="EP")
    causal_results = make_causal_results(["EP"], ["BP"])
    classification = classify_factors(train, test, causal_results)
    assert "matrix" in classification
    assert "categories" in classification
    assert "EP" in classification["causal_significant"]


def test_factor_decay_analysis():
    df = make_panel(n_dates=20, n_stocks=30, factor_name="EP")
    decay_df, rate_df = factor_decay_analysis(df, ["EP"], [])
    assert not decay_df.empty
    assert not rate_df.empty
    assert np.isfinite(rate_df["decay_rate"]).all()


def test_factor_crowding_analysis():
    df = make_panel(n_dates=20, n_stocks=30, factor_name="EP")
    crowd_df = factor_crowding_analysis(df, ["EP"], [])
    assert not crowd_df.empty
    assert "low_crowd_ic" in crowd_df.columns
    assert "high_crowd_ic" in crowd_df.columns


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
