"""Shared fixtures and utilities for the causal factor research test suite."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from causal_factor_research.config import FACTORS, RANDOM_SEED, TRAIN_END, TRAIN_START
from causal_factor_research.utils import compute_ic, compute_ic_summary, load_features, set_random_seeds, split_train_test


@pytest.fixture(scope="session", autouse=True)
def seed():
    """Set the global random seed once per test session."""
    set_random_seeds(RANDOM_SEED)


@pytest.fixture(scope="session")
def feature_path():
    """Return the path to the processed features."""
    return PROJECT_ROOT / "data" / "processed" / "causal_features.parquet"


@pytest.fixture(scope="session")
def df_full(feature_path):
    """Load the full feature dataframe once per session."""
    if not feature_path.exists():
        pytest.skip(f"Feature file not found: {feature_path}")
    return load_features()


@pytest.fixture(scope="session")
def train_df(df_full):
    """Return the training split."""
    train, _ = split_train_test(df_full)
    return train


@pytest.fixture(scope="session")
def test_df(df_full):
    """Return the test split."""
    _, test = split_train_test(df_full)
    return test


@pytest.fixture(scope="session")
def sample_factor():
    """Return a representative factor for tests."""
    return "EP"


def _add_missing_factors(df: pd.DataFrame, active_factor: str = "EP", rng=None) -> pd.DataFrame:
    """Add all required FACTOR columns with default random values so modules looping over FACTORS do not fail."""
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)
    df = df.copy()
    for f in FACTORS:
        if f not in df.columns:
            if f == active_factor:
                continue
            # Use modest random values to avoid dominating the active factor
            df[f] = rng.normal(0, 1, size=len(df))
    return df


def _add_market_and_industry_controls(df: pd.DataFrame, rng=None) -> pd.DataFrame:
    """Add market context and industry dummies required by the pipeline."""
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)
    df = df.copy()
    if "mkt_ret_20" not in df.columns:
        df["mkt_ret_20"] = rng.normal(0, 0.001, size=len(df))
    if "mkt_vol_20" not in df.columns:
        df["mkt_vol_20"] = 0.15
    if "mkt_trend" not in df.columns:
        df["mkt_trend"] = rng.normal(0, 0.02, size=len(df))
    # Add a dummy industry column
    if not any(c.startswith("ind_") for c in df.columns):
        df["ind_main_board"] = rng.integers(0, 2, size=len(df))
    return df


@pytest.fixture
def synthetic_ic_data():
    """Create a small synthetic panel with a known factor-return relationship."""
    rng = np.random.default_rng(RANDOM_SEED)
    dates = pd.date_range("2020-01-01", periods=10, freq="B")
    records = []
    for d in dates:
        for stock in range(20):
            factor = rng.normal(0, 1)
            label = 0.1 * factor + rng.normal(0, 0.5)
            records.append({"trade_date": d, "ts_code": f"s{stock}", "EP": factor, "label": label})
    df = pd.DataFrame(records)
    df = _add_missing_factors(df, active_factor="EP", rng=rng)
    df = _add_market_and_industry_controls(df, rng=rng)
    return df


@pytest.fixture
def backtest_returns():
    """Simple daily return series for backtest metric tests."""
    rng = np.random.default_rng(RANDOM_SEED)
    return pd.Series(rng.normal(0.0005, 0.02, 252))


@pytest.fixture
def robustness_panel():
    """Small panel for robustness tests, includes market controls."""
    rng = np.random.default_rng(RANDOM_SEED)
    dates = pd.date_range("2020-01-01", periods=20, freq="B")
    records = []
    for d in dates:
        for s in range(50):
            ep = rng.normal(0, 1)
            records.append({
                "trade_date": d,
                "ts_code": f"s{s:03d}",
                "EP": ep,
                "label": 0.05 * ep + rng.normal(0, 0.3),
                "mkt_ret_20": 0.0,
                "mkt_vol_20": 0.15,
                "mkt_trend": 0.0,
            })
    df = pd.DataFrame(records)
    df = _add_missing_factors(df, active_factor="EP", rng=rng)
    df = _add_market_and_industry_controls(df, rng=rng)
    return df
