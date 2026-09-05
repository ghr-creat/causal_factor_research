"""Unit tests for causal effect estimation module."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from causal_factor_research.causal_effect import (
    _get_controls,
    causal_forest_effect,
    double_ml_effect,
    front_door_effect,
    iv_effect,
)
from causal_factor_research.config import RANDOM_SEED


def make_synthetic_panel(n_stocks: int = 30, n_dates: int = 20, add_mediator: bool = False):
    """Create a small panel with factor, label, and minimal controls."""
    rng = np.random.default_rng(RANDOM_SEED)
    dates = pd.date_range("2020-01-01", periods=n_dates, freq="B")
    records = []
    for d in dates:
        for sid in range(n_stocks):
            factor = rng.normal(0, 1)
            if add_mediator:
                mediator = 0.3 * factor + rng.normal(0, 0.5)
                label = 0.2 * mediator + rng.normal(0, 0.5)
            else:
                mediator = np.nan
                label = 0.1 * factor + rng.normal(0, 0.5)
            records.append({
                "trade_date": d,
                "ts_code": f"s{sid:03d}",
                "EP": factor,
                "future_ret_5": mediator,
                "label": label,
                # Controls need non-zero variance: _get_controls drops constant
                # columns as collinear-degenerate.
                "mkt_ret_20": rng.normal(0, 0.001),
                "mkt_vol_20": abs(rng.normal(0.15, 0.02)),
                "mkt_trend": rng.normal(0, 0.02),
                "VOL20": abs(rng.normal(0.15, 0.02)),
                "size": rng.lognormal(0, 0.1),
            })
    return pd.DataFrame(records)


def test_get_controls():
    df = make_synthetic_panel(n_stocks=10, n_dates=10)
    X = _get_controls(df, "EP")
    assert "EP" not in X.columns
    assert "mkt_vol_20" in X.columns


def test_dml_effect(synthetic_ic_data):
    df = synthetic_ic_data.rename(columns={"factor": "EP"})
    df["mkt_ret_20"] = 0.0
    df["mkt_vol_20"] = 0.15
    df["mkt_trend"] = 0.0
    res = double_ml_effect(df, "EP")
    assert res["factor"] == "EP"
    assert res["method"] == "DML"
    assert res["n"] > 0


def test_iv_effect(synthetic_ic_data):
    df = synthetic_ic_data.rename(columns={"factor": "EP"})
    df["mkt_ret_20"] = 0.0
    df["mkt_vol_20"] = 0.15
    df["mkt_trend"] = 0.0
    res = iv_effect(df, "EP")
    assert res["factor"] == "EP"
    assert res["method"] == "IV"
    assert res["n"] > 0
    assert res["first_stage_f"] >= 0


def test_front_door_effect():
    df = make_synthetic_panel(n_stocks=40, n_dates=20, add_mediator=True)
    res = front_door_effect(df, "EP")
    assert res["factor"] == "EP"
    assert res["method"] == "FrontDoor"
    assert res["n"] > 0


def test_causal_forest_effect(synthetic_ic_data):
    df = synthetic_ic_data.rename(columns={"factor": "EP"})
    df["mkt_ret_20"] = 0.0
    df["mkt_vol_20"] = 0.15
    df["mkt_trend"] = 0.0
    df["size"] = 1.0
    df["VOL20"] = 0.15
    res = causal_forest_effect(df, "EP")
    assert res["factor"] == "EP"
    assert res["method"] == "CausalForest"
    assert res["n"] > 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
