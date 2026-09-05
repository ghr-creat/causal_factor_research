"""Unit tests for backtest module."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from causal_factor_research.backtest import (
    _backtest_signs,
    _build_score,
    _metrics_from_returns,
    _period_return,
    _train_ic_signs,
    backtest_portfolio,
)
from causal_factor_research.config import FACTOR_PRIOR_SIGN, FACTORS, RANDOM_SEED


def make_backtest_panel(n_dates=12, n_stocks=100, factor_name="EP"):
    """Create a panel with a clear factor-score return relationship."""
    rng = np.random.default_rng(RANDOM_SEED)
    dates = pd.date_range("2020-01-01", periods=n_dates, freq="B")
    records = []
    for d in dates:
        # Rank stocks by factor; higher factor -> higher label
        for rank in range(n_stocks):
            factor = (rank - n_stocks / 2) / (n_stocks / 2)
            label = 0.05 * factor + rng.normal(0, 0.02)
            row = {"trade_date": d, "ts_code": f"s{rank:03d}", factor_name: factor, "label": label}
            for f in FACTORS:
                if f not in row:
                    row[f] = rng.normal(0, 1)
            records.append(row)
    return pd.DataFrame(records)


def test_train_ic_signs():
    df = make_backtest_panel(n_dates=10, n_stocks=100, factor_name="EP")
    signs = _train_ic_signs(df)
    assert signs.get("EP", 0) == 1


def test_build_score():
    df = make_backtest_panel(n_dates=3, n_stocks=100, factor_name="EP")
    score = _build_score(df, ["EP"], {"EP": 1})
    assert len(score) == len(df)
    assert score.isna().sum() == 0


def test_period_return():
    df = make_backtest_panel(n_dates=1, n_stocks=100, factor_name="EP")
    df["score"] = _build_score(df, ["EP"], {"EP": 1})
    ret = _period_return(df, "score")
    assert ret > 0


def test_backtest_signs():
    """Backtest signs are always the fixed economic priors, independent of causal consensus."""
    records = [
        {"factor": "EP", "method": "DML", "ate": 0.01, "pvalue": 0.01, "dml_valid": True, "iv_valid": False, "fd_valid": False, "cf_valid": False},
        {"factor": "EP", "method": "IV", "ate": 0.015, "pvalue": 0.01, "dml_valid": False, "iv_valid": True, "fd_valid": False, "cf_valid": False},
        {"factor": "MOM60", "method": "DML", "ate": 0.01, "pvalue": 0.01, "dml_valid": True, "iv_valid": False, "fd_valid": False, "cf_valid": False},
        {"factor": "MOM60", "method": "IV", "ate": -0.02, "pvalue": 0.01, "dml_valid": False, "iv_valid": True, "fd_valid": False, "cf_valid": False},
        {"factor": "VOL20", "method": "DML", "ate": -0.01, "pvalue": 0.01, "dml_valid": True, "iv_valid": False, "fd_valid": False, "cf_valid": False},
        {"factor": "VOL20", "method": "CausalForest", "ate": -0.02, "pvalue": 0.01, "dml_valid": False, "iv_valid": False, "fd_valid": False, "cf_valid": True},
        {"factor": "TURN20", "method": "DML", "ate": 0.02, "pvalue": 0.01, "dml_valid": True, "iv_valid": False, "fd_valid": False, "cf_valid": False},
        {"factor": "TURN20", "method": "IV", "ate": 0.03, "pvalue": 0.08, "dml_valid": False, "iv_valid": True, "fd_valid": False, "cf_valid": False},
    ]
    causal_results = pd.DataFrame(records)
    signs = _backtest_signs(causal_results)
    assert signs["EP"] == 1
    assert signs["MOM60"] == FACTOR_PRIOR_SIGN["MOM60"]
    assert signs["VOL20"] == -1
    assert signs["TURN20"] == FACTOR_PRIOR_SIGN["TURN20"]


def test_metrics_from_returns():
    rng = np.random.default_rng(RANDOM_SEED)
    returns = pd.Series(rng.normal(0.01, 0.05, 24))
    metrics = _metrics_from_returns(returns, periods_per_year=12)
    assert "sharpe" in metrics
    assert "max_drawdown" in metrics
    assert "calmar" in metrics


def test_backtest_portfolio():
    df = make_backtest_panel(n_dates=12, n_stocks=100, factor_name="EP")
    signs = _train_ic_signs(df)
    bt = backtest_portfolio(df, ["EP"], signs, "test")
    assert not bt.empty
    assert "net_ret" in bt.columns
    assert "turnover" in bt.columns


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
