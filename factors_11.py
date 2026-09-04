"""Build the 11 canonical factors for the causal factor study.

Because the raw parquet only contains daily OHLCV, PE/PB and turnover_rate,
we derive the necessary fundamentals (total_share, total_mktcap, net_profit,
total_equity) from the daily market data itself.  This is a documented proxy
for the quarterly fundamental data used in the multi-factor project.
"""
from typing import List

import numpy as np
import pandas as pd

from causal_factor_research.config import FACTORS

FACTOR_LIST = FACTORS


_MIN_OBS_FOR_SHARE_EST = 30  # minimum observations to estimate total_share
_DERIVED_FUNDAMENTAL_COLS = [
    "total_share",
    "total_mktcap",
    "net_profit",
    "total_equity",
    "roe_proxy",
    "roa_proxy",
    "net_profit_yoy",
]


def _derive_fundamentals(df: pd.DataFrame) -> pd.DataFrame:
    """Derive daily fundamentals from price, volume, PE/PB and turnover_rate.

    We estimate total_share as the median of vol / (turnover_rate / 100).
    This is approximate (turnover_rate is based on float shares in most data
    vendors), but it gives a stable cross-sectional measure of equity size.
    """
    df = df.sort_values(["ts_code", "trade_date"]).copy()

    # Turnover rate is in percent in the raw data; convert to decimal.
    turnover_decimal = df["turnover_rate"] / 100.0
    turnover_decimal = turnover_decimal.replace(0, np.nan)

    # Implied shares from each day's volume and turnover rate.
    implied_shares = df["vol"] / turnover_decimal
    implied_shares = implied_shares.where(implied_shares > 0, np.nan)

    # Use median per stock to avoid daily noise.
    median_shares = (
        implied_shares.groupby(df["ts_code"])
        .transform(lambda x: x.median() if x.count() >= _MIN_OBS_FOR_SHARE_EST else np.nan)
    )
    df["total_share"] = median_shares

    df["total_mktcap"] = df["close"] * df["total_share"]
    # Guard against zero/negative market cap.
    df["total_mktcap"] = df["total_mktcap"].where(df["total_mktcap"] > 0, np.nan)

    # Only use PE/PB that are positive and finite.
    pe = pd.to_numeric(df["pe"], errors="coerce").replace([0, np.inf, -np.inf], np.nan)
    pb = pd.to_numeric(df["pb"], errors="coerce").replace([0, np.inf, -np.inf], np.nan)

    df["net_profit"] = df["total_mktcap"] / pe
    df["total_equity"] = df["total_mktcap"] / pb

    df["roe_proxy"] = df["net_profit"] / df["total_equity"].replace(0, np.nan)
    # Approximate ROA = ROE / 1.5 (equity/assets ~ 2/3 for non-financials).
    df["roa_proxy"] = df["roe_proxy"] / 1.5

    # Year-over-year net profit for GROWTH.  Use ~252 trading days shift.
    df["net_profit_yoy"] = df.groupby("ts_code")["net_profit"].shift(252)
    df["GROWTH"] = (df["net_profit"] - df["net_profit_yoy"]) / df["net_profit_yoy"].abs()
    df["GROWTH"] = df["GROWTH"].replace([np.inf, -np.inf], np.nan)

    return df


def _compute_momentum(close: pd.Series, window: int) -> pd.Series:
    """Classic momentum excluding the most recent day."""
    return close.shift(1) / close.shift(window) - 1


def _compute_volatility(close: pd.Series, window: int = 20) -> pd.Series:
    """Annualised standard deviation of log returns."""
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(window=window, min_periods=10).std() * np.sqrt(252)


def build_11_factors(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the 11 canonical factors from the raw daily table.

    Input columns required: trade_date, ts_code, open, high, low, close, vol,
    amount, turnover_rate, pe, pb.

    Returns the same dataframe with additional factor columns and derived
    fundamentals (total_mktcap, net_profit, ...).
    """
    df = df.sort_values(["ts_code", "trade_date"]).copy()

    # Ensure numeric.
    for col in ["close", "vol", "amount", "turnover_rate", "pe", "pb"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = _derive_fundamentals(df)

    # Value factors.
    df["EP"] = 1.0 / df["pe"].replace(0, np.nan)
    df["BP"] = 1.0 / df["pb"].replace(0, np.nan)

    # Momentum factors.
    df["MOM20"] = df.groupby("ts_code")["close"].transform(lambda x: _compute_momentum(x, 20))
    df["MOM60"] = df.groupby("ts_code")["close"].transform(lambda x: _compute_momentum(x, 60))

    # Quality factors (use derived proxies).
    df["ROE"] = df["roe_proxy"]
    df["ROA"] = df["roa_proxy"]

    # Volatility factor.
    df["VOL20"] = df.groupby("ts_code")["close"].transform(lambda x: _compute_volatility(x, 20))

    # Turnover factor: average daily turnover rate (decimal) over 20 days.
    df["TURN20"] = (
        df.groupby("ts_code")["turnover_rate"]
        .transform(lambda x: x.rolling(window=20, min_periods=10).mean() / 100.0)
    )

    # Liquidity factor: average volume over 20 days, in log.
    df["LIQ20"] = np.log(
        df.groupby("ts_code")["vol"]
        .transform(lambda x: x.rolling(window=20, min_periods=10).mean())
        .replace(0, np.nan)
    )

    # Short-term reversal.
    df["REV5"] = -df.groupby("ts_code")["close"].pct_change(5)

    # Ensure all expected factors are present.
    missing = [f for f in FACTOR_LIST if f not in df.columns]
    if missing:
        raise ValueError(f"Missing 11-factor columns after computation: {missing}")

    return df


if __name__ == "__main__":
    import pandas as pd
    from causal_factor_research.config import RAW_DIR

    basic = pd.read_parquet(RAW_DIR / "basic_all.parquet")
    basic["trade_date"] = pd.to_datetime(basic["trade_date"])
    out = build_11_factors(basic)
    print("11-factor matrix shape:", out.shape)
    print(out[FACTOR_LIST].describe().T)
