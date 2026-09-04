"""Build the 11-factor real-data dataset for causal inference research."""
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from causal_factor_research.config import (
    FACTORS,
    FIGURES_DIR,
    LABEL_HORIZON,
    MAD_MULTIPLE,
    PROCESSED_DIR,
    RAW_DIR,
    REBALANCE_DAYS,
)
from causal_factor_research.factors_11 import build_11_factors


def load_raw_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load daily price/basic data and index data."""
    basic_path = RAW_DIR / "basic_all.parquet"
    index_path = RAW_DIR / "index_daily_000300.SH.parquet"

    basic = pd.read_parquet(basic_path)
    index_df = pd.read_parquet(index_path)

    basic["trade_date"] = pd.to_datetime(basic["trade_date"])
    index_df["trade_date"] = pd.to_datetime(index_df["trade_date"])

    # Ensure numeric columns.
    numeric_cols = [
        "open", "high", "low", "close", "pre_close",
        "vol", "amount", "turnover_rate", "pct_chg", "pe", "pb",
    ]
    for col in numeric_cols:
        if col in basic.columns:
            basic[col] = pd.to_numeric(basic[col], errors="coerce")

    # PE/PB must be positive.
    for col in ["pe", "pb"]:
        if col in basic.columns:
            basic[col] = basic[col].where(basic[col] > 0, np.nan)

    return basic, index_df


def add_market_factors(df: pd.DataFrame, index_df: pd.DataFrame) -> pd.DataFrame:
    """Add market return and rolling market stats."""
    index_df = index_df.sort_values("trade_date").copy()
    index_df["mkt_ret"] = index_df["close"] / index_df["pre_close"] - 1
    index_df["mkt_ret_20"] = index_df["mkt_ret"].rolling(window=20, min_periods=10).mean()
    index_df["mkt_vol_20"] = index_df["mkt_ret"].rolling(window=20, min_periods=10).std()
    index_df["mkt_trend"] = index_df["close"] / index_df["close"].shift(20) - 1
    df = df.merge(
        index_df[["trade_date", "mkt_ret_20", "mkt_vol_20", "mkt_trend"]],
        on="trade_date",
        how="left",
    )
    return df


def add_industry_controls(df: pd.DataFrame) -> pd.DataFrame:
    """Add industry/board dummy controls from ts_code prefix."""
    df = df.copy()

    def _board(code: str) -> str:
        if not isinstance(code, str) or len(code) < 6:
            return "other"
        prefix = code[:3]
        if prefix in ("600", "601", "603", "605", "000"):
            return "main_board"
        elif prefix in ("002", "003"):
            return "sme"
        elif prefix == "300":
            return "chinext"
        elif prefix == "688":
            return "star"
        else:
            return "other"

    df["industry"] = df["ts_code"].apply(_board)
    dummies = pd.get_dummies(df["industry"], prefix="ind", drop_first=True)
    df = pd.concat([df, dummies], axis=1)
    df = df.drop(columns=["industry"], errors="ignore")
    return df


def generate_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Generate 20-day forward return without lookahead."""
    df = df.sort_values(["ts_code", "trade_date"]).copy()
    df["next_open"] = df.groupby("ts_code")["open"].shift(-1)
    df["future_close"] = df.groupby("ts_code")["close"].shift(-LABEL_HORIZON)
    df["label"] = df["future_close"] / df["next_open"] - 1

    # Mediator for front-door adjustment: 5-day forward return.
    df["future_close_5"] = df.groupby("ts_code")["close"].shift(-5)
    df["future_ret_5"] = df["future_close_5"] / df["next_open"] - 1

    df = df.drop(columns=["next_open", "future_close", "future_close_5"])
    return df


def select_rebalance_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Select every 20th trading day, starting after enough history."""
    dates = sorted(df["trade_date"].unique())
    # 120 days needed for 60-day momentum and 20-day volatility.
    start_idx = 120
    rebalance_dates = pd.to_datetime(dates[start_idx::REBALANCE_DAYS])
    df = df[df["trade_date"].isin(rebalance_dates)].copy()
    return df


def industry_neutralize(df: pd.DataFrame, factor_cols: List[str]) -> pd.DataFrame:
    """Cross-sectionally neutralize factors with respect to size and industry dummies.

    We regress each factor on log market cap and industry dummies per date and
    keep the residuals.  This is equivalent to the standard A-share factor
    preprocessing pipeline.
    """
    df = df.sort_values(["trade_date", "ts_code"]).copy()
    ind_cols = [c for c in df.columns if c.startswith("ind_")]
    if "total_mktcap" not in df.columns or df["total_mktcap"].isna().all():
        # No size information available; skip neutralization.
        return df

    df["size"] = np.log(df["total_mktcap"].replace(0, np.nan))

    for col in factor_cols:
        if col not in df.columns:
            continue
        residuals = []
        for date, g in df.groupby("trade_date"):
            y = g[col].values
            valid = np.isfinite(y)
            X_list = [g["size"].values]
            for ind in ind_cols:
                X_list.append(g[ind].values)
            X = np.column_stack(X_list)
            # Keep only rows with finite y and all X finite.
            x_finite = np.all(np.isfinite(X), axis=1)
            use = valid & x_finite
            if use.sum() < 5:
                residuals.append(pd.Series(np.nan, index=g.index))
                continue
            y_sub = y[use]
            X_sub = X[use]
            # Solve OLS and compute residuals on all rows (including those with
            # missing y by predicting them with available X).
            beta, *_ = np.linalg.lstsq(X_sub, y_sub, rcond=None)
            pred = X.dot(beta)
            resid = y - pred
            residuals.append(pd.Series(resid, index=g.index))
        df[col] = pd.concat(residuals).reindex(df.index)
    return df


def handle_missing_values(df: pd.DataFrame, factor_cols: List[str]) -> pd.DataFrame:
    """Cross-sectional median fill per rebalance date, then forward fill per stock."""
    df = df.sort_values(["ts_code", "trade_date"]).copy()
    for col in factor_cols:
        if col not in df.columns:
            continue
        df[col] = df.groupby("ts_code")[col].transform(lambda x: x.ffill(limit=3))
        df[col] = df.groupby("trade_date")[col].transform(lambda x: x.fillna(x.median()))
    return df


def extreme_value_process(df: pd.DataFrame, factor_cols: List[str]) -> pd.DataFrame:
    """MAD winsorization cross-sectionally per rebalance date."""
    for col in factor_cols:
        if col not in df.columns:
            continue
        median = df.groupby("trade_date")[col].transform("median")
        mad = df.groupby("trade_date")[col].transform(
            lambda x: np.median(np.abs(x - np.median(x)))
        )
        upper = median + MAD_MULTIPLE * mad
        lower = median - MAD_MULTIPLE * mad
        df[col] = np.clip(df[col], lower, upper)
    return df


def standardize(df: pd.DataFrame, factor_cols: List[str]) -> pd.DataFrame:
    """Cross-sectional z-score standardization."""
    for col in factor_cols:
        if col not in df.columns:
            continue
        mean = df.groupby("trade_date")[col].transform("mean")
        std = df.groupby("trade_date")[col].transform("std")
        df[col] = (df[col] - mean) / (std + 1e-8)
    return df


def build_dataset() -> pd.DataFrame:
    """Main pipeline to build the 11-factor monthly dataset."""
    df, index_df = load_raw_data()
    df = build_11_factors(df)
    df = add_market_factors(df, index_df)
    df = add_industry_controls(df)
    df = generate_labels(df)
    df = select_rebalance_dates(df)

    missing = [f for f in FACTORS if f not in df.columns]
    if missing:
        raise ValueError(f"Missing factors after computation: {missing}")

    df = handle_missing_values(df, FACTORS)
    df = industry_neutralize(df, FACTORS)
    df = extreme_value_process(df, FACTORS)

    # Save raw (unstandardized) version for reference.
    raw_cols = (
        ["ts_code", "trade_date", "close", "open"]
        + FACTORS
        + ["label", "future_ret_5"]
        + ["mkt_ret_20", "mkt_vol_20", "mkt_trend"]
    )
    raw_cols = [c for c in raw_cols if c in df.columns]
    df_raw = df[raw_cols].copy()
    df_raw.to_parquet(PROCESSED_DIR / "causal_features_raw.parquet", index=False)

    df = standardize(df, FACTORS)

    final_cols = (
        ["ts_code", "trade_date", "close", "open"]
        + FACTORS
        + ["label", "future_ret_5"]
        + ["mkt_ret_20", "mkt_vol_20", "mkt_trend"]
    )
    ind_cols = [c for c in df.columns if c.startswith("ind_")]
    final_cols = list(dict.fromkeys(final_cols + ind_cols))
    final_cols = [c for c in final_cols if c in df.columns]
    df = df[final_cols].copy()
    df = df.dropna(subset=FACTORS + ["label"]).sort_values(
        ["trade_date", "ts_code"]
    ).reset_index(drop=True)

    df.to_parquet(PROCESSED_DIR / "causal_features.parquet", index=False)
    return df


if __name__ == "__main__":
    data = build_dataset()
    print(
        f"Built causal features: {len(data)} rows, {len(FACTORS)} factors, "
        f"dates {data['trade_date'].min()} ~ {data['trade_date'].max()}"
    )
