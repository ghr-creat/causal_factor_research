"""Utility helpers for the causal factor research project."""
import json
import os
import random
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from loguru import logger

from causal_factor_research.config import (
    FACTORS,
    FIGURES_DIR,
    PROCESSED_DIR,
    RANDOM_SEED,
    RESULTS_DIR,
    TEST_END,
    TEST_START,
    TRAIN_END,
    TRAIN_START,
)




def load_features() -> pd.DataFrame:
    """Load the standardized causal features dataset."""
    path = PROCESSED_DIR / "causal_features.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Feature file not found: {path}. Run data_builder.py first.")
    logger.info(f"Loading features from {path}")
    with _suppress_runtime_warnings():
        df = pd.read_parquet(path)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    logger.info(f"Loaded {len(df)} rows, {len(FACTORS)} factors, dates {df['trade_date'].min().date()} ~ {df['trade_date'].max().date()}")
    return df


@contextmanager
def _suppress_runtime_warnings():
    """Context manager to suppress RuntimeWarning inside numerical operations."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        yield


@contextmanager
def suppress_warnings():
    """Public context manager to suppress warnings in a scoped way."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def set_random_seeds(seed: int = RANDOM_SEED):
    """Set random seeds across all libraries for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    try:
        import xgboost as xgb
        xgb.set_config(verbosity=0)
    except Exception:
        pass


def split_train_test(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split by date strictly."""
    train = df[(df["trade_date"] >= TRAIN_START) & (df["trade_date"] <= TRAIN_END)].copy()
    test = df[(df["trade_date"] >= TEST_START) & (df["trade_date"] <= TEST_END)].copy()
    logger.info(f"Train {TRAIN_START}~{TRAIN_END}: {len(train)} rows; Test {TEST_START}~{TEST_END}: {len(test)} rows")
    return train, test


def compute_ic(df: pd.DataFrame, factor_col: str, label_col: str = "label", method: str = "spearman") -> pd.DataFrame:
    """Compute rank IC (Spearman) or Pearson IC per date.

    Args:
        method: "spearman" (default, matching the multi-factor project) or "pearson".
    """
    corr_func = stats.spearmanr if method == "spearman" else stats.pearsonr
    ic_list = []
    for date, g in df.groupby("trade_date"):
        x = g[factor_col].values
        y = g[label_col].values
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() < 5:
            continue
        x = x[valid]
        y = y[valid]
        r, p = corr_func(x, y)
        ic_list.append({"trade_date": date, "ic": r, "p": p})
    return pd.DataFrame(ic_list)


def _newey_west_se(ic: pd.Series, max_lags: int = 3) -> float:
    """Newey-West standard error for the mean of a time series."""
    try:
        import statsmodels.api as sm

        n = len(ic)
        if n < max_lags + 2:
            return float(ic.std() / np.sqrt(n))
        ic_const = sm.add_constant(np.ones(n))
        ols = sm.OLS(ic.values, ic_const).fit(
            cov_type="HAC", cov_kwds={"maxlags": max_lags}
        )
        return float(ols.bse[0])
    except Exception:
        n = len(ic)
        return float(ic.std() / np.sqrt(n)) if n > 0 else np.nan


def compute_ic_summary(df: pd.DataFrame, factor_col: str, label_col: str = "label", method: str = "spearman", newey_west: bool = True) -> Dict:
    """Compute mean IC, t-stat, positive rate.

    Default is Spearman rank IC with Newey-West standard errors to account for
    serial correlation in the monthly IC series.
    """
    ic = compute_ic(df, factor_col, label_col, method=method)
    if ic.empty:
        return {"mean_ic": np.nan, "tstat": np.nan, "pos_rate": np.nan, "pvalue": np.nan, "method": method}
    mean_ic = ic["ic"].mean()
    se = _newey_west_se(ic["ic"]) if newey_west else ic["ic"].std() / np.sqrt(len(ic))
    tstat = mean_ic / se if se > 0 else np.nan
    pvalue = 2 * (1 - stats.t.cdf(abs(tstat), df=len(ic) - 1)) if not np.isnan(tstat) else np.nan
    pos_rate = (ic["ic"] > 0).mean()
    return {
        "mean_ic": float(mean_ic),
        "tstat": float(tstat),
        "pos_rate": float(pos_rate),
        "pvalue": float(pvalue),
        "method": method,
    }


def save_json(obj: dict, name: str):
    """Save a JSON object to results dir."""
    path = RESULTS_DIR / f"{name}.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    logger.debug(f"Saved JSON: {path}")


def save_table(df: pd.DataFrame, name: str):
    """Save a DataFrame to CSV."""
    path = RESULTS_DIR / f"{name}.csv"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    logger.debug(f"Saved CSV: {path}")


def save_figure(fig, name: str):
    """Save a matplotlib figure."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    logger.debug(f"Saved figure: {path}")


def winsorize_series(s: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """Winsorize a series at given percentiles."""
    lo = s.quantile(lower)
    hi = s.quantile(upper)
    return s.clip(lo, hi)


def load_causal_effects_train() -> pd.DataFrame:
    """Load train-only causal effects; raise if not found to prevent data leakage."""
    path = RESULTS_DIR / "causal_effects_train.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Train-only causal effects not found: {path}. "
            "Run causal_effect.estimate_all_effects(train_df) first to avoid data leakage."
        )
    return pd.read_csv(path)


def add_constant(df: pd.DataFrame, cols: List[str] = None) -> pd.DataFrame:
    """Add a constant column to a dataframe."""
    df = df.copy()
    df["_const"] = 1.0
    return df


if __name__ == "__main__":
    df = load_features()
    print(f"Loaded {len(df)} rows, factors: {df[FACTORS].shape}")
