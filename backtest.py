"""Backtest three portfolios: all-factor, IC-significant, and causal-significant."""
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats

from causal_factor_research.causal_effect import double_ml_effect, estimate_all_effects
from causal_factor_research.config import (
    FACTORS,
    FACTOR_PRIOR_SIGN,
    FIGURES_DIR,
    RESULTS_DIR,
    TEST_END,
    TEST_START,
    TRAIN_END,
    TRAIN_START,
)
from causal_factor_research.utils import compute_ic, compute_ic_summary, load_features, save_figure, save_json, save_table, split_train_test



def _train_ic_signs(train: pd.DataFrame) -> Dict[str, int]:
    """Determine empirical sign for each factor from training IC.

    Note: kept for backward compatibility/tests.  Production backtests now use
    ``_backtest_signs`` which blends economic priors with consensus causal signs.
    """
    signs = {}
    for f in FACTORS:
        ic = compute_ic(train, f)
        if not ic.empty and np.isfinite(ic["ic"].mean()):
            signs[f] = int(np.sign(ic["ic"].mean()))
        else:
            signs[f] = FACTOR_PRIOR_SIGN.get(f, 1)
    return signs


def _causal_signs(causal_results: pd.DataFrame) -> Dict[str, int]:
    """Return the consensus causal sign for each factor.

    A factor gets a causal sign if at least one of the four estimation methods is
    both statistically significant (p<0.05) and passes its diagnostic flag.  This
    matches the production-standard definition used in classify_factors and does
    not impose a minimum-count rule beyond the significance threshold itself.
    """
    method_list = ["DML", "IV", "FrontDoor", "CausalForest"]
    valid_map = {"DML": "dml_valid", "IV": "iv_valid", "FrontDoor": "fd_valid", "CausalForest": "cf_valid"}
    signs: Dict[str, int] = {}
    for f in causal_results["factor"].unique():
        sub = causal_results[causal_results["factor"] == f]
        sig_ates = []
        for m in method_list:
            m_sub = sub[sub["method"] == m]
            if m_sub.empty:
                continue
            valid_col = valid_map.get(m)
            valid = bool(m_sub[valid_col].values[0]) if valid_col in m_sub.columns else True
            p = m_sub["pvalue"].values[0]
            if valid and p < 0.05:
                sig_ates.append(m_sub["ate"].values[0])
        if sig_ates:
            consensus = np.sign(np.mean(sig_ates))
            signs[f] = int(consensus) if consensus != 0 else 1
    return signs


def _backtest_signs(causal_results: pd.DataFrame) -> Dict[str, int]:
    """Backtest signs come from fixed economic priors (see config.FACTOR_PRIOR_SIGN).

    The sign of each factor is an economic hypothesis fixed ex-ante, before
    touching the out-of-sample data: e.g. higher EP/BP -> higher expected
    return, higher volatility/turnover -> lower. Using priors rather than any
    sample-estimated sign avoids look-ahead and sign-flipping noise from short
    windows. The causal analysis is used to select *which* factors enter the
    causal portfolio, not to determine their direction.
    """
    return FACTOR_PRIOR_SIGN.copy()


def _build_score(df: pd.DataFrame, factors: List[str], signs: Dict[str, int]) -> pd.Series:
    """Build equal-weight score for selected factors."""
    score = np.zeros(len(df))
    for f in factors:
        if f in df.columns:
            score += signs.get(f, 1) * df[f].fillna(0).values
    return pd.Series(score, index=df.index)


def _period_return(df_date: pd.DataFrame, score_col: str) -> float:
    """Long top 30 / short bottom 30 equal-weight return."""
    df_date = df_date.dropna(subset=[score_col])
    if len(df_date) < 60:
        return 0.0
    top = df_date.nlargest(30, score_col)
    bottom = df_date.nsmallest(30, score_col)
    long_ret = top["label"].mean()
    short_ret = bottom["label"].mean()
    return float(long_ret - short_ret)


def _turnover(weights_new: pd.Series, weights_old: pd.Series) -> float:
    """One-way turnover between two weight vectors."""
    all_assets = weights_new.index.union(weights_old.index)
    w_new = weights_new.reindex(all_assets, fill_value=0)
    w_old = weights_old.reindex(all_assets, fill_value=0)
    return float(np.abs(w_new - w_old).sum() / 2)


def _costs(one_way_cost: float) -> float:
    """Round-trip cost = buy + sell + stamp + slippage."""
    return one_way_cost * 2


ROUND_TRIP_COST = 0.0003 + 0.0003 + 0.001 + 2 * 0.0001


def backtest_portfolio(
    df: pd.DataFrame,
    factors: List[str],
    signs: Dict[str, int],
    portfolio_name: str,
    one_way_cost: float = None,
) -> pd.DataFrame:
    """Run monthly long-short backtest for a given factor list."""
    if one_way_cost is None:
        round_trip = ROUND_TRIP_COST
    else:
        round_trip = _costs(one_way_cost)

    df = df.copy()
    df["score"] = _build_score(df, factors, signs)

    records = []
    prev_weights = pd.Series(dtype=float)
    for date, g in df.groupby("trade_date"):
        g = g.copy()
        g = g.dropna(subset=["score"])
        if len(g) < 60:
            continue
        top = g.nlargest(30, "score")
        bottom = g.nsmallest(30, "score")
        long_stocks = top["ts_code"].tolist()
        short_stocks = bottom["ts_code"].tolist()

        all_stocks = df["ts_code"].unique()
        weights = pd.Series(0.0, index=all_stocks)
        weights.loc[long_stocks] = 0.5 / len(long_stocks)
        weights.loc[short_stocks] = -0.5 / len(short_stocks)

        turnover = _turnover(weights, prev_weights)
        cost = turnover * round_trip
        if prev_weights.empty:
            cost = 0.0

        gross_ret = top["label"].mean() - bottom["label"].mean()
        net_ret = gross_ret - cost

        records.append({
            "trade_date": date,
            "portfolio": portfolio_name,
            "gross_ret": gross_ret,
            "cost": cost,
            "net_ret": net_ret,
            "benchmark_ret": float(g["label"].mean()),
            "turnover": turnover,
            "n_long": len(long_stocks),
            "n_short": len(short_stocks),
        })
        prev_weights = weights

    return pd.DataFrame(records)


def _metrics_from_returns(returns, periods_per_year: int = 12) -> Dict:
    """Compute annualized metrics from period returns.

    If ``returns`` is a DataFrame with columns ``net_ret`` and ``benchmark_ret``,
    the information ratio is computed from active returns relative to the
    equal-weight market benchmark.  If ``returns`` is a Series, it is treated as
    net returns and the information ratio equals the Sharpe ratio.
    """
    if isinstance(returns, pd.DataFrame):
        df = returns[["net_ret", "benchmark_ret"]].dropna()
        net_ret = df["net_ret"]
        active_ret = df["net_ret"] - df["benchmark_ret"]
        ret_for_nav = net_ret
    else:
        net_ret = returns.dropna()
        active_ret = net_ret
        ret_for_nav = net_ret

    if net_ret.empty or net_ret.std() == 0:
        return {
            "annual_return": 0.0,
            "annual_vol": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "total_return": 0.0,
            "calmar": 0.0,
            "information_ratio": 0.0,
            "sortino": 0.0,
        }

    total = (1 + net_ret).prod()
    n = len(net_ret)
    ann_ret = total ** (periods_per_year / n) - 1
    ann_vol = net_ret.std() * np.sqrt(periods_per_year)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    nav = (1 + ret_for_nav).cumprod()
    running_max = nav.cummax()
    mdd = float(((nav - running_max) / running_max).min())
    calmar = ann_ret / abs(mdd) if mdd < 0 else np.nan
    downside = net_ret[net_ret < 0].std() * np.sqrt(periods_per_year)
    sortino = ann_ret / downside if downside > 0 else np.nan

    ann_active = active_ret.mean() * periods_per_year
    tracking_error = active_ret.std() * np.sqrt(periods_per_year)
    information_ratio = ann_active / tracking_error if tracking_error > 0 else 0.0

    return {
        "annual_return": float(ann_ret),
        "annual_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(mdd),
        "win_rate": float((net_ret > 0).mean()),
        "total_return": float(nav.iloc[-1] - 1),
        "calmar": float(calmar) if not np.isnan(calmar) else 0.0,
        "information_ratio": float(information_ratio),
        "sortino": float(sortino) if not np.isnan(sortino) else 0.0,
    }


def _select_causal_factors(causal_results: pd.DataFrame) -> List[str]:
    """Select causally significant factors using diagnostic flags."""
    return list(_causal_signs(causal_results).keys())


def _select_ic_factors(ic_summary: pd.DataFrame) -> List[str]:
    """Select IC-significant factors using the production standard |t|>2."""
    if "tstat" in ic_summary.columns:
        return ic_summary.index[abs(ic_summary["tstat"]) > 2].tolist()
    if "pvalue" in ic_summary.columns:
        return ic_summary.index[ic_summary["pvalue"] < 0.05].tolist()
    return []


def run_all_backtests(train_df: Optional[pd.DataFrame] = None, test_df: Optional[pd.DataFrame] = None) -> Dict:
    """Run three portfolios on the out-of-sample test data using only training-set factor selection."""
    if train_df is None or test_df is None:
        df = load_features()
        train_df, test_df = split_train_test(df)

    # Load or compute causal effects on train data only.
    causal_results_path = RESULTS_DIR / "causal_effects_train.csv"
    if causal_results_path.exists():
        causal_results = pd.read_csv(causal_results_path)
    else:
        causal_results = estimate_all_effects(train_df)

    # Signs are fixed economic priors (see _backtest_signs); short-window
    # sample-estimated signs are noisy and can invert the long/short legs.
    signs = _backtest_signs(causal_results)
    causal_factors = _select_causal_factors(causal_results)

    # Load or compute IC summary on train data only.
    ic_summary_path = RESULTS_DIR / "ic_summary_train.csv"
    if ic_summary_path.exists():
        ic_summary = pd.read_csv(ic_summary_path).set_index("factor")
    else:
        ic_summary = pd.DataFrame([{"factor": f, **compute_ic_summary(train_df, f)} for f in FACTORS]).set_index("factor")
    ic_significant = _select_ic_factors(ic_summary)

    all_factors = [f for f in FACTORS]

    # Fallbacks if selection is empty.
    if not causal_factors:
        dml = causal_results[causal_results["method"] == "DML"]
        if "dml_valid" in dml.columns:
            dml = dml[dml["dml_valid"]].copy()
        if dml.empty:
            dml = causal_results[causal_results["method"] == "DML"].copy()
        dml = dml.sort_values("pvalue")
        causal_factors = dml.head(3)["factor"].tolist()
        logger.warning(f"No causal-significant factors; using DML top-3 fallback: {causal_factors}")

    if not ic_significant:
        ic_sorted = ic_summary.copy()
        ic_sorted["abs_t"] = ic_sorted["tstat"].abs()
        ic_significant = ic_sorted.sort_values("abs_t", ascending=False).head(3).index.tolist()
        logger.warning(f"No IC-significant factors; using top-3 |t| fallback: {ic_significant}")

    logger.info(f"All factors: {len(all_factors)}")
    logger.info(f"Causal factors: {causal_factors}")
    logger.info(f"IC factors: {ic_significant}")

    # Run on OOS test set.
    bt_all = backtest_portfolio(test_df, all_factors, signs, "全因子等权")
    bt_causal = backtest_portfolio(test_df, causal_factors, signs, "因果显著因子")
    bt_ic = backtest_portfolio(test_df, ic_significant, signs, "IC显著因子")

    combined = pd.concat([bt_all, bt_causal, bt_ic], ignore_index=True)
    save_table(combined, "backtest_returns")

    metrics = {}
    nav_curves = {}
    for name, group in combined.groupby("portfolio"):
        group = group.sort_values("trade_date")
        metrics[name] = _metrics_from_returns(group[["net_ret", "benchmark_ret"]])
        nav = (1 + group["net_ret"]).cumprod()
        nav_curves[name] = pd.Series(nav.values, index=group["trade_date"].values, name=name)

    save_json(metrics, "backtest_metrics")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))
    for name, nav in nav_curves.items():
        ax.plot(nav.index, nav.values, label=name)
    ax.axhline(1.0, color="black", linewidth=0.8)
    ax.set_title("三组组合样本外净值曲线")
    ax.set_xlabel("日期")
    ax.set_ylabel("净值")
    ax.legend()
    save_figure(fig, "backtest_nav_curves")

    metrics_df = pd.DataFrame(metrics).T
    save_table(metrics_df.reset_index().rename(columns={"index": "portfolio"}), "backtest_metrics")
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics_df["sharpe"].plot(kind="bar", ax=ax, color=["gray", "green", "orange"])
    ax.set_title("样本外夏普比率对比")
    ax.set_ylabel("Sharpe")
    ax.tick_params(axis="x", rotation=15)
    save_figure(fig, "backtest_sharpe_comparison")

    # Cost sensitivity: 0, 10, 20, 30 bps one-way.
    cost_records = []
    for cost_bps in [0, 10, 20, 30]:
        cost = cost_bps / 10000.0
        for name, factors, label in [
            ("全因子等权", all_factors, "全因子等权"),
            ("因果显著因子", causal_factors, "因果显著因子"),
            ("IC显著因子", ic_significant, "IC显著因子"),
        ]:
            bt = backtest_portfolio(test_df, factors, signs, label, one_way_cost=cost)
            m = _metrics_from_returns(bt.sort_values("trade_date")[["net_ret", "benchmark_ret"]])
            cost_records.append({
                "portfolio": name,
                "one_way_cost_bps": cost_bps,
                "annual_return": m["annual_return"],
                "sharpe": m["sharpe"],
                "max_drawdown": m["max_drawdown"],
            })
    cost_df = pd.DataFrame(cost_records)
    save_table(cost_df, "cost_sensitivity")

    return {
        "metrics": metrics,
        "causal_factors": causal_factors,
        "ic_factors": ic_significant,
        "returns": combined,
    }


if __name__ == "__main__":
    run_all_backtests()
