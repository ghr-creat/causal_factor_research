"""Robustness checks: placebo, time permutation, market regimes, sensitivity analysis."""
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats
from sklearn.linear_model import LogisticRegression

from causal_factor_research.causal_effect import (
    causal_forest_effect,
    double_ml_effect,
    front_door_effect,
    iv_effect,
)
from causal_factor_research.config import FACTORS, RANDOM_SEED, RESULTS_DIR
from causal_factor_research.utils import load_features, save_figure, save_json, save_table, split_train_test



def _sensitivity_bounds(paired_diff: np.ndarray, gamma: float) -> Tuple[float, float]:
    """Approximate Rosenbaum-style bounds for Wilcoxon signed-rank p-value."""
    n = len(paired_diff)
    if n == 0:
        return np.nan, np.nan
    abs_diff = np.abs(paired_diff)
    nonzero = abs_diff > 1e-12
    abs_diff = abs_diff[nonzero]
    signs = np.sign(paired_diff[nonzero])
    ranks = stats.rankdata(abs_diff)
    W = np.sum(ranks * signs)

    upper_prob = gamma / (1.0 + gamma)
    lower_prob = 1.0 / (1.0 + gamma)
    mean_sign_upper = 2 * upper_prob - 1
    mean_sign_lower = 2 * lower_prob - 1

    E_upper = np.sum(ranks * mean_sign_upper)
    E_lower = np.sum(ranks * mean_sign_lower)
    var = np.sum(ranks ** 2 * (1 - mean_sign_upper ** 2))
    sd = np.sqrt(max(var, 1e-12))

    z_upper = (W - E_upper) / sd
    z_lower = (W - E_lower) / sd

    p_min = 2 * min(1 - stats.norm.cdf(abs(z_lower)), stats.norm.cdf(-abs(z_lower)))
    p_max = 2 * min(1 - stats.norm.cdf(abs(z_upper)), stats.norm.cdf(-abs(z_upper)))
    p_min = max(0.0, p_min)
    p_max = min(1.0, p_max)
    return p_min, p_max


def rosenbaum_sensitivity(df: pd.DataFrame, factor: str) -> Dict:
    """Rosenbaum-style sensitivity bounds for top vs bottom factor portfolio."""
    df = df.copy()
    df["treated"] = df.groupby("trade_date")[factor].transform(lambda x: (x >= x.quantile(0.7)).astype(int))
    df["control"] = df.groupby("trade_date")[factor].transform(lambda x: (x <= x.quantile(0.3)).astype(int))

    control_cols = [c for c in df.columns if c not in [factor, "label", "treated", "control", "trade_date", "ts_code", "future_ret_5", "close", "open"]]
    control_cols = [c for c in control_cols if c in df.columns]

    gammas = np.linspace(1.0, 2.0, 11)
    bounds = []
    paired_diffs = []

    for date, g in df.groupby("trade_date"):
        treat = g[g["treated"] == 1]
        ctrl = g[g["control"] == 1]
        if len(treat) < 5 or len(ctrl) < 5:
            continue
        X_t = treat[control_cols].fillna(0).values
        X_c = ctrl[control_cols].fillna(0).values
        pool = pd.concat([treat, ctrl])
        y = pool["treated"].values
        X = pool[control_cols].fillna(0).values
        try:
            logit = LogisticRegression(max_iter=500, solver="lbfgs")
            logit.fit(X, y)
            p_t = logit.predict_proba(X_t)[:, 1]
            p_c = logit.predict_proba(X_c)[:, 1]
        except Exception:
            p_t = treat[factor].values
            p_c = ctrl[factor].values
        for i, pt in enumerate(p_t):
            j = np.argmin(np.abs(p_c - pt))
            diff = treat.iloc[i]["label"] - ctrl.iloc[j]["label"]
            paired_diffs.append(diff)

    paired_diffs = np.array(paired_diffs)
    if len(paired_diffs) < 10:
        return {"factor": factor, "error": "too few pairs"}

    # Baseline exact Wilcoxon p-value at gamma=1.
    try:
        w_stat, p_baseline = stats.wilcoxon(paired_diffs)
    except Exception:
        p_baseline = np.nan

    for gamma in gammas:
        p_min, p_max = _sensitivity_bounds(paired_diffs, gamma)
        bounds.append({"gamma": float(gamma), "p_min": float(p_min), "p_max": float(p_max)})

    return {"factor": factor, "n_pairs": len(paired_diffs), "baseline_pvalue": float(p_baseline), "bounds": bounds}


def placebo_test(df: pd.DataFrame, n_seeds: int = 20) -> pd.DataFrame:
    """Run full causal pipeline on multiple placebo factor constructions.

    Each method's own validity flag (dml_valid/iv_valid/fd_valid/cf_valid) is recorded
    so that significance can be filtered by method validity in the summary.
    """
    methods = [
        ("DML", double_ml_effect),
        ("IV", iv_effect),
        ("FrontDoor", front_door_effect),
        ("CausalForest", causal_forest_effect),
    ]
    valid_flag_map = {
        "DML": "dml_valid",
        "IV": "iv_valid",
        "FrontDoor": "fd_valid",
        "CausalForest": "cf_valid",
    }
    records = []

    for seed in range(n_seeds):
        rng = np.random.default_rng(RANDOM_SEED + seed)
        df_noise = df.copy()
        df_noise["placebo_noise"] = rng.standard_normal(len(df_noise))
        for name, func in methods:
            try:
                res = func(df_noise, "placebo_noise")
                valid_flag = valid_flag_map.get(name)
                valid = bool(res.get(valid_flag, False)) if valid_flag else False
                records.append({
                    "placebo_type": "noise",
                    "seed": seed,
                    "method": name,
                    "ate": res["ate"],
                    "pvalue": res["pvalue"],
                    "tvalue": res.get("tvalue", np.nan),
                    "valid": valid,
                    "dml_valid": res.get("dml_valid"),
                    "iv_valid": res.get("iv_valid"),
                    "fd_valid": res.get("fd_valid"),
                    "cf_valid": res.get("cf_valid"),
                })
            except Exception as e:
                records.append({
                    "placebo_type": "noise",
                    "seed": seed,
                    "method": name,
                    "ate": np.nan,
                    "pvalue": np.nan,
                    "tvalue": np.nan,
                    "valid": False,
                    "dml_valid": False,
                    "iv_valid": False,
                    "fd_valid": False,
                    "cf_valid": False,
                    "error": str(e),
                })

    # Cross-sectionally shuffled version of the first factor.
    for seed in range(n_seeds):
        df_shuffled = _shuffle_factor_cross_section(df, FACTORS[0], seed=RANDOM_SEED + 100 + seed)
        for name, func in methods:
            try:
                res = func(df_shuffled, FACTORS[0])
                valid_flag = valid_flag_map.get(name)
                valid = bool(res.get(valid_flag, False)) if valid_flag else False
                records.append({
                    "placebo_type": "shuffled",
                    "seed": seed,
                    "method": name,
                    "ate": res["ate"],
                    "pvalue": res["pvalue"],
                    "tvalue": res.get("tvalue", np.nan),
                    "valid": valid,
                    "dml_valid": res.get("dml_valid"),
                    "iv_valid": res.get("iv_valid"),
                    "fd_valid": res.get("fd_valid"),
                    "cf_valid": res.get("cf_valid"),
                })
            except Exception as e:
                records.append({
                    "placebo_type": "shuffled",
                    "seed": seed,
                    "method": name,
                    "ate": np.nan,
                    "pvalue": np.nan,
                    "tvalue": np.nan,
                    "valid": False,
                    "dml_valid": False,
                    "iv_valid": False,
                    "fd_valid": False,
                    "cf_valid": False,
                    "error": str(e),
                })
    return pd.DataFrame(records)


def _summarize_placebo(placebo: pd.DataFrame) -> pd.DataFrame:
    """Summarize placebo tests with FDR correction, validity filter, and a binomial audit.

    Reports, per placebo type and method, the number of raw significant seeds,
    the FDR-corrected significant count, and the count after filtering by the
    method's own validity flag.  In addition, a binomial test is applied to the
    raw seed-level p-values to determine whether the observed false-positive rate
    is systematically above the nominal 5% level.  This turns isolated
    seed-level p-values into a rigorous Type-I-error audit and prevents a single
    lucky seed from being interpreted as a real effect.
    """
    from statsmodels.stats.multitest import multipletests

    df = placebo.dropna(subset=["pvalue"]).copy()
    if df.empty:
        return pd.DataFrame()
    _, qvals, _, _ = multipletests(df["pvalue"].values, alpha=0.05, method="fdr_bh")
    df["qvalue"] = qvals
    summary = []
    for (ptype, method), g in df.groupby(["placebo_type", "method"]):
        n = len(g)
        sig_raw = (g["pvalue"] < 0.05).sum()
        sig_fdr = (g["qvalue"] < 0.05).sum()
        sig_valid = ((g["pvalue"] < 0.05) & g["valid"].fillna(False)).sum()

        # Binomial test: is the observed number of raw false positives above the
        # expected 5% under the null?  A small p-value here would mean the
        # method has inflated Type-I error.
        try:
            # scipy >= 1.7 provides binomtest; the legacy binom_test was
            # deprecated in 1.7 and removed in 1.12.
            binom_p = stats.binomtest(int(sig_raw), n, p=0.05, alternative="greater").pvalue
        except AttributeError:
            binom_p = stats.binom_test(int(sig_raw), n, p=0.05, alternative="greater")
        except Exception:
            binom_p = np.nan

        summary.append({
            "placebo_type": ptype,
            "method": method,
            "n_seeds": int(n),
            "n_sig_raw": int(sig_raw),
            "fpr_raw": float(sig_raw / n) if n > 0 else np.nan,
            "n_sig_fdr": int(sig_fdr),
            "n_sig_valid": int(sig_valid),
            "binom_test_pvalue": float(binom_p),
            "fpr_inflated": bool(binom_p < 0.05),
        })
    return pd.DataFrame(summary)


def _shuffle_factor_cross_section(df: pd.DataFrame, factor: str, seed: int = 42) -> pd.DataFrame:
    """Shuffle factor values within each date (destroying stock-level information but preserving time structure)."""
    rng = np.random.default_rng(seed)
    df = df.copy()
    shuffled = []
    for date, g in df.groupby("trade_date"):
        g = g.copy()
        values = g[factor].values.copy()
        rng.shuffle(values)
        g[factor] = values
        shuffled.append(g)
    return pd.concat(shuffled, ignore_index=True)


def time_permutation_test(df: pd.DataFrame, factors: List[str] = None) -> pd.DataFrame:
    """Time-permutation test: future factor should not cause current returns.

    We shift the factor forward by one rebalance period (factor_{t+1}) and estimate
    its effect on the label at time t.  A significant effect indicates that the
    factor is exploiting future information (look-ahead).  The rebalance period
    spacing ensures that factor_{t+1} and the forward-looking label_t do not
    overlap mechanically.
    """
    if factors is None:
        factors = FACTORS
    df = df.sort_values(["ts_code", "trade_date"]).copy()
    records = []
    for factor in factors:
        df_perm = df.copy()
        df_perm[f"{factor}_perm"] = df_perm.groupby("ts_code")[factor].shift(-1)
        # Keep only rows where the shifted factor exists.
        df_perm = df_perm[df_perm[f"{factor}_perm"].notna()].copy()
        if len(df_perm) < 50:
            records.append({"factor": factor, "ate": np.nan, "pvalue": np.nan, "tvalue": np.nan, "error": "too few obs"})
            continue
        try:
            df_perm = df_perm.drop(columns=[factor]).rename(columns={f"{factor}_perm": factor})
            res = double_ml_effect(df_perm, factor)
            records.append({"factor": factor, "ate": res["ate"], "pvalue": res["pvalue"], "tvalue": res.get("tvalue", np.nan)})
        except Exception as e:
            records.append({"factor": factor, "ate": np.nan, "pvalue": np.nan, "tvalue": np.nan, "error": str(e)})
    return pd.DataFrame(records)


def market_regime_analysis(df: pd.DataFrame, factors: List[str] = None) -> pd.DataFrame:
    """Split by market regime and run DML for all selected factors."""
    if factors is None:
        factors = FACTORS
    df = df.copy()
    df["regime"] = pd.cut(
        df["mkt_trend"],
        bins=[-np.inf, -0.05, 0.05, np.inf],
        labels=["bear", "neutral", "bull"],
    )
    records = []
    for regime, g in df.groupby("regime"):
        if len(g) < 200:
            continue
        for factor in factors:
            try:
                res = double_ml_effect(g, factor)
                records.append({
                    "regime": regime,
                    "factor": factor,
                    "ate": res["ate"],
                    "pvalue": res["pvalue"],
                    "n": res["n"],
                })
            except Exception:
                records.append({"regime": regime, "factor": factor, "ate": np.nan, "pvalue": np.nan, "n": 0})
    return pd.DataFrame(records)


def run_all_robustness(train_df: Optional[pd.DataFrame] = None, n_seeds: int = 20):
    """Run all robustness checks on training data only to avoid leakage.

    Args:
        n_seeds: Number of random seeds for the placebo test.  Default is 20;
            smaller values can be used for quick tests.
    """
    if train_df is None:
        df = load_features()
        train_df, _ = split_train_test(df)

    # Causal-significant factors from train-only causal effects.
    dml = pd.read_csv(RESULTS_DIR / "causal_effects_train.csv")
    dml = dml[dml["method"] == "DML"].sort_values("pvalue")
    top_factors = dml.head(3)["factor"].tolist()

    results = {}

    # Placebo
    placebo = placebo_test(train_df, n_seeds=n_seeds)
    save_table(placebo, "placebo_test")
    results["placebo"] = placebo.to_dict(orient="records")
    placebo_summary = _summarize_placebo(placebo)
    if not placebo_summary.empty:
        save_table(placebo_summary, "placebo_test_summary")
        results["placebo_summary"] = placebo_summary.to_dict(orient="records")

    # Time permutation (use all factors for comprehensive check)
    perm = time_permutation_test(train_df, factors=FACTORS)
    if not perm.empty:
        save_table(perm, "time_permutation_test")
        results["time_permutation"] = perm.to_dict(orient="records")

    # Market regime (all factors)
    regimes = market_regime_analysis(train_df, factors=FACTORS)
    save_table(regimes, "market_regime_dml")
    results["market_regime"] = regimes.to_dict(orient="records")

    import matplotlib.pyplot as plt

    pivot = regimes.pivot(index="factor", columns="regime", values="ate")
    fig, ax = plt.subplots(figsize=(14, 8))
    pivot.plot(kind="bar", ax=ax)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("DML 因果效应分市场状态对比")
    ax.set_ylabel("ATE")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(title="市场状态")
    save_figure(fig, "market_regime_comparison")

    # Rosenbaum sensitivity for top DML factors.
    rosenbaum_records = []
    for factor in top_factors:
        logger.info(f"Rosenbaum sensitivity for {factor}...")
        try:
            res = rosenbaum_sensitivity(train_df, factor)
            bounds = res.get("bounds", [])
            for b in bounds:
                rosenbaum_records.append({
                    "factor": factor,
                    "gamma": b["gamma"],
                    "p_min": b["p_min"],
                    "p_max": b["p_max"],
                    "baseline_pvalue": res.get("baseline_pvalue", np.nan),
                })
        except Exception as e:
            logger.warning(f"Rosenbaum sensitivity for {factor} failed: {e}")
    rosenbaum_df = pd.DataFrame(rosenbaum_records)
    save_table(rosenbaum_df, "rosenbaum_bounds")
    results["rosenbaum"] = rosenbaum_df.to_dict(orient="records")

    if not rosenbaum_df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        for factor in rosenbaum_df["factor"].unique():
            sub = rosenbaum_df[rosenbaum_df["factor"] == factor]
            ax.plot(sub["gamma"], sub["p_max"], label=factor)
            ax.fill_between(sub["gamma"], sub["p_min"], sub["p_max"], alpha=0.2)
        ax.axhline(0.05, color="red", linestyle="--", linewidth=1, label="p=0.05")
        ax.set_xlabel("Γ (混杂强度)")
        ax.set_ylabel("p 值范围")
        ax.set_title("Rosenbaum 敏感性边界：未观测混杂强度 vs p 值")
        ax.legend()
        save_figure(fig, "rosenbaum_bounds")

    save_json(results, "robustness_summary")
    return results


if __name__ == "__main__":
    run_all_robustness()
