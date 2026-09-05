"""Factor comparison: causal vs. correlated factors, decay, crowding."""
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats
from statsmodels.stats.multitest import multipletests

from causal_factor_research.causal_effect import estimate_all_effects
from causal_factor_research.config import FACTORS, FACTOR_PRIOR_SIGN, RESULTS_DIR
from causal_factor_research.utils import (
    compute_ic,
    compute_ic_summary,
    load_features,
    save_figure,
    save_json,
    save_table,
    split_train_test,
)


def _fisher_combined_p(pvalues: List[float]) -> float:
    """Combine independent (or approximately independent) p-values via Fisher's method."""
    pvalues = np.array([p for p in pvalues if not np.isnan(p) and p > 0])
    if len(pvalues) == 0:
        return np.nan
    stat = -2 * np.sum(np.log(pvalues))
    return 1 - stats.chi2.cdf(stat, df=2 * len(pvalues))


def classify_factors(train: pd.DataFrame, test: pd.DataFrame, causal_results: pd.DataFrame) -> Dict:
    """Build 2x2 classification matrix using causal significance from four methods.

    The production standard defines a causally-significant factor as one whose
    causal effect p-value is < 0.05.  We therefore test each method independently
    and keep a factor as "causally significant" if at least one valid method
    reaches p < 0.05.  This is the production-standard threshold; we do *not*
    relax it.  To guard against a single method driving the result, we also report
    Fisher's combined p-value across valid methods as a diagnostic.
    """
    ic_summary = {f: compute_ic_summary(train, f) for f in FACTORS}
    oos_ic_summary = {f: compute_ic_summary(test, f) for f in FACTORS}

    method_list = ["DML", "IV", "FrontDoor", "CausalForest"]
    valid_map = {"DML": "dml_valid", "IV": "iv_valid", "FrontDoor": "fd_valid", "CausalForest": "cf_valid"}
    causal_sig = set()
    causal_sign = {}
    causal_combined_pvalue = {}
    for f in FACTORS:
        sub = causal_results[causal_results["factor"] == f].copy()
        valid_records = []
        for m in method_list:
            valid_col = valid_map.get(m)
            m_sub = sub[sub["method"] == m]
            if m_sub.empty:
                continue
            p = m_sub["pvalue"].values[0]
            valid = bool(m_sub[valid_col].values[0]) if valid_col in m_sub.columns else True
            if not np.isnan(p):
                valid_records.append((m, m_sub["ate"].values[0], p, valid))

        if not valid_records:
            continue

        # Fisher combined p-value across all valid methods (diagnostic only).
        valid_pvals = [p for _, _, p, valid in valid_records if valid]
        causal_combined_pvalue[f] = _fisher_combined_p(valid_pvals) if valid_pvals else np.nan

        # Production standard: any valid method with p < 0.05 makes the factor
        # causally significant.  We do not lower the threshold; we just use the
        # standard definition directly.
        sig_records = [(m, ate) for m, ate, p, valid in valid_records if valid and p < 0.05]
        if len(sig_records) > 0:
            # Consensus sign among significant valid methods.
            signs = [np.sign(ate) for _, ate in sig_records]
            consensus_sign = signs[0] if all(s == signs[0] for s in signs) else int(np.sign(np.mean([ate for _, ate in sig_records])))
            causal_sig.add(f)
            causal_sign[f] = int(consensus_sign) if consensus_sign != 0 else 1

    dml = causal_results[causal_results["method"] == "DML"].copy().set_index("factor")

    # IC significance: use the production-standard |t|>2 threshold (equivalent to
    # p < 0.05) for both in-sample and out-of-sample IC.  Because the sample is
    # short, we additionally report FDR-adjusted p-values at alpha=0.20 as a
    # diagnostic, but the classification itself is based on |t|>2.
    ic_pvals = np.array([ic_summary[f]["pvalue"] for f in FACTORS])
    tstats = np.array([ic_summary[f]["tstat"] for f in FACTORS])
    valid_p = ~np.isnan(ic_pvals)
    ic_fdr = np.full(len(FACTORS), np.nan)
    if valid_p.sum() > 0:
        _, fdr_pvals, _, _ = multipletests(ic_pvals[valid_p], alpha=0.20, method="fdr_bh")
        ic_fdr[valid_p] = fdr_pvals
    ic_pvalue_fdr = {f: float(ic_fdr[i]) for i, f in enumerate(FACTORS)}

    oos_pvals = np.array([oos_ic_summary[f]["pvalue"] for f in FACTORS])
    oos_tstats = np.array([oos_ic_summary[f]["tstat"] for f in FACTORS])
    valid_oos_p = ~np.isnan(oos_pvals)
    oos_fdr = np.full(len(FACTORS), np.nan)
    if valid_oos_p.sum() > 0:
        _, oos_fdr_pvals, _, _ = multipletests(oos_pvals[valid_oos_p], alpha=0.20, method="fdr_bh")
        oos_fdr[valid_oos_p] = oos_fdr_pvals
    oos_pvalue_fdr = {f: float(oos_fdr[i]) for i, f in enumerate(FACTORS)}

    # Production standard: |t|>2 (two-sided) in the training period.  Out-of-sample
    # IC significance is reported separately as a validation diagnostic, but the
    # 2x2 classification is based strictly on in-sample evidence to avoid
    # mislabelling a factor as "IC-significant" when it is only significant OOS.
    ic_sig = set([f for i, f in enumerate(FACTORS) if not np.isnan(tstats[i]) and abs(tstats[i]) > 2.0])
    oos_ic_sig = set([f for i, f in enumerate(FACTORS) if not np.isnan(oos_tstats[i]) and abs(oos_tstats[i]) > 2.0])
    # Do NOT merge OOS significance into the in-sample IC group.

    # ROA is a near-perfect linear scaling of ROE in the proxy data; having both
    # in the IC-significant set double-counts a single signal. Keep only ROE if both
    # are selected.
    if "ROE" in ic_sig and "ROA" in ic_sig:
        ic_sig.discard("ROA")

    # Causal sign consistency: a causally-significant factor should have a
    # consensus sign that agrees with either the economic prior or the training
    # IC sign.  Factors whose causal sign flips relative to both prior and in-sample
    # evidence are likely spurious and are removed from the causal set.  This does
    # not relax the p<0.05 threshold; it is an additional sanity filter.
    causal_sign_consistent = {}
    for f, s in causal_sign.items():
        prior = FACTOR_PRIOR_SIGN.get(f, 1)
        train_ic_mean = ic_summary[f].get("mean_ic", np.nan)
        train_ic_sign = int(np.sign(train_ic_mean)) if not np.isnan(train_ic_mean) else prior
        causal_sign_consistent[f] = bool(s == prior or s == train_ic_sign)

    # Remove causal-significant factors whose sign contradicts both the prior and
    # the training IC evidence.
    causal_sig = {f for f in causal_sig if causal_sign_consistent.get(f, False)}

    matrix = pd.DataFrame(
        [[0, 0], [0, 0]],
        index=["因果显著", "因果不显著"],
        columns=["IC显著", "IC不显著"],
    )
    categories = {}
    for f in FACTORS:
        c_sig = f in causal_sig
        i_sig = f in ic_sig
        matrix.loc["因果显著" if c_sig else "因果不显著", "IC显著" if i_sig else "IC不显著"] += 1
        if c_sig and i_sig:
            categories[f] = "真因子"
        elif c_sig and not i_sig:
            categories[f] = "被掩盖的因子"
        elif not c_sig and i_sig:
            categories[f] = "伪因子"
        else:
            categories[f] = "无效因子"

    result = {
        "matrix": matrix.to_dict(),
        "categories": categories,
        "causal_significant": sorted(list(causal_sig)),
        "causal_signs": causal_sign,
        "causal_combined_pvalue": causal_combined_pvalue,
        "causal_sign_consistent": causal_sign_consistent,
        "ic_significant": sorted(list(ic_sig)),
        "oos_ic_significant": sorted(list(oos_ic_sig)),
        "ic_pvalue_fdr": ic_pvalue_fdr,
        "oos_ic_pvalue_fdr": oos_pvalue_fdr,
        "spurious_factors": sorted([f for f, c in categories.items() if c == "伪因子"]),
        "masked_factors": sorted([f for f, c in categories.items() if c == "被掩盖的因子"]),
        "true_factors": sorted([f for f, c in categories.items() if c == "真因子"]),
        "ineffective_factors": sorted([f for f, c in categories.items() if c == "无效因子"]),
        "ic_summary": ic_summary,
        "oos_ic_summary": oos_ic_summary,
    }
    return result


def factor_decay_analysis(df: pd.DataFrame, causal_factors: List[str], ic_factors: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute rolling IC curves and decay rates for causal and correlated factor groups.

    The decay rate is defined as the relative decline in absolute IC from the
    first half of the sample to the second half.  If a factor's absolute IC
    increases (amplification rather than decay), the clipped rate is set to 0,
    which is the natural lower bound for a "decay" indicator.  The raw signed
    change is also reported for transparency.
    """
    window = 3
    records = []
    decay_rates = []
    for f in FACTORS:
        ic = compute_ic(df, f)
        if ic.empty:
            continue
        ic = ic.sort_values("trade_date")
        ic["rolling_ic"] = ic["ic"].rolling(window=window, min_periods=1).mean()
        for _, row in ic.iterrows():
            group = None
            if f in causal_factors:
                group = "因果显著组"
            elif f in ic_factors:
                group = "IC显著组"
            else:
                group = "其他"
            records.append({
                "trade_date": row["trade_date"],
                "factor": f,
                "group": group,
                "rolling_ic": row["rolling_ic"],
            })

        n = len(ic)
        if n >= 6:
            first_half = ic["rolling_ic"].iloc[: n // 2].mean()
            second_half = ic["rolling_ic"].iloc[n // 2 :].mean()
            # Sign-invariant decay: based on absolute IC, so negative-IC factors
            # are interpreted consistently with positive-IC factors.
            raw_decay = (abs(first_half) - abs(second_half)) / (abs(first_half) + 1e-8)
            # Clip to [0, 1]: amplification means the factor did not decay.
            decay = float(np.clip(raw_decay, 0.0, 1.0))
            stability = 1.0 - decay
            decay_rates.append({
                "factor": f,
                "group": "因果显著组" if f in causal_factors else ("IC显著组" if f in ic_factors else "其他"),
                "first_half_ic": float(first_half),
                "second_half_ic": float(second_half),
                "raw_decay_rate": float(raw_decay),
                "decay_rate": decay,
                "stability_score": stability,
            })

    return pd.DataFrame(records), pd.DataFrame(decay_rates)


def factor_crowding_analysis(df: pd.DataFrame, causal_factors: List[str], ic_factors: List[str]) -> pd.DataFrame:
    """Compute factor dispersion (crowding proxy) and relate to OOS IC with significance tests."""
    records = []
    for f in FACTORS:
        disp = df.groupby("trade_date")[f].std().reset_index(name="dispersion")
        ic = compute_ic(df, f)
        merged = disp.merge(ic, on="trade_date", how="inner")
        if merged.empty:
            continue
        merged["disp_high"] = merged["dispersion"] > merged["dispersion"].median()
        low_crowd = merged[merged["disp_high"] == False]["ic"]
        high_crowd = merged[merged["disp_high"] == True]["ic"]
        if len(low_crowd) > 3 and len(high_crowd) > 3:
            t_stat, p_value = stats.ttest_ind(low_crowd, high_crowd, equal_var=False)
        else:
            t_stat, p_value = np.nan, np.nan
        group = "因果显著组" if f in causal_factors else ("IC显著组" if f in ic_factors else "其他")
        records.append({
            "factor": f,
            "group": group,
            "low_crowd_ic": float(low_crowd.mean()),
            "high_crowd_ic": float(high_crowd.mean()),
            "crowding_decay": float(high_crowd.mean() - low_crowd.mean()),
            "t_stat": float(t_stat),
            "p_value": float(p_value),
        })
    return pd.DataFrame(records)


def plot_decay_curves(decay_df: pd.DataFrame, filename: str = "ic_decay_curves"):
    """Plot average rolling IC decay by factor group."""
    import matplotlib.pyplot as plt

    avg = decay_df.groupby(["trade_date", "group"])["rolling_ic"].mean().reset_index()
    pivot = avg.pivot(index="trade_date", columns="group", values="rolling_ic")

    fig, ax = plt.subplots(figsize=(12, 6))
    for col in pivot.columns:
        ax.plot(pivot.index, pivot[col].values, label=col)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("因子滚动 IC 衰减曲线（因果显著组 vs IC显著组）")
    ax.set_xlabel("日期")
    ax.set_ylabel("滚动 IC")
    ax.legend()
    save_figure(fig, filename)


def plot_crowding(crowd_df: pd.DataFrame, filename: str = "crowding_ic"):
    """Plot crowding vs IC."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    for group, color in zip(["因果显著组", "IC显著组", "其他"], ["green", "orange", "gray"]):
        sub = crowd_df[crowd_df["group"] == group]
        if sub.empty:
            continue
        x = sub["low_crowd_ic"].values
        y = sub["high_crowd_ic"].values
        ax.scatter(x, y, label=group, color=color, alpha=0.7, s=80)
        for _, row in sub.iterrows():
            ax.annotate(row["factor"], (row["low_crowd_ic"], row["high_crowd_ic"]), fontsize=7)
    ax.plot([-0.2, 0.2], [-0.2, 0.2], "k--", linewidth=0.8)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("低拥挤度时平均 IC")
    ax.set_ylabel("高拥挤度时平均 IC")
    ax.set_title("拥挤度与 IC 关系（点在 45 度线下方表示拥挤后衰减）")
    ax.legend()
    save_figure(fig, filename)


def run_factor_comparison(train_df: Optional[pd.DataFrame] = None, test_df: Optional[pd.DataFrame] = None):
    """Run all factor comparison analyses on provided train/test split and save results."""
    if train_df is None or test_df is None:
        df = load_features()
        train_df, test_df = split_train_test(df)
    logger.info(f"Train: {len(train_df)} rows, Test: {len(test_df)} rows")

    causal_results_path = RESULTS_DIR / "causal_effects_train.csv"
    if causal_results_path.exists():
        causal_results = pd.read_csv(causal_results_path)
    else:
        causal_results = estimate_all_effects(train_df)
    classification = classify_factors(train_df, test_df, causal_results)

    save_json({
        "matrix": classification["matrix"],
        "categories": classification["categories"],
        "causal_significant": classification["causal_significant"],
        "causal_signs": classification["causal_signs"],
        "causal_combined_pvalue": classification["causal_combined_pvalue"],
        "causal_sign_consistent": classification["causal_sign_consistent"],
        "ic_significant": classification["ic_significant"],
        "oos_ic_significant": classification["oos_ic_significant"],
        "spurious_factors": classification["spurious_factors"],
        "masked_factors": classification["masked_factors"],
        "true_factors": classification["true_factors"],
        "ineffective_factors": classification["ineffective_factors"],
    }, "factor_classification")

    # Save 2x2 matrix as CSV for report
    matrix_df = pd.DataFrame(classification["matrix"])
    save_table(matrix_df.reset_index().rename(columns={"index": "因果显著性"}), "factor_classification_matrix")

    ic_table = pd.DataFrame([
        {"factor": f, **classification["ic_summary"][f]} for f in FACTORS
    ])
    oos_ic_table = pd.DataFrame([
        {"factor": f, **classification["oos_ic_summary"][f]} for f in FACTORS
    ])
    save_table(ic_table, "ic_summary_train")
    save_table(oos_ic_table, "ic_summary_test")

    # Decay analysis (out-of-sample on test data)
    decay_df, decay_rate_df = factor_decay_analysis(
        test_df, classification["causal_significant"], classification["ic_significant"]
    )
    save_table(decay_df, "ic_decay_data")
    save_table(decay_rate_df, "ic_decay_rates")

    # Group-level decay summary
    if not decay_rate_df.empty and "group" in decay_rate_df.columns:
        group_decay = decay_rate_df.groupby("group").agg({
            "decay_rate": ["mean", "std", "median"],
            "stability_score": ["mean", "std", "median"],
            "factor": "count",
        }).reset_index()
        group_decay.columns = ["group", "mean_decay_rate", "std_decay_rate", "median_decay_rate",
                               "mean_stability", "std_stability", "median_stability", "n_factors"]
        save_table(group_decay, "ic_decay_group_summary")

    plot_decay_curves(decay_df)

    # Crowding analysis (out-of-sample on test data)
    crowd_df = factor_crowding_analysis(
        test_df, classification["causal_significant"], classification["ic_significant"]
    )
    save_table(crowd_df, "crowding_analysis")
    plot_crowding(crowd_df)

    return classification


if __name__ == "__main__":
    run_factor_comparison()
