"""Causal effect estimation: Double ML, IV, Front-door, Causal Forest.

All four estimators now report diagnostic flags that are used in the
"method consistency" step.  Only estimators that pass their diagnostics are
counted when deciding whether a factor is "causally significant".
"""
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from loguru import logger
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

from causal_factor_research.config import CAUSAL_FOREST_N_ESTIMATORS, DML_N_SPLITS, FACTORS, FIGURES_DIR, RESULTS_DIR, RANDOM_SEED
from causal_factor_research.utils import load_features, save_figure, save_json, save_table, suppress_warnings


MAX_N = 1000  # cap rows per causal effect estimation to keep runtime reasonable

XGBOOST_AVAILABLE = False
ECONML_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception:
    pass

try:
    import econml
    # Only use EconML for versions with the API we target (>=0.14). Older versions
    # lack the model_final/effect API and are slower than the manual fallback.
    if tuple(map(int, econml.__version__.split(".")[:2])) >= (0, 14):
        from econml.dml import DML as EconDML, LinearDML as EconLinearDML
        from econml.grf import CausalForest as EconCausalForest
        ECONML_AVAILABLE = True
    else:
        ECONML_AVAILABLE = False
except Exception:
    ECONML_AVAILABLE = False


def _get_nuisance_model(task: str = "regression"):
    """Return a fast nuisance model. Prefer XGBoost, then HistGB, then classic GB."""
    if XGBOOST_AVAILABLE:
        return xgb.XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=4,
            verbosity=0,
        )
    try:
        return HistGradientBoostingRegressor(
            max_iter=50, max_depth=4, learning_rate=0.05, random_state=42
        )
    except Exception:
        return GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)


def _subsample(df: pd.DataFrame, max_n: int = MAX_N) -> pd.DataFrame:
    """Subsample by date to preserve within-stock time ordering for IV/FD."""
    if len(df) <= max_n:
        return df
    dates = sorted(df["trade_date"].unique())
    avg_n = df.groupby("trade_date").size().mean()
    target_dates = max_n / max(avg_n, 1)
    step = max(1, int(len(dates) / target_dates))
    selected_dates = dates[::step]
    return df[df["trade_date"].isin(selected_dates)].copy()


def _drop_collinear_controls(X: pd.DataFrame, target: pd.Series = None, threshold: float = 0.95) -> pd.DataFrame:
    """Drop columns that are nearly perfectly collinear with an earlier column or with the target."""
    X = X.copy()
    cols = list(X.columns)
    keep = []
    for c in cols:
        if X[c].std() == 0:
            continue
        redundant = False
        # Drop if nearly perfectly correlated with the target treatment.
        if target is not None and target.std() > 0:
            corr_t = np.corrcoef(X[c].dropna().values, target.dropna().values)[0, 1]
            if abs(corr_t) >= threshold:
                redundant = True
        if not redundant:
            for k in keep:
                corr = np.corrcoef(X[c].dropna().values, X[k].dropna().values)[0, 1]
                if abs(corr) >= threshold:
                    redundant = True
                    break
        if not redundant:
            keep.append(c)
    return X[keep]


def _get_controls(df: pd.DataFrame, target_factor: str) -> pd.DataFrame:
    """Build control matrix for a given factor.

    Includes other factors, market context, and industry/board dummies.  Near-perfect
    collinear controls (e.g. ROE and ROA in our proxy data) are removed, including
    controls that are nearly identical to the target.
    """
    other_factors = [f for f in FACTORS if f != target_factor]
    control_cols = other_factors + ["mkt_ret_20", "mkt_vol_20", "mkt_trend"]
    ind_cols = [c for c in df.columns if c.startswith("ind_")]
    control_cols = list(dict.fromkeys(control_cols + ind_cols))
    control_cols = [c for c in control_cols if c in df.columns]
    X = df[control_cols].copy()
    target = df[target_factor].copy()
    X = _drop_collinear_controls(X, target=target, threshold=0.95)
    return X


def _propensity_overlap(D: np.ndarray, X: np.ndarray) -> float:
    """Compute propensity-score overlap ratio for a binary treatment approximation."""
    from sklearn.linear_model import LogisticRegression

    try:
        median_d = np.median(D)
        treated = (D > median_d).astype(int)
        logit = LogisticRegression(max_iter=500, solver="lbfgs")
        logit.fit(X, treated)
        ps = logit.predict_proba(X)[:, 1]
        t_ps = ps[treated == 1]
        c_ps = ps[treated == 0]
        overlap = (
            (min(t_ps.max(), c_ps.max()) - max(t_ps.min(), c_ps.min()))
            / (max(t_ps.max(), c_ps.max()) - min(t_ps.min(), c_ps.min()))
        ) if (max(t_ps.max(), c_ps.max()) - min(t_ps.min(), c_ps.min())) > 0 else 0.0
        return float(overlap)
    except Exception:
        return np.nan


def _plot_residual_balance(D_resid: np.ndarray, factor: str, method: str):
    """Plot and save the distribution of residualized treatment (E[D|X] residuals)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(D_resid, bins=50, edgecolor="k", alpha=0.7)
    ax.axvline(0, color="red", linestyle="--", linewidth=1)
    ax.set_title(f"{method}: 残差化 {factor} 分布 (E[D|X] residuals)")
    ax.set_xlabel("Residual")
    ax.set_ylabel("Frequency")
    save_figure(fig, f"residual_balance_{method}_{factor}")


def _plot_propensity_overlap(D: np.ndarray, X: np.ndarray, factor: str, method: str):
    """Plot propensity score overlap (treated vs control) for binary treatment approximation."""
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LogisticRegression

    try:
        median_d = np.median(D)
        treated = (D > median_d).astype(int)
        logit = LogisticRegression(max_iter=500, solver="lbfgs")
        logit.fit(X, treated)
        ps = logit.predict_proba(X)[:, 1]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(ps[treated == 1], bins=30, alpha=0.6, label="Treated (high factor)")
        ax.hist(ps[treated == 0], bins=30, alpha=0.6, label="Control (low factor)")
        ax.set_title(f"{method}: {factor} 倾向得分共同支撑域")
        ax.set_xlabel("Propensity score")
        ax.set_ylabel("Frequency")
        ax.legend()
        save_figure(fig, f"propensity_overlap_{method}_{factor}")
    except Exception as e:
        logger.warning(f"Propensity plot failed for {factor}: {e}")


def _dml_residuals(D: np.ndarray, X: np.ndarray, n_splits: int = 5) -> np.ndarray:
    """Cross-fit residualized treatment."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    D_resid = np.zeros_like(D)
    for train_idx, test_idx in kf.split(X):
        d_mod = _get_nuisance_model()
        d_mod.fit(X[train_idx], D[train_idx])
        D_resid[test_idx] = D[test_idx] - d_mod.predict(X[test_idx])
    return D_resid


def double_ml_effect(df: pd.DataFrame, factor: str) -> Dict:
    """Estimate ATE using Double Machine Learning with diagnostic checks."""
    sub = df[[factor, "label"] + [c for c in df.columns if c not in [factor, "label"]]].copy()
    sub = sub.dropna()
    sub = _subsample(sub)
    if len(sub) < 50:
        return _empty_result(factor, "DML")

    D = sub[factor].values
    Y = sub["label"].values
    X = _get_controls(sub, factor)
    X_arr = sm.add_constant(X).values
    X_no_const = X.values

    result = {"factor": factor, "method": "DML"}

    # Try EconML LinearDML first (provides inference for linear final model)
    if ECONML_AVAILABLE:
        try:
            dml_kwargs = dict(
                model_y=_get_nuisance_model(),
                model_t=_get_nuisance_model(),
                cv=DML_N_SPLITS,
                random_state=42,
            )
            # EconML >= 0.14 expects an explicit linear final model.
            if hasattr(EconLinearDML, "__module__") and tuple(
                map(int, econml.__version__.split(".")[:2])
            ) >= (0, 14):
                dml_kwargs["model_final"] = sm.OLS
            est = EconLinearDML(**dml_kwargs)
            est.fit(Y, D, X=X_no_const)
            if hasattr(est, "effect"):
                ate = est.effect(X=X_no_const)
                ci = est.effect_interval(X=X_no_const, alpha=0.05)
            else:
                ate = est.ate(X=X_no_const)
                ci = est.ate_interval(X=X_no_const, alpha=0.05)
            ate = float(ate) if not isinstance(ate, (np.floating, float)) else float(ate)
            se = float((ci[1] - ci[0]) / 3.92) if ci is not None else np.nan
            tstat = ate / se if se > 0 else np.nan
            pvalue = 2 * (1 - stats.t.cdf(abs(tstat), df=len(sub) - 1)) if not np.isnan(tstat) else np.nan
            result.update({
                "ate": ate,
                "se": se,
                "tvalue": tstat,
                "pvalue": pvalue,
                "ci_lower": float(ci[0]) if ci is not None else np.nan,
                "ci_upper": float(ci[1]) if ci is not None else np.nan,
                "n": len(sub),
                "econml_used": True,
            })
            D_resid = _dml_residuals(D, X_arr, n_splits=DML_N_SPLITS)
            _plot_residual_balance(D_resid, factor, "DML")
            overlap = _propensity_overlap(D, X_no_const)
            _plot_propensity_overlap(D, X_no_const, factor, "DML")
            result.update(_dml_diagnostics(D, D_resid, overlap))
            return result
        except Exception as e:
            logger.warning(f"  EconML DML failed for {factor}, falling back to manual DML: {e}")

    # Manual DML fallback
    kf = KFold(n_splits=DML_N_SPLITS, shuffle=True, random_state=42)
    Y_resid = np.zeros_like(Y)
    D_resid = np.zeros_like(D)

    for train_idx, test_idx in kf.split(X_arr):
        y_mod = _get_nuisance_model()
        d_mod = _get_nuisance_model()
        y_mod.fit(X_arr[train_idx], Y[train_idx])
        d_mod.fit(X_arr[train_idx], D[train_idx])
        Y_resid[test_idx] = Y[test_idx] - y_mod.predict(X_arr[test_idx])
        D_resid[test_idx] = D[test_idx] - d_mod.predict(X_arr[test_idx])

    # Winsorize predictions to avoid extreme residuals
    D_resid = np.clip(D_resid, np.quantile(D_resid, 0.001), np.quantile(D_resid, 0.999))

    X2 = sm.add_constant(D_resid)
    model = sm.OLS(Y_resid, X2).fit()
    ate = model.params[1]
    se = model.bse[1]
    pvalue = model.pvalues[1]
    ci = model.conf_int()[1].tolist()

    X_ols = sm.add_constant(np.column_stack([D, X_arr[:, 1:]]))
    ols = sm.OLS(Y, X_ols).fit()
    ols_beta = ols.params[1]
    ols_se = ols.bse[1]
    ols_p = ols.pvalues[1]

    _plot_residual_balance(D_resid, factor, "DML")
    overlap = _propensity_overlap(D, X_no_const)
    _plot_propensity_overlap(D, X_no_const, factor, "DML")

    result.update({
        "ate": float(ate),
        "se": float(se),
        "tvalue": float(ate / se) if se > 0 else np.nan,
        "pvalue": float(pvalue),
        "ci_lower": float(ci[0]),
        "ci_upper": float(ci[1]),
        "ols_beta": float(ols_beta),
        "ols_se": float(ols_se),
        "ols_pvalue": float(ols_p),
        "n": len(sub),
        "resid_D_mean": float(D_resid.mean()),
        "resid_D_std": float(D_resid.std()),
        "econml_used": False,
    })
    result.update(_dml_diagnostics(D, D_resid, overlap))
    return result


def _dml_diagnostics(D: np.ndarray, D_resid: np.ndarray, overlap: float) -> Dict:
    """Return DML diagnostic flags and metrics."""
    resid_mean = float(D_resid.mean())
    resid_std = float(D_resid.std())
    # For a standardized treatment, residual std should be in a reasonable range.
    balanced = (abs(resid_mean) < 0.1) and (0.3 < resid_std < 1.5)
    overlap_ok = (not np.isnan(overlap)) and overlap > 0.3
    return {
        "resid_D_mean": resid_mean,
        "resid_D_std": resid_std,
        "propensity_overlap": float(overlap),
        "dml_balanced": bool(balanced),
        "dml_overlap_ok": bool(overlap_ok),
        "dml_valid": bool(balanced and overlap_ok),
    }


def _board(code: str) -> str:
    """Infer board/exchange from ts_code."""
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
    return "other"


def _build_iv_instruments(df: pd.DataFrame, factor: str) -> pd.DataFrame:
    """Build two instruments: lagged factor and industry/board-average factor."""
    df = df.sort_values(["ts_code", "trade_date"]).copy()
    df["board"] = df["ts_code"].apply(_board)
    df[f"{factor}_lag1"] = df.groupby("ts_code")[factor].shift(1)

    # Industry average excluding self.
    def _exclude_self_avg(x):
        total = x.sum() - x.values
        return total / (len(x) - 1)

    df[f"{factor}_ind_avg"] = df.groupby(["trade_date", "board"])[factor].transform(_exclude_self_avg)
    return df


def iv_effect(df: pd.DataFrame, factor: str) -> Dict:
    """Estimate ATE using IVs with diagnostic validity checks."""
    sub = df[[factor, "label"] + [c for c in df.columns if c not in [factor, "label"]]].copy()
    sub = _build_iv_instruments(sub, factor)
    sub = _subsample(sub)

    control_cols = [c for c in _get_controls(sub, factor).columns]
    sub = sub[[factor, "label", f"{factor}_lag1", f"{factor}_ind_avg"] + control_cols].dropna()
    if len(sub) < 50:
        return _empty_result(factor, "IV")

    D = sub[factor].values
    Y = sub["label"].values
    Z = sub[[f"{factor}_lag1", f"{factor}_ind_avg"]].values
    X = sub[control_cols].values
    X = sm.add_constant(X)

    # First stage
    X_first = sm.add_constant(np.column_stack([Z, X[:, 1:]]))
    first_stage = sm.OLS(D, X_first).fit()
    D_hat = first_stage.predict(X_first)
    f_test = first_stage.f_test("x1 = x2 = 0")
    f_stat = float(f_test.fvalue[0, 0]) if hasattr(f_test.fvalue, "shape") else float(f_test.fvalue)
    f_pvalue = float(f_test.pvalue)

    # Second stage
    X_second = sm.add_constant(np.column_stack([D_hat, X[:, 1:]]))
    second_stage = sm.OLS(Y, X_second).fit()
    ate = second_stage.params[1]
    se = second_stage.bse[1]
    pvalue = second_stage.pvalues[1]
    ci = second_stage.conf_int()[1].tolist()

    # Sargan overidentification test
    residuals = second_stage.resid
    X_sargan = sm.add_constant(np.column_stack([Z, X[:, 1:]]))
    sargan_mod = sm.OLS(residuals, X_sargan).fit()
    r2 = sargan_mod.rsquared
    sargan_stat = len(residuals) * r2
    sargan_p = 1 - stats.chi2.cdf(sargan_stat, df=Z.shape[1] - 1)

    # Validity: strong instruments (Staiger-Stock rule F > 10), an upper bound to
    # avoid instruments that are nearly the treatment itself, a relaxed Sargan
    # threshold (0.10) because the overidentification test has low power with only
    # two instruments, and a sanity bound on the estimated ATE magnitude.
    strong_iv = f_stat > 10
    not_treatment_itself = f_stat < 1000
    overid_ok = sargan_p > 0.10
    reasonable_magnitude = abs(ate) < 1.0
    iv_valid = strong_iv and not_treatment_itself and overid_ok and reasonable_magnitude

    return {
        "factor": factor,
        "method": "IV",
        "ate": float(ate),
        "se": float(se),
        "tvalue": float(ate / se) if se > 0 else np.nan,
        "pvalue": float(pvalue),
        "ci_lower": float(ci[0]),
        "ci_upper": float(ci[1]),
        "first_stage_f": float(f_stat),
        "first_stage_f_pvalue": float(f_pvalue),
        "sargan_stat": float(sargan_stat),
        "sargan_pvalue": float(sargan_p),
        "iv_valid": bool(iv_valid),
        "n": len(sub),
    }


def front_door_effect(df: pd.DataFrame, factor: str) -> Dict:
    """Front-door adjustment using 5-day future return as mediator."""
    if "future_ret_5" not in df.columns:
        return _empty_result(factor, "FrontDoor")
    sub = df[[factor, "label", "future_ret_5"] + [c for c in df.columns if c not in [factor, "label", "future_ret_5"]]].copy()
    sub = sub.dropna()
    sub = _subsample(sub)
    control_cols = [c for c in _get_controls(sub, factor).columns]
    sub = sub[[factor, "label", "future_ret_5"] + control_cols]
    if len(sub) < 50:
        return _empty_result(factor, "FrontDoor")

    D = sub[factor].values
    M = sub["future_ret_5"].values
    Y = sub["label"].values
    X = sub[control_cols].values
    X = sm.add_constant(X)

    X1 = sm.add_constant(np.column_stack([D, X[:, 1:]]))
    step1 = sm.OLS(M, X1).fit()
    beta1 = step1.params[1]
    se1 = step1.bse[1]
    pvalue1 = step1.pvalues[1]

    X2 = sm.add_constant(np.column_stack([M, D, X[:, 1:]]))
    step2 = sm.OLS(Y, X2).fit()
    beta2 = step2.params[1]
    se2 = step2.bse[1]
    pvalue2 = step2.pvalues[1]

    ate = beta1 * beta2
    var = (beta2 ** 2) * (se1 ** 2) + (beta1 ** 2) * (se2 ** 2)
    se = np.sqrt(var)
    tvalue = ate / se if se > 0 else np.nan
    pvalue = 2 * (1 - stats.t.cdf(abs(tvalue), df=len(sub) - 2)) if not np.isnan(tvalue) else np.nan
    ci = [ate - 1.96 * se, ate + 1.96 * se]

    # Front-door identification requires both paths to be strong and the mediator
    # to block the direct effect of D on Y.  We tighten the path p-value threshold
    # to 0.01, require the mediator not to be too collinear with the outcome (0.5),
    # and check that the direct effect of D on Y controlling for M is insignificant.
    identification_pass = (pvalue1 < 0.01) and (pvalue2 < 0.01)
    # If the mediator is highly correlated with the outcome, the front-door
    # criterion is violated regardless of statistical significance.
    m_y_corr = np.corrcoef(M, Y)[0, 1] if np.std(M) > 0 and np.std(Y) > 0 else np.nan
    mediator_collinear = (not np.isnan(m_y_corr)) and abs(m_y_corr) > 0.5
    # Direct effect of D on Y should be mediated through M.
    direct_pvalue = step2.pvalues[2]
    direct_effect_small = direct_pvalue > 0.05
    fd_valid = identification_pass and (not mediator_collinear) and direct_effect_small and (not np.isnan(ate)) and (np.isfinite(ate))

    return {
        "factor": factor,
        "method": "FrontDoor",
        "ate": float(ate),
        "se": float(se),
        "tvalue": float(tvalue),
        "pvalue": float(pvalue),
        "ci_lower": float(ci[0]),
        "ci_upper": float(ci[1]),
        "beta1": float(beta1),
        "beta2": float(beta2),
        "beta1_pvalue": float(pvalue1),
        "beta2_pvalue": float(pvalue2),
        "identification_pass": bool(identification_pass),
        "fd_valid": bool(fd_valid),
        "n": len(sub),
    }


def causal_forest_effect(df: pd.DataFrame, factor: str) -> Dict:
    """Estimate heterogeneous ITE using Causal Forest with winsorization."""
    sub = df[[factor, "label"] + [c for c in df.columns if c not in [factor, "label"]]].copy()
    sub = sub.dropna()
    sub = _subsample(sub)
    if len(sub) < 50:
        return _empty_result(factor, "CausalForest")

    D = sub[factor].values
    Y = sub["label"].values
    X = _get_controls(sub, factor)
    X_df = X.copy()
    X_arr = X.values

    result = {"factor": factor, "method": "CausalForest"}

    if ECONML_AVAILABLE:
        try:
            cf = EconCausalForest(
                n_estimators=CAUSAL_FOREST_N_ESTIMATORS,
                max_depth=6,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=4,
            )
            cf.fit(X_arr, D, Y)
            # CausalForest API differs across EconML versions.
            if hasattr(cf, "effect"):
                ate_arr = cf.effect(X=X_arr)
                ci_arr = cf.effect_interval(X=X_arr, alpha=0.05)
            elif hasattr(cf, "ate"):
                ate_arr = cf.ate(X=X_arr)
                ci_arr = cf.ate_interval(X=X_arr, alpha=0.05)
            else:
                ate_arr = cf.predict(X=X_arr)
                ci_arr = cf.predict_interval(X=X_arr, alpha=0.05)
            ate = float(ate_arr.mean())
            se = float((ci_arr[1].mean() - ci_arr[0].mean()) / 3.92) if ci_arr is not None else np.nan
            tvalue = ate / se if se > 0 else np.nan
            pvalue = 2 * (1 - stats.t.cdf(abs(tvalue), df=len(sub) - 1)) if not np.isnan(tvalue) else np.nan
            ite = ate_arr
            ite_w = _winsorize(ite)
            result.update({
                "ate": ate,
                "se": se,
                "tvalue": tvalue,
                "pvalue": pvalue,
                "ci_lower": float(ci_arr[0].mean()) if ci_arr is not None else np.nan,
                "ci_upper": float(ci_arr[1].mean()) if ci_arr is not None else np.nan,
                "ite_mean": float(ite_w.mean()),
                "ite_std": float(ite_w.std()),
                "ite_q10": float(np.quantile(ite_w, 0.1)),
                "ite_q90": float(np.quantile(ite_w, 0.9)),
                "n": len(sub),
                "econml_used": True,
            })
            _plot_ite_distribution(ite_w, factor, X_df)
            result.update(_cf_diagnostics(ite_w, X_df, factor))
            return result
        except Exception as e:
            logger.warning(f"  EconML CausalForest failed for {factor}, falling back to manual: {e}")

    # Manual residual-forest fallback.  We keep the ratio ITE for visualisation
    # and heterogeneity analysis, but we estimate the ATE and its SE with a robust
    # residual-on-residual regression.  The ratio mean is unstable when the
    # predicted treatment residual is close to zero, which inflates Type-I error in
    # the placebo test.  The OLS coefficient avoids this division-by-small-number
    # problem while still delivering a valid ATE and ITE distribution.
    kf = KFold(n_splits=DML_N_SPLITS, shuffle=True, random_state=42)
    Y_resid = np.zeros_like(Y)
    D_resid = np.zeros_like(D)

    for train_idx, test_idx in kf.split(X_arr):
        y_mod = _get_nuisance_model()
        d_mod = _get_nuisance_model()
        y_mod.fit(X_arr[train_idx], Y[train_idx])
        d_mod.fit(X_arr[train_idx], D[train_idx])
        Y_resid[test_idx] = Y[test_idx] - y_mod.predict(X_arr[test_idx])
        D_resid[test_idx] = D[test_idx] - d_mod.predict(X_arr[test_idx])

    rf_y = RandomForestRegressor(n_estimators=50, max_depth=6, random_state=42, n_jobs=4)
    rf_d = RandomForestRegressor(n_estimators=50, max_depth=6, random_state=42, n_jobs=4)
    rf_y.fit(X_arr, Y_resid)
    rf_d.fit(X_arr, D_resid)

    y_pred = rf_y.predict(X_arr)
    d_pred = rf_d.predict(X_arr)
    ite = np.where(np.abs(d_pred) > 1e-8, y_pred / d_pred, 0.0)
    ite_w = _winsorize(ite)

    # Robust ATE: residual-on-residual regression (no intercept because residuals
    # are already centered).
    X_ols = sm.add_constant(D_resid) if np.abs(D_resid.mean()) > 1e-8 else D_resid[:, None]
    ols = sm.OLS(Y_resid, X_ols).fit()
    ate = float(ols.params[-1])
    ate_se = float(ols.bse[-1])
    tvalue = float(ols.tvalues[-1])
    pvalue = float(ols.pvalues[-1])
    ci = [ate - 1.96 * ate_se, ate + 1.96 * ate_se]

    # Causal-forest validity: require a stable predicted treatment residual surface
    # and a reasonable ATE magnitude.  This prevents unstable ratio estimates from
    # driving false positives in placebo tests.
    d_pred_stable = (
        np.isfinite(d_pred).all()
        and np.std(d_pred) > 0.05
        and np.mean(np.abs(d_pred) > 0.01) > 0.5
    )
    ate_reasonable = abs(ate) < 0.5
    cf_valid = (
        np.isfinite(ite_w).all()
        and (ite_w.std() > 0)
        and (len(ite_w) > 0)
        and d_pred_stable
        and ate_reasonable
    )

    result.update({
        "ate": ate,
        "se": ate_se,
        "tvalue": tvalue,
        "pvalue": pvalue,
        "ci_lower": ci[0],
        "ci_upper": ci[1],
        "ite_mean": float(ite_w.mean()),
        "ite_std": float(ite_w.std()),
        "ite_q10": float(np.quantile(ite_w, 0.1)),
        "ite_q90": float(np.quantile(ite_w, 0.9)),
        "n": len(sub),
        "econml_used": False,
    })
    _plot_ite_distribution(ite_w, factor, X_df)
    result.update(_cf_diagnostics(ite_w, X_df, factor, cf_valid_override=cf_valid))
    return result


def _winsorize(arr: np.ndarray, lower: float = 0.01, upper: float = 0.99) -> np.ndarray:
    """Winsorize an array at given percentiles."""
    lo = np.quantile(arr, lower)
    hi = np.quantile(arr, upper)
    return np.clip(arr, lo, hi)


def _cf_diagnostics(ite: np.ndarray, X_df: pd.DataFrame, factor: str, cf_valid_override: bool = None) -> Dict:
    """Compute Causal Forest diagnostic metrics and heterogeneity.

    The caller can pass a pre-computed validity decision based on the fitted
    residual surface and ATE magnitude; otherwise we apply the minimal checks.
    """
    if cf_valid_override is None:
        cf_valid = np.isfinite(ite).all() and (ite.std() > 0) and (len(ite) > 0)
    else:
        cf_valid = bool(cf_valid_override)
    hetero_records = []
    for col in ["VOL20", "TURN20", "LIQ20"]:
        if col in X_df.columns:
            x_col = X_df[col].values
            median_x = np.median(x_col)
            high = x_col > median_x
            low = ~high
            hetero_records.append({
                "factor": factor,
                "characteristic": col,
                "low_group_ite_mean": float(ite[low].mean()),
                "high_group_ite_mean": float(ite[high].mean()),
                "diff": float(ite[high].mean() - ite[low].mean()),
            })
    if hetero_records:
        hetero_df = pd.DataFrame(hetero_records)
        path = RESULTS_DIR / f"ite_heterogeneity_{factor}.csv"
        hetero_df.to_csv(path, index=False, encoding="utf-8")
    return {
        "cf_valid": bool(cf_valid),
        "ite_q01": float(np.quantile(ite, 0.01)),
        "ite_q99": float(np.quantile(ite, 0.99)),
    }


def _plot_ite_distribution(ite: np.ndarray, factor: str, X_df: pd.DataFrame):
    """Plot ITE distribution and heterogeneity."""
    import matplotlib.pyplot as plt

    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(ite, bins=50, edgecolor="k", alpha=0.7)
        ax.set_title(f"ITE 分布: {factor}")
        ax.set_xlabel("个体处理效应")
        ax.set_ylabel("频数")
        save_figure(fig, f"ite_distribution_{factor}")
    except Exception as e:
        logger.warning(f"ITE distribution plot failed for {factor}: {e}")


def _empty_result(factor: str, method: str) -> Dict:
    base = {
        "factor": factor,
        "method": method,
        "ate": np.nan,
        "se": np.nan,
        "tvalue": np.nan,
        "pvalue": np.nan,
        "ci_lower": np.nan,
        "ci_upper": np.nan,
        "n": 0,
    }
    if method == "DML":
        base.update({"dml_valid": False, "dml_balanced": False, "dml_overlap_ok": False})
    elif method == "IV":
        base.update({"iv_valid": False})
    elif method == "FrontDoor":
        base.update({"fd_valid": False, "identification_pass": False})
    elif method == "CausalForest":
        base.update({"cf_valid": False})
    return base


def estimate_all_effects(df: pd.DataFrame, output_name: str = "causal_effects_train") -> pd.DataFrame:
    """Run all causal effect methods for all factors and save train-only results."""
    logger.info("Starting causal effect estimation for all factors...")
    records = []
    for factor in FACTORS:
        logger.info(f"Estimating causal effects for {factor}...")
        try:
            records.append(double_ml_effect(df, factor))
        except Exception as e:
            logger.warning(f"DML failed for {factor}: {e}")
            records.append(_empty_result(factor, "DML"))
        try:
            records.append(iv_effect(df, factor))
        except Exception as e:
            logger.warning(f"IV failed for {factor}: {e}")
            records.append(_empty_result(factor, "IV"))
        try:
            records.append(front_door_effect(df, factor))
        except Exception as e:
            logger.warning(f"Front-door failed for {factor}: {e}")
            records.append(_empty_result(factor, "FrontDoor"))
        try:
            records.append(causal_forest_effect(df, factor))
        except Exception as e:
            logger.warning(f"Causal Forest failed for {factor}: {e}")
            records.append(_empty_result(factor, "CausalForest"))

    results = pd.DataFrame(records)
    save_table(results, output_name)
    _save_method_consistency(results)
    logger.info(f"Causal effect estimation completed; {len(results)} records saved")
    return results


def _save_method_consistency(results: pd.DataFrame):
    """Analyze consistency across the four causal effect methods, using diagnostics."""
    methods = ["DML", "IV", "FrontDoor", "CausalForest"]
    pivot = results.pivot(index="factor", columns="method", values="ate").reindex(columns=methods)
    sig_pivot = results.pivot(index="factor", columns="method", values="pvalue").reindex(columns=methods)
    valid_map = {
        "DML": "dml_valid",
        "IV": "iv_valid",
        "FrontDoor": "fd_valid",
        "CausalForest": "cf_valid",
    }

    records = []
    for factor in pivot.index:
        ates = pivot.loc[factor]
        pvals = sig_pivot.loc[factor]
        valid_flags = {}
        for m in methods:
            flag_col = valid_map.get(m)
            if flag_col and flag_col in results.columns:
                valid_flags[m] = bool(
                    results[(results["factor"] == factor) & (results["method"] == m)][flag_col].values[0]
                ) if len(results[(results["factor"] == factor) & (results["method"] == m)]) else False
            else:
                valid_flags[m] = True

        sig_count = sum((pvals[m] < 0.05) and valid_flags.get(m, False) for m in methods)
        sig_ates = [ates[m] for m in methods if (pvals[m] < 0.05) and valid_flags.get(m, False)]
        sign_consistent = False
        if len(sig_ates) > 1:
            signs = [np.sign(a) for a in sig_ates]
            sign_consistent = all(s == signs[0] for s in signs)
        mean_ate = np.nanmean(ates)
        std_ate = np.nanstd(ates)
        cv = std_ate / abs(mean_ate) if mean_ate != 0 else np.nan

        label = "一致" if sign_consistent and sig_count >= 2 else ("部分一致" if sig_count >= 1 else "不一致")
        records.append({
            "factor": factor,
            "n_methods_significant": int(sig_count),
            "sign_consistent": bool(sign_consistent),
            "mean_ate": float(mean_ate),
            "std_ate": float(std_ate),
            "cv": float(cv),
            "consistency_label": label,
        })
    consistency_df = pd.DataFrame(records).sort_values("n_methods_significant", ascending=False)
    save_table(consistency_df, "causal_method_consistency")
    save_json(consistency_df.to_dict(orient="records"), "causal_method_consistency")


def plot_method_comparison(results: pd.DataFrame, filename: str = "causal_effect_comparison"):
    """Plot ATE comparison across methods for each factor, masking invalid methods."""
    import matplotlib.pyplot as plt

    methods = ["DML", "IV", "FrontDoor", "CausalForest"]
    valid_map = {"DML": "dml_valid", "IV": "iv_valid", "FrontDoor": "fd_valid", "CausalForest": "cf_valid"}

    pivot = results.pivot(index="factor", columns="method", values="ate").reindex(columns=methods)
    # Mask invalid methods by setting to NaN so they don't appear in the bar chart.
    for m in methods:
        flag_col = valid_map.get(m)
        if flag_col and flag_col in results.columns:
            valid = results.pivot(index="factor", columns="method", values=flag_col)[m].fillna(False).astype(bool)
            pivot.loc[~valid, m] = np.nan

    fig, ax = plt.subplots(figsize=(14, 10))
    pivot.plot(kind="bar", ax=ax)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("因果效应估计四维方法对比（ATE，无效方法已屏蔽）")
    ax.set_xlabel("因子")
    ax.set_ylabel("ATE")
    ax.legend(title="方法")
    ax.tick_params(axis="x", rotation=45)
    save_figure(fig, filename)


def plot_ite_distribution(df: pd.DataFrame, results: pd.DataFrame, top_factors: List[str] = None):
    """Ensure ITE distributions are generated for selected factors."""
    if top_factors is None:
        dml = results[(results["method"] == "DML") & (results.get("dml_valid", False))].copy()
        dml = dml.sort_values("pvalue").head(3)
        top_factors = dml["factor"].tolist() if not dml.empty else ["EP", "BP", "MOM20"]

    for factor in top_factors:
        try:
            causal_forest_effect(df, factor)
        except Exception as e:
            logger.warning(f"ITE plot failed for {factor}: {e}")


if __name__ == "__main__":
    df = load_features()
    res = estimate_all_effects(df)
    plot_method_comparison(res)
    plot_ite_distribution(df, res)
    print("Causal effect estimation completed.")
