"""Identification diagnostics: back-door criterion, do-calculus, and assumption checks."""
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import LogisticRegression

from causal_factor_research.config import CAUSAL_DISCOVERY_FACTORS, FACTORS, RESULTS_DIR
from causal_factor_research.utils import load_features, save_figure, save_json, save_table, split_train_test



def back_door_adjustment_formula(factor: str, controls: List[str]) -> str:
    """Return the back-door adjustment formula for a factor-outcome pair."""
    controls_str = ", ".join(controls) if controls else "Z"
    return f"P(label | do({factor})) = Σ_{{{controls_str}}} P(label | {factor}, {controls_str}) P({controls_str})"


def backdoor_diagnostics(df: pd.DataFrame, factor: str) -> Dict:
    """Compute back-door diagnostic: propensity score overlap and adjustment formula."""
    sub = df[[factor, "label"] + [c for c in df.columns if c not in [factor, "label"]]].dropna()
    if len(sub) < 50:
        return {"factor": factor, "error": "too few observations"}

    controls = [c for c in sub.columns if c not in [factor, "label", "ts_code", "trade_date", "future_ret_5", "close", "open"]]
    X = sub[controls].values
    D = sub[factor].values
    Y = sub["label"].values

    # Propensity score overlap (binary treatment by median split)
    median_d = np.median(D)
    treated = (D > median_d).astype(int)
    try:
        ps = LogisticRegression(max_iter=500, solver="lbfgs").fit(X, treated).predict_proba(X)[:, 1]
    except Exception:
        ps = None

    overlap_ratio = None
    if ps is not None:
        t_ps = ps[treated == 1]
        c_ps = ps[treated == 0]
        overlap_ratio = float(
            (min(t_ps.max(), c_ps.max()) - max(t_ps.min(), c_ps.min()))
            / (max(t_ps.max(), c_ps.max()) - min(t_ps.min(), c_ps.min()))
        ) if (max(t_ps.max(), c_ps.max()) - min(t_ps.min(), c_ps.min())) > 0 else 0.0

    return {
        "factor": factor,
        "backdoor_formula": back_door_adjustment_formula(factor, controls),
        "n_controls": len(controls),
        "controls": controls,
        "treated_n": int((treated == 1).sum()),
        "control_n": int((treated == 0).sum()),
        "propensity_overlap_ratio": overlap_ratio,
    }


def sutva_diagnostic(df: pd.DataFrame, factor: str) -> Dict:
    """Assess SUTVA by checking whether top-factor stocks cluster in same industry/board."""
    ind_cols = [c for c in df.columns if c.startswith("ind_")]
    if not ind_cols:
        return {"factor": factor, "note": "no industry/board dummies available"}

    df = df.copy()
    df["high_factor"] = df.groupby("trade_date")[factor].transform(lambda x: x >= x.quantile(0.7))
    sub = df[df["high_factor"]].copy()
    ind_shares = {col: float(sub[col].mean()) for col in ind_cols}
    max_share = max(ind_shares.values())
    dominant_ind = max(ind_shares, key=ind_shares.get)
    return {
        "factor": factor,
        "dominant_board": dominant_ind,
        "dominant_share": max_share,
        "sutva_concern": "高" if max_share > 0.5 else "中" if max_share > 0.35 else "低",
        "board_shares": ind_shares,
    }


def draw_factor_dag(df: pd.DataFrame, factor: str, filename: str):
    """Draw a small causal DAG for the factor-outcome relationship with controls."""
    try:
        controls = [c for c in df.columns if c not in [factor, "label", "ts_code", "trade_date", "future_ret_5", "close", "open"]]
        # Limit to a few key controls for readability.
        key_controls = [c for c in controls if c in ["MOM20", "MOM60", "VOL20", "TURN20", "LIQ20", "EP", "BP", "GROWTH", "mkt_ret_20", "mkt_trend"] or c.startswith("ind_")]
        key_controls = key_controls[:8]

        G = nx.DiGraph()
        G.add_node(factor, color="lightgreen")
        G.add_node("label", color="lightcoral")
        for c in key_controls:
            G.add_node(c, color="lightblue")
            G.add_edge(c, factor)
            G.add_edge(c, "label")
        G.add_edge(factor, "label")

        pos = nx.spring_layout(G, seed=42)
        fig, ax = plt.subplots(figsize=(10, 8))
        node_color = [G.nodes[n].get("color", "white") for n in G.nodes]
        nx.draw_networkx_nodes(G, pos, node_color=node_color, node_size=1500, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=9, ax=ax)
        nx.draw_networkx_edges(G, pos, arrowstyle="->", arrowsize=15, ax=ax, node_size=1500)
        ax.set_title(f"因果图: {factor} → label")
        ax.axis("off")
        save_figure(fig, filename)
    except Exception as e:
        logger.warning(f"Failed to draw DAG for {factor}: {e}")


def run_all_identification(df: pd.DataFrame = None) -> Dict:
    """Run identification diagnostics for all factors."""
    if df is None:
        df = load_features()
        df, _ = split_train_test(df)

    # Use causal-discovery factors for DAGs (ROA excluded due to collinearity).
    dag_factors = CAUSAL_DISCOVERY_FACTORS

    backdoor_records = []
    sutva_records = []
    for factor in dag_factors:
        backdoor_records.append(backdoor_diagnostics(df, factor))
        sutva_records.append(sutva_diagnostic(df, factor))
        draw_factor_dag(df, factor, f"dag_{factor}")

    backdoor_df = pd.DataFrame(backdoor_records)
    save_table(backdoor_df, "backdoor_diagnostics")
    save_json(backdoor_records, "backdoor_diagnostics")

    sutva_df = pd.DataFrame(sutva_records)
    save_table(sutva_df, "sutva_diagnostics")
    save_json(sutva_records, "sutva_diagnostics")

    # Identification summary for the report
    avg_overlap = backdoor_df["propensity_overlap_ratio"].dropna().mean() if "propensity_overlap_ratio" in backdoor_df.columns else None
    high_sutva = [r["factor"] for r in sutva_records if r.get("sutva_concern") == "高"]
    all_controls = []
    for r in backdoor_records:
        all_controls.extend(r.get("controls", []))
    common_controls = sorted(set(all_controls))[:8]
    identification_summary = {
        "ignorability": {
            "description": "给定行业、市值、动量、波动率等控制变量后，处理分配（因子暴露）与潜在结果（未来收益）近似独立。未观测的市场情绪、宏观冲击仍可能构成残余混杂。",
            "controls": common_controls,
            "assessment": "中等：已纳入主要观测控制变量，但不可观测混杂无法完全排除。"
        },
        "common_support": {
            "description": "处理组（高因子暴露）与对照组（低因子暴露）在控制变量上需有重叠。通过 Logistic 回归倾向得分计算重叠度。",
            "mean_overlap_ratio": float(avg_overlap) if avg_overlap is not None else None,
            "assessment": "良好" if (avg_overlap is not None and avg_overlap > 0.5) else "中等" if (avg_overlap is not None and avg_overlap > 0.3) else "不足"
        },
        "sutva": {
            "description": "稳定单位处理值假设：一只股票的高因子暴露不会显著影响其他股票的收益。金融市场中通过因子组合可能存在溢出效应。",
            "high_concern_factors": high_sutva,
            "assessment": "近似成立" if len(high_sutva) <= 1 else "需谨慎，部分因子高暴露股票行业集中度高"
        }
    }
    save_json(identification_summary, "identification_summary")

    return {"backdoor": backdoor_records, "sutva": sutva_records, "summary": identification_summary}


if __name__ == "__main__":
    run_all_identification()
