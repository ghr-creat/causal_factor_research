"""Causal discovery: PC, FCI, Bootstrap stability, and CI test comparison."""
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Disable tqdm progress bars before importing causal-learn
os.environ["TQDM_DISABLE"] = "1"

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats

from causal_factor_research.config import (
    CAUSAL_DISCOVERY_FACTORS,
    ENABLE_KCI,
    FACTORS,
    FIGURES_DIR,
    N_BOOTSTRAP,
    N_BOOTSTRAP_FCI,
    RESULTS_DIR,
    RANDOM_SEED,
    RUN_FULL_PRODUCTION,
)
from causal_factor_research.utils import (
    load_features,
    save_figure,
    save_json,
    save_table,
    split_train_test,
    suppress_warnings,
)


CAUSAL_LEARN_AVAILABLE = False
GRAPHVIZ_AVAILABLE = False


try:
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.search.ConstraintBased.FCI import fci
    from causallearn.utils.cit import fisherz, kci
    from causallearn.utils.GraphUtils import GraphUtils
    from causallearn.graph.Edge import Edge
    from causallearn.graph.Endpoint import Endpoint

    CAUSAL_LEARN_AVAILABLE = True
except Exception as e:
    print(f"[Warning] causal-learn not available ({e}); will use fallback skeleton methods.")


try:
    import graphviz

    GRAPHVIZ_AVAILABLE = True
except Exception:
    pass


MAX_N = 500  # cap sample size for PC/FCI to keep computation feasible


@contextmanager
def _suppress_stdout():
    """Redirect stdout and stderr to devnull to suppress causal-learn progress bars."""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr



def load_graph_data(df: pd.DataFrame, variables: List[str], max_n: int = MAX_N) -> Tuple[np.ndarray, List[str]]:
    """Prepare data matrix for causal discovery algorithms."""
    data = df[variables].dropna()
    if len(data) > max_n:
        data = data.sample(n=max_n, random_state=RANDOM_SEED)
    return data.values, variables


def _endpoint_symbol(endpoint_val: int) -> str:
    """Map causal-learn endpoint codes to a readable symbol."""
    # Endpoint.TAIL = -1, Endpoint.ARROW = 1, Endpoint.CIRCLE = 2/3, Endpoint.STAR = 4
    if endpoint_val == -1:
        return "-"
    elif endpoint_val == 1:
        return ">"
    elif endpoint_val in (2, 3):
        return "o"
    elif endpoint_val == 4:
        return "*"
    return "?"


def _edge_symbol(endpoint_i: int, endpoint_j: int) -> str:
    """Map causal-learn endpoint pair to a readable symbol."""
    sym_i = _endpoint_symbol(endpoint_i)
    sym_j = _endpoint_symbol(endpoint_j)
    if sym_i == "-" and sym_j == ">":
        return "->"
    if sym_i == ">" and sym_j == "-":
        return "<-"
    if sym_i == "-" and sym_j == "-":
        return "-"
    if sym_i == ">" and sym_j == ">":
        return "<->"
    if sym_i == "o" and sym_j == "o":
        return "o-o"
    if sym_i == "o" and sym_j == ">":
        return "o->"
    if sym_i == ">" and sym_j == "o":
        return "<-o"
    if sym_i == "*" or sym_j == "*":
        return "*-*"
    return f"{sym_i}-{sym_j}"


def parse_causal_learn_graph(G, var_names: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[Tuple[str, str, str]]]:
    """Convert causal-learn graph to adjacency matrix, endpoint matrix, edge list, and v-structures.

    Returns
    -------
    adj : pd.DataFrame
        Binary adjacency matrix (undirected).
    endpoints : pd.DataFrame
        Endpoint code matrix (i row, j col) for PAG semantics.
    edges : list of str
        Human-readable edge list with endpoint symbols.
    v_structures : list of (i, j, k)
        V-structures where i -> j <- k.
    """
    # Unwrap CausalGraph if needed
    if hasattr(G, "G"):
        graph = G.G
    else:
        graph = G

    n = len(var_names)
    adj = np.zeros((n, n), dtype=int)
    endpoints = np.full((n, n), -99, dtype=int)
    edges = []
    v_structures = []

    edge_list = []
    if hasattr(graph, "get_graph_edges"):
        edge_list = graph.get_graph_edges()

    for edge in edge_list:
        node1 = node2 = ep1 = ep2 = None
        try:
            node1 = edge.get_node1()
            node2 = edge.get_node2()
            ep1 = edge.get_endpoint1()
            ep2 = edge.get_endpoint2()
        except Exception:
            try:
                node1 = edge.node1
                node2 = edge.node2
                ep1 = edge.endpoint1
                ep2 = edge.endpoint2
            except Exception:
                continue
        if node1 is None or node2 is None:
            continue
        i = node1.get_name() if hasattr(node1, "get_name") else str(node1)
        j = node2.get_name() if hasattr(node2, "get_name") else str(node2)
        if isinstance(i, int):
            i = var_names[i]
        if isinstance(j, int):
            j = var_names[j]
        if isinstance(i, str) and i.startswith("X") and i[1:].isdigit():
            i = var_names[int(i[1:]) - 1]
        if isinstance(j, str) and j.startswith("X") and j[1:].isdigit():
            j = var_names[int(j[1:]) - 1]
        ep1_val = ep1.value if hasattr(ep1, "value") else int(ep1)
        ep2_val = ep2.value if hasattr(ep2, "value") else int(ep2)
        sym = _edge_symbol(ep1_val, ep2_val)
        edges.append(f"{i} {sym} {j}")
        ii, jj = var_names.index(i), var_names.index(j)
        adj[ii, jj] = 1
        adj[jj, ii] = 1
        endpoints[ii, jj] = ep1_val
        endpoints[jj, ii] = ep2_val

    # Detect v-structures: i -> j <- k (j is pointed to from both ends).
    # Build adjacency of directed edges.
    directed = set()
    for edge in edges:
        parts = edge.split(" ")
        if len(parts) != 3:
            continue
        a, sym, b = parts
        if sym == "->":
            directed.add((a, b))
        elif sym == "<-":
            directed.add((b, a))

    for j in var_names:
        preds = [i for i in var_names if (i, j) in directed]
        for idx_i, i in enumerate(preds):
            for k in preds[idx_i + 1:]:
                # Check that i and k are not adjacent (no undirected edge).
                if (i, k) not in directed and (k, i) not in directed:
                    # Also need to check if there is an undirected edge. We only have directed set.
                    # We can use adjacency matrix: if adj[i,k]==0 then no edge.
                    ii, kk = var_names.index(i), var_names.index(k)
                    if adj[ii, kk] == 0:
                        v_structures.append((i, j, k))

    endpoint_df = pd.DataFrame(endpoints, index=var_names, columns=var_names)
    return pd.DataFrame(adj, index=var_names, columns=var_names), endpoint_df, edges, v_structures


def run_pc(data: np.ndarray, var_names: List[str], alpha: float = 0.05, indep_test="fisherz", stable: bool = True):
    """Run PC algorithm and return parsed graph + edges."""
    if not CAUSAL_LEARN_AVAILABLE:
        raise RuntimeError("causal-learn is not available")
    if indep_test == "fisherz":
        itest = fisherz
    elif indep_test == "kci":
        itest = kci
    else:
        raise ValueError(f"Unsupported CI test: {indep_test}")
    with _suppress_stdout():
        G = pc(data, alpha, indep_test=itest, stable=stable, uc_rule=0, mvpc=False)
    adj, endpoints, edges, v_structures = parse_causal_learn_graph(G, var_names)
    return adj, endpoints, edges, v_structures, G


def run_fci(data: np.ndarray, var_names: List[str], alpha: float = 0.05, indep_test="fisherz"):
    """Run FCI algorithm and return parsed graph + edges."""
    if not CAUSAL_LEARN_AVAILABLE:
        raise RuntimeError("causal-learn is not available")
    if indep_test == "fisherz":
        itest = fisherz
    elif indep_test == "kci":
        itest = kci
    else:
        raise ValueError(f"Unsupported CI test: {indep_test}")
    with _suppress_stdout():
        G, edges_internal = fci(data, alpha=alpha, independence_test_method=itest, verbose=False, show_progress=False)
    adj, endpoints, edges, v_structures = parse_causal_learn_graph(G, var_names)
    return adj, endpoints, edges, v_structures, G


def _edge_key(method: str, edge_str: str) -> str:
    """Normalize edge string to undirected key for stability counting."""
    parts = edge_str.split(" ")
    if len(parts) != 3:
        return edge_str
    i, sym, j = parts
    # Treat all edge types as undirected for stability counting
    return f"{method}: {min(i, j)} {sym} {max(i, j)}"


def _undirected_edge_key(method: str, edge_str: str) -> str:
    """Strictly undirected key (ignore direction)."""
    parts = edge_str.split(" ")
    if len(parts) != 3:
        return edge_str
    i, _, j = parts
    return f"{method}: {min(i, j)} -- {max(i, j)}"


def bootstrap_causal_discovery(
    df: pd.DataFrame,
    variables: List[str],
    n_boot: int = 1000,
    alpha: float = 0.05,
    methods: List[str] = ("pc",),
    indep_test: str = "fisherz",
    seed: int = RANDOM_SEED,
    sample_size: int = MAX_N,
    stability_threshold: float = 0.70,
) -> Dict:
    """Bootstrap stability analysis for causal discovery.

    Returns a dict with edge stability, edges above threshold, directed stability,
    and method-specific summaries.
    """
    rng = np.random.default_rng(seed)
    data = df[variables].dropna().values
    n = min(len(data), sample_size)
    if len(data) > sample_size:
        idx_full = rng.choice(len(data), size=sample_size, replace=False)
        data = data[idx_full]

    edge_counts = {}
    directed_counts = {}
    undirected_counts = {}
    v_structure_counts = {}

    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sample = data[idx]
        try:
            for method in methods:
                if method == "pc":
                    adj, endpoints, edges, v_structures, _ = run_pc(
                        sample, variables, alpha, indep_test
                    )
                elif method == "fci":
                    adj, endpoints, edges, v_structures, _ = run_fci(
                        sample, variables, alpha, indep_test
                    )
                else:
                    continue
                for e in edges:
                    key = _edge_key(method, e)
                    edge_counts[key] = edge_counts.get(key, 0) + 1
                    undirected_key = _undirected_edge_key(method, e)
                    undirected_counts[undirected_key] = undirected_counts.get(undirected_key, 0) + 1
                    if "->" in e and "<-" not in e:
                        directed_counts[key] = directed_counts.get(key, 0) + 1
                for v in v_structures:
                    v_key = f"{method}: {v[0]} -> {v[1]} <- {v[2]}"
                    v_structure_counts[v_key] = v_structure_counts.get(v_key, 0) + 1
        except Exception as e:
            # Some bootstrap samples may be degenerate.
            logger.debug(f"Bootstrap {b+1} failed: {e}")
            continue
        if (b + 1) % 100 == 0 or (b + 1) == n_boot:
            logger.info(f"Bootstrap {b+1}/{n_boot} completed ({methods}, {indep_test})")

    stability = {k: v / n_boot for k, v in edge_counts.items()}
    undirected_stability = {k: v / n_boot for k, v in undirected_counts.items()}
    directed_stability = {k: v / edge_counts.get(k, 1) for k, v in directed_counts.items()}
    stable_edges = {k: v for k, v in stability.items() if v >= stability_threshold}
    v_stability = {k: v / n_boot for k, v in v_structure_counts.items()}

    return {
        "stability": stability,
        "undirected_stability": undirected_stability,
        "stable_edges": stable_edges,
        "directed_stability": directed_stability,
        "v_structures": v_stability,
        "n_boot": n_boot,
        "threshold": stability_threshold,
        "variables": variables,
    }


def distance_correlation_matrix(df: pd.DataFrame, variables: List[str]) -> pd.DataFrame:
    """Compute pairwise distance correlation matrix (unconditional)."""
    n = len(variables)
    mat = np.zeros((n, n))

    def _dcentroid(X):
        n = X.shape[0]
        D = np.abs(X[:, None] - X[None, :])
        row_mean = D.mean(axis=1, keepdims=True)
        col_mean = D.mean(axis=0, keepdims=True)
        grand = D.mean()
        return D - row_mean - col_mean + grand

    for i in range(n):
        for j in range(n):
            x = df[variables[i]].dropna().values
            y = df[variables[j]].dropna().values
            if len(x) != len(y):
                min_len = min(len(x), len(y))
                x = x[:min_len]
                y = y[:min_len]
            A = _dcentroid(x)
            B = _dcentroid(y)
            denom = np.sqrt((A * A).sum() * (B * B).sum())
            if denom > 0:
                mat[i, j] = np.sqrt(np.sum(A * B) / denom)
    return pd.DataFrame(mat, index=variables, columns=variables)


def conditional_independence_distance_covariance(
    x: np.ndarray, y: np.ndarray, z: Optional[np.ndarray] = None, alpha: float = 0.05
) -> Tuple[float, bool]:
    """Distance-covariance based conditional independence test.

    Uses partial distance correlation: regress x and y on z (linearly), then compute
    distance correlation of residuals. If no z, returns unconditional distance correlation.
    """
    from sklearn.linear_model import LinearRegression

    x = np.asarray(x).reshape(-1, 1)
    y = np.asarray(y).reshape(-1, 1)
    if z is not None:
        z = np.asarray(z)
        if z.ndim == 1:
            z = z.reshape(-1, 1)
        mx = LinearRegression().fit(z, x)
        my = LinearRegression().fit(z, y)
        x_resid = x - mx.predict(z)
        y_resid = y - my.predict(z)
    else:
        x_resid = x
        y_resid = y

    dcor = distance_correlation_matrix(
        pd.DataFrame({"x": x_resid.ravel(), "y": y_resid.ravel()}), ["x", "y"]
    ).loc["x", "y"]
    n = len(x_resid)
    stat = n * dcor ** 2
    pvalue = 1 - stats.chi2.cdf(stat, df=1)
    independent = pvalue > alpha
    return pvalue, independent


def draw_graph(adj: pd.DataFrame, edges: List[str], var_names: List[str], title: str, filename: str):
    """Draw a graph using graphviz or networkx."""
    import matplotlib.pyplot as plt

    try:
        import networkx as nx

        G = nx.DiGraph()
        G.add_nodes_from(var_names)
        for edge in edges:
            parts = edge.split(" ")
            if len(parts) != 3:
                continue
            i, sym, j = parts
            if sym == "->":
                G.add_edge(i, j)
            elif sym == "<-":
                G.add_edge(j, i)
            elif sym in ("-", "o-o"):
                G.add_edge(i, j)
                G.add_edge(j, i)
            elif sym == "<->":
                G.add_edge(i, j)
                G.add_edge(j, i)
        pos = nx.shell_layout(G) if len(var_names) <= 15 else nx.spring_layout(G, seed=42, k=0.5)
        fig, ax = plt.subplots(figsize=(14, 12))
        nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=1200, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=9, ax=ax)
        nx.draw_networkx_edges(G, pos, arrowstyle="->", arrowsize=15, ax=ax, node_size=1200)
        ax.set_title(title, fontsize=14)
        ax.axis("off")
        save_figure(fig, filename)
    except Exception as e:
        logger.warning(f"Failed to draw graph {filename}: {e}")


def _draw_pag_graph(adj: pd.DataFrame, endpoints: pd.DataFrame, edges: List[str], var_names: List[str], title: str, filename: str):
    """Draw PAG graph preserving edge endpoint styles where possible."""
    import matplotlib.pyplot as plt

    try:
        import networkx as nx

        G = nx.DiGraph()
        G.add_nodes_from(var_names)
        for edge in edges:
            parts = edge.split(" ")
            if len(parts) != 3:
                continue
            i, sym, j = parts
            if "->" in sym and "<-" not in sym:
                G.add_edge(i, j)
            elif sym == "<->":
                G.add_edge(i, j)
                G.add_edge(j, i)
            elif sym in ("-", "o-o"):
                G.add_edge(i, j)
                G.add_edge(j, i)
        pos = nx.shell_layout(G) if len(var_names) <= 15 else nx.spring_layout(G, seed=42, k=0.5)
        fig, ax = plt.subplots(figsize=(14, 12))
        nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=1200, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=9, ax=ax)
        nx.draw_networkx_edges(G, pos, arrowstyle="->", arrowsize=15, ax=ax, node_size=1200)
        ax.set_title(title, fontsize=14)
        ax.axis("off")
        save_figure(fig, filename)
    except Exception as e:
        logger.warning(f"Failed to draw PAG graph {filename}: {e}")


def _adjacency_from_distance_correlation(
    df: pd.DataFrame, variables: List[str], threshold: float = 0.15
) -> pd.DataFrame:
    """Build a simple adjacency matrix from distance correlation threshold."""
    df_sub = df[variables].dropna()
    if len(df_sub) > MAX_N:
        df_sub = df_sub.sample(n=MAX_N, random_state=RANDOM_SEED)
    dcor = distance_correlation_matrix(df_sub, variables)
    adj = (dcor.abs() > threshold).astype(int)
    np.fill_diagonal(adj.values, 0)
    return adj


def _compare_ci_methods(
    df: pd.DataFrame, variables: List[str], results: Dict
) -> pd.DataFrame:
    """Compare adjacency matrices from Fisher-Z, KCI, and Distance Correlation."""
    # PC-FisherZ adjacency from results.
    adj_pc = pd.DataFrame(
        np.zeros((len(variables), len(variables))),
        index=variables,
        columns=variables,
    )
    for e in results.get("pc_fisherz_edges", []):
        parts = e.split(" ")
        if len(parts) == 3:
            i, _, j = parts
            if i in variables and j in variables:
                adj_pc.loc[i, j] = 1
                adj_pc.loc[j, i] = 1

    # KCI adjacency.
    adj_kci = pd.DataFrame(
        np.zeros((len(variables), len(variables))),
        index=variables,
        columns=variables,
    )
    for e in results.get("pc_kci_edges", []):
        parts = e.split(" ")
        if len(parts) == 3:
            i, _, j = parts
            if i in variables and j in variables:
                adj_kci.loc[i, j] = 1
                adj_kci.loc[j, i] = 1

    # Distance correlation adjacency.
    adj_dc = _adjacency_from_distance_correlation(df, variables)

    # Edge-level comparison.
    records = []
    for i, j in [(variables[ii], variables[jj]) for ii in range(len(variables)) for jj in range(ii + 1, len(variables))]:
        records.append({
            "i": i,
            "j": j,
            "fisherz": int(adj_pc.loc[i, j]),
            "kci": int(adj_kci.loc[i, j]),
            "dcorr": int(adj_dc.loc[i, j]),
            "agreement": int(adj_pc.loc[i, j]) + int(adj_kci.loc[i, j]) + int(adj_dc.loc[i, j]),
        })
    return pd.DataFrame(records)


def _plot_bootstrap_heatmap(stab_df: pd.DataFrame, var_names: List[str], filename: str, title: str):
    """Plot bootstrap stability heatmap."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    n = len(var_names)
    stab_mat = np.zeros((n, n))
    for _, row in stab_df.iterrows():
        parts = row["edge"].split(": ")
        if len(parts) != 2:
            continue
        edge_part = parts[1]
        tokens = edge_part.split(" ")
        if len(tokens) != 3:
            continue
        i, _, j = tokens
        if i in var_names and j in var_names:
            ii, jj = var_names.index(i), var_names.index(j)
            stab_mat[ii, jj] = max(stab_mat[ii, jj], row["stability"])
            stab_mat[jj, ii] = stab_mat[ii, jj]

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        stab_mat,
        xticklabels=var_names,
        yticklabels=var_names,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        ax=ax,
    )
    ax.set_title(title, fontsize=14)
    save_figure(fig, filename)


def run_all_discovery(df: Optional[pd.DataFrame] = None):
    """Run the full causal discovery workflow on training data and save results."""
    if df is None:
        df = load_features()
        df, _ = split_train_test(df)
    variables = CAUSAL_DISCOVERY_FACTORS + ["label"]
    data, var_names = load_graph_data(df, variables)
    logger.info(f"Causal discovery started on {data.shape[0]} samples x {len(var_names)} variables")

    results = {}

    # PC with Fisher-Z
    try:
        adj_pc, endpoints_pc, edges_pc, v_pc, _ = run_pc(
            data, var_names, alpha=0.05, indep_test="fisherz"
        )
        results["pc_fisherz_edges"] = edges_pc
        results["pc_fisherz_endpoints"] = endpoints_pc.to_dict()
        results["pc_fisherz_v_structures"] = [f"{v[0]} -> {v[1]} <- {v[2]}" for v in v_pc]
        draw_graph(adj_pc, edges_pc, var_names, "PC (Fisher-Z) CPDAG", "pc_dag")
    except Exception as e:
        results["pc_fisherz_error"] = str(e)
        logger.error(f"PC Fisher-Z failed: {e}")

    # PC with KCI on a small subsample (production mode).
    if ENABLE_KCI:
        try:
            data_kci, _ = load_graph_data(df, var_names, max_n=100)
            adj_kci, endpoints_kci, edges_kci, v_kci, _ = run_pc(
                data_kci, var_names, alpha=0.05, indep_test="kci"
            )
            results["pc_kci_edges"] = edges_kci
            results["pc_kci_endpoints"] = endpoints_kci.to_dict()
            results["pc_kci_v_structures"] = [f"{v[0]} -> {v[1]} <- {v[2]}" for v in v_kci]
            draw_graph(adj_kci, edges_kci, var_names, "PC (KCI) CPDAG", "pc_kci_dag")
        except Exception as e:
            results["pc_kci_error"] = str(e)
            logger.error(f"PC KCI failed: {e}")
    else:
        results["pc_kci_note"] = "KCI skipped in fast mode; enable ENABLE_KCI"
        logger.info("KCI skipped in fast mode")

    # FCI with Fisher-Z (small sample due to exponential worst-case complexity).
    try:
        data_fci, _ = load_graph_data(df, var_names, max_n=300)
        adj_fci, endpoints_fci, edges_fci, v_fci, _ = run_fci(
            data_fci, var_names, alpha=0.05, indep_test="fisherz"
        )
        results["fci_fisherz_edges"] = edges_fci
        results["fci_fisherz_endpoints"] = endpoints_fci.to_dict()
        results["fci_fisherz_v_structures"] = [f"{v[0]} -> {v[1]} <- {v[2]}" for v in v_fci]
        results["fci_fisherz_pag_legend"] = {
            "->": "directed (i causes j)",
            "o-o": "latent confounding possible",
            "o->": "possibly directed",
            "<->": "bidirected (latent confounding)",
        }
        _draw_pag_graph(adj_fci, endpoints_fci, edges_fci, var_names, "FCI (Fisher-Z) PAG", "fci_pag")
    except Exception as e:
        results["fci_fisherz_error"] = str(e)
        logger.error(f"FCI failed: {e}")

    # Distance correlation matrix (sample for speed).
    try:
        df_sub = df[var_names].dropna()
        if len(df_sub) > MAX_N:
            df_sub = df_sub.sample(n=MAX_N, random_state=RANDOM_SEED)
        dcor_mat = distance_correlation_matrix(df_sub, var_names)
        save_table(dcor_mat, "distance_correlation_matrix")
        results["distance_correlation"] = dcor_mat.to_dict()
    except Exception as e:
        results["distance_correlation_error"] = str(e)
        logger.error(f"Distance correlation failed: {e}")

    # Bootstrap stability: PC and FCI controlled by config.
    try:
        n_boot_pc = N_BOOTSTRAP
        logger.info(f"Starting PC bootstrap stability: {n_boot_pc} iterations")
        pc_stab = bootstrap_causal_discovery(
            df,
            var_names,
            n_boot=n_boot_pc,
            alpha=0.05,
            methods=("pc",),
            indep_test="fisherz",
            sample_size=500,
            stability_threshold=0.70,
        )
        stab_df = pd.DataFrame(
            [{"edge": k, "stability": v} for k, v in pc_stab["stability"].items()]
        )
        stab_df = stab_df.sort_values("stability", ascending=False)
        save_table(stab_df, "bootstrap_stability_pc")
        results["bootstrap_pc"] = {
            "n_boot": pc_stab["n_boot"],
            "threshold": pc_stab["threshold"],
            "top_edges": stab_df.head(20).to_dict(orient="records"),
            "stable_edges": list(pc_stab["stable_edges"].keys()),
            "directed_stability": pc_stab["directed_stability"],
            "v_structures": pc_stab["v_structures"],
        }
        _plot_bootstrap_heatmap(
            stab_df, var_names, "bootstrap_stability_heatmap_pc", "PC Bootstrap Edge Stability (Fisher-Z)"
        )

        if N_BOOTSTRAP_FCI > 0:
            logger.info(f"Starting FCI bootstrap stability: {N_BOOTSTRAP_FCI} iterations")
            fci_stab = bootstrap_causal_discovery(
                df,
                var_names,
                n_boot=N_BOOTSTRAP_FCI,
                alpha=0.05,
                methods=("fci",),
                indep_test="fisherz",
                sample_size=300,
                stability_threshold=0.70,
            )
            fci_stab_df = pd.DataFrame(
                [{"edge": k, "stability": v} for k, v in fci_stab["stability"].items()]
            )
            fci_stab_df = fci_stab_df.sort_values("stability", ascending=False)
            save_table(fci_stab_df, "bootstrap_stability_fci")
            results["bootstrap_fci"] = {
                "n_boot": fci_stab["n_boot"],
                "threshold": fci_stab["threshold"],
                "top_edges": fci_stab_df.head(20).to_dict(orient="records"),
                "stable_edges": list(fci_stab["stable_edges"].keys()),
                "v_structures": fci_stab["v_structures"],
            }
            _plot_bootstrap_heatmap(
                fci_stab_df, var_names, "bootstrap_stability_heatmap_fci", "FCI Bootstrap Edge Stability (Fisher-Z)"
            )

        # Direction certainty: for each edge, report directed stability ratio.
        dir_records = []
        for edge, stab in pc_stab["stability"].items():
            d_stab = pc_stab["directed_stability"].get(edge, 0.0)
            dir_records.append({
                "edge": edge,
                "stability": stab,
                "directed_stability": d_stab,
                "direction_certain": bool(d_stab >= 0.7),
            })
        dir_df = pd.DataFrame(dir_records).sort_values("stability", ascending=False)
        save_table(dir_df, "edge_direction_certainty")
        results["edge_direction_certainty"] = dir_df.to_dict(orient="records")
    except Exception as e:
        results["bootstrap_error"] = str(e)
        logger.error(f"Bootstrap failed: {e}")

    # Compare CI methods.
    try:
        ci_compare = _compare_ci_methods(df, var_names, results)
        save_table(ci_compare, "ci_method_comparison")
        results["ci_method_comparison"] = ci_compare.to_dict(orient="records")
    except Exception as e:
        results["ci_method_comparison_error"] = str(e)
        logger.error(f"CI method comparison failed: {e}")

    results["bootstrap_n"] = n_boot_pc
    results["bootstrap_stable_edges"] = results.get("bootstrap_pc", {}).get("stable_edges", [])

    save_json(results, "causal_discovery_summary")
    logger.info(f"Causal discovery completed; saved {len(results)} result entries")
    return results


if __name__ == "__main__":
    run_all_discovery()
