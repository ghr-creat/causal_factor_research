import sys

path = r"C:\Users\郭浩然\ZCodeProject\QuantResearch\causal_factor_research\backtest.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

old_text = '''def _causal_signs(causal_results: pd.DataFrame) -> Dict[str, int]:
    """Return the consensus causal sign for each factor.

    A factor gets a causal sign only if at least two of the four estimation
    methods are both statistically significant (p<0.05) and pass their diagnostic
    flag, and all such significant ATEs share the same sign.
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
        if len(sig_ates) >= 2 and all(np.sign(a) == np.sign(sig_ates[0]) for a in sig_ates):
            signs[f] = int(np.sign(sig_ates[0]))
    return signs


def _select_ic_factors(ic_summary: pd.DataFrame) -> List[str]:
    """Select IC-significant factors (relaxed p<0.10 due to short training window)."""
    if "pvalue" in ic_summary.columns:
        return ic_summary.index[ic_summary["pvalue"] < 0.10].tolist()
    if "tstat" in ic_summary.columns:
        return ic_summary.index[abs(ic_summary["tstat"]) > 2].tolist()
    return []'''

new_text = '''def _causal_signs(causal_results: pd.DataFrame) -> Dict[str, int]:
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


def _select_ic_factors(ic_summary: pd.DataFrame) -> List[str]:
    """Select IC-significant factors using the production standard |t|>2."""
    if "tstat" in ic_summary.columns:
        return ic_summary.index[abs(ic_summary["tstat"]) > 2].tolist()
    if "pvalue" in ic_summary.columns:
        return ic_summary.index[ic_summary["pvalue"] < 0.05].tolist()
    return []'''

if old_text not in content:
    print("ERROR: old text not found")
    sys.exit(1)

content = content.replace(old_text, new_text)
with open(path, "w", encoding="utf-8") as f:
    f.write(content)
print("Replacement done")
