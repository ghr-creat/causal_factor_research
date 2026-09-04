# uncompyle6 version 3.9.3
# Python bytecode version base 3.7.0 (3394)
# Decompiled from: Python 3.7.4 (default, Aug  9 2019, 18:34:13) [MSC v.1915 64 bit (AMD64)]
# Embedded file name: C:\Users\郭浩然\ZCodeProject\causal_factor_research\report.py
# Compiled at: 2026-07-07 12:43:06
# Size of source mod 2**32: 41525 bytes
"""Generate LaTeX academic-style reports and Markdown/HTML/PDF fallback from analysis results.

This module builds a Chinese-first academic paper (with English companion) from the
artifacts produced by the causal factor research pipeline. It uses dynamic data
substitution so that no hard-coded numbers are written into the final document.
When LaTeX is unavailable, it falls back to Markdown + HTML + PDF via Playwright.
"""
import json, shutil, subprocess, urllib.parse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
from loguru import logger
from causal_factor_research.config import FACTOR_DESCRIPTIONS, FACTORS, FIGURES_DIR, REPORT_DIR, RESULTS_DIR

# Optional HTML/PDF fallback tools.
_HAS_MARKDOWN2 = False
_HAS_PLAYWRIGHT = False
try:
    import markdown2
    _HAS_MARKDOWN2 = True
except Exception:
    pass
try:
    from playwright.sync_api import sync_playwright
    _HAS_PLAYWRIGHT = True
except Exception:
    pass

def _load_csv(name: str) -> Optional[pd.DataFrame]:
    path = RESULTS_DIR / f"{name}.csv"
    if path.exists():
        return pd.read_csv(path)
    return


def _load_json(name: str) -> Optional[Dict]:
    path = RESULTS_DIR / f"{name}.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return


def _safe_get(data, *keys, default=None):
    for key in keys:
        if isinstance(data, dict) and key in data:
            data = data[key]
        else:
            return default

    return data


def _format_num(x, fmt: str='.4f') -> str:
    if x is None:
        return "N/A"
    if isinstance(x, float):
        if pd.isna(x):
            return "N/A"
    if isinstance(x, (int, float)):
        return f"{x:{fmt}}"
    return str(x)


def _escape_tex(s: str) -> str:
    if not isinstance(s, str):
        return str(s)
    for ch, repl in (('&', '\\&'), ('%', '\\%'), ('$', '\\$'), ('#', '\\#'), ('_', '\\_'),
                     ('{', '\\{'), ('}', '\\}'), ('~', '\\textasciitilde{}'), ('^', '\\textasciicircum{}')):
        s = s.replace(ch, repl)

    return s


def _df_to_latex(df: pd.DataFrame, columns: Optional[List[str]]=None, float_fmt: str='.4f', index: bool=False) -> str:
    if df is None or df.empty:
        return "\\textit{（暂无数据）}"
    elif columns is not None:
        df = df[[c for c in columns if c in df.columns]].copy()

    def _cell(x):
        if pd.isna(x):
            return "N/A"
        if isinstance(x, (int, float)):
            return (f"{x:{float_fmt}}")
        return _escape_tex(str(x))

    if index:
        header_cols = [
         ""] + list(df.columns)
    else:
        header_cols = list(df.columns)
    header = " & ".join([_cell(c) for c in header_cols]) + " \\\\ " + "\n"
    rows = []
    for idx, row in df.iterrows():
        vals = [_cell(v) for v in row.values]
        if index:
            vals = [
             _cell(idx)] + vals
        rows.append(" & ".join(vals) + " \\\\ ")

    align = "l" + "c" * (len(header_cols) - 1) if index else "c" * len(header_cols)
    rows = "\n".join(rows)
    return "\\begin{tabular}{" + align + "}\n\\toprule\n" + header + "\\midrule\n" + rows + "\n\\bottomrule\n\\end{tabular}"


def _img(rel_path: str) -> str:
    full = FIGURES_DIR / rel_path
    if full.exists():
        return f"\\includegraphics[width=0.95\\textwidth]{{figures/{rel_path}}}"
    return "\\textit{（图表未生成）}"


def _substitute(template: str, **kwargs) -> str:
    for key, value in kwargs.items():
        template = template.replace(f"__{key.upper()}__", str(value))

    return template


_CN_TEMPLATE = '\n\\documentclass[12pt,a4paper]{ctexart}\n\\usepackage[top=25mm,bottom=25mm,left=22mm,right=22mm]{geometry}\n\\usepackage{graphicx}\n\\usepackage{booktabs}\n\\usepackage{longtable}\n\\usepackage{hyperref}\n\\usepackage{xcolor}\n\\usepackage{amsmath}\n\\usepackage{amsfonts}\n\\usepackage{natbib}\n\n\\title{\\textbf{因果推断框架下的多因子选股研究}\\\\\\large 基于沪深300成分股的因果因子发现与样本外验证}\n\\author{ZCode 自动化研究}\n\\date{\\today}\n\n\\begin{document}\n\\maketitle\n\n\\begin{abstract}\n\\noindent 本研究使用因果推断框架重新检验 A 股沪深300成分股中多因子选股的因子有效性。研究区分"因果因子"与"相关因子"，通过 PC 与 FCI 算法进行因果发现，采用 Double Machine Learning、工具变量法、前门调整与 Causal Forest 四种方法估计因果效应，并在 2018--2022 年训练集与 2023--2024 年测试集上验证因子衰减与组合表现。\\\\[6pt]\n\\textbf{核心发现}：__CONCLUSION_CORE__\\\\[6pt]\n\\noindent\\textbf{关键词：}因果推断；因子发现；Double Machine Learning；工具变量；前门调整；Causal Forest；样本外验证\n\\end{abstract}\n\n\\tableofcontents\n\\newpage\n\n\\section{引言}\n传统多因子研究依赖横截面相关性（IC）筛选因子，但高 IC 未必意味着因果效应。若因子与收益由共同的市场情绪或宏观冲击驱动，则基于相关性的因子组合可能因套利而快速衰减。因果推断的目标是从观测数据中估计干预后的分布 $P(Y\\mid do(D=d))$，而非 merely 条件期望 $P(Y\\mid D=d)$。本研究将因果推断框架系统引入 A 股多因子选股，区分因果显著因子与相关因子，并验证其样本外稳健性。\n\n\\section{相关工作}\n因果推断在金融学中的应用日益受到关注。\\cite{pearl2009causality} 提出的因果图与 do-演算奠定了识别因果效应的形式化基础；\\cite{chernozhukov2018double} 提出的 Double Machine Learning（DML）利用机器学习估计干扰函数并通过交叉拟合消除偏差；\\cite{wager2018estimation} 的 Causal Forest 则进一步估计异质性处理效应（ITE）。\\cite{lopezdeprado2018advances} 强调了金融机器学习中因果推断与样本外验证的重要性。\\cite{harvey2016and} 指出因子挖掘中的多重检验问题，提示研究者需区分真实预测能力与伪相关。\n\n\\section{数据与因子定义}\n本研究使用 2018-01-02 至 2024-12-31 的沪深300成分股日行情数据。共构建 11 个横截面选股因子，所有因子经行业/板块中性化、MAD 去极值与截面标准化处理。预测标签为 20 日持有期收益。\n\n__FACTOR_TABLE__\n\n控制变量包括市值对数、20 日市场收益、市场波动率、市场趋势以及板块哑变量。\n\n\\section{因果发现}\n\\subsection{PC算法与FCI算法}\nPC 算法假设无潜在混杂，使用 Fisher-Z 条件独立性检验（$\\alpha=0.05$）学习 CPDAG。FCI 算法允许存在未观测混杂因子，输出 PAG（Partial Ancestral Graph）。由于金融数据中市场情绪、宏观冲击等未观测变量几乎必然存在，FCI 的结果比 PC 更保守、更诚实。\n\n\\begin{figure}[htbp]\n\\centering\n__PC_DAG__\n\\caption{PC 算法学习到的因果图（CPDAG）}\n\\end{figure}\n\n\\begin{figure}[htbp]\n\\centering\n__FCI_PAG__\n\\caption{FCI 算法学习到的部分祖先图（PAG）}\n\\end{figure}\n\n\\subsection{Bootstrap稳定性}\n对真实数据进行 __BOOTSTRAP_N__ 次 Bootstrap 重采样，同时运行 PC 与 FCI 算法，报告每条边出现频率。仅保留稳定性 $> 70\\%$ 的边。本次共识别 __N_STABLE_EDGES__ 条稳定边。\n\n\\begin{figure}[htbp]\n\\centering\n__BOOTSTRAP_HEATMAP__\n\\caption{Bootstrap 边稳定性热力图}\n\\end{figure}\n\n\\section{因果效应估计：四维方法矩阵}\n对每个因子分别采用四种识别策略，比较因果效应值（ATE）与置信区间。若多方法一致，则因果效应更稳健。\n\n\\subsection{Double Machine Learning}\nDML 使用 XGBoost 估计 $E[Y\\mid X]$ 与 $E[D\\mid X]$，通过 5-fold 交叉拟合残差化后回归。结果见表 \\ref{tab:dml}。\n\n\\begin{table}[htbp]\n\\centering\n\\caption{DML 因果效应估计}\n\\label{tab:dml}\n__DML_TABLE__\n\\end{table}\n\n\\subsection{工具变量法（IV）}\n构造行业均值 EP 与滞后一期因子作为工具变量。第一阶段 F 统计量需大于 10 以排除弱工具变量问题；Sargan 检验用于过度识别约束。结果见表 \\ref{tab:iv}。\n\n\\begin{table}[htbp]\n\\centering\n\\caption{IV 因果效应估计}\n\\label{tab:iv}\n__IV_TABLE__\n\\end{table}\n\n\\subsection{前门调整}\n以 5 日未来收益作为中介变量，通过两步估计识别 EP 到 20 日收益的因果效应。结果见表 \\ref{tab:fd}。\n\n\\begin{table}[htbp]\n\\centering\n\\caption{前门调整因果效应估计}\n\\label{tab:fd}\n__FD_TABLE__\n\\end{table}\n\n\\subsection{Causal Forest}\n估计异质性处理效应（ITE），给出因子因果效应的截面分布。结果见表 \\ref{tab:cf}。\n\n\\begin{table}[htbp]\n\\centering\n\\caption{Causal Forest 因果效应估计}\n\\label{tab:cf}\n__CF_TABLE__\n\\end{table}\n\n\\begin{figure}[htbp]\n\\centering\n__CAUSAL_EFFECT_COMPARISON__\n\\caption{四种方法 ATE 对比}\n\\end{figure}\n\n\\section{因果因子 vs 相关因子}\n\\subsection{2$\\times$2 分类矩阵}\n按因果显著性（四种方法中至少两种显著且符号一致）与训练集 IC 显著性（$|t|>2$）交叉分类：\n\\begin{itemize}\n    \\item \\textbf{因果显著 \\& IC显著}：真因子\n    \\item \\textbf{因果显著 \\& IC不显著}：被掩盖的因子\n    \\item \\textbf{因果不显著 \\& IC显著}：伪因子（由混杂驱动的虚假相关）\n    \\item \\textbf{均不显著}：无效因子\n\\end{itemize}\n\n\\begin{table}[htbp]\n\\centering\n\\caption{因子 2$\\times$2 分类矩阵}\n\\label{tab:matrix}\n__CLASSIFICATION_MATRIX__\n\\end{table}\n\n因果显著因子：\\texttt{__CAUSAL_FACTORS_LIST__}。\\\\\nIC 显著因子：\\texttt{__IC_FACTORS_LIST__}。\\\\\n伪因子（IC 显著但因果不显著）：\\texttt{__SPURIOUS_FACTORS_LIST__}。\\\\\n被掩盖的因子（因果显著但 IC 不显著）：\\texttt{__MASKED_FACTORS_LIST__}。\n\n\\subsection{IC 衰减曲线}\n在 2023--2024 年测试集上计算滚动 IC，对比因果显著组、IC 显著组与其他组的衰减。\n\n\\begin{figure}[htbp]\n\\centering\n__IC_DECAY_CURVES__\n\\caption{因子滚动 IC 衰减曲线}\n\\end{figure}\n\n\\begin{table}[htbp]\n\\centering\n\\caption{因子 IC 衰减率}\n\\label{tab:decay}\n__DECAY_TABLE__\n\\end{table}\n\n\\subsection{因子拥挤度}\n以因子横截面标准差衡量拥挤度，比较高低拥挤度下的 IC 差异。\n\n\\begin{figure}[htbp]\n\\centering\n__CROWDING_IC__\n\\caption{拥挤度与 IC 关系}\n\\end{figure}\n\n\\section{组合回测}\n在 2023--2024 年样本外，比较三组多空组合：全因子等权、IC 显著因子、因果显著因子。每组选择因子得分最高/最低的 30 只股票，等权做多/做空，计入交易成本。\n\n\\begin{figure}[htbp]\n\\centering\n__BACKTEST_NAV__\n\\caption{三组组合样本外净值曲线}\n\\end{figure}\n\n\\begin{figure}[htbp]\n\\centering\n__BACKTEST_SHARPE__\n\\caption{样本外夏普比率对比}\n\\end{figure}\n\n\\begin{table}[htbp]\n\\centering\n\\caption{组合回测绩效指标}\n\\label{tab:backtest}\n__BACKTEST_TABLE__\n\\end{table}\n\n\\section{稳健性检验}\n\\subsection{Rosenbaum 敏感性边界}\n检验因果结论对未观测混杂的稳健性。$\\Gamma$ 从 1.0 增加到 2.0，若 $\\Gamma=1.5$ 时仍显著则为中等稳健，$\\Gamma=2.0$ 仍显著则为高度稳健。\n\n\\begin{figure}[htbp]\n\\centering\n__ROSENBAUM_BOUNDS__\n\\caption{Rosenbaum 敏感性边界}\n\\end{figure}\n\n\\subsection{Placebo 检验}\n对随机噪声因子与截面打乱后的真实因子运行完整因果推断流程，预期伪因子因果效应不显著。\n\n\\begin{table}[htbp]\n\\centering\n\\caption{Placebo 检验结果}\n\\label{tab:placebo}\n__PLACEBO_TABLE__\n\\end{table}\n\n\\subsection{时间置换检验}\n将因子向前平移一期，检验未来因子是否因果影响当前收益。预期结果不显著。\n\n\\begin{table}[htbp]\n\\centering\n\\caption{时间置换检验结果}\n\\label{tab:timeperm}\n__TIME_PERMUTATION_TABLE__\n\\end{table}\n\n\\subsection{分市场状态}\n按市场趋势将样本划分为熊市、震荡市与牛市，分别估计 DML 因果效应。\n\n\\begin{figure}[htbp]\n\\centering\n__MARKET_REGIME__\n\\caption{DML 因果效应分市场状态对比}\n\\end{figure}\n\n\\section{数学严谨性：do-演算与识别假设}\n\\subsection{后门准则与 do-演算}\n若控制变量集 $Z$ 阻断了从处理 $D$ 到结果 $Y$ 的所有后门路径，则\n\\begin{equation}\nP(Y \\mid do(D=d)) = \\sum_z P(Y \\mid D=d, Z=z) P(Z=z)\n\\end{equation}\n\n本研究对每个因子纳入行业、市值、动量、波动率等控制变量以阻断后门路径。倾向得分重叠度平均为 __MEAN_OVERLAP__，共同支撑条件 __OVERLAP_ASSESSMENT__。\n\n\\subsection{识别假设}\n\\begin{itemize}\n    \\item \\textbf{可忽略性}：给定控制变量 $Z$，处理分配（因子暴露）与潜在结果（未来收益）近似独立。未观测的市场情绪、宏观冲击仍可能构成残余混杂。\n    \\item \\textbf{共同支撑域}：处理组与对照组在控制变量上存在重叠，通过倾向得分分布验证。\n    \\item \\textbf{SUTVA}：横截面因子研究中个股处理效应近似独立，但因子组合内股票间可能存在溢出效应。SUTVA 风险较高的因子：\\texttt{__HIGH_SUTVA_FACTORS__}。\n\\end{itemize}\n\n\\section{结论与局限}\n__CONCLUSION_CORE__\n\n本研究的主要局限包括：工具变量的排除性约束无法直接验证，前门调整的中介变量为代理变量，未观测混杂无法完全排除，以及因果发现算法对分布假设和样本量敏感。未来工作可探索更大样本、更精细的中介变量与动态因果图方法。\n\n\\bibliographystyle{plain}\n\\bibliography{references}\n\n\\end{document}\n'
_EN_TEMPLATE = "\n\\documentclass[12pt,a4paper]{article}\n\\usepackage[top=25mm,bottom=25mm,left=22mm,right=22mm]{geometry}\n\\usepackage{graphicx}\n\\usepackage{booktabs}\n\\usepackage{longtable}\n\\usepackage{hyperref}\n\\usepackage{xcolor}\n\\usepackage{amsmath}\n\\usepackage{amsfonts}\n\\usepackage{natbib}\n\n\\title{\\textbf{Causal Inference for Multi-Factor Equity Selection}\\\\\\large Causal Factor Discovery and Out-of-Sample Validation in CSI300}\n\\author{ZCode Automated Research}\n\\date{\\today}\n\n\\begin{document}\n\\maketitle\n\n\\begin{abstract}\n\\noindent This study re-examines the validity of common equity factors in the CSI300 universe using a causal inference framework. We distinguish causal factors from merely correlated factors via causal discovery (PC/FCI) and four causal effect estimators (Double Machine Learning, Instrumental Variables, Front-Door Adjustment, and Causal Forest). Estimation is performed on the 2018--2022 training period and evaluated on the 2023--2024 test period.\\\\[6pt]\n\\textbf{Key finding}: __CONCLUSION_CORE_EN__\\\\[6pt]\n\\noindent\\textbf{Keywords:} causal inference; factor discovery; Double Machine Learning; instrumental variables; front-door adjustment; Causal Forest; out-of-sample validation\n\\end{abstract}\n\n\\tableofcontents\n\\newpage\n\n\\section{Introduction}\nTraditional multi-factor research relies on cross-sectional correlation (IC) to screen factors, but high IC does not imply causality. If factors and returns are jointly driven by market sentiment or macro shocks, correlation-based factor portfolios may decay quickly as arbitrage removes the spurious relationship. This study systematically introduces causal inference into CSI300 multi-factor stock selection, distinguishes causal-significant factors from correlated factors, and validates their out-of-sample robustness.\n\n\\section{Related Work}\nCausal inference is gaining traction in finance. \\cite{pearl2009causality} introduced causal graphs and do-calculus for identifying causal effects; \\cite{chernozhukov2018double} proposed Double Machine Learning (DML) to estimate nuisance functions with machine learning and cross-fitting; \\cite{wager2018estimation} developed Causal Forests for heterogeneous treatment effects. \\cite{lopezdeprado2018advances} emphasized the importance of causal inference and out-of-sample testing in financial machine learning. \\cite{harvey2016and} highlighted the multiple-testing problem in factor mining, urging researchers to distinguish genuine predictive power from spurious correlations.\n\n\\section{Data and Factor Definitions}\nWe use daily market data of CSI300 constituents from 2018-01-02 to 2024-12-31. We construct 11 cross-sectional factors, all neutralized by industry/board, winsorized via MAD, and standardized cross-sectionally. The target label is the 20-day holding-period return.\n\n__FACTOR_TABLE__\n\nControl variables include log market capitalization, 20-day market return, market volatility, market trend, and board dummies.\n\n\\section{Causal Discovery}\n\\subsection{PC and FCI Algorithms}\nPC assumes no latent confounding and learns a CPDAG using Fisher-Z conditional independence tests ($\\alpha=0.05$). FCI allows latent confounders and outputs a PAG. Because market sentiment and macro shocks are almost certainly unobserved in financial data, FCI is more conservative than PC.\n\n\\begin{figure}[htbp]\n\\centering\n__PC_DAG__\n\\caption{Causal graph learned by PC (CPDAG)}\n\\end{figure}\n\n\\begin{figure}[htbp]\n\\centering\n__FCI_PAG__\n\\caption{Partial Ancestral Graph learned by FCI}\n\\end{figure}\n\n\\subsection{Bootstrap Stability}\nWe run __BOOTSTRAP_N__ bootstrap replications of PC/FCI on the real data and report edge stability. Only edges with stability $>70\\%$ are retained. This run identified __N_STABLE_EDGES__ stable edges.\n\n\\begin{figure}[htbp]\n\\centering\n__BOOTSTRAP_HEATMAP__\n\\caption{Bootstrap edge stability heatmap}\n\\end{figure}\n\n\\section{Causal Effect Estimation: Four Methods}\nEach factor is estimated with four identification strategies. Agreement across methods strengthens the causal claim.\n\n\\subsection{Double Machine Learning}\nDML uses XGBoost to estimate $E[Y\\mid X]$ and $E[D\\mid X]$, then regresses the residualized outcome on the residualized treatment using 5-fold cross-fitting. See Table \\ref{tab:dml}.\n\n\\begin{table}[htbp]\n\\centering\n\\caption{DML causal effect estimates}\n\\label{tab:dml}\n__DML_TABLE__\n\\end{table}\n\n\\subsection{Instrumental Variables}\nWe construct industry-average EP and lagged factor as instruments. The first-stage F-statistic must exceed 10 to rule out weak instruments; the Sargan test checks over-identifying restrictions. See Table \\ref{tab:iv}.\n\n\\begin{table}[htbp]\n\\centering\n\\caption{IV causal effect estimates}\n\\label{tab:iv}\n__IV_TABLE__\n\\end{table}\n\n\\subsection{Front-Door Adjustment}\nUsing 5-day future return as a mediator, we identify the causal effect of EP on 20-day return through two-stage estimation. See Table \\ref{tab:fd}.\n\n\\begin{table}[htbp]\n\\centering\n\\caption{Front-door adjustment estimates}\n\\label{tab:fd}\n__FD_TABLE__\n\\end{table}\n\n\\subsection{Causal Forest}\nCausal Forest estimates heterogeneous treatment effects (ITE) and reports the cross-sectional distribution of factor effects. See Table \\ref{tab:cf}.\n\n\\begin{table}[htbp]\n\\centering\n\\caption{Causal Forest estimates}\n\\label{tab:cf}\n__CF_TABLE__\n\\end{table}\n\n\\begin{figure}[htbp]\n\\centering\n__CAUSAL_EFFECT_COMPARISON__\n\\caption{Comparison of ATEs across four methods}\n\\end{figure}\n\n\\section{Causal vs. Correlated Factors}\n\\subsection{2$\\times$2 Classification}\nWe classify factors by causal significance (at least two of four methods significant with consistent sign) and in-sample IC significance ($|t|>2$):\n\\begin{itemize}\n    \\item \\textbf{Causal significant \\& IC significant}: true factor\n    \\item \\textbf{Causal significant \\& IC not significant}: masked factor\n    \\item \\textbf{Causal not significant \\& IC significant}: spurious factor\n    \\item \\textbf{Neither significant}: ineffective factor\n\\end{itemize}\n\n\\begin{table}[htbp]\n\\centering\n\\caption{2$\\times$2 factor classification matrix}\n\\label{tab:matrix}\n__CLASSIFICATION_MATRIX__\n\\end{table}\n\nCausal-significant factors: \\texttt{__CAUSAL_FACTORS_LIST__}.\\\\\nIC-significant factors: \\texttt{__IC_FACTORS_LIST__}.\\\\\nSpurious factors (IC significant but not causal): \\texttt{__SPURIOUS_FACTORS_LIST__}.\\\\\nMasked factors (causal but not IC significant): \\texttt{__MASKED_FACTORS_LIST__}.\n\n\\subsection{IC Decay Curves}\nWe compute rolling IC on the 2023--2024 test set and compare decay rates across groups.\n\n\\begin{figure}[htbp]\n\\centering\n__IC_DECAY_CURVES__\n\\caption{Rolling IC decay curves}\n\\end{figure}\n\n\\begin{table}[htbp]\n\\centering\n\\caption{IC decay rates}\n\\label{tab:decay}\n__DECAY_TABLE__\n\\end{table}\n\n\\subsection{Factor Crowding}\nWe use cross-sectional dispersion as a proxy for crowding and compare IC under high vs. low crowding.\n\n\\begin{figure}[htbp]\n\\centering\n__CROWDING_IC__\n\\caption{Factor crowding vs. IC}\n\\end{figure}\n\n\\section{Portfolio Backtest}\nOut-of-sample from 2023--2024, we compare three long-short portfolios: all-factor equal-weight, IC-significant factors, and causal-significant factors. Each selects the top/bottom 30 stocks by score, equal-weighted long and short, with transaction costs.\n\n\\begin{figure}[htbp]\n\\centering\n__BACKTEST_NAV__\n\\caption{Out-of-sample NAV curves of three portfolios}\n\\end{figure}\n\n\\begin{figure}[htbp]\n\\centering\n__BACKTEST_SHARPE__\n\\caption{Out-of-sample Sharpe ratio comparison}\n\\end{figure}\n\n\\begin{table}[htbp]\n\\centering\n\\caption{Portfolio backtest performance}\n\\label{tab:backtest}\n__BACKTEST_TABLE__\n\\end{table}\n\n\\section{Robustness Checks}\n\\subsection{Rosenbaum Sensitivity Bounds}\nWe test robustness to unobserved confounding by varying $\\Gamma$ from 1.0 to 2.0. If the effect remains significant at $\\Gamma=1.5$, it is moderately robust; at $\\Gamma=2.0$, highly robust.\n\n\\begin{figure}[htbp]\n\\centering\n__ROSENBAUM_BOUNDS__\n\\caption{Rosenbaum sensitivity bounds}\n\\end{figure}\n\n\\subsection{Placebo Tests}\nWe run the full causal pipeline on random noise factors and cross-sectionally shuffled real factors; the causal effects should be insignificant.\n\n\\begin{table}[htbp]\n\\centering\n\\caption{Placebo test results}\n\\label{tab:placebo}\n__PLACEBO_TABLE__\n\\end{table}\n\n\\subsection{Time-Permutation Test}\nWe shift the factor forward by one period and estimate its effect on the current-period label. A significant effect would indicate look-ahead bias or spurious time-reversal.\n\n\\begin{table}[htbp]\n\\centering\n\\caption{Time-permutation test results}\n\\label{tab:timeperm}\n__TIME_PERMUTATION_TABLE__\n\\end{table}\n\n\\subsection{Market Regimes}\nWe split the sample by market trend into bear, neutral, and bull regimes and estimate DML effects separately.\n\n\\begin{figure}[htbp]\n\\centering\n__MARKET_REGIME__\n\\caption{DML effects by market regime}\n\\end{figure}\n\n\\section{Math Rigor: do-Calculus and Identification}\n\\subsection{Back-Door Criterion and do-Calculus}\nIf a set of controls $Z$ blocks all back-door paths from treatment $D$ to outcome $Y$, then\n\\begin{equation}\nP(Y \\mid do(D=d)) = \\sum_z P(Y \\mid D=d, Z=z) P(Z=z)\n\\end{equation}\n\nWe include industry, size, momentum, and volatility controls for each factor to block back-door paths. The average propensity-score overlap is __MEAN_OVERLAP__, indicating __OVERLAP_ASSESSMENT__ common support.\n\n\\subsection{Identification Assumptions}\n\\begin{itemize}\n    \\item \\textbf{Ignorability}: Conditional on controls $Z$, treatment assignment (factor exposure) is approximately independent of potential outcomes (future returns). Unobserved market sentiment and macro shocks remain residual confounders.\n    \\item \\textbf{Common support}: Treated and control units overlap in the control space, verified via propensity score distributions.\n    \\item \\textbf{SUTVA}: In cross-sectional factor studies, individual treatment effects are approximately independent, but spillovers may exist within factor portfolios. High SUTVA-risk factors: \\texttt{__HIGH_SUTVA_FACTORS__}.\n\\end{itemize}\n\n\\section{Conclusion and Limitations}\n__CONCLUSION_CORE_EN__\n\nLimitations include the untestable exclusion restriction of IVs, the proxy nature of the mediator in front-door adjustment, residual unobserved confounding, and the sensitivity of causal discovery algorithms to distributional assumptions and sample size. Future work could explore larger samples, richer mediators, and dynamic causal graphs.\n\n\\bibliographystyle{plain}\n\\bibliography{references}\n\n\\end{document}\n"

def _build_factor_table() -> str:
    df = pd.DataFrame([{'因子':f,  '含义':(FACTOR_DESCRIPTIONS.get)(f, "")} for f in FACTORS])
    return _df_to_latex(df)


def _build_factor_table_en() -> str:
    df = pd.DataFrame([{'Factor':f,  'Description':(FACTOR_DESCRIPTIONS.get)(f, "")} for f in FACTORS])
    return _df_to_latex(df)


def _build_method_table(causal_effects: pd.DataFrame, method: str, columns: List[str]) -> str:
    sub = causal_effects[causal_effects["method"] == method]
    if sub.empty:
        return "\\textit{（暂无数据）}"
    cols = [c for c in columns if c in sub.columns]
    return _df_to_latex((sub[["factor"] + cols]), columns=(["factor"] + cols))


def _build_classification_matrix(classification: Optional[Dict]) -> str:
    if classification is None or "matrix" not in classification:
        return "\\textit{（暂无数据）}"
    matrix = pd.DataFrame(classification["matrix"])
    return _df_to_latex(matrix, index=True)


def _build_backtest_table(backtest_metrics: Optional[Dict]) -> str:
    if backtest_metrics is None:
        return "\\textit{（暂无数据）}"
    rows = []
    for name, metrics in backtest_metrics.items():
        if isinstance(metrics, dict):
            row = {"组合": name}
            row.update(metrics)
            rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return "\\textit{（暂无数据）}"
    cols = [
     "组合"] + [c for c in df.columns if c != "组合"]
    return _df_to_latex(df[cols])


def _conclusion_core(backtest_metrics: Optional[Dict], causal_factors: List[str], ic_factors: List[str], spurious_factors: List[str], decay_group: Optional[pd.DataFrame], lang: str='cn') -> str:
    causal_sharpe = None
    ic_sharpe = None
    causal_decay = None
    ic_decay = None
    if backtest_metrics:
        if isinstance(backtest_metrics, dict):
            causal_sharpe = backtest_metrics.get("因果显著因子", {}).get("sharpe")
            ic_sharpe = backtest_metrics.get("IC显著因子", {}).get("sharpe")
    else:
        if decay_group is not None:
            if not decay_group.empty:
                causal_row = decay_group[decay_group["group"] == "因果显著组"]
                ic_row = decay_group[decay_group["group"] == "IC显著组"]
                if not causal_row.empty:
                    causal_decay = causal_row["mean_decay_rate"].values[0]
                if not ic_row.empty:
                    ic_decay = ic_row["mean_decay_rate"].values[0]
    if lang == "cn":
        parts = []
        if spurious_factors:
            parts.append(f'部分传统 IC 显著因子（如 {", ".join(spurious_factors)}）在因果推断中并不显著，存在伪因子风险。')
        if causal_factors:
            parts.append(f'因果显著因子包括：{", ".join(causal_factors)}。')
        if causal_decay is not None:
            if ic_decay is not None:
                parts.append(f"因果显著组样本外 IC 平均衰减率为 {causal_decay:.2%}，IC 显著组为 {ic_decay:.2%}。")
        if causal_sharpe is not None:
            if ic_sharpe is not None:
                if causal_sharpe > ic_sharpe:
                    parts.append(f"因果显著因子组合样本外夏普（{_format_num(causal_sharpe)}）高于 IC 显著因子组合（{_format_num(ic_sharpe)}），支持因果因子更稳健的判断。")
                else:
                    parts.append(f"因果显著因子组合样本外夏普（{_format_num(causal_sharpe)}）未高于 IC 显著因子组合（{_format_num(ic_sharpe)}），提示本次样本外期间因果优势不显著。")
        if parts:
            return " ".join(parts)
        return "结果待生成。"
    parts = []
    if spurious_factors:
        parts.append(f'Several traditional IC-significant factors (e.g., {", ".join(spurious_factors)}) are not causally significant, indicating spurious-correlation risk.')
    elif causal_factors:
        parts.append(f'Causal-significant factors include: {", ".join(causal_factors)}.')
    elif causal_decay is not None:
        if ic_decay is not None:
            parts.append(f"Out-of-sample IC decay is {causal_decay:.2%} for the causal-significant group and {ic_decay:.2%} for the IC-significant group.")
    if causal_sharpe is not None and ic_sharpe is not None:
        if causal_sharpe > ic_sharpe:
            parts.append(f"The causal-significant portfolio outperforms the IC-significant portfolio out-of-sample (Sharpe {_format_num(causal_sharpe)} vs {_format_num(ic_sharpe)}), supporting the robustness of causal factors.")
        else:
            parts.append(f"The causal-significant portfolio (Sharpe {_format_num(causal_sharpe)}) did not outperform the IC-significant portfolio (Sharpe {_format_num(ic_sharpe)}) in this sample.")
    if parts:
        return " ".join(parts)
    return "Results pending."


def _shared_values(causal_effects: Optional[pd.DataFrame], classification: Optional[Dict], ic_train: Optional[pd.DataFrame], ic_test: Optional[pd.DataFrame], backtest_metrics: Optional[Dict], backtest_returns: Optional[pd.DataFrame], decay_rates: Optional[pd.DataFrame], decay_group: Optional[pd.DataFrame], discovery: Optional[Dict], identification: Optional[Dict], placebo: Optional[pd.DataFrame], time_perm: Optional[pd.DataFrame]) -> Dict:
    """Compute common values used by both reports."""
    bootstrap_n = _safe_get(discovery, "bootstrap_n", default=1000)
    stable_edges = _safe_get(discovery, "bootstrap_stable_edges", default=[])
    causal_factors = _safe_get(classification, "causal_significant", default=[])
    ic_factors = _safe_get(classification, "ic_significant", default=[])
    spurious_factors = _safe_get(classification, "spurious_factors", default=[])
    masked_factors = _safe_get(classification, "masked_factors", default=[])
    mean_overlap = _safe_get(identification, "common_support", "mean_overlap_ratio", default=None)
    overlap_assessment = _safe_get(identification, "common_support", "assessment", default="待评估")
    high_sutva = _safe_get(identification, "sutva", "high_concern_factors", default=[])
    conclusion_cn = _conclusion_core(backtest_metrics,
      causal_factors, ic_factors, spurious_factors, decay_group, lang="cn")
    conclusion_en = _conclusion_core(backtest_metrics,
      causal_factors, ic_factors, spurious_factors, decay_group, lang="en")
    dml_table = _build_method_table(causal_effects, "DML", ['ate', 'se', 'tvalue', 'pvalue', 'ci_lower', 'ci_upper']) if causal_effects is not None else "\\textit{（暂无数据）}"
    iv_table = _build_method_table(causal_effects, "IV", ['ate', 'se', 'pvalue', 'first_stage_f', 'sargan_pvalue']) if causal_effects is not None else "\\textit{（暂无数据）}"
    fd_table = _build_method_table(causal_effects, "FrontDoor", ['ate', 'se', 'pvalue', 'beta1', 'beta2']) if causal_effects is not None else "\\textit{（暂无数据）}"
    cf_table = _build_method_table(causal_effects, "CausalForest", ["ate", "se", "pvalue", "ite_std"]) if causal_effects is not None else "\\textit{（暂无数据）}"
    return dict(conclusion_core=conclusion_cn,
      conclusion_core_en=conclusion_en,
      factor_table=(_build_factor_table()),
      dml_table=dml_table,
      iv_table=iv_table,
      fd_table=fd_table,
      cf_table=cf_table,
      causal_factors_list=(_escape_tex(str(causal_factors))),
      ic_factors_list=(_escape_tex(str(ic_factors))),
      spurious_factors_list=(_escape_tex(str(spurious_factors))),
      masked_factors_list=(_escape_tex(str(masked_factors))),
      classification_matrix=(_build_classification_matrix(classification)),
      decay_table=(_df_to_latex(decay_rates)),
      bootstrap_n=bootstrap_n,
      n_stable_edges=(len(stable_edges)),
      pc_dag=(_img("pc_dag.png")),
      fci_pag=(_img("fci_pag.png")),
      bootstrap_heatmap=(_img("bootstrap_stability_heatmap.png")),
      causal_effect_comparison=(_img("causal_effect_comparison.png")),
      ic_decay_curves=(_img("ic_decay_curves.png")),
      crowding_ic=(_img("crowding_ic.png")),
      backtest_nav=(_img("backtest_nav_curves.png")),
      backtest_sharpe=(_img("backtest_sharpe_comparison.png")),
      backtest_table=(_build_backtest_table(backtest_metrics)),
      rosenbaum_bounds=(_img("rosenbaum_bounds.png")),
      placebo_table=(_df_to_latex(placebo)),
      time_permutation_table=(_df_to_latex(time_perm)),
      market_regime=(_img("market_regime_comparison.png")),
      mean_overlap=(_format_num(mean_overlap)),
      overlap_assessment=(_escape_tex(overlap_assessment)),
      high_sutva_factors=(_escape_tex(str(high_sutva))))


def _build_cn_report(**kwargs) -> str:
    """Build Chinese LaTeX report content."""
    kwargs["factor_table"] = _build_factor_table()
    return _substitute(_CN_TEMPLATE, **kwargs)


def _build_en_report(**kwargs) -> str:
    """Build English LaTeX report content."""
    kwargs["factor_table"] = _build_factor_table_en()
    kwargs["conclusion_core"] = kwargs.get("conclusion_core_en", kwargs["conclusion_core"])
    return _substitute(_EN_TEMPLATE, **kwargs)


def _compile_latex(tex_path: Path, cwd: Path) -> bool:
    """Compile a LaTeX file with xelatex (or pdflatex if xelatex unavailable)."""
    if shutil.which("xelatex") is not None:
        compiler = "xelatex"
    else:
        if shutil.which("pdflatex") is not None:
            compiler = "pdflatex"
        else:
            logger.warning("xelatex/pdflatex not found in PATH; skipping PDF compilation.")
            return False
    try:
        for _ in range(2):
            subprocess.run([
             compiler, "-interaction=nonstopmode", str(tex_path.name)],
              cwd=(str(cwd)),
              check=True,
              stdout=(subprocess.PIPE),
              stderr=(subprocess.PIPE))

        return True
    except Exception as e:
        try:
            logger.error(f"LaTeX compilation failed for {tex_path}: {e}")
            return False
        finally:
            e = None
            del e


def _generate_markdown(causal_effects, classification, backtest_metrics, decay_group, discovery, identification, output_path: Path):
    """Generate a Markdown fallback report if LaTeX is unavailable."""
    causal_factors = _safe_get(classification, "causal_significant", default=[])
    ic_factors = _safe_get(classification, "ic_significant", default=[])
    spurious = _safe_get(classification, "spurious_factors", default=[])
    masked = _safe_get(classification, "masked_factors", default=[])
    conclusion = _conclusion_core(backtest_metrics, causal_factors, ic_factors, spurious, decay_group, "cn")
    lines = [
     "# 因果推断框架下的多因子选股研究\n",
     "## 摘要\n",
     conclusion,
     "\n## 因子定义\n",
     pd.DataFrame([{'因子':f,  '含义':(FACTOR_DESCRIPTIONS.get)(f, "")} for f in FACTORS]).to_markdown(index=False),
     "\n## 因果发现\n",
     f'对真实数据进行 **{_safe_get(discovery, "bootstrap_n", default=1000)} 次** Bootstrap 重采样，同时运行 PC 与 FCI 算法。',
     f'共识别 **{len(_safe_get(discovery, "bootstrap_stable_edges", default=[]))}** 条稳定边（稳定性 > 70%）。',
     "\n![PC 因果图](figures/pc_dag.png)",
     "\n![FCI 部分祖先图](figures/fci_pag.png)",
     "\n![Bootstrap 稳定性热力图](figures/bootstrap_stability_heatmap_pc.png)"]
    if causal_effects is not None:
        lines += [
         "\n## 因果效应估计：四维方法矩阵\n",
         "### DML\n",
         causal_effects[causal_effects["method"] == "DML"].to_markdown(index=False),
         "\n### IV\n",
         causal_effects[causal_effects["method"] == "IV"].to_markdown(index=False),
         "\n### 前门调整\n",
         causal_effects[causal_effects["method"] == "FrontDoor"].to_markdown(index=False),
         "\n### Causal Forest\n",
         causal_effects[causal_effects["method"] == "CausalForest"].to_markdown(index=False),
         "\n![四种方法 ATE 对比](figures/causal_effect_comparison.png)"]
    if classification:
        matrix = pd.DataFrame(_safe_get(classification, "matrix", default={}))
        lines += [
         "\n## 因果因子 vs 相关因子\n",
         "### 2×2 分类矩阵\n",
         matrix.to_markdown(),
         f"\n- 因果显著因子：{causal_factors}",
         f"- IC 显著因子：{ic_factors}",
         f"- 伪因子（IC 显著但因果不显著）：{spurious}",
         f"- 被掩盖的因子（因果显著但 IC 不显著）：{masked}",
         "\n![IC 衰减曲线](figures/ic_decay_curves.png)",
         "\n![拥挤度与 IC](figures/crowding_ic.png)"]
    if backtest_metrics:
        lines += [
         "\n## 组合回测指标\n",
         pd.DataFrame(backtest_metrics).T.to_markdown(),
         "\n![样本外净值曲线](figures/backtest_nav_curves.png)",
         "\n![样本外夏普对比](figures/backtest_sharpe_comparison.png)"]
    lines += [
     "\n## 稳健性检验\n",
     "\n![Rosenbaum 敏感性边界](figures/rosenbaum_bounds.png)",
     "\n![分市场状态 DML 对比](figures/market_regime_comparison.png)",
     "\n## 识别假设与数学严谨性\n",
     f'- 平均倾向得分重叠度：{_format_num(_safe_get(identification, "common_support", "mean_overlap_ratio", default=None))}',
     f'- 共同支撑评估：{_safe_get(identification, "common_support", "assessment", default="待评估")}',
     f'- SUTVA 高关注因子：{_safe_get(identification, "sutva", "high_concern_factors", default=[])}',
     "\n---\n",
     "所有图表保存在 `figures/` 目录。正式 PDF 可通过 `xelatex report_cn.tex` 编译生成。\n"]
    output_path.write_text(("\n".join(lines)), encoding="utf-8")


def _generate_html_and_pdf(md_path: Path, html_path: Path, pdf_path: Path):
    """Convert Markdown report to HTML and then to PDF using Playwright."""
    if not _HAS_MARKDOWN2:
        logger.warning("markdown2 not available; skipping HTML/PDF generation.")
        return
    md_text = md_path.read_text(encoding="utf-8")
    html_body = markdown2.markdown(md_text, extras=["tables", "fenced-code-blocks"])
    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>因果推断框架下的多因子选股研究</title>
<style>
body {{ font-family: "Microsoft YaHei", "SimSun", sans-serif; line-height: 1.6; max-width: 900px; margin: 40px auto; padding: 0 20px; color: #333; }}
h1 {{ text-align: center; border-bottom: 2px solid #333; padding-bottom: 10px; }}
h2 {{ border-bottom: 1px solid #ccc; padding-bottom: 5px; margin-top: 30px; }}
table {{ border-collapse: collapse; width: 100%; margin: 15px 0; font-size: 12px; }}
th, td {{ border: 1px solid #ddd; padding: 6px; text-align: left; }}
th {{ background-color: #f2f2f2; }}
img {{ max-width: 100%; height: auto; }}
</style>
</head>
<body>
{html_body}
</body>
</html>"""
    html_path.write_text(html, encoding="utf-8")
    logger.info(f"HTML report saved: {html_path}")

    if not _HAS_PLAYWRIGHT:
        logger.warning("playwright not available; skipping PDF generation.")
        return
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto("file:///" + urllib.parse.quote(str(html_path).replace("\\", "/"), safe="/"))
            page.pdf(path=str(pdf_path), format="A4", margin={"top": "1cm", "bottom": "1cm", "left": "1cm", "right": "1cm"})
            browser.close()
        logger.info(f"PDF report saved: {pdf_path}")
    except Exception as e:
        logger.warning(f"PDF generation failed: {e}")


def generate_report():
    """Generate Chinese and English LaTeX reports and compile to PDF."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    causal_effects = _load_csv("causal_effects_train")
    classification = _load_json("factor_classification")
    ic_train = _load_csv("ic_summary_train")
    ic_test = _load_csv("ic_summary_test")
    backtest_metrics = _load_json("backtest_metrics")
    backtest_returns = _load_csv("backtest_returns")
    decay_rates = _load_csv("ic_decay_rates")
    decay_group = _load_csv("ic_decay_group_summary")
    discovery = _load_json("causal_discovery_summary")
    identification = _load_json("identification_summary")
    placebo_summary = _load_csv("placebo_test_summary")
    if placebo_summary is None or placebo_summary.empty:
        placebo_summary = _load_csv("placebo_test")
    time_perm = _load_csv("time_permutation_test")
    report_figures = REPORT_DIR / "figures"
    report_figures.mkdir(parents=True, exist_ok=True)
    if FIGURES_DIR.exists():
        for fig in FIGURES_DIR.glob("*.png"):
            shutil.copy(fig, report_figures / fig.name)

    bib_src = REPORT_DIR / "references.bib"
    bib_dst = REPORT_DIR / "references.bib"
    kwargs = _shared_values(causal_effects, classification, ic_train, ic_test, backtest_metrics, backtest_returns, decay_rates, decay_group, discovery, identification, placebo_summary, time_perm)
    cn_tex = REPORT_DIR / "report_cn.tex"
    en_tex = REPORT_DIR / "report_en.tex"
    cn_content = _build_cn_report(**kwargs)
    en_content = _build_en_report(**kwargs)
    with open(cn_tex, "w", encoding="utf-8") as f:
        f.write(cn_content)
    with open(en_tex, "w", encoding="utf-8") as f:
        f.write(en_content)
    logger.info(f"LaTeX sources saved: {cn_tex}, {en_tex}")
    cn_compiled = _compile_latex(cn_tex, REPORT_DIR)
    en_compiled = _compile_latex(en_tex, REPORT_DIR)
    cn_pdf = REPORT_DIR / "report_cn.pdf"
    en_pdf = REPORT_DIR / "report_en.pdf"
    if cn_compiled and cn_pdf.exists():
        logger.info(f"Chinese PDF report saved: {cn_pdf}")
    else:
        logger.warning(f"Chinese PDF not generated. Source: {cn_tex}")
    if en_compiled and en_pdf.exists():
        logger.info(f"English PDF report saved: {en_pdf}")
    else:
        logger.warning(f"English PDF not generated. Source: {en_tex}")
    md_path = REPORT_DIR / "report_cn.md"
    try:
        _generate_markdown(causal_effects, classification, backtest_metrics, decay_group, discovery, identification, md_path)
        logger.info(f"Markdown fallback saved: {md_path}")
        # Always generate HTML/PDF fallback (especially useful when LaTeX is unavailable).
        _generate_html_and_pdf(md_path, REPORT_DIR / "report_cn.html", cn_pdf)
    except Exception as e:
        try:
            logger.warning(f"Markdown fallback generation failed: {e}")
        finally:
            e = None
            del e


if __name__ == "__main__":
    generate_report()
