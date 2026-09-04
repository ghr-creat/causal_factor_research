"""Configuration for the causal factor research project."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results" / "causal_final"
FIGURES_DIR = RESULTS_DIR / "figures"
REPORT_DIR = PROJECT_ROOT / "report" / "causal_final"

TRAIN_START = "2018-01-01"
TRAIN_END = "2022-12-31"
TEST_START = "2023-01-01"
TEST_END = "2024-12-31"

REBALANCE_DAYS = 20
LABEL_HORIZON = 20
MAD_MULTIPLE = 3.0

RANDOM_SEED = 42

# Production-level switch: controls bootstrap iterations, KCI, FCI, etc.
RUN_FULL_PRODUCTION = True

# Production-level hyper-parameters.
N_BOOTSTRAP = 1000          # PC algorithm bootstrap replications
N_BOOTSTRAP_FCI = 100       # FCI algorithm bootstrap replications (slower)
ENABLE_KCI = True           # Run KCI conditional independence test alongside Fisher-Z
DML_N_SPLITS = 5            # Cross-fitting folds for Double ML
CAUSAL_FOREST_N_ESTIMATORS = 200  # Number of trees in Causal Forest

# 11 canonical factors from the multi-factor project.
FACTORS: List[str] = [
    "EP",
    "BP",
    "MOM20",
    "MOM60",
    "ROE",
    "ROA",
    "VOL20",
    "TURN20",
    "LIQ20",
    "GROWTH",
    "REV5",
]

# Because ROA is derived as a perfect linear scaling of ROE in our proxy data,
# including both in the causal-discovery covariance matrix makes it singular.
# We keep ROA in the modelling factor set but exclude it from causal discovery.
CAUSAL_DISCOVERY_FACTORS: List[str] = [f for f in FACTORS if f != "ROA"]

FACTOR_DESCRIPTIONS: Dict[str, str] = {
    "EP": "盈利收益率 Earnings Yield = 1/PE",
    "BP": "账面市值比 Book-to-Price = 1/PB",
    "MOM20": "20日动量 (剔除最近1日)",
    "MOM60": "60日动量 (剔除最近1日)",
    "ROE": "净资产收益率",
    "ROA": "总资产收益率",
    "VOL20": "20日收益波动率 (年化)",
    "TURN20": "20日平均换手率",
    "LIQ20": "20日平均成交量 (对数)",
    "GROWTH": "净利润同比增长率",
    "REV5": "5日短期反转",
}

# Prior signs: +1 means higher factor -> higher expected return.
# EP/BP/MOM/ROE/ROA/GROWTH are positive; VOL/TURN are negative;
# LIQ20 is ambiguous but often negative (liquidity premium);
# REV5 is positive (past losers rebound).
FACTOR_PRIOR_SIGN: Dict[str, int] = {
    "EP": 1,
    "BP": 1,
    "MOM20": 1,
    "MOM60": 1,
    "ROE": 1,
    "ROA": 1,
    "VOL20": -1,
    "TURN20": -1,
    "LIQ20": -1,
    "GROWTH": 1,
    "REV5": 1,
}


def ensure_dirs():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_dirs()
    print("Config OK")
