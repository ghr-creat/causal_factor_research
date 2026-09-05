"""End-to-end orchestration for the causal factor research project (final production version).

This version adds centralized logging, reproducible random seeds, and extended tests,
and generates a LaTeX-based bilingual PDF report instead of an HTML/Playwright report.
"""
import argparse
import os
import sys
import time
import warnings
from pathlib import Path

# Disable tqdm progress bars from causal-learn and other libraries before any imports
os.environ["TQDM_DISABLE"] = "1"

# Use a non-interactive matplotlib backend to avoid tkinter threading errors in
# headless / multi-threaded environments (e.g., XGBoost/CausalForest plotting).
import matplotlib
matplotlib.use("Agg")

from loguru import logger

# Add project root to path so existing backtest/config modules can be imported
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from causal_factor_research.config import PROCESSED_DIR, RESULTS_DIR, RANDOM_SEED, ensure_dirs
from causal_factor_research.data_builder import build_dataset
from causal_factor_research.causal_discovery import run_all_discovery
from causal_factor_research.causal_effect import estimate_all_effects, plot_method_comparison, plot_ite_distribution
from causal_factor_research.factor_comparison import run_factor_comparison
from causal_factor_research.backtest import run_all_backtests
from causal_factor_research.robustness import run_all_robustness
from causal_factor_research.identification import run_all_identification
from causal_factor_research.report import generate_report
from causal_factor_research.utils import load_features, set_random_seeds, split_train_test
from causal_factor_research.logging_config import setup_logging


def main(seed: int = RANDOM_SEED):
    # Setup logging first so every module logs to file and console.
    setup_logging()
    # Suppress noisy third-party warnings (sklearn, statsmodels, causal-learn) to keep logs readable.
    warnings.filterwarnings("ignore")
    set_random_seeds(seed)
    logger.info(f"Random seed set to {seed}")

    logger.info("=" * 60)
    logger.info("因果推断 + 因子发现 · 生产级全流程")
    logger.info("=" * 60)

    ensure_dirs()

    # Step 1: Build dataset (skip if processed features already exist and raw data is unavailable)
    processed_path = PROCESSED_DIR / "causal_features.parquet"
    if processed_path.exists():
        logger.info("[Step 1/8] 已存在 processed 数据，跳过 build_dataset()")
    else:
        logger.info("[Step 1/8] 构建 11 因子真实数据集...")
        t0 = time.time()
        build_dataset()
        logger.info(f"Dataset built in {time.time() - t0:.2f}s")

    # Step 2: Causal discovery (use training data to avoid data leakage)
    logger.info("[Step 2/8] 因果发现...")
    df = load_features()
    train_df, test_df = split_train_test(df)
    run_all_discovery(train_df)

    # Step 3: Causal effect estimation (strictly on training data)
    logger.info("[Step 3/8] 因果效应估计...")
    res = estimate_all_effects(train_df)
    plot_method_comparison(res)
    plot_ite_distribution(train_df, res)

    # Step 4: Factor comparison (train/test split already loaded)
    logger.info("[Step 4/8] 因果 vs 相关因子对比...")
    run_factor_comparison(train_df=train_df, test_df=test_df)

    # Step 5: Backtest (out-of-sample on test data only)
    logger.info("[Step 5/8] 组合回测...")
    run_all_backtests(train_df=train_df, test_df=test_df)

    # Step 6: Robustness (on training data)
    logger.info("[Step 6/8] 稳健性检验...")
    run_all_robustness(train_df=train_df, n_seeds=20)

    # Step 7: Identification diagnostics
    logger.info("[Step 7/8] 数学严谨性诊断...")
    run_all_identification(train_df)

    # Step 8: Report
    logger.info("[Step 8/8] 生成 LaTeX PDF 报告（中英双版本）...")
    generate_report()

    logger.info("=" * 60)
    logger.info(f"全部完成。结果目录：{RESULTS_DIR}")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="因果因子发现生产级流程")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="全局随机种子")
    args = parser.parse_args()
    main(seed=args.seed)
