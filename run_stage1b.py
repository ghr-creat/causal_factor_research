import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import warnings
warnings.filterwarnings("ignore")

from causal_factor_research.config import RANDOM_SEED, ensure_dirs
from causal_factor_research.utils import load_features, set_random_seeds, split_train_test
from causal_factor_research.causal_effect import estimate_all_effects, plot_method_comparison, plot_ite_distribution
from causal_factor_research.factor_comparison import run_factor_comparison
from causal_factor_research.backtest import run_all_backtests
from causal_factor_research.logging_config import setup_logging

setup_logging()
set_random_seeds(RANDOM_SEED)
ensure_dirs()

df = load_features()
train_df, test_df = split_train_test(df)

res = estimate_all_effects(train_df)
plot_method_comparison(res)
plot_ite_distribution(train_df, res)
run_factor_comparison(train_df=train_df, test_df=test_df)
run_all_backtests(train_df=train_df, test_df=test_df)
print("Stage 1b completed: causal effects -> comparison -> backtest")
