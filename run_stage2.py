import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import warnings
warnings.filterwarnings("ignore")

from causal_factor_research.config import RANDOM_SEED, ensure_dirs
from causal_factor_research.utils import load_features, set_random_seeds, split_train_test
from causal_factor_research.robustness import run_all_robustness
from causal_factor_research.identification import run_all_identification
from causal_factor_research.report import generate_report
from causal_factor_research.logging_config import setup_logging

setup_logging()
set_random_seeds(RANDOM_SEED)
ensure_dirs()

df = load_features()
train_df, _ = split_train_test(df)

run_all_robustness(train_df=train_df)
run_all_identification(train_df)
generate_report()
print("Stage 2 completed: robustness -> identification -> report")
