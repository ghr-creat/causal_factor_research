import sys
sys.path.insert(0, r'C:\Users\郭浩然\ZCodeProject\QuantResearch\causal_factor_research')
import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')

from causal_factor_research.utils import load_features, split_train_test
from causal_factor_research.factor_comparison import run_factor_comparison
from causal_factor_research.backtest import run_all_backtests
from causal_factor_research.robustness import run_all_robustness
from causal_factor_research.config import RESULTS_DIR
import pandas as pd

print('Loading...')
df = load_features()
train, test = split_train_test(df)

print('Factor comparison...')
run_factor_comparison(train_df=train, test_df=test)

print('Backtest...')
run_all_backtests(train_df=train, test_df=test)

print('Robustness with reduced seeds...')
run_all_robustness(train_df=train, n_seeds=5)

print('Done')
print('Checking outputs:')
perm = pd.read_csv(RESULTS_DIR / 'time_permutation_test.csv')
print('Time permutation:')
print(perm[['factor', 'ate', 'pvalue']].to_string(index=False))
print()
placebo_summary = pd.read_csv(RESULTS_DIR / 'placebo_test_summary.csv')
print('Placebo summary:')
print(placebo_summary.to_string(index=False))
print()
matrix = pd.read_csv(RESULTS_DIR / 'factor_classification_matrix.csv')
print('2x2 matrix:')
print(matrix.to_string(index=False))
print()
decay = pd.read_csv(RESULTS_DIR / 'ic_decay_group_summary.csv')
print('Decay group summary:')
print(decay.to_string(index=False))
