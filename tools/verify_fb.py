import sys
sys.path.insert(0, r'C:\Users\郭浩然\ZCodeProject\QuantResearch\causal_factor_research')
import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')

from causal_factor_research.utils import load_features, split_train_test
from causal_factor_research.factor_comparison import run_factor_comparison
from causal_factor_research.backtest import run_all_backtests
from causal_factor_research.config import RESULTS_DIR
import pandas as pd
import time

print('Loading...')
df = load_features()
train, test = split_train_test(df)

print('Factor comparison...')
t0 = time.time()
run_factor_comparison(train_df=train, test_df=test)
print(f'  done in {time.time()-t0:.1f}s')

print('Backtest...')
t0 = time.time()
run_all_backtests(train_df=train, test_df=test)
print(f'  done in {time.time()-t0:.1f}s')

print('Done')
print('Checking outputs:')
matrix = pd.read_csv(RESULTS_DIR / 'factor_classification_matrix.csv')
print('2x2 matrix:')
print(matrix.to_string(index=False))
print()
decay = pd.read_csv(RESULTS_DIR / 'ic_decay_group_summary.csv')
print('Decay group summary:')
print(decay.to_string(index=False))
print()
metrics = pd.read_csv(RESULTS_DIR / 'backtest_metrics.csv')
print('Backtest metrics:')
print(metrics.to_string(index=False))
