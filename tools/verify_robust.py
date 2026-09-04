import sys
sys.path.insert(0, r'C:\Users\郭浩然\ZCodeProject\QuantResearch\causal_factor_research')
import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')

from causal_factor_research.utils import load_features, split_train_test
from causal_factor_research.robustness import run_all_robustness
import time

print('Loading...')
df = load_features()
train, _ = split_train_test(df)

print('Running robustness with n_seeds=5...')
t0 = time.time()
run_all_robustness(train_df=train, n_seeds=5)
print(f'  done in {time.time()-t0:.1f}s')
print('Done')
