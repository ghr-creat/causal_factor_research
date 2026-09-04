import sys
sys.path.insert(0, r'C:\Users\郭浩然\ZCodeProject\QuantResearch\causal_factor_research')
import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')

from causal_factor_research.utils import load_features, split_train_test
from causal_factor_research.robustness import time_permutation_test
import time

print('Loading...')
df = load_features()
train, _ = split_train_test(df)
print('Running time permutation test for all 11 factors...')
t0 = time.time()
res = time_permutation_test(train, factors=['EP', 'BP', 'MOM20', 'MOM60', 'ROE', 'ROA', 'VOL20', 'TURN20', 'LIQ20', 'GROWTH', 'REV5'])
print(f'Done in {time.time()-t0:.1f}s')
print(res[['factor', 'ate', 'pvalue']].to_string(index=False))
