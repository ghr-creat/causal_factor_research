import sys
sys.path.insert(0, r'C:\Users\郭浩然\ZCodeProject\QuantResearch')
import warnings
warnings.filterwarnings('ignore')
from causal_factor_research.utils import load_features, split_train_test
from causal_factor_research.robustness import time_permutation_test
import pandas as pd

print('Loading...')
df = load_features()
print('Splitting...')
train, _ = split_train_test(df)
print('Running time permutation test...')
res = time_permutation_test(train, factors=['EP', 'BP', 'MOM60', 'TURN20'])
print(res.to_string(index=False))
print('Done')
