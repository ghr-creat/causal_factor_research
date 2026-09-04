import sys
sys.path.insert(0, r'C:\Users\郭浩然\ZCodeProject\QuantResearch\causal_factor_research')
import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
from causal_factor_research.utils import load_features, split_train_test
from causal_factor_research.causal_effect import causal_forest_effect
from causal_factor_research.config import RANDOM_SEED

# Load data
print('Loading...')
df = load_features()
train, _ = split_train_test(df)

# Create a noise factor
rng = np.random.default_rng(RANDOM_SEED)
train_noise = train.copy()
train_noise['noise'] = rng.standard_normal(len(train_noise))

print('Running CausalForest on noise...')
res = causal_forest_effect(train_noise, 'noise')
print('ATE:', res['ate'])
print('p-value:', res['pvalue'])
print('cf_valid:', res['cf_valid'])
print('ite_std:', res['ite_std'])
print('ite_q01:', res['ite_q01'])
print('ite_q99:', res['ite_q99'])

# Run on a real factor for comparison
print('Running CausalForest on TURN20...')
res2 = causal_forest_effect(train, 'TURN20')
print('ATE:', res2['ate'])
print('p-value:', res2['pvalue'])
print('cf_valid:', res2['cf_valid'])
print('ite_std:', res2['ite_std'])
print('ite_q01:', res2['ite_q01'])
print('ite_q99:', res2['ite_q99'])
