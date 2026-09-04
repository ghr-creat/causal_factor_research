import sys
sys.path.insert(0, r'C:\Users\郭浩然\ZCodeProject\QuantResearch')

from causal_factor_research.utils import load_features, split_train_test
from causal_factor_research.robustness import time_permutation_test
import pandas as pd
import numpy as np
import traceback

df = load_features()
train_df, _ = split_train_test(df)

try:
    res = time_permutation_test(train_df, factors=['EP', 'BP'])
    print(res)
except Exception as e:
    print('Error in time_permutation_test:', e)
    traceback.print_exc()
