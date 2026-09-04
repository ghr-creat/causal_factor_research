import sys
sys.path.insert(0, r'C:\Users\郭浩然\ZCodeProject\QuantResearch\causal_factor_research_extracted\causal_factor_research')

from causal_factor_research.utils import load_features, split_train_test
from causal_factor_research.robustness import time_permutation_test
from causal_factor_research.causal_effect import double_ml_effect
import pandas as pd
import numpy as np
import traceback

df = load_features()
train_df, _ = split_train_test(df)

# Test a single factor with full traceback
factor = 'EP'
try:
    df_perm = train_df.copy()
    df_perm[f'{factor}_perm'] = df_perm.groupby('ts_code')[factor].shift(-1)
    df_perm = df_perm[df_perm[f'{factor}_perm'].notna()].copy()
    df_perm = df_perm.drop(columns=[factor]).rename(columns={f'{factor}_perm': factor})
    res = double_ml_effect(df_perm, factor)
    print('Success:', res)
except Exception as e:
    print('Error:', e)
    traceback.print_exc()
