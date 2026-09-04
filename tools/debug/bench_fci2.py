import pandas as pd
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.cit import fisherz
import time

df = pd.read_parquet('C:/Users/郭浩然/ZCodeProject/attention_factor_system/data/processed/causal_features.parquet')
variables = [c for c in df.columns if c not in ['ts_code','trade_date','close','open','label','future_ret_5']] + ['label']
print('vars', len(variables))
data = df[variables].dropna().sample(n=500, random_state=42).values
t0 = time.time()
G, _ = fci(data, alpha=0.05, independence_test_method=fisherz, verbose=False, show_progress=False)
print('FCI real 500 time', time.time() - t0, flush=True)
