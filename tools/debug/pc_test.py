import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, "C:/Users/郭浩然/ZCodeProject/attention_factor_system")

from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import fisherz

print("Loading data...", flush=True)
df = pd.read_parquet('C:/Users/郭浩然/ZCodeProject/attention_factor_system/data/processed/causal_features.parquet')
df = df[df['trade_date'] < '2023-01-01']
cols = ['ep', 'bp', 'mom_20', 'vol_20', 'size', 'label']
data = df[cols].dropna().sample(300, random_state=42).values
print(f"Data shape: {data.shape}", flush=True)
print("Starting PC...", flush=True)
G = pc(data, 0.05, indep_test=fisherz, stable=True, uc_rule=0, mvpc=False)
print("PC done", flush=True)
