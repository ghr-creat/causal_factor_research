import sys
sys.path.insert(0, r'C:\Users\郭浩然\ZCodeProject\QuantResearch')
import warnings
warnings.filterwarnings('ignore')
from causal_factor_research.utils import load_features, split_train_test
from causal_factor_research.factor_comparison import classify_factors, factor_decay_analysis
from causal_factor_research.config import RESULTS_DIR
import pandas as pd

print('Loading...')
df = load_features()
print('Splitting...')
train, test = split_train_test(df)
print('Loading causal results...')
causal_results = pd.read_csv(RESULTS_DIR / 'causal_effects_train.csv')
print('Classifying...')
classification = classify_factors(train, test, causal_results)
print('Matrix:')
print(pd.DataFrame(classification['matrix']))
print('Categories:', classification['categories'])
print('Causal significant:', classification['causal_significant'])
print('IC significant:', classification['ic_significant'])
print('True factors:', classification['true_factors'])
print('Masked factors:', classification['masked_factors'])
print('Spurious factors:', classification['spurious_factors'])
print('Ineffective factors:', classification['ineffective_factors'])
print('Combined p-values:', classification['causal_combined_pvalue'])
print('Decay analysis...')
decay_df, decay_rate_df = factor_decay_analysis(test, classification['causal_significant'], classification['ic_significant'])
print(decay_rate_df[['factor', 'group', 'decay_rate', 'stability_score']])
print('Group summary:')
print(decay_rate_df.groupby('group').agg({'decay_rate': ['mean', 'median'], 'stability_score': ['mean', 'median'], 'factor': 'count'}))
print('Done')
