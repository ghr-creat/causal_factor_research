import sys
sys.path.insert(0, r'C:\Users\郭浩然\ZCodeProject\QuantResearch\causal_factor_research')
import warnings
warnings.filterwarnings('ignore')

from causal_factor_research.robustness import _summarize_placebo
from causal_factor_research.config import RESULTS_DIR
import pandas as pd

placebo = pd.read_csv(RESULTS_DIR / 'placebo_test.csv')
summary = _summarize_placebo(placebo)
print(summary.to_string(index=False))
summary.to_csv(RESULTS_DIR / 'placebo_test_summary.csv', index=False, encoding='utf-8')
print('Updated placebo_test_summary.csv saved')
