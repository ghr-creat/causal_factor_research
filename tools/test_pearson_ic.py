import sys
sys.path.insert(0, r'C:\Users\郭浩然\ZCodeProject\QuantResearch')
import warnings
warnings.filterwarnings('ignore')
from causal_factor_research.utils import load_features, split_train_test, compute_ic_summary
from causal_factor_research.config import FACTORS

print('Loading...')
df = load_features()
print('Splitting...')
train, test = split_train_test(df)
print('Pearson IC train:')
for f in FACTORS:
    s = compute_ic_summary(train, f, method='pearson')
    print(f'{f}: mean={s["mean_ic"]:.4f}, t={s["tstat"]:.2f}, p={s["pvalue"]:.4f}')
print('Pearson IC test:')
for f in FACTORS:
    s = compute_ic_summary(test, f, method='pearson')
    print(f'{f}: mean={s["mean_ic"]:.4f}, t={s["tstat"]:.2f}, p={s["pvalue"]:.4f}')
print('Done')
