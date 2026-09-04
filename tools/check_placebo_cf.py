import pandas as pd
placebo = pd.read_csv(r'C:\Users\郭浩然\ZCodeProject\QuantResearch\causal_factor_research\results\causal_final\placebo_test.csv')
noise_cf = placebo[(placebo['placebo_type']=='noise') & (placebo['method']=='CausalForest')]
print(noise_cf[['seed', 'ate', 'pvalue', 'cf_valid', 'ite_std', 'ite_q01', 'ite_q99']].to_string(index=False))
shuffled_cf = placebo[(placebo['placebo_type']=='shuffled') & (placebo['method']=='CausalForest')]
print('--- shuffled ---')
print(shuffled_cf[['seed', 'ate', 'pvalue', 'cf_valid', 'ite_std', 'ite_q01', 'ite_q99']].to_string(index=False))
