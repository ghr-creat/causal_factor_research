import sys

path = r"C:\Users\郭浩然\ZCodeProject\QuantResearch\causal_factor_research\backtest.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

start = content.find('def _causal_signs')
end = content.find('def _backtest_signs')
actual = content[start:end]
print('Start:', start, 'End:', end)
print('Actual length:', len(actual))
print('Actual first 200:', repr(actual[:200]))
print('Actual last 200:', repr(actual[-200:]))
print('---')
print(actual)
