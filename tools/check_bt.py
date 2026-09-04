import sys

path = r"C:\Users\郭浩然\ZCodeProject\QuantResearch\causal_factor_research\backtest.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

old = "def _causal_signs"
print("Old found:", old in content)
print("Position:", content.find(old))
print("Lines 42-66:")
lines = content.splitlines()
for i, line in enumerate(lines[41:67], 42):
    print(f"{i}: {repr(line)}")
