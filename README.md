# causal_factor_research

生产级因果推断因子研究。在 11 个经典 A 股因子上结合**因果发现**（PC / FCI / Bootstrap 稳定性 / 距离相关性）与**因果效应估计**（DML / IV / Front-door / Causal Forest），回答"哪些因子真正因果驱动收益"，并提供：

- 2×2 因子分类（因果显著 × 传统 IC 显著）
- 三组组合回测
- 完整稳健性检验（Placebo / 时间置换 / 分市场 / Rosenbaum 敏感性）
- 统一的日志系统（loguru + 文件滚动）与全局随机种子管理（可复现）
- 单元 + 集成测试体系
- 基于 LaTeX 的中英双语 PDF 报告

## 输入

仓库已附带构建好的特征数据（21 个候选因子 + 收益标签，2018–2024）：

```text
data/processed/causal_features.parquet
```

如需从原始行情重建特征，可使用 `data_builder.py`。

## 输出

```text
results/causal_final/          # CSV / JSON 结果与图表
report/causal_final/           # LaTeX 源文件与 PDF 报告
results/causal_final/logs/     # 运行日志
```

## 环境安装

```bash
pip install -r requirements-causal.txt
```

> Windows 用户若使用 Python 3.7，建议从预编译 wheel 安装 `causal-learn`/`econml` 以避免编译问题。

## 运行完整流程

项目以包形式组织（模块内使用 `causal_factor_research.*` 绝对导入），克隆后请保持文件夹名 `causal_factor_research` 不变，然后在仓库的**父目录**下以模块方式运行：

```bash
git clone https://github.com/ghr-creat/causal_factor_research.git
cd causal_factor_research/..
python -m causal_factor_research.main --seed 42
```

也可以直接进入仓库目录运行入口脚本（脚本会自动把父目录加入 `sys.path`）：

```bash
cd causal_factor_research
python main.py --seed 42
```

日志会同时输出到控制台与：

```text
results/causal_final/logs/causal_pipeline.log
```

## 分阶段运行

```bash
python run_stage1a.py   # 阶段 1a：因果发现（PC / FCI / Bootstrap / 距离相关性）
python run_stage1b.py   # 阶段 1b：因果效应估计（DML / IV / Front-door / Causal Forest）
python run_stage2.py    # 阶段 2：稳健性检验
```

## 运行测试

```bash
pytest tests -v
```

## 报告生成

默认生成 `report/causal_final/report_cn.tex` 与 `report/causal_final/report_en.tex`。若系统已安装 XeLaTeX（并配置 PATH），会自动编译为 PDF：

```text
report_cn.pdf
report_en.pdf
```

未安装 XeLaTeX 时会保留 `.tex` 源文件，可在安装 MiKTeX/TeX Live 后手动编译：

```bash
cd report/causal_final
xelatex -interaction=nonstopmode report_cn.tex
xelatex -interaction=nonstopmode report_en.tex
```

## 结构

```text
causal_factor_research/
├── main.py                 # 入口，含日志初始化、随机种子、CLI
├── logging_config.py       # 统一日志配置
├── config.py               # 路径、因子、随机种子常量
├── utils.py                # 数据加载、IC、复现种子
├── data_builder.py         # 21 因子特征构建
├── causal_discovery.py     # PC / FCI / Bootstrap / Distance Correlation
├── causal_effect.py        # DML / IV / Front-door / Causal Forest
├── factor_comparison.py    # 2×2 分类 / 衰减 / 拥挤度
├── backtest.py             # 三组组合回测
├── robustness.py           # Placebo / 时间置换 / 分市场 / Rosenbaum
├── identification.py       # 后门 / SUTVA 诊断
├── report.py               # LaTeX 中英报告生成
├── requirements-causal.txt
├── README.md
└── tests/
    ├── conftest.py
    ├── test_basic.py
    ├── test_causal_discovery.py
    ├── test_causal_effect.py
    ├── test_factor_comparison.py
    ├── test_backtest.py
    ├── test_robustness.py
    └── test_report.py
```

## 许可证

MIT，见 [LICENSE](LICENSE)。
