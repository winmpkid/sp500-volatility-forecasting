# 基于 HAR-RV 的标普 500 指数实现波动率预测

本项目用日度实现方差衡量标普 500 指数的波动，预测下一交易日的对数实现方差。建模从 Persistence 持续性基准出发，比较 HAR-RV 的 Day、Week、Month 特征，再根据验证集残差诊断检验 Lag2 是否有用。特征和模型在开发阶段确定，独立测试集用于最后的样本外评估。

最终模型为 HAR + Lag2。独立测试集上，其均方误差（MSE）和平均绝对误差（MAE）相较 Persistence 分别下降约 11.60% 和 7.16%；相较基础 HAR-RV，分别下降约 1.49% 和 0.61%。Lag2 的增益较小，测试期残差也仍存在自相关。

## 项目目标

- 仅用预测时点可获得的数据构造特征；
- 按时间顺序比较 Persistence、基础 HAR-RV 和 HAR + Lag2；
- 结合预测误差和残差诊断判断增加特征是否有用；
- 保存模型、元数据和预测记录，便于复查与复现。

## 数据与预测目标

项目数据位于：

```text
datasets/oxfordmanrealizedvolatilityindices.csv
```

当前建模对象为 `.SPX`。代码将原始变量 `rv5` 作为日度实现方差 RV，预测目标为：

```text
target(t) = log(RV(t+1))
```

对数变换压缩了实现方差的取值范围。本文将“波动”用于描述研究对象；模型中的 RV 及误差指标均以实现方差及其对数为口径。

## 特征构造

预测 `log(RV(t+1))` 时，只使用 t 日及以前能够获得的信息：

| 特征 | 定义 | 含义 |
|---|---|---|
| `log_rv_day` | `log(RV(t))` | 当日波动信息 |
| `log_rv_lag2` | `log(RV(t-1))` | 相对预测目标滞后两个交易日 |
| `log_rv_week` | `log(mean(RV(t-4:t)))` | 最近 5 个交易日平均实现方差的对数 |
| `log_rv_month` | `log(mean(RV(t-21:t)))` | 最近 22 个交易日平均实现方差的对数 |

最终模型形式为：

```text
log(RV(t+1))
    = beta_0
    + beta_day   * log_rv_day
    + beta_lag2  * log_rv_lag2
    + beta_week  * log_rv_week
    + beta_month * log_rv_month
    + error(t+1)
```

周度和月度特征先对实现方差求平均，再取对数。训练与预测共用特征计算函数，保持数据口径一致。

## 为什么使用时间序列验证

随机交叉验证可能用较晚的数据训练，再预测较早的样本，不符合实际预测顺序。本项目的每行特征截至 t 日，目标是 t+1 日；随机拆分相邻记录，还可能让验证目标出现在训练特征中。

项目采用以下措施减少信息泄漏：

- 按日期顺序，以 60%、20%、20% 为初始比例划分训练集、验证集和测试集，并剔除目标跨越边界的记录；
- 划分边界同时检查特征日期和目标日期；
- 开发阶段使用 `TimeSeriesSplit(n_splits=5, gap=1)`；
- 测试集不参与特征选择和模型选择；
- 模型确定后仅进行一次最终测试。

## 模型选择依据

Persistence 直接用当日 `log(RV)` 预测下一交易日。以此为基准，开发阶段依次比较 Day、Day + Week 和 Day + Week + Month，检查周度、月度特征是否降低预测误差。

基础 HAR-RV 的验证集残差在第二滞后期的 ACF 和 PACF 约为 0.088，因此将 `log_rv_lag2` 列为候选特征。加入 Lag2 后，验证集 5 阶和 10 阶 Ljung-Box 检验的 p 值从 0.05 以下升至 0.05 以上，即在 5% 显著性水平下，这两项检验不再拒绝残差无自相关的原假设。这并不说明残差完全独立。

按当前代码计算，加入 Lag2 后的五折平均 MSE 从约 0.350979 降至 0.345829，下降约 1.47%，五个验证折的 MSE 均下降。该比较保留 Day、Week 和 Month，仅增加 Lag2，支持在当前设定下保留这一特征。

残差诊断用于提出候选特征，交叉验证和消融比较用于选模。独立测试只检验已确定的模型，不用于反复调整特征。

## 最终测试结果

模型确定后，合并训练集和验证集重新拟合，再在 924 条独立测试样本上评估。下表指标均在对数实现方差尺度上计算，百分比表示误差相对 Persistence 的降幅。

| 模型 | MSE | MAE | MSE 降幅 | MAE 降幅 |
|---|---:|---:|---:|---:|
| Persistence | 0.478135 | 0.551313 | 0.00% | 0.00% |
| HAR | 0.429034 | 0.514996 | 10.27% | 6.59% |
| HAR + Lag2 | **0.422655** | **0.511852** | **11.60%** | **7.16%** |

相较基础 HAR，HAR + Lag2 的 MSE 下降约 1.49%，MAE 下降约 0.61%。验证、交叉验证和最终测试的改善方向一致，说明 Lag2 在现有样本中有小幅增益；是否具有统计显著性，还需进一步检验。

## 项目结构

```text
Quant/
├── reports/
│   └── HAR_RV项目报告.docx
├── datasets/
│   └── oxfordmanrealizedvolatilityindices.csv
├── outputs/
│   ├── har_lag2_model.joblib
│   ├── har_lag2_model_info.json
│   └── predictions.csv
├── src/volatility/
│   ├── data.py
│   ├── features.py
│   ├── validation.py
│   ├── models.py
│   ├── evaluation.py
│   ├── diagnostics.py
│   ├── cli.py
│   ├── benchmark.py
│   ├── final_test.py
│   └── predict.py
├── tests/
│   ├── test_features.py
│   ├── test_model_artifacts.py
│   └── test_validation.py
├── requirements.txt
├── run_pipeline.py
└── README.md
```

主要模块说明：

| 文件 | 作用 |
|---|---|
| `data.py` | 读取原始数据 |
| `features.py` | 构造训练和预测特征 |
| `validation.py` | 按时间顺序划分训练、验证和测试数据 |
| `models.py` | 训练模型并保存模型及元数据 |
| `evaluation.py` | 计算 MSE 和 MAE |
| `diagnostics.py` | 描述性统计、ACF、PACF、Q-Q 图和 Ljung-Box 检验 |
| `benchmark.py` | 特征消融、时间序列交叉验证和候选模型比较 |
| `final_test.py` | 独立测试、最终诊断和模型保存 |
| `predict.py` | 加载已保存模型并生成下一交易日预测 |

## 环境安装

本项目已在 Python 3.14.6 环境完成验证。在项目根目录创建虚拟环境并安装依赖：

```bash
cd /Users/wimpykid/Quant
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

模型信息文件记录了保存时的 scikit-learn 版本。加载时使用相同版本，可减少序列化兼容问题。

## 运行方法

在项目根目录运行完整流程：

```bash
cd /Users/wimpykid/Quant
python run_pipeline.py --stage all
```

默认使用无界面绘图模式，适合自动化运行。如果需要显示图形窗口：

```bash
python run_pipeline.py --stage all --interactive-plots
```

如只需重新运行某个阶段，可使用：

```bash
python run_pipeline.py --stage diagnostics
python run_pipeline.py --stage benchmark
python run_pipeline.py --stage final-test
python run_pipeline.py --stage predict
```

### 1. 查看训练期统计和收益率诊断

```bash
cd src/volatility
python cli.py
```

### 2. 运行特征消融和时间序列交叉验证

```bash
python benchmark.py
```

该步骤仅使用开发数据。完成模型选择后，再运行最终测试；不要根据测试结果反复调整特征。

### 3. 进行最终测试并保存模型

```bash
MPLBACKEND=Agg python final_test.py
```

执行后会更新：

```text
outputs/har_lag2_model.joblib
outputs/har_lag2_model_info.json
```

其中：

- `har_lag2_model.joblib` 保存可直接加载的回归模型；
- `har_lag2_model_info.json` 保存特征顺序、回归系数、训练区间、测试区间和评估指标。

### 4. 使用已保存模型生成预测

```bash
python predict.py
```

预测结果保存至：

```text
outputs/predictions.csv
```

预测记录按特征日期和模型保存时间去重。同一保存版本对同一特征日期重复预测时，会更新原记录；重新训练并保存的模型按新版本保留记录。

## 自动化测试

回到项目根目录后运行：

```bash
cd /Users/wimpykid/Quant
python -m unittest discover -s tests -v
```

测试检查以下行为：

- 预测特征保留最新的有效数据日期；
- 训练特征会删除尚无下一日目标的最后一行；
- `target` 确实等于下一条观测的 `log(RV)`；
- 训练、验证和测试边界不存在未来目标重叠；
- 保存模型的特征数量和顺序与 JSON 元数据一致；
- 保存后的模型能够重新加载并执行预测。

## 预测值的尺度

模型直接输出下一交易日的对数实现方差预测值：

```text
predicted_log_rv
```

对预测值取指数，得到原方差尺度上的数值：

```python
predicted_rv = np.exp(predicted_log_rv)
```

再取平方根，得到对应的波动率数值：

```python
predicted_volatility = np.sqrt(predicted_rv)
```

当前预测记录的特征时间为 `2018-06-26T23:00:00+00:00`，对应原始数据中的 `2018-06-27 00:00:00+01:00`。代码统一使用 UTC 时间。因此，`predictions.csv` 保存的是历史样本末端的下一交易日预测，不是当前市场预测。

## 模型局限

- 最终测试残差的 Ljung-Box 检验仍显著，说明模型尚未充分捕捉测试期的时间依赖；
- 线性模型对非线性关系和市场状态切换的刻画有限，也未单独纳入 VIX、成交量、宏观变量、跳跃或杠杆效应；
- `exp(predicted_log_rv)` 不直接等于原尺度的条件均值。若需要均值预测，应评估对数变换的偏差修正；
- 保存的模型仅用训练集和验证集拟合。如用于部署，可在冻结特征和模型设定后，另行用全部可用历史数据训练部署版本，并保留当前评估记录；
- 历史回测结果不代表未来表现，本项目不构成投资建议。

## 后续改进

- 增加 QLIKE 等波动率专用评价指标；
- 使用 Diebold-Mariano 检验比较模型预测误差；
- 比较 Ridge、Lasso、GARCH、EGARCH 或非线性模型；
- 加入外部市场变量并继续使用相同的时间序列验证框架；
- 扩充时间划分边界、无效数据和预测记录去重的测试；
- 对滚动窗口和扩展窗口重新训练方案进行比较。

## 项目结论

在当前样本和验证设定下，HAR-RV 的测试误差低于 Persistence，加入 Lag2 后进一步小幅下降。现有结果支持保留 Lag2，但尚不能据此判断其在其他市场阶段的表现；测试期残差自相关仍是需要继续处理的问题。
