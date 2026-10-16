# 直接预测网络线性输出头对照

本实验为论文的 DSM 残差修正架构提供补充对照。新增 M1-linear，复用既有 M1-softplus 和 M2 的逐种子结果。

## 文件放置与运行

将 `linear_head_control.py` 放到 `cc_0902` 目录，使用原项目的 Python 环境运行：

```powershell
python linear_head_control.py --check-only
python linear_head_control.py
```

脚本也支持从其他目录运行：

```powershell
python linear_head_control.py --project-dir "D:\PyCharm\Py_Projects\pinn\cc.pro\cc_0902"
```

默认结果写入 `cc_0902/linear_head_control_results/时间戳/`。每完成一次训练就更新结果表。运行目录包含脚本、数据和参照文件的 SHA-256，以及运行环境、特征名称、配置和数据划分。

## 对照定义

| 名称 | 预测方式 | 本次数据来源 |
|---|---|---|
| M1-softplus | 相同骨干的输出经 Softplus 后预测总承载力 | 原 `ablation_decompose_results.csv` 的 `MLP_F` |
| M1-linear | 使用 `ResidualCorrectionNet` 的线性输出直接预测总承载力，不叠加 DSM | 本次新增训练 |
| M2 | DSM 加网络输出的残差，无软约束 | 原 `ablation_decompose_results.csv` 的 `Full_RF` |

新训练使用 `ablation_efficiency.py` 的 `EFF_CFG` 和 `train_variant`，包括 400 轮上限、早停耐心 40、训练 Huber 阈值 30、验证 Huber 默认阈值 1。不同的训练与验证阈值沿用原实验设置，避免在本次输出头对照中混入其他修改。

规模为 50 和 725，各使用下采样种子 0–4，以及对应的网络种子 1000–1004。输入为含 DSM 特征的 26 维矩阵。验证集和测试集各 156 个试件，标准化器仅在每次训练子样本上拟合。M1-linear 的测试预测与原实验一样，进行下限为 0.001 kN 的裁剪；原始预测也单独保存。

脚本只在当前进程内临时替换直接预测网络的构造器，并在训练后恢复；主模型及原实验源文件不会被编辑。

## 输出与解释

- `linear_results.csv`：本次 10 次训练的指标、耗时和裁剪前负预测比例。
- `all_results.csv`：本次结果和历史 M1、M2 参照，`origin` 列区分两者。
- `summary.csv`：各规模、模型的均值、样本标准差和记录数。
- `paired_comparison.csv`：同规模、同种子的 R2 增益及 RMSE 降幅。
- `splits.csv`：逐次训练的训练、验证、测试试件编号。
- `predictions/`：每次训练的逐试件预测。
- `checkpoints/`：各次 M1-linear 最佳模型的参数；恢复预测时还需按清单拟合相应训练子集的标准化器。
- `manifest.json`：配置、运行环境和文件哈希。

`R2_gain_M2_over_linear` 大于零，表示 M2 的 R2 高于匹配线性输出头的直接模型；`RMSE_reduction_M2_over_linear` 大于零，表示 M2 的 RMSE 更低。

旧 M2 的 10 条结果在两份历史结果文件中经过一致性检查。原始历史运行没有保存划分和环境哈希，因此它们仍属于历史参照，本次没有重新训练 M1-softplus 或 M2。论文应说明参照的来源；不将五种子均值差直接称为统计显著差异，也不外推为所有网络或训练配置下的结论。
