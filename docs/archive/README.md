# docs/archive — 已归档文档

本目录存放**不再作为当前方法主线**的历史文档。归档只表示它们描述的方法路线已被取代，不代表其中记录的实验数字或建模结论作废：每一份文档仍然是它当时所测配置的真实记录。

## 归档原因

方法方向已确定为**滤波器式 token correction**，即把 token 修正形式化为测量空间中的贝叶斯滤波更新（增益 × 新息），可训练的只是滤波器模型本身。当前主线文档见 [`../bridgevla_filter_token_correction_design.md`](../bridgevla_filter_token_correction_design.md)。

下列文档描述的是被取代的 **hidden-state / prior-observation** 路线（$A_\psi(H_t,y_t^+)$ 形式的任意残差修正），因此归档保留。

## 归档清单

| 文件 | 标题 | 类型 | 归档前日期 |
|---|---|---|---|
| `bridgevla_probabilistic_robotics_model.md` | BridgeVLA 的概率机器人建模 | 设计 / 建模 | 2026-09-02 |
| `bridgevla_experiment_comparison.md` | BridgeVLA 当前实验、论文指标与训练成本对照 | 实验 | 2026-09-02 |
| `bridgevla_hidden_state_five_run_eval_report.md` | BridgeVLA 纯 token-only H-token 路线五轮完整 RLBench 实验报告 | 实验 | 2026-09-01 |
| `bridgevla_prior_observation_prediction_eval_report.md` | BridgeVLA Prior-Observation Hidden-State 实验结果汇报 | 实验 | 2026-09-02 |

## 使用方式

- 需要复核历史实验协议、checkpoint 路径或训练成本时，从本目录查阅；
- 不要基于这些文档推导当前方法的结论——它们描述的是已停止演进的路线；
- 引用本目录文档中的数字时，请连同该文档记录的配置一起引用。
