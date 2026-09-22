# filt3r_akf 归档范围

主脚本只保留当前 `event_neutral`、`diagonal_robust_mahalanobis`、reset、decoder protection gate、sparse evidence 和 temporal/spatial/innovation/guarded evidence。

以下内容不再进入运行时主路径：

- 旧 Q：`constant`、`motion_jitter_adaptive`、`drift_adaptive_sigmoid`、`gain_space_adaptive`、`event_adaptive`、APNE 系列；
- 旧 drift scope、W4 四帧窗口和速度预测；
- jitter-aware/adaptive-R；
- trust-region 实际裁剪；当前 `token_shift_ratio` 只作为诊断量；
- APNE、selective gate、nominal floor、额外 Q penalty 及其配置诊断。
- 可选的 metric audit/W4 统计旁路；旧调用仍保留 no-op hook，不再生成 audit 结果。

历史 metric 实现仍保存在 [metric_candidates.py](./metric_candidates.py)。这些路径只供阅读和追溯，不应作为新的运行入口。

旧配置脚本如果仍设置 `q_mode=constant`、`drift_adaptive_sigmoid`、`gain_space_adaptive`、`r_mode=jitter_aware` 或速度预测参数，应一并视为历史实验配置；当前主脚本会明确拒绝这些设置。
