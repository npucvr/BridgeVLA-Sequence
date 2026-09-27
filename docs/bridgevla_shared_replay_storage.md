# BridgeVLA 共享数据存储

## 工程定位

共享存储、episode 编码和 NFS 直读是为了支持快速、可复现的轻量化微调迭代的工程实现，不是模型或训练方法创新，也不改变既有任务、采样和评估协议。性能结果用于运维和工程选型，不应被解释为算法增益。

## 目的

`/remote_databuffer` 只维护原始 RLBench 数据和经过验收的正式派生数据。原始数据是重建训练编码和评估封装的唯一权威来源；训练和评估统一从 NFS 只读，节点本地目录不承载数据副本。

## 当前目录约定

```text
/remote_databuffer/BridgeVLA/
  original/v1/                 # RLBench_TRAIN_DATA 和 RLBench_EVAL_DATA，只读
  encoded_train_v2/            # episode-aware 训练编码（验收后才启用）
  eval_pack/                   # 规划中的评估封装，尚未实现
```

`encoded_v1` 是已废弃的历史产物，不属于正式数据源，也不属于任何兼容或回滚路径。训练入口会拒绝 `BRIDGEVLA_REPLAY_FORMAT=encoded_v1`；新的编码器不从它转换，也不以它作为验收基线。已有目录可以保留用于历史审计，但不得用于新训练、评估或性能基准。

原始数据使用可断点续传的脚本同步：

```bash
source scripts/bridgevla_runtime.sh
bash scripts/sync_bridgevla_data_to_databuffer.sh \
  /data/local_userdata/sunguodong/BridgeVLA/data \
  /remote_databuffer/BridgeVLA/original/v1
```

脚本默认只同步 `RLBench_TRAIN_DATA` 和 `RLBench_EVAL_DATA`，不会同步 legacy replay。脚本不使用 `--delete`，不会删除目标端已有文件；同步完成后应单独核对文件数、字节数和校验清单。原始目录的一次性迁移可以设置 `BRIDGEVLA_SYNC_JOBS=8` 按 task 并行断点续传，以减少单进程 metadata 瓶颈；这只是迁移参数，训练和评估不使用本地源目录。

## 训练编码设计

`encoded_train_v2` 从 `RLBench_TRAIN_DATA` 重新构建，最小存储单位是完整 episode 或包含若干完整 episode 的 chunk。manifest 至少记录：

- `split=train`、task、episode ID 和起止位置；
- action、observation、terminal/timeout 的字段 schema、shape、dtype 和预处理版本；
- chunk byte range、校验值以及构建参数；
- 18-task uniform 所需的 task/episode 索引。

FilterCorrection 的 posterior、covariance、hidden state 和其他依赖 checkpoint 的状态不写入编码文件，必须在训练时从 episode 起点重建。训练 reader 应提供完整 episode、固定窗口加 burn-in 和必要的 transition smoke 接口，但不需要兼容 `encoded_v1` 的字段布局。

在 `encoded_train_v2` 完成 reader、sequence mask、18-task 采样和 NFS cold/warm 基准前，不启用新的正式 encoded 训练路径。基础 chunk reader 位于 [episode_store.py](/remote_userdata/sunguodong/repos/BridgeVLA/finetune/RLBench/utils/episode_store.py)，当前布局是一个 `manifest.json` 加少量 `chunks/chunk-*.bin`，episode 索引内嵌在 manifest 中。原始 episode 到该基础格式的入口是 [build_bridgevla_episode_store.py](/remote_userdata/sunguodong/repos/BridgeVLA/scripts/build_bridgevla_episode_store.py)；它只接受 `/remote_databuffer` 的 NFS 路径。现有 legacy replay 不作为共享依赖，也不复制到节点本地。

直读 benchmark 位于 [benchmark_bridgevla_episode_store.py](/remote_userdata/sunguodong/repos/BridgeVLA/scripts/benchmark_bridgevla_episode_store.py)，会拒绝非 NFS 文件系统，并输出 hostname、mount、chunk open 次数和 episode 吞吐。例如在一条已同步的 `close_jar/episode0` 上，原始布局包含 2328 个文件、约 32.5 MB；两次逐文件读取耗时约 4.07 s/0.61 s，临时单 chunk 读取耗时约 0.098 s/0.083 s。这是只针对 NFS 小文件与 chunk I/O 的 smoke，不代表模型训练吞吐，也不替代完整 v2 字段和 sequence 验收。

构建前先确认原始同步已完成，再使用类似命令做小范围 smoke：

```bash
source scripts/bridgevla_runtime.sh
python scripts/build_bridgevla_episode_store.py \
  --data-root /remote_databuffer/BridgeVLA/original/v1/RLBench_TRAIN_DATA \
  --destination /remote_databuffer/BridgeVLA/encoded_train_v2_smoke \
  --tasks close_jar \
  --max-episodes-per-task 1
python scripts/benchmark_bridgevla_episode_store.py \
  --store /remote_databuffer/BridgeVLA/encoded_train_v2_smoke \
  --episodes 1 --repeats 3 --cache-episodes 2 --verify-chunks
```

smoke 通过后仍需完成 18-task 构建、逐字段对照、sequence adapter 和多 worker cold/warm 测试，才能把目标目录命名为正式 `encoded_train_v2` 并接入训练。

## 评估数据设计

当前 [eval.py](/remote_userdata/sunguodong/repos/BridgeVLA/finetune/RLBench/eval.py) 将 `RLBench_EVAL_DATA` 作为 RLBench 环境的 `dataset_root`，按 task、variation 和 episode reset 环境，而不是读取训练 replay。因此评估侧第一版继续保留原始 task/variation/episode 布局，不转换为训练用 replay。

若确认评估端的小文件 I/O 成为瓶颈，再设计独立的 `eval_pack`，以 task/variation/episode 为不可分单元，并提供兼容 RLBench `dataset_root` 的只读 adapter。它必须通过环境启动、固定 episode reset、逐帧观测和视频/CSV 对照后才能使用。

训练和评估必须分别维护 manifest、source hash 和 split 标记。评估数据不得出现在训练编码的 fallback 搜索路径中；每次结果必须记录评估 manifest hash、episode 范围和 25/35 步协议。

## 运行规则

- 训练、评估或基准启动前，在同一 shell 核对 hostname、仓库路径、GPU、NFS 挂载、数据路径、manifest hash 和可读性。
- NFS 是唯一正式数据读取路径；优化应针对 NFS 的 chunk、range read、cache 和 worker 预取，不再比较或依赖节点本地数据 staging。
- NFS 不可用、manifest 不完整或校验失败时直接停止任务；正式结果、manifest 和日志仍写回 NFS。
- 不删除或覆盖历史 `encoded_v1` 产物，但不得重新启用它。
