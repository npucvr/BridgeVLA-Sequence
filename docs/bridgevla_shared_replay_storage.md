# BridgeVLA 共享数据存储

## 工程定位

共享存储、episode 编码和 NFS 直读是为了支持快速、可复现的轻量化微调迭代的工程实现，不是模型或训练方法创新，也不改变既有任务、采样和评估协议。性能结果用于运维和工程选型，不应被解释为算法增益。

## 目的

`/remote_databuffer` 只维护原始 RLBench 数据和经过验收的正式派生数据。原始数据是重建训练编码和评估封装的唯一权威来源；训练和评估统一从 NFS 只读，节点本地目录不承载数据副本。

模型权重不在 `/remote_databuffer`，而在仓库 NFS 路径 `/remote_userdata/sunguodong/repos/BridgeVLA/data/bridgevla_ckpt`（PaliGemma、BridgeVLA checkpoint 等）。节点本地不保留数据或权重副本；本地仅允许运行环境与短期缓存。

## 当前目录约定

```text
/remote_databuffer/BridgeVLA/
  original/archives/           # 权威原始数据：按 task 打包的 RLBench 归档，只读
    RLBench_TRAIN_DATA/<task>.tar.xz
    RLBench_EVAL_DATA/<task>.tar.xz
  encoded_train_v2/            # episode-aware 训练编码（已构建）
  eval_pack/                   # 评估元数据 pack（reset 高效直读，已构建）
```

原始数据的唯一权威形态是 `original/archives` 下的 task 归档，不再维护展开的小文件树（原 `original/v1` 已删除）。训练侧从归档流式读取并构建 `encoded_train_v2`；评估侧按需从归档 materialize 需要的 task/episode。

`encoded_v1` 已删除，不属于正式数据源，也不属于任何兼容或回滚路径。训练入口会拒绝 `BRIDGEVLA_REPLAY_FORMAT=encoded_v1`；新的编码器不从它转换，也不以它作为验收基线。

原始归档使用可断点续传的脚本同步：

```bash
source scripts/bridgevla_runtime.sh
bash scripts/sync_bridgevla_archives_to_databuffer.sh \
  /share/datasets/RLBench \
  /remote_databuffer/BridgeVLA/original/archives
```

脚本默认同步 `RLBench_TRAIN_DATA` 和 `RLBench_EVAL_DATA` 的全部 task 归档；按大小跳过已完成项，中断后可重跑。注意评估侧归档扩展名虽为 `.tar.xz`，实际可能是 gzip，且内部带历史 HDFS 前缀，读取时需自动识别。`BRIDGEVLA_ARCHIVE_JOBS` 控制并行度。旧的展开树同步脚本 `sync_bridgevla_data_to_databuffer.sh` 仅保留作历史参考，不再使用。

## 训练编码设计

`encoded_train_v2` 从 `original/archives` 中的 `RLBench_TRAIN_DATA` 归档重新构建，最小存储单位是完整 episode 或包含若干完整 episode 的 chunk。manifest 至少记录：

- `split=train`、task、episode ID 和起止位置；
- action、observation、terminal/timeout 的字段 schema、shape、dtype 和预处理版本；
- chunk byte range、校验值以及构建参数；
- 18-task uniform 所需的 task/episode 索引。

FilterCorrection 的 posterior、covariance、hidden state 和其他依赖 checkpoint 的状态不写入编码文件，必须在训练时从 episode 起点重建。训练 reader 应提供完整 episode、固定窗口加 burn-in 和必要的 transition smoke 接口，但不需要兼容 `encoded_v1` 的字段布局。

在 `encoded_train_v2` 完成 reader、sequence mask、18-task 采样和 NFS cold/warm 基准前，不启用新的正式 encoded 训练路径。基础 chunk reader 位于 [episode_store.py](/remote_userdata/sunguodong/repos/BridgeVLA/finetune/RLBench/utils/episode_store.py)，当前布局是一个 `manifest.json` 加少量 `chunks/chunk-*.bin`，episode 索引内嵌在 manifest 中。归档到该基础格式的入口是 [build_bridgevla_episode_store.py](/remote_userdata/sunguodong/repos/BridgeVLA/scripts/build_bridgevla_episode_store.py) 和 [tar_episode_source.py](/remote_userdata/sunguodong/repos/BridgeVLA/finetune/RLBench/utils/tar_episode_source.py)；它们只接受 `/remote_databuffer` 的 NFS 路径，直接从 task 归档流式读取 episode，不展开小文件。现有 legacy replay 不作为共享依赖，也不复制到节点本地。

直读 benchmark 位于 [benchmark_bridgevla_episode_store.py](/remote_userdata/sunguodong/repos/BridgeVLA/scripts/benchmark_bridgevla_episode_store.py)，会拒绝非 NFS 文件系统，并输出 hostname、mount、chunk open 次数和 episode 吞吐。例如在一条已同步的 `close_jar/episode0` 上，原始布局包含 2328 个文件、约 32.5 MB；两次逐文件读取耗时约 4.07 s/0.61 s，临时单 chunk 读取耗时约 0.098 s/0.083 s。这是只针对 NFS 小文件与 chunk I/O 的 smoke，不代表模型训练吞吐，也不替代完整 v2 字段和 sequence 验收。

构建前先确认原始归档同步已完成，再使用类似命令做小范围 smoke：

```bash
source scripts/bridgevla_runtime.sh
python scripts/build_bridgevla_episode_store.py \
  --archives-root /remote_databuffer/BridgeVLA/original/archives/RLBench_TRAIN_DATA \
  --destination /remote_databuffer/BridgeVLA/encoded_train_v2_smoke \
  --tasks close_jar \
  --max-episodes-per-task 1
python scripts/benchmark_bridgevla_episode_store.py \
  --store /remote_databuffer/BridgeVLA/encoded_train_v2_smoke \
  --episodes 1 --repeats 3 --cache-episodes 2 --verify-chunks
```

smoke 通过后仍需完成 18-task 构建、逐字段对照、sequence adapter 和多 worker cold/warm 测试，才能把目标目录命名为正式 `encoded_train_v2` 并接入训练。

## 评估数据设计

评估 reset 的确定访问单位是 episode；`reset_to_demo` 只需要 `low_dim_obs.pkl` 中的 `Demo.random_seed` / 观测、`variation_number` 与语言描述，**不读取相机 PNG**。因此评估不使用展开小文件树，也不在评估时临时解压。

`eval_pack` 是评估侧正式编码（`/remote_databuffer/BridgeVLA/eval_pack`）：

- 每个 episode 一个小 tar，仅含 `low_dim_obs.pkl`、`variation_number.pkl`、`variation_descriptions.pkl`；
- [eval_pack.py](/remote_userdata/sunguodong/repos/BridgeVLA/finetune/RLBench/utils/eval_pack.py) 提供与官方 `get_stored_demos` 语义一致的加载接口；
- [custom_rlbench_env.py](/remote_userdata/sunguodong/repos/BridgeVLA/finetune/RLBench/utils/custom_rlbench_env.py) 的 `reset_to_demo` 在 `BRIDGEVLA_EVAL_PACK` 或 pack `manifest.json` 存在时直接读取 pack，路径兼容官方 `dataset_root` 展开树；
- 构建入口：[build_bridgevla_eval_pack.py](/remote_userdata/sunguodong/repos/BridgeVLA/scripts/build_bridgevla_eval_pack.py)，源为 `original/archives/RLBench_EVAL_DATA`。

已验收：18 task × 25 = 450 episodes，约 535MB；与官方归档加载对比 `random_seed` / `variation_number` / 描述 / 步数一致；单 episode 元数据加载约 15–140ms。完整评估若还需要图像可视化，再从 `original/archives` 按需读取，不进入 reset 快速路径。

训练和评估必须分别维护 manifest、source hash 和 split 标记。评估数据不得出现在训练编码的 fallback 搜索路径中；每次结果必须记录评估 manifest hash、episode 范围和 25/35 步协议。

## 运行规则

- 训练、评估或基准启动前，在同一 shell 核对 hostname、仓库路径、GPU、NFS 挂载、数据路径、manifest hash 和可读性。
- NFS 是唯一正式数据读取路径；优化应针对 NFS 的 chunk、range read、cache 和 worker 预取，不再比较或依赖节点本地数据 staging。
- NFS 不可用、manifest 不完整或校验失败时直接停止任务；正式结果、manifest 和日志仍写回 NFS。
- 历史 `encoded_v1` 与展开小文件树已删除；不得重新引入或启用。
