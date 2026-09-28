# BridgeVLA replay 编码方案设计

## 工程定位

本方案是数据工程与运行基础设施，用于减少 NFS 小文件 metadata 开销、支持 episode 级读取并缩短轻量化微调的迭代等待。它不改变模型结构、FilterCorrection 算法、采样语义或评估协议，也不作为方法创新或论文贡献；所有吞吐和延迟数据只用于工程选型、回归和运行容量规划。

## 目标和当前边界

新的训练编码从 `original/archives` 中的 `RLBench_TRAIN_DATA` task 归档重新构建，直接面向 episode sequence 和 FilterCorrection 微调。历史 `encoded_v1` 已删除，不再作为训练输入、基线、回滚路径或兼容目标。

长期共享的数据源只保留 `original/archives` 下的 `RLBench_TRAIN_DATA` 和 `RLBench_EVAL_DATA` 归档，以及通过验收流程生成的独立训练/评估派生包。新的编码格式不从 legacy replay 或 `encoded_v1` 转换，也不承担旧布局兼容。

训练和评估统一从 `/remote_databuffer` 的 NFS 权威路径只读，不再设计节点本地数据 staging。v2 的 chunk 布局应优先减少小文件和随机 metadata 操作，并把 NFS 的连续 range read、worker cache 和预取作为主要优化对象。基础的长度前缀 chunk reader 已放在 [episode_store.py](/remote_userdata/sunguodong/repos/BridgeVLA/finetune/RLBench/utils/episode_store.py)，后续 encoder 和 sequence adapter 以它为底层，不再复用旧 v1 reader。

## 已确认的训练访问模式

| 路径 | 当前行为 | 对存储的要求 |
| --- | --- | --- |
| 默认训练 | `task_uniform`，batch size 4，`num_workers=3`；每个 transition 还会读取 next transition | 低延迟随机读取；避免每条记录一次 NFS open |
| 固定窗口 sequence | 目标 transition 前的 burn-in 加连续 forward window | 同一 task 内的连续 range read；不能跨 episode |
| 完整 episode sequence | 从 episode 起点读取整条 episode，短 episode padding 并生成 `valid_mask/loss_mask` | episode 起止索引；尽量一次或少数几次读取 |
| 验证 enumerate | 按 transition 顺序遍历 | 顺序吞吐；不能被随机索引结构拖慢 |
| DDP/DataLoader | 每个 rank 的 worker 是独立进程，各自维护文件句柄和 cache | 只读并发安全；避免共享写锁；控制每 worker 的 cache 和 RSS |

历史 `encoded_v1` 的统计只属于旧产物审计，不是新格式的设计约束。v2 的 episode 长度、chunk 大小和字段体积应从原始数据重新测量。

## 历史格式处理

`encoded_v1` 已删除。它的格式、读取性能和字段布局不属于 v2 的设计约束；任何显式选择 `encoded_v1` 的训练入口都必须失败，并提示从归档原始数据重建新的 `encoded_train_v2`。

## 工程实现方案（v2 原型）

先实现一个不依赖新的数据库服务的 **episode-aware、可索引 chunk** 格式：

```text
encoded_train_v2/
  manifest.json
  chunks/chunk-000000.bin
  chunks/chunk-000001.bin
  ...
```

当前基础 reader 将 task/episode 索引内嵌在 `manifest.json`，每条记录是长度前缀的 pickle payload；后续字段分组和更细的 byte range 可以在语义对照通过后加入同一契约。每个 task 的索引至少保存：

1. episode 的起止 transition、有效长度和 terminal/timeout；
2. global index 到 task/episode/chunk/local row 的映射；
3. 每个 chunk 内各字段组的 byte range、shape、dtype 和校验值；
4. schema/version、原始数据版本和构建参数。

每个 chunk 聚合一小组完整 episode，而不是固定切断 transition。初始候选取 16、32、64 个 episode 三档，实际大小由 NFS cold/warm 读和 worker RSS 决定。episode 长度变化时，index 仍然按真实边界工作，padding 只在 sampler 层完成。

字段先分成三组：

- `state_action`：action、reward、terminal、timeout、pose、low-dim、keypoint 和 action index；
- `visual`：各相机 RGB、depth、point cloud；
- `language_camera`：语言、语言 embedding、相机内外参。

reader 对随机 transition 可以只取需要的字段组，对完整 episode 则对同一 chunk 发起连续 range read；一个 batch 先按 chunk 排序合并读取，再恢复原始样本顺序。所有字段仍保持当前 dtype/shape，不在第一版中做量化、图像重编码或动作修改。

### NFS 直读实现约束

第一版不把 NFS 当作一堆小文件的共享磁盘，而把它当作只读的 chunk 存储：

- 训练启动只读取一次 manifest 和索引，不递归扫描 episode 目录；
- 一个 worker 保持少量 chunk 文件描述符，按 batch 将相邻 range 合并后用 `os.pread` 读取；
- 完整 episode 优先一次读取或少数几次读取，再在 CPU 侧解码成 sequence；不对每个 transition 单独 `open/stat`；
- DataLoader 只做有限的 worker 预取和 batch cache，避免多个 worker 同时重复拉取同一 chunk；
- 第一版以无压缩 chunk 作为稳定基线，之后再测 chunk 级 zstd，不能默认压缩一定更快；
- NFS 读失败、短读、manifest/hash 不一致必须让任务失败，不得回退到本地数据副本。

当前的 [build_bridgevla_episode_store.py](/remote_userdata/sunguodong/repos/BridgeVLA/scripts/build_bridgevla_episode_store.py) 已能从原始 RLBench 直接构建单 episode smoke：它按现有 heuristic keypoint 生成 action/observation 序列，预计算语言 embedding，并把 FilterCorrection hidden state 留在运行时。它还没有接入 `get_dataset.py` 的正式训练 adapter；在字段逐项对照、18-task uniform、sequence mask 和 multi-worker NFS 验收完成前，不能把它当作正式训练数据。

## FilterCorrection 的联合数据与训练契约

FilterCorrection 不是普通的独立 transition head。它在 episode 起点初始化 belief，使用当前观测做 measurement update，再用示范 action 预测下一步 prior。因此，滤波路线的训练单位应是**有序 episode 或不跨边界的有序窗口**，而不是把 transition 完全打散后逐条训练。

### 存储层与运行时状态的边界

- v2 从权威的原始 RLBench episode 构建；可以缓存经过固定版本预处理的视觉、语言和 action 字段，但必须记录预处理版本、shape、dtype 和校验值。
- `FilterCorrection` 的 posterior mean/covariance、hidden state 和任何由当前模型参数产生的滤波轨迹不写入数据集。它们依赖当前 checkpoint、随机增强和采样顺序，必须在训练或评估时从 episode 起点重建。
- 每个 episode 保存明确的起止位置、terminal/timeout、有效长度和 task；padding 只由 sampler 生成，并通过 `valid_mask`/`loss_mask` 排除。
- 一个 episode 可以作为一个连续 range 放入 CPU 缓冲或 chunk，不要求整个数据集或所有 episode 同时进入 GPU。

### 18-task uniform 的 episode 版本

滤波训练仍保持 18-task uniform，但采样层定义为：先均匀选择 task，再在该 task 内选择 episode，再连续读取该 episode。这样不会因为某个 task 的 episode 更长而隐式增加 task 权重。若实验需要 transition 加权或 episode-length 加权，必须显式记录为另一个 sampler 配置。

### 训练接口

1. 默认滤波路线使用完整 episode sequence；序列起点初始化 hidden state，按时间顺序执行观测更新和 action transition。
2. `bptt_length=1` 可以作为显存友好的初始设置：它截断旧历史的梯度，但不截断 hidden state 的数值传递。更长的 BPTT 作为独立消融，不改变数据顺序契约。
3. episode 过长时才使用固定窗口加 burn-in。窗口必须由 terminal/timeout 截断，burn-in 只恢复状态，不计 action loss。
4. 普通 action head 或 action LoRA 可以继续使用 transition sampler；只有启用 hidden-state/filter 路线时才切换到 chronological sequence sampler。
5. action loss、filter innovation loss 和可选 LoRA 分支的开关、权重、更新步数分别记录，避免把数据读取变化和参数路线变化混为一个实验。

这意味着 v2 reader 应优先提供 `sample_episode(task_uniform)` 和连续字段读取接口，同时保留 `sample_transition` 作为兼容路径。训练器接收的是 `[B, L, ...]` 及其 mask，reader 负责高效加载，滤波状态只在模型运行时产生。

## 训练编码与评估数据分离

训练和评估虽然可以共享 task 名称、相机配置和字段定义，但不应共享同一个物理数据包或采样器。两者的访问契约不同：

| 数据 | 当前入口 | 主要目标 | 推荐布局 |
| --- | --- | --- | --- |
| `RLBench_TRAIN_DATA` | replay/sequence trainer | 18-task uniform、重复采样、episode 序列、action loss | `encoded_train_v2`：episode-aware、字段分组、可连续读取 |
| `RLBench_EVAL_DATA` | `eval.py` 的 `CustomMultiTaskRLBenchEnv(dataset_root=...)` | 固定 held-out episode、环境 reset、可追溯视频和 CSV | 原始 task/variation/episode 布局；必要时再做逐 episode 的只读封装 |

当前 `eval.py` 不从 training replay 取样，而是把 `eval_datafolder` 交给 RLBench 环境，并通过 `eval_demo_seed` 和 episode 编号选择评估样例。因此，评估数据第一版不应转换成训练用的 transition replay，也不应接受训练增强、task sampler 或预计算 hidden state。

评估侧如果以后遇到大量小文件带来的 I/O 瓶颈，可以单独设计 `eval_pack`：以 task/variation/episode 为最小不可分单元，保存原始环境需要的文件和索引，并提供兼容 `dataset_root` 的只读 adapter。该封装必须先通过环境启动、固定 episode reset、观测逐帧一致和视频/CSV 对照，不能因为打包方便而改变 episode 选择或仿真语义。

两套编码必须共享但分别校验以下元数据：task 列表、相机和分辨率、预处理版本、episode ID、source hash、horizon 协议和 terminal/timeout 定义。训练 manifest 记录 `split=train`，评估 manifest 记录 `split=held_out`；评估数据不能出现在训练编码的 fallback 搜索路径中。每次结果报告对应的 manifest hash、episode 范围以及 25/35 步协议，避免把新编码的分数混入旧汇总。

### 压缩策略

第一轮同时测 `none` 和低级别 zstd/deflate：

- 无压缩作为延迟基线；
- chunk 级压缩减少 NFS 字节数，但随机单行会付出整 chunk 解压成本；
- 不把每条 transition 单独压缩，否则会重新引入大量小随机操作；
- 只有在 exact-value 对照通过后，才考虑对 float64 depth/camera 做无损压缩或安全的字段去重。

相机矩阵、语言 embedding 等是否可以按 episode 去重，必须先统计同一 episode 内的值是否恒定；在证明之前不做语义假设。

## 候选技术取舍

| 候选 | 优点 | 当前风险 | 决定 |
| --- | --- | --- | --- |
| episode-aware 自描述 chunk | 直接匹配训练访问；无外部服务；可做 batch range read | 需要维护 index/header 和 reader | 当前工程候选 |
| Parquet/Arrow row group | 列投影和压缩成熟 | 大型多维 tensor、对象字符串和当前依赖栈适配成本较高 | 作为对照，不先绑定 |
| Zarr/HDF5/LMDB | 有现成 chunk 或 KV 语义 | NFS 多进程锁、metadata、mmap/page fault 行为需要单独验证 | 不作为第一版默认 |

## 验收实验

所有候选使用相同 task、相同 index、相同随机 seed 和相同 NFS 路径，分别记录 cold cache 和 warm cache：

1. 随机 transition：batch 4，`task_uniform`，读取 current/next；
2. 固定窗口：sequence length 4，burn-in 0/2；
3. 完整 episode：batch 4，包含短 episode padding；
4. enumerate：完整 task 顺序遍历；
5. worker 并发：`num_workers=0/3`，再用 2/4 个独立 rank 模拟 DDP；
6. 长序列：覆盖 `stack_blocks`、`place_cups`、`put_item_in_drawer` 等较长 episode。

每组至少记录：

- samples/s、单 batch P50/P95/P99 latency；
- NFS read bytes、shard/chunk open 次数、cache hit rate；
- worker RSS、GPU 等待比例和 DataLoader queue 空转比例；
- 与原始 RLBench episode 的逐字段语义、episode boundary、`valid_mask/loss_mask` 和 task distribution。

候选只有在原始数据语义对照通过，并满足 sequence/multi-worker 的吞吐验收后，才进入训练小规模验证。正式训练通过显式的 `encoded_train_v2` 配置选择格式；manifest 不完整、split 错误或 schema/hash 不匹配时必须拒绝启动。

## 实施顺序

1. 固定原始 RLBench episode 的字段、边界和 18-task 采样契约；
2. 从原始 RLBench 数据构建 `encoded_train_v2` 最小 encoder/index/reader；
3. 在 `/remote_databuffer` 上完成 cold/warm 与多 worker 对照；
4. 在两个代表性任务上跑短 sequence 微调，确认 GPU 等待、hidden-state 轨迹和 loss 行为；
5. 通过后将 v2 作为唯一正式 encoded 训练格式；
6. 历史 `encoded_v1` 与展开小文件树保持删除状态，不重新引入。
