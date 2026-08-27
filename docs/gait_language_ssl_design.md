# Gait-Language SSL 当前端到端系统设计

本文档只描述当前仓库已经实现的主流程，并将尚未实现的内容明确列入后续工作。代码依据为：

```text
run.py
run_exp.py
analyze_vq.py
knee_kinematics/data/loading.py
knee_kinematics/data/repository.py
knee_kinematics/data/transforms/gait_cycle.py
knee_kinematics/data/transforms/quality_control.py
knee_kinematics/gait_language/data.py
knee_kinematics/gait_language/vq.py
knee_kinematics/gait_language/sentence.py
knee_kinematics/gait_language/models.py
knee_kinematics/gait_language/downstream.py
knee_kinematics/gait_language/trainer.py
```

## 1. 当前系统目标

当前系统先使用健康数据学习离散 gait-word vocabulary 和上下文 Sentence Encoder，再使用 KGKD 训练集中的健康受试者建立固定的 embedding-space normative reference，最后用分层偏移特征训练 Healthy/ACLD/KOA 三分类器。

```text
健康 SSL 数据
→ 自适应分周期、质量控制、统一标准化
→ 六个 DOF-specific VQ codebook
→ 健康 Gait-Language Sentence Encoder
→ KGKD-train Healthy normative reference
→ word / DOF / side / subject deviation
→ Healthy / ACLD / KOA classifier
→ internal / external evaluation
```

当前实现可以回答：

1. 每个有效周期、DOF 和侧别在健康 embedding 坐标中向什么方向偏移；
2. 偏移程度的 mean、RMS、std 和 max 分别多大；
3. 左右共同偏移和左右有符号不对称分别是什么；
4. 受试者属于 Healthy、ACLD 或 KOA 的预测概率；
5. 对具有患侧标签的样本，预测患侧为 left 或 right。

当前尚未实现 code-frequency surprise、transition surprise、按类别导出的 deviation table、概率校准和医学可视化，不能把这些规划项描述成现有系统输出。

## 2. 数据边界、冻结规则与坐标系

### 2.1 数据用途

| 数据 | 当前用途 | 是否参与参数拟合 |
|---|---|---|
| SSL Healthy train | 分周期/QC、标准化、VQ、SSL | 是 |
| SSL Healthy validation | VQ/SSL early stopping | 否，仅选择 checkpoint |
| KGKD `dev_data` | downstream QC、Healthy reference、三分类训练 | 是 |
| KGKD `dev_validation_data` | downstream checkpoint selection | 否，仅选择 checkpoint |
| KGKD internal test | 内部评估 | 否 |
| KGKD external test | 外部评估 | 否 |

标签映射在 `loading.py` 中固定为：

```text
Healthy = 0
ACLD    = 1
KOA     = 2
```

CSV 首先按真实受试者记录聚合为 subject array；`source_file` 只进入 trace/center metadata，不进入模型张量。

### 2.2 当前真实冻结与信息隔离规则

当前代码不是“完成 VQ 后把所有 encoder/decoder 永久统一冻结”，而是分阶段处理：

| 组件 | VQ Stage | SSL Stage | Downstream Stage |
|---|---|---|---|
| VQ `word_encoder` | 训练 | 作为 `target_tokenizer` 冻结 | 不直接用于 classifier |
| VQ codebook | EMA 更新 | 冻结并产生 target code | 不直接更新 |
| VQ decoder | 训练 | 冻结在 `target_tokenizer` 内 | 不进入 downstream model |
| Sentence Encoder 的 word encoder | 尚未创建 | 从 VQ word encoder 深拷贝后继续训练 | 默认冻结，可显式解冻 |
| Sentence Transformer blocks | 尚未创建 | 训练 | 默认冻结，可显式解冻 |
| SSL prediction/rhythm/pair heads | 尚未创建 | 训练 | 不进入 downstream model |
| KGKD Healthy normative reference | 尚未创建 | 尚未创建 | 训练前拟合一次，之后作为 buffer 固定 |
| Deviation projections/classifier | 尚未创建 | 尚未创建 | 有监督训练 |

`GaitLanguageSSLModel` 初始化时将整个 VQ tokenizer 作为 `target_tokenizer`，并把其所有参数设置为 `requires_grad=False`。Sentence Encoder 使用 `copy.deepcopy(target_tokenizer.word_encoder)`，因此其 word encoder 从健康 VQ 初始化，但在 SSL 中是可训练副本。

Downstream 当前默认：

```text
--freeze-sentence-encoder
```

即固定 SSL embedding 坐标，只训练 deviation projections 和分类头。使用 `--no-freeze-sentence-encoder` 可以解冻，但 KGKD Healthy reference 仍只在训练开始前拟合一次，不会随 encoder 更新而重新拟合。

### 2.3 统一波形坐标系

原始运动学标准化参数只使用通过 SSL-train 自适应分周期和 SSL-train QC 的健康周期拟合：

```text
x_std[d] = (x[d] - μ_SSL[d]) / σ_SSL[d]
```

同一 `KinematicDOFStandardizer` 应用于 SSL validation、KGKD train/validation、internal test 和 external test。KGKD 数据不会重新拟合波形 mean/std。

需要区分两类参考：

1. `KinematicDOFStandardizer`：SSL Healthy train 拟合，作用于 100 点波形；
2. `reference_mean/reference_std`：KGKD `dev_data` Healthy 拟合，作用于冻结 Sentence Encoder 的 contextual token。

二者处于不同层级，不能混为同一个标准化过程。

## 3. 全局符号、DOF 与张量定义

### 3.1 符号

| 符号 | 含义 | 默认值 |
|---|---|---:|
| `B` | batch size | 32 |
| `S` | side 数量 | 2 |
| `W` | 当前 batch 左右合并后的最大周期数 | 可变，且 `W≤32` |
| `C` | DOF 数量 | 6 |
| `T` | 每个 time-normalized cycle 的采样点数 | 100 |
| `D` | word/token embedding dimension | 128 |
| `K` | 每个 DOF 的 code 数量 | 128 |
| `H_dof` | downstream per-DOF hidden dimension | 64 |

Side index 固定为：

```text
0 = left
1 = right
```

### 3.2 原始 DOF 顺序与模型顺序

Repository 原始单侧顺序为：

```text
[VV, IE, FE, AP, SI/PD, ML]
```

`MODEL_DOF_ORDER=[2,0,1,3,5,4]` 将其重排为模型顺序：

```text
0 flexion_extension
1 adduction_abduction / varus_valgus
2 internal_external_rotation
3 anterior_posterior_translation
4 medial_lateral_translation
5 superior_inferior_translation
```

自适应分周期发生在原始顺序上，因此 `reference_dof_index=2` 指原始 FE；进入模型后的 index 2 已是 IE，二者不能混用。

### 3.3 当前 Bilateral sentence 的真实结构

Repository collate 先产生左右独立的 padded cycle tensor：

```text
left_cycles      [B, W_left_max, 6, T]
right_cycles     [B, W_right_max, 6, T]
left_cycle_mask  [B, W_left_max]
right_cycle_mask [B, W_right_max]
```

`build_language_batch` 再构造统一 sentence：

```text
words         [B, 2, W, 6, T]
word_mask     [B, 2, W]       bool
timing        [B, 2, W, 4]
disease_label [B]
affected_side_label      [B]
affected_side_valid_mask [B]
```

当前送入模型的 `timing[...,4]` 只有四个通道：

| Index | 内容 | 单位/范围 | 代码定义 |
|---:|---|---|---|
| 0 | cycle duration | seconds | `(end-start)/sampling_rate` |
| 1 | continuous center position | 约 `[0,1]` | `((start+end)/2)/recording_length` |
| 2 | preceding center interval | seconds | 当前 center 减前一 center；首周期复制第二周期 interval，单周期时使用 duration |
| 3 | segmentation quality | correlation score | adaptive similarity score |

`start_time`、`end_time` 和原始 boundary 保存在 `metadata.segmentation`，但不作为独立 timing channel 输入模型。左右周期独立检测，不复制、不截断成相同有效数量，也不假设采集同步；仅在 batch 维度 padding 到同一个 `W`。

## 4. 当前端到端执行入口

`run.py` 支持：

| `--stage` | 当前行为 |
|---|---|
| `all` | 训练 VQ → 训练 SSL → 拟合 reference/训练 downstream → 评估 |
| `vq` | 训练 VQ 后停止 |
| `ssl` | 加载 VQ，训练 SSL 后停止 |
| `ssl_downstream` | 加载固定 VQ，训练 SSL 和 downstream，随后评估 |
| `downstream` | 加载固定 VQ/SSL，训练 downstream，随后评估 |
| `evaluate` | 加载 VQ/SSL/downstream，只运行 internal/external evaluation |

下面将当前主流程整理为 Stage 0～7；Stage 8 是已经实现但不在 `run.py` 主链中自动执行的 VQ 离线诊断。

## 5. Stage 0：CSV 解析与 subject-level split

### 5.1 输入

```text
Dataset/SSL_Healthy/ssl_healthy_dataset.csv
Dataset/KGKD/dev_dataset.csv
Dataset/KGKD/test_dataset.csv
```

每个受试者解析后形成：

```text
raw_subject [12, T_raw]
```

前 6 个 channel 为 left，后 6 个为 right；默认 `T_raw=600`、`sampling_rate=60 Hz`。

### 5.2 Split 实现

1. KGKD dev CSV 先按 label stratify，使用 `internal_test_size=0.20` 和 `seed=42` 分出 internal test；
2. 剩余 dev subject 按每个类别分别 shuffle，使用 `seed+1`，每类约 15% 进入 `dev_validation_data`；
3. SSL subject 使用 `seed=42` 随机排列，约 10% 进入 `ssl_validation_data`；
4. external CSV 全部进入 `ext_test_data`。

### 5.3 输出

内存中的 NumPy partition：

```text
ssl_data / ssl_label / ssl_trace_info
dev_data / dev_label / dev_trace_info
dev_test_data / dev_test_label / dev_test_trace_info
ext_test_data / ext_test_label / ext_test_trace_info
```

这些 subject array 随后由 `RepositoryKinematicDataset` 转换为 cycle dataset。当前代码没有单独写出 `split_manifest.json`；实际运行会把 seed、CSV 路径和 split fraction 写入 `args.json`。

### 5.4 本 Stage 无 loss

非有限受试者会在加载阶段移除；标签只作为 downstream metadata，VQ/SSL forward 不读取疾病标签。

## 6. Stage 1：自适应分周期、QC、标准化与 batch 构造

### 6.1 输入

单侧原始信号：

```text
x_side [6, T_raw]
```

左右侧分别调用同一个 `AdaptiveGaitCycleSegmenter`。

### 6.2 自适应分周期参数与算法

默认参数：

| 参数 | 默认值 |
|---|---:|
| sampling rate | 60 Hz |
| target length | 100 |
| raw reference DOF index | 2 (FE) |
| min/max duration | 0.4 / 4.0 s |
| Savitzky–Golay window | 0.15 s |
| polynomial order | 3 |
| peak prominence fraction | 0.15 |
| peak distance fraction | 0.55 |
| period correlation floor | 0.05 |
| relative period interval | 0.55～1.60 |
| similarity MAD scale | 3.0 |

算法顺序：

```text
FE Savitzky–Golay smoothing
→ autocorrelation period + swing-peak spacing period
→ swing peaks 之间寻找 extension minima
→ 相邻 minima 形成 candidate intervals
→ absolute 与 relative duration filtering
→ 六个 raw DOF 使用同一边界切分
→ linear interpolation 到 100 点
→ 六 DOF 联合 waveform correlation similarity filtering
```

每个保留周期输出：

```text
cycle       [6,100]
boundary    [start_sample,end_sample]
quality     scalar median pairwise waveform correlation
```

### 6.3 Population-level quality control

QC 使用训练 partition 的分割结果拟合每个 raw DOF 的 subject dispersion 上界。默认：

```text
min_cycles_per_side = 2
robust_z_threshold = 6
min_upper_scale_factor = 3
min_reference_subjects = 20
```

对每个 DOF：

```text
center = median(scale)
sigma_MAD = 1.4826 * median(|scale-center|)
sigma_IQR = (Q75-Q25)/1.349
sigma = max(sigma_MAD, sigma_IQR, numerical_floor)
upper = max(center + 6*sigma, 3*center)
```

任一侧周期数不足、存在非有限 dispersion 或任一 DOF dispersion 超过上界时，subject 被对应 dataset 排除。

SSL QC 只由 SSL train 拟合；downstream QC 只由 KGKD `dev_data` 拟合，随后应用于 KGKD validation/internal/external。QC 不使用疾病 label 计算阈值，但 downstream QC 的拟合 partition 包含 KGKD train 的全部三类。

### 6.4 波形标准化

SSL standardizer 使用通过 SSL-train QC 的全部 time-normalized cycle 拟合每个模型顺序 DOF 的 mean/std：

```text
x_standardized = (x - μ_d) / max(σ_d, 1e-6)
```

输出周期：

```text
left_cycles  [W_left,6,100]
right_cycles [W_right,6,100]
```

### 6.5 Stage 输出与指标

最终 DataLoader batch 和 language batch 形状见 3.3。运行目录写出 `word_statistics.json`，当前包含：

- retained subject 数；
- 左右周期数 min/median/mean/max；
- duration 分布；
- quality score 分布；
- 左周期到最近右周期中心的时间偏移分布。

这些是数据审计指标，不参与训练 loss。`word_mask` 是后续所有重建、attention、偏移聚合的有效性依据。

## 7. Stage 2：健康 DOF-Specific VQ tokenizer

### 7.1 输入

```text
words     [B,2,W,6,100]
word_mask [B,2,W]
timing    [B,2,W,4]
```

VQ 只使用 `ssl_data` 训练，使用 `ssl_validation_data` early stopping。

### 7.2 Word Encoder 架构

对每个 DOF waveform 独立展平到 Conv1d batch：

```text
[N,1,100]
→ Conv1d(1,64,kernel=7,stride=2,padding=3)
→ GroupNorm(groups=1,channels=64)
→ GELU
→ Conv1d(64,128,kernel=5,stride=2,padding=2)
→ GroupNorm(groups=1,channels=128)
→ GELU
→ Conv1d(128,128,kernel=3,stride=1,padding=1)
→ GELU
→ AdaptiveAvgPool1d(1)
→ [N,128]
```

随后加入 `dof_embedding [6,128]`，并经过六个独立 residual adapter：

```text
adapter_d: Linear(128,32) → GELU → Linear(32,128)
e_d = stem(x_d) + dof_embedding_d + adapter_d(stem(x_d)+dof_embedding_d)
```

输出：

```text
encoded [B,2,W,6,128]
```

### 7.3 六个独立 EMA codebook

Codebook buffer：

```text
embedding    [6,128,128]  # [DOF,K,D]
cluster_size [6,128]
embedding_sum[6,128,128]
```

Encoder 与 codebook prototype 先进行 L2 normalization，以 cosine similarity 选择 code：

```text
y = argmax_k <normalize(e), c[d,k]>
q = c[d,y]
```

训练时 codebook 使用 EMA 更新，默认 `decay=0.99`、`epsilon=1e-5`。`cluster_size<1.0` 的 dead code 使用当前 batch 对应 DOF 的 normalized encoder value 替换。Encoder 使用 straight-through estimator：

```text
q_ST = e_norm + stop_gradient(q-e_norm)
```

输出：

```text
quantized [B,2,W,6,128]
indices   [B,2,W,6] int64
```

### 7.4 当前默认 Decoder 架构

默认 `vq_decoder=local_context_sentence`，由 local anchor 和 contextual residual 两部分组成。

Local branch：

```text
q + local_dof_embedding
→ Linear(128,256)
→ GELU
→ Linear(256,100)
→ x_local [B,2,W,6,100]
```

Context branch token：

```text
q
+ side_embedding [2,128]
+ cycle_embedding [max_words=32,128]
+ dof_embedding [6,128]
+ timing_projection(timing)
```

`timing_projection` 为 `Linear(4,128) → GELU → Linear(128,128)`。所有 `[2,W,6]` token 展平为长度 `2*W*6` 的序列，输入：

```text
TransformerEncoder depth=2
d_model=128
heads=4
head_dim=32
feedforward_dim=512
dropout=0.1
activation=GELU
norm_first=true
final LayerNorm(128)
```

六个 DOF output head 均为：

```text
Linear(128,128) → Tanh → Linear(128,100)
```

最后一层在初始化时置零。最终重建：

```text
x_residual = context_decoder(q,mask,timing)
x_hat = x_local + 0.5 * x_residual
```

### 7.5 VQ loss

令 `M[b,s,w,d]` 为有效 DOF-word mask，`T=100`。

最终重建 MSE：

```text
L_rec = Σ M * (x_hat-x)² / (ΣM * T)
```

Local anchor MSE：

```text
L_local = Σ M * (x_local-x)² / (ΣM * T)
```

一阶差分 velocity loss：

```text
L_vel = Σ M * (Δx_hat-Δx)² / (ΣM * (T-1))
```

Commitment loss：

```text
L_commit = Σ M * ||normalize(e)-stop_gradient(q)||² / (ΣM * D)
```

Residual-energy loss 对已经乘 residual scale 的分支计算：

```text
L_res = Σ M * ||0.5*x_residual||² / (ΣM * T)
```

当前默认总损失：

```text
L_VQ = L_rec
     + 2.0 * L_local
     + 0.20 * L_vel
     + 0.25 * L_commit
     + 0.01 * L_res
```

### 7.6 输出与指标

Forward 输出包括：

| 输出 | 形状/类型 | 意义 |
|---|---|---|
| `indices` | `[B,2,W,6]` | 六个 vocabulary 的离散 word ID |
| `reconstructed` | `[B,2,W,6,100]` | 最终标准化波形重建 |
| `loss` | scalar | checkpoint 使用的总目标 |
| reconstruction/local/velocity/commitment/residual loss | scalar | 分项诊断 |
| raw/scaled context residual RMS | scalar | contextual branch 强度 |
| active code ratio | scalar | 六个 DOF 平均使用 code 比例 |
| perplexity | scalar | assignment entropy 的有效 code 数 |

训练默认 `AdamW(lr=3e-4, weight_decay=1e-4)`、最多 100 epoch、gradient clip 1.0、patience 10；minimum validation total loss 保存为 `best_vq.pt`。

## 8. Stage 3：健康 Gait-Language Sentence Encoder 与 SSL

### 8.1 输入与固定 target

```text
words     [B,2,W,6,100]
word_mask [B,2,W]
timing    [B,2,W,4]
```

冻结 VQ tokenizer 产生：

```text
target_code [B,2,W,6]
```

Sentence Encoder 的 `DOFWordEncoder` 从 VQ word encoder 深拷贝，随后参与 SSL 优化。

### 8.2 Token 构造

对每个 `[b,s,w,d]`：

```text
token = shape_embedding
      + sentence_dof_embedding
      + side_embedding
      + cycle_position_embedding
      + continuous_center_embedding
      + duration_embedding
      + interval_embedding
      + quality_embedding
```

具体参数：

| 组件 | 架构/形状 |
|---|---|
| shape encoder | 与 Stage 2 `DOFWordEncoder` 相同，输出 D=128 |
| sentence DOF embedding | `[6,128]` |
| side embedding | `[2,128]` |
| cycle position embedding | `[32,128]` |
| center projection | `Linear(1,128)→GELU→Linear(128,128)` |
| duration projection | 同上 |
| interval projection | 同上 |
| quality projection | 同上 |
| shape mask token | `[6,128]` |
| timing mask embedding | `[3,128]`，分别替代 center/duration/interval；quality 不被替代 |

输出初始 token：

```text
tokens [B,2,W,6,128]
```

### 8.3 GaitSentenceBlock 架构

默认 `sentence_depth=2`，每个 block 依次执行：

1. Temporal axis self-attention：每个 side×DOF 独立在 W 个 cycle 上 attention；
2. DOF axis self-attention：每个 side×cycle 独立在 6 个 DOF 上 attention；
3. 仅在任务显式要求时执行双侧 content-based cross-attention。

Temporal 与 DOF layer 参数相同：

```text
TransformerEncoderLayer
d_model=128
heads=4
head_dim=32
feedforward=512
dropout=0.1
activation=GELU
batch_first=true
norm_first=true
```

Bilateral cross-attention 参数：

```text
Q/K/V: Linear(128,128)
heads=4, head_dim=32
scaled dot-product attention
output: Linear(128,128) + Dropout(0.1)
residual connection
FFN: Linear(128,512)→GELU→Dropout→Linear(512,128)→Dropout
LayerNorm on query/context/output
```

双侧 attention 不使用 relative-time bias，因为左右数据不是同步采集。

### 8.4 Sentence Encoder 输出

最终 token 经过 `LayerNorm(128)`：

```text
tokens [B,2,W,6,128]
```

每侧在有效 W 和全部 6 DOF 上平均：

```text
left_embedding  [B,128]
right_embedding [B,128]
```

进一步输出：

```text
shared_embedding = MLP([left+right, left*right])       [B,128]
directional_difference = Linear(left-right)            [B,128]
absolute_difference = MLP([abs(left-right),left*right])[B,128]
left_difference_map  [B,W,6,128]
right_difference_map [B,W,6,128]
```

### 8.5 Prediction heads

六个独立 code head：

```text
Linear(128,128 codes)
→ logits [B,2,W,6,128]
```

六个独立 prototype head：

```text
Linear(128,128 embedding dims)
→ predicted_prototype [B,2,W,6,128]
```

Rhythm head：

```text
mean over 6 DOF token
→ LayerNorm(128)
→ Linear(128,128)
→ GELU
→ Linear(128,2)
→ [predicted_duration,predicted_interval]
```

Pair head：

```text
[abs(left-right),left*right] [B,256]
→ LayerNorm(256)
→ Linear(256,128)
→ GELU
→ Dropout(0.1)
→ Linear(128,1)
```

### 8.6 Default SSL tasks

#### A. Within-DOF masked code prediction

对有效 `[side,word,dof]` 以 0.30 概率选 loss target；输入 mask 以 `span_length=2` 向后扩展，但 CE 只在原始 target 位置计算：

```text
L_within = CE(logits,target_code)
```

指标 `within_accuracy` 是 target 位置 exact code accuracy。

#### B. Cross-DOF relaxed prediction

每个 subject×side 随机选择一个 target DOF，将该 DOF 的全部有效 cycle morphology token mask，使用其余五个 DOF，以及 target token 仍保留的 side/position/timing 元信息预测；输入中不存在未遮挡的 target-DOF morphology。

令真实 code prototype 为 `c_y`，同 DOF codebook 为 `{c_k}`，取 cosine similarity 最高的 top-5 邻域 `N_5(y)`：

```text
p_k = softmax(cos(c_y,c_k)/τ), k∈N_5(y), τ=0.10
L_soft = -mean Σ p_k log softmax(logits)_k
L_hard = CE(logits,y)
L_proto = mean(1-cos(predicted_prototype,c_y))
```

默认：

```text
L_cross = 0*L_hard + 1*L_soft + 1*L_proto
```

指标同时报告 exact accuracy 和 top-5 是否包含真实 code。

#### C. Explicit rhythm prediction

以 0.30 概率选择有效 cycle，使用 timing mask embedding 替换 center、duration、interval embedding；quality embedding 保留。预测该 cycle 的 duration 和 preceding center interval。

PyTorch Smooth-L1（beta=1）定义：

```text
smoothL1(e) = 0.5e², |e|<1
              |e|-0.5, otherwise
```

```text
L_duration = SmoothL1(duration_hat,duration)
L_interval = SmoothL1(interval_hat,interval)
L_rhythm = 1.0*L_duration + 1.0*L_interval
```

指标 `duration_mae`、`interval_mae` 以秒为单位。

#### D. Pure contralateral relaxed prediction

每个 subject 随机选择 left 或 right 为 target side。该侧所有有效 token 在输入中被 mask，启用 content-based bilateral cross-attention；loss 只在该侧随机 0.30 的 target 位置计算。

损失形式与 Cross-DOF 相同：

```text
L_contra = 0*L_hard + 1*L_soft + 1*L_proto
```

由于输入 target side 全部被遮挡，主要信息来自对侧内容，不假设左右同步相位。

#### E. Healthy bilateral pair discrimination

Positive 是同一 batch subject 的 left/right pooled embedding；negative 将 right embedding 按随机非零 batch shift 循环移动，形成不同 subject 配对。

```text
pair_feature = [abs(left-right),left*right]
L_pair = BCEWithLogits(pair_logit,pair_label)
```

指标 `bilateral_pair_accuracy` 的随机基线为 50%。

### 8.7 已实现但默认关闭的任务

```text
ssl_bilateral_context_task = false
ssl_swap_task = false
```

`bilateral_context_task` 在选中侧随机 mask token，启用双侧 cross-attention，并用 hard CE 预测 code。`swap_task` 将左右输入交换，约束 shared/absolute 不变、directional 取反：

```text
L_swap = MSE(shared,shared_swap)
       + MSE(absolute,absolute_swap)
       + MSE(directional,-directional_swap)
```

### 8.8 SSL 总损失

当前默认外层权重：

```text
L_SSL = 1.0*L_within
      + 1.0*L_cross
      + 0.5*L_rhythm
      + 1.0*L_bilateral_context  # task disabled，当前该项返回 0
      + 0.5*L_contra
      + 1.0*L_pair
      + 0.1*L_swap               # task disabled，当前该项返回 0
```

关闭任务时对应 forward 指标返回 0。不同任务的自然 loss 尺度不同，不能只根据 total loss 数值判断某项任务是否有效。

### 8.9 输出、指标与 checkpoint

`GaitLanguageSSLModel.forward` 输出：

- total 与所有 task/component loss；
- within/cross/contralateral exact 与 top-5 accuracy；
- duration/interval MAE；
- pair accuracy；
- `bilateral_surprise [B,2,W,6]`，但默认 bilateral task 关闭时为零；
- shared/directional/absolute embedding `[B,128]`；
- left/right difference map `[B,W,6,128]`。

训练默认 `AdamW(lr=3e-4, weight_decay=1e-2)`、最多 100 epoch、clip 1.0、patience 10。Train mask 随机；validation 使用固定 `validation_mask_seed=seed+10000`。Minimum validation total SSL loss 保存为 `best_ssl.pt`。

## 9. Stage 4：KGKD-train Healthy normative reference

### 9.1 输入

加载 `best_ssl.pt` 后，使用 KGKD `dev_data` language batch：

```text
words         [B,2,W,6,100]
word_mask     [B,2,W]
timing        [B,2,W,4]
disease_label [B]
```

Sentence Encoder 输出 contextual：

```text
z [B,2,W,6,128]
```

只选择 `disease_label==0` 的 Healthy token。

### 9.2 Reference 定义

按 side×DOF×embedding coordinate 计算：

```text
μ[s,d,j] = Σ valid_healthy z / count[s,d]
σ²[s,d,j] = Σ valid_healthy z²/count - μ²
σ_ref = max(sqrt(max(σ²,0)), 0.05)
```

### 9.3 输出

`HierarchicalGaitDeviationEncoder` buffer：

```text
reference_mean [2,6,128]
reference_std  [2,6,128]
```

它们作为 `best_downstream.pt` 的 model state 保存。当前不会单独写出 normative-reference JSON，也不会使用 KGKD validation/test 更新 reference。

本 Stage 没有 optimizer 和 loss。

## 10. Stage 5：word→DOF→side→subject 分层偏移

### 10.1 Word-level

输入 `z [B,2,W,6,128]`，标准化有符号方向：

```text
r = (z-μ_ref)/σ_ref
word_direction [B,2,W,6,128]
```

程度为 embedding coordinate RMS：

```text
m_word = sqrt(mean_j r_j²)
word_magnitude [B,2,W,6]
```

### 10.2 DOF-level

对有效 W 聚合：

```text
dof_direction = mean_w(r)                         [B,2,6,128]
dof_magnitude_mean = mean_w(m_word)               [B,2,6]
dof_magnitude_rms = sqrt(mean_w(m_word²))         [B,2,6]
dof_magnitude_std = sqrt(mean_w((m_word-mean)²))  [B,2,6]
dof_magnitude_max = max_w(m_word)                 [B,2,6]
dof_direction_strength = sqrt(mean_j direction²) [B,2,6]
```

每个 DOF feature 长度为 `128+5=133`：

```text
LayerNorm(133)
→ Linear(133,64)
→ GELU
→ Dropout(0.2)
→ dof_embedding [B,2,6,64]
```

### 10.3 Side-level

Side feature 拼接：

```text
flatten(six dof embeddings) = 6*64
side_direction = mean_dof(dof_direction) = 128
side magnitude mean/rms/max = 3
side direction strength = 1
total = 516
```

架构：

```text
LayerNorm(516)
→ Linear(516,256)
→ GELU
→ Dropout(0.2)
→ Linear(256,128)
→ side_embedding [B,2,128]
```

同时输出 side direction `[B,2,128]` 和 magnitude mean/rms/max `[B,2,1]`。

### 10.4 Subject-level

疾病分类使用左右交换不变的输入：

```text
mean(left_side,right_side)             128
abs(left_side-right_side)              128
elementwise_max(left_side,right_side)  128
subject_direction                      128
abs(left_direction-right_direction)    128
subject magnitude mean/rms/max           3
abs(left-right magnitude gap)            1
total                                  644
```

Subject projection：

```text
LayerNorm(644)
→ Linear(644,256)
→ GELU
→ Dropout(0.2)
→ Linear(256,128)
→ subject_embedding [B,128]
```

带左右符号的输出单独保留：

```text
bilateral_deviation_direction [B,128] = left_direction-right_direction
bilateral_magnitude_gap       [B,1]   = left_magnitude-right_magnitude
```

这些 signed feature 不直接进入 disease head，而进入 affected-side head。

### 10.5 本 Stage 输出和指标意义

Forward 返回全部 word/DOF/side/subject direction、magnitude 和 learned embedding tensor。训练日志只汇总：

```text
word_deviation_magnitude
dof_deviation_magnitude
side_deviation_magnitude
subject_deviation_magnitude
bilateral_magnitude_gap
```

前三个高层全局 mean 在代数上可能相同，因为 side/subject 继续对 DOF/side mean 求均值；真正进行层级解释时应使用完整 per-DOF/per-side tensor，而不是仅比较全局 scalar。

本 Stage 的 projection 参数由 Stage 6 分类 loss 端到端训练，没有单独 deviation loss。

## 11. Stage 6：有监督 Healthy/ACLD/KOA downstream

### 11.1 输入

```text
subject_embedding [B,128]
disease_label     [B]
bilateral_direction [B,128]
bilateral_magnitude_gap [B,1]
affected_side_label [B]
affected_side_valid_mask [B]
```

### 11.2 Disease head

```text
LayerNorm(128)
→ Linear(128,128)
→ GELU
→ Dropout(0.2)
→ Linear(128,3)
→ disease_logits [B,3]
```

### 11.3 Affected-side head

```text
concat([bilateral_direction,bilateral_magnitude_gap]) [B,129]
→ LayerNorm(129)
→ Linear(129,64)
→ GELU
→ Dropout(0.2)
→ Linear(64,2)
→ affected_side_logits [B,2]
```

患侧 loss 只在 `affected_side_valid_mask=True` 的样本上计算；当前 trace 中主要是具有 left/right ACLD 标注的样本。

### 11.4 Class-balanced loss

令 KGKD train 三类计数为 `n_c`、总数为 `N`：

```text
u_c = N / max(n_c,1)
w_c = u_c / mean_j(u_j)
L_disease = weighted_CE(disease_logits,y,w)
L_side = CE(affected_logits[valid],affected_y[valid])
L_downstream = L_disease + 0.20*L_side
```

### 11.5 优化、冻结与 checkpoint

默认冻结 Sentence Encoder；以下模块训练：

- DOF projection；
- side projection；
- subject projection；
- disease head；
- affected-side head。

默认：

```text
AdamW(lr=3e-4, weight_decay=1e-2)
epochs=50
patience=10
gradient_clip=1.0
checkpoint criterion=max validation macro-F1
```

输出 `best_downstream.pt`，其中包括 reference buffer、所有 downstream projection/head、Sentence Encoder state 和 optimizer state。

### 11.6 训练输出指标

| 指标 | 作用 |
|---|---|
| total/disease/affected-side loss | 优化与泛化差距诊断 |
| accuracy | 总体正确率，受类别比例影响 |
| macro-F1 | 三类等权的 precision/recall 综合指标，也是 checkpoint criterion |
| macro-AUROC | 三个 one-vs-rest 排序能力的宏平均 |
| affected-side accuracy | 有有效患侧标签子集的左右正确率 |
| deviation magnitudes | 监控不同 split 相对 KGKD Healthy reference 的整体偏移 |

## 12. Stage 7：Internal/External evaluation 与运行产物

### 12.1 输入

```text
best_downstream.pt
dev_test_data DataLoader
ext_test_data DataLoader
```

Evaluation 调用与训练相同的 forward；`affected_side_weight=0`，所以 evaluation total loss 等于 disease loss，但仍单独计算并报告 affected-side loss/accuracy。

### 12.2 输出

`evaluation.json`：

```text
internal_test:
  loss / disease_loss / affected_side_loss
  deviation summary scalars
  accuracy / macro_f1 / macro_auroc / affected_side_accuracy
external_test:
  same fields
vq_checkpoint / ssl_checkpoint / downstream_checkpoint
```

当前没有自动输出 confusion matrix、per-class recall/F1、subject prediction table 或 calibrated probability。

### 12.3 每个 run 的实际文件

```text
args.json
word_statistics.json
metrics.jsonl
best_vq.pt              # 若该 run 训练 VQ
best_ssl.pt             # 若该 run 训练 SSL
best_downstream.pt      # 若该 run 训练 downstream
evaluation.json         # 若执行最终 evaluation
```

`run.py` 默认写入：

```text
Results/gait_language/dev_exp/dev_exp_MMDD_HHMM
```

`run_exp.py` 的 SSL task ablation 默认写入：

```text
Results/gait_language/ablation_exp/ablation_exp_MMDD_HHMM
```

## 13. Stage 8：已实现的离线 VQ 诊断（独立入口）

`analyze_vq.py` 不由 `run.py` 自动调用，但已实现并可分析任一具有 `best_vq.pt` 的 run。

### 13.1 输入

```text
run args.json
best_vq.pt
指定 split 的 DataLoader
```

可选 split 包含 SSL train/validation、KGKD train/validation/internal/external。

### 13.2 输出与意义

```text
code_usage_and_variance.csv
dof_reconstruction_metrics.csv
subject_cycle_consistency.csv
dof_cycle_consistency_summary.csv
prototype_pair_similarity.csv
prototype_duplicate_summary.csv
code_waveform_statistics.npz
summary.json
```

主要指标：

- per-DOF active code、perplexity、code occupancy：检查 code collapse/imbalance；
- RMSE、MAE、correlation、velocity RMSE：检查波形与动态重建；
- same/near-code rate：检查相似周期 assignment consistency；
- prototype cosine 与 decoded waveform correlation：检查重复或 alias prototype；
- code 内 waveform variance：检查一个 code 是否混入过多异质周期。

该 Stage 只读取 checkpoint，不更新模型。

## 14. 当前实现状态

| 模块 | 状态 | 说明 |
|---|---|---|
| Subject-level CSV parsing/split | 已实现 | stratified internal split、class-wise downstream validation split |
| Adaptive bilateral segmentation | 已实现 | 左右独立、FE proxy、共享六 DOF boundary |
| Train-fitted QC | 已实现 | SSL 与 downstream 各自训练 partition 拟合 QC |
| SSL Healthy waveform standardizer | 已实现 | 所有后续 split 复用 |
| DOF-specific VQ | 已实现 | K=128、EMA、dead-code replacement |
| Local + context residual decoder | 已实现且默认 | residual scale=0.5、energy weight=0.01 |
| VQ offline diagnostics | 已实现 | 独立 `analyze_vq.py` |
| Sentence token 完整构造 | 已实现 | shape+DOF+side+position+center+duration+interval+quality |
| Temporal/DOF attention | 已实现 | depth=2、heads=4 |
| Content-based bilateral attention | 已实现 | 只在相应 SSL task 启用 |
| Within/Cross-DOF/Rhythm/Contralateral/Pair SSL | 已实现且默认启用 | Cross/Contra 使用 soft+prototype |
| Bilateral contextual/Swap SSL | 已实现但默认关闭 | 可通过 CLI 启用 |
| SSL ablation runner | 已实现 | 六组固定 VQ 消融 |
| KGKD Healthy normative reference | 已实现 | side×DOF token mean/std |
| Word/DOF/side/subject deviation | 已实现 | direction+mean/RMS/std/max |
| Deviation-aware three-class classifier | 已实现 | 默认冻结 Sentence Encoder |
| Affected-side auxiliary classifier | 已实现 | signed bilateral feature |
| Internal/external aggregate evaluation | 已实现 | accuracy/F1/AUROC/side accuracy |
| Per-class/per-DOF deviation export | 未实现 | 当前 forward 有 tensor，但不写表 |
| Confusion matrix/per-class recall | 未实现 | evaluation 只写 aggregate metrics |
| Code frequency/transition surprise | 未实现 | 旧文档中的规划，当前无代码 |
| Rhythm normative reference | 未实现 | SSL 有 rhythm task，但 downstream 未单独计算健康 z-score |
| Probability calibration | 未实现 | 无 temperature scaling/ECE 输出 |
| ACLD/KOA waveform/heatmap visualization | 未实现 | 无自动科研图表入口 |
| Multi-seed downstream experiment runner | 未实现 | `run_exp.py` 当前只定义 SSL task ablation |

## 15. 下一步

### 15.1 第一优先级：固定上游 checkpoint 的 downstream 对照

首次 deviation-aware downstream 同时重新训练了 VQ/SSL，并改变了冻结策略，不能隔离架构效应。下一组应固定完全相同的 `best_vq.pt` 和 `best_ssl.pt`：

```text
A. frozen Sentence Encoder + 旧 shared/absolute pooled head
B. frozen Sentence Encoder + 当前 deviation-aware head
C. frozen Sentence Encoder + 线性/单层 deviation classifier
D. deviation head + 只解冻最后一个 Sentence block，encoder LR=head LR/10～1/30
```

### 15.2 第二优先级：补齐结构化偏移输出

应从现有 forward tensor 导出：

```text
subject_id / split / label
word_deviation_direction/magnitude
per-side per-DOF mean/RMS/std/max/direction_strength
subject direction/magnitude
bilateral signed direction/gap
```

并按 Healthy/ACLD/KOA 计算 class centroid、effect size 和分布，而不是只保留全 split global mean。

### 15.3 第三优先级：控制 downstream 过拟合

当前冻结 encoder 后 classifier 仍快速达到接近 100% train F1。应优先比较：

- `deviation_dof_dim` 更小；
- 线性或单 hidden-layer classifier；
- 更高 dropout/weight decay；
- 仅在固定 validation protocol 上选择容量。

### 15.4 后续解释与评估

在严格 downstream 对照完成后，再实现：

1. confusion matrix、per-class precision/recall/F1；
2. validation-only temperature scaling、NLL、ECE；
3. high-deviation word 到 decoded VQ prototype waveform 的映射；
4. per-DOF/per-side ACLD 与 KOA deviation heatmap；
5. 3～5 个固定 seed 的均值、标准差和方向一致性。

External test 只用于最终审计，不继续承担反复选择模型或 task weight 的功能。

## 16. 当前最终系统定义

基于当前已经实现的代码，最终系统应准确表述为：

> 一个以健康双侧膝关节 6-DOF 运动学周期为 gait words、以六个 DOF-specific EMA codebook 为离散 vocabulary、以 temporal/DOF attention 和任务级 content-based bilateral attention 为上下文建模器，并以 KGKD-train Healthy contextual-token 分布为 normative reference 的分层偏移三分类系统。

其完整函数关系为：

```text
f_segment:
  [12,T_raw]
  → left/right variable cycles [W_side,6,100]

f_vq:
  cycle waveform [6,100]
  → continuous morphology [6,128]
  → code IDs [6]
  → reconstructed waveform [6,100]

f_sentence:
  words [B,2,W,6,100] + timing [B,2,W,4]
  → contextual tokens [B,2,W,6,128]
  → bilateral sentence embeddings

f_reference:
  KGKD-train Healthy contextual tokens
  → μ_ref,σ_ref [2,6,128]

f_deviation:
  contextual tokens + μ_ref,σ_ref
  → word/DOF/side/subject directions and magnitudes
  → subject_embedding [B,128]

f_classifier:
  subject_embedding
  → disease_logits [B,3]

f_side:
  signed bilateral deviation [B,129]
  → affected_side_logits [B,2]
```

当前系统的主要产物是 VQ/SSL/downstream checkpoint、aggregate evaluation metrics，以及 forward 中可访问的分层 deviation tensor。它还不是一个已经完成概率校准、自动生成 per-class deviation table 和临床可视化的最终科研报告系统；这些能力属于第 15 节列出的后续实现。
