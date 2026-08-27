# Gait-Language Downstream 偏移表征与三分类设计

## 1. 目标

VQ 与健康 Gait-Language Sentence Encoder 已经能够把每个受试者表示为：

```text
[side, cycle word, DOF, embedding]
```

Downstream 不再直接把左右 pooled embedding 拼接后分类，而是先回答两个问题：

1. 当前 gait word 相对 KGKD 训练集中的健康参考向哪个方向偏移；
2. 这种偏移在 word、DOF、side 和 subject 四个层级上有多大。

在此基础上，用 KGKD 训练集的 Healthy、ACLD、KOA 标签有监督训练三分类器。设计目标不仅是提高分类能力，还要保留偏移定位信息，使模型可以进一步解释“哪一侧、哪个 DOF、哪些周期偏离健康”。

## 2. 总体流程

```text
固定 VQ / 加载 SSL Sentence Encoder
                │
                ▼
contextual word tokens [B, 2, W, 6, D]
                │
                ▼
KGKD train Healthy side×DOF reference
                │
                ▼
word signed direction + word magnitude
                │
                ▼
DOF direction + mean/RMS/std/max magnitude
                │
                ▼
side direction + magnitude + six-DOF signature
                │
                ▼
subject common deviation + bilateral asymmetry
                │
                ▼
supervised Healthy / ACLD / KOA classifier
```

当前实现位于：

```text
knee_kinematics/gait_language/downstream.py
knee_kinematics/gait_language/models.py
knee_kinematics/gait_language/trainer.py
run.py
```

## 3. 数据边界与健康参考

### 3.1 使用的数据

KGKD 开发数据仍按 subject 分为：

```text
dev_data             downstream train
dev_validation_data  checkpoint selection
dev_test_data        internal evaluation
ext_test_data        external evaluation
```

健康参考只能由 `dev_data` 中 `disease_label=0` 的 Healthy subject 拟合。ACLD、KOA、validation、internal test 和 external test 均不参与健康参考估计。

### 3.2 为什么按 side 和 DOF 建参考

Sentence Encoder 输出：

```text
Z ∈ R[B, 2, W, 6, D]
```

其中 token 已包含 shape、DOF、side、cycle position、continuous timing、duration、interval 和 quality 信息。由于不同 DOF 的语义与数值分布不同，左右侧也带有独立的 side embedding 和采集差异，因此健康参考不能只使用一个全局中心。

对健康训练 token，分别计算：

```text
μ[s,d,j] = Healthy token coordinate mean
σ[s,d,j] = Healthy token coordinate standard deviation
```

其中：

```text
s ∈ {left, right}
d ∈ {0, ..., 5}
j ∈ {0, ..., D-1}
```

实现中的 buffer 形状为：

```text
reference_mean [2, 6, D]
reference_std  [2, 6, D]
```

标准差使用 `deviation_std_floor=0.05` 作为最小尺度，防止一个几乎不变化的健康 embedding coordinate 主导全部偏移量。参考统计在 downstream 训练开始前拟合一次，并随 `best_downstream.pt` 保存；validation/test 只读取该参考，不重新估计。

### 3.3 Encoder 是否更新

`run.py` 当前默认：

```text
freeze_sentence_encoder = true
```

冻结可以保持 SSL embedding 坐标与健康均值/标准差一致，使偏移量在整个分类训练过程中具有固定含义。可以通过 `--no-freeze-sentence-encoder` 做全量微调对照，但此时 encoder 坐标会改变，而训练开始时拟合的健康参考保持不变，因此不作为当前默认方案。

## 4. Word-level 偏移

对受试者 token `z[b,s,w,d]`，定义逐坐标标准化残差：

```text
r[b,s,w,d,j]
    = (z[b,s,w,d,j] - μ[s,d,j]) / σ[s,d,j]
```

### 4.1 偏移方向

```text
word_deviation_direction = r ∈ R[B,2,W,6,D]
```

`r` 保留完整符号：正负方向表示该 token 在 SSL embedding coordinate 上位于健康中心的哪一侧。这里的方向是健康表示空间中的方向，不等同于原始关节角增大或减小；若需要映射回物理波形，可在后续使用 VQ prototype decoder 解释相应 embedding direction。

### 4.2 偏移程度

```text
m_word = sqrt(mean_j(r_j²))
```

输出：

```text
word_deviation_magnitude ∈ R[B,2,W,6]
```

该值是相对健康标准差归一化后的 RMS 距离：

- 接近 0：位于相应 side/DOF 的健康中心附近；
- 较大：多个 embedding coordinate 或少数强 coordinate 明显偏离健康；
- 不依赖 embedding dimension 的简单扩张。

Padding word 由 `word_mask` 排除，不参与任何层级聚合。

## 5. DOF-level 偏移

每侧、每个 DOF 在有效周期上聚合 word 指标。

### 5.1 DOF 方向

```text
v_dof = mean_w(r_word)
direction_strength_dof = sqrt(mean_j(v_dof,j²))
```

`v_dof` 表示该 DOF 跨周期一致的有符号偏移；如果不同周期向相反方向变化，平均方向会相互抵消，而 word magnitude 仍会保留异常程度。

### 5.2 DOF 程度

实现同时计算：

```text
mean(m_word)   平均偏移负荷
RMS(m_word)    对较大偏移更敏感
std(m_word)    周期间不稳定性
max(m_word)    局灶或极端异常周期
```

输出包括：

```text
dof_deviation_direction       [B,2,6,D]
dof_deviation_magnitude_mean  [B,2,6]
dof_deviation_magnitude_rms   [B,2,6]
dof_deviation_magnitude_std   [B,2,6]
dof_deviation_magnitude_max   [B,2,6]
dof_direction_strength        [B,2,6]
```

随后共享的 per-DOF projection 将：

```text
[signed direction, mean, RMS, std, max, direction strength]
```

映射为 `dof_embedding [B,2,6,deviation_dof_dim]`。六个 DOF 的顺序不会在 side 聚合前丢失。

## 6. Side-level 偏移

Side 层将同侧六个 DOF 的 embedding 按固定 DOF 顺序展开，同时加入：

```text
side_deviation_direction       六个 DOF direction 的均值
side_deviation_magnitude_mean  六个 DOF 平均偏移
side_deviation_magnitude_rms   六个 DOF RMS 偏移
side_deviation_magnitude_max   同侧最大局部偏移
side_direction_strength        同侧净方向强度
```

Side projection 输出：

```text
side_embedding ∈ R[B,2,D]
```

这样每一侧同时保留：

- 六个 DOF 各自的偏移模式；
- 整侧共同偏移方向；
- 平均、整体和极端偏移程度。

## 7. Subject-level 偏移

### 7.1 共同疾病偏移

左右相对各自健康参考的平均方向：

```text
subject_deviation_direction
    = (left_direction + right_direction) / 2
```

它表示两侧共享的受试者级偏移方向。

### 7.2 双侧不对称

```text
bilateral_deviation_direction
    = left_direction - right_direction

bilateral_deviation_magnitude_gap
    = left_magnitude - right_magnitude
```

这两个量保留左右符号，主要服务 affected-side auxiliary head。

### 7.3 三分类使用的左右交换不变特征

Healthy/ACLD/KOA 是疾病类别，不应因为把左右两侧交换就改变。因此 disease subject feature 仅使用：

```text
mean(left_side_embedding, right_side_embedding)
abs(left_side_embedding - right_side_embedding)
elementwise_max(left_side_embedding, right_side_embedding)
subject_deviation_direction
abs(bilateral_deviation_direction)
subject magnitude mean / RMS / max
abs(left-right magnitude gap)
```

其中 `subject_deviation_direction` 虽然保留健康 embedding coordinate 的正负方向，但它对左右交换不变；带有明确 left-minus-right 符号的指标只以绝对值进入 disease head。

Subject projection 最终输出：

```text
subject_embedding ∈ R[B,D]
```

## 8. 有监督三分类器

### 8.1 Disease head

```text
subject_embedding
→ LayerNorm
→ Linear(D,D)
→ GELU
→ Dropout
→ Linear(D,3)
→ Healthy / ACLD / KOA logits
```

标签定义保持为：

```text
Healthy = 0
ACLD    = 1
KOA     = 2
```

训练使用 KGKD `dev_data` 的三类标签和 class-balanced cross entropy。类别权重仍由 downstream train 的类别计数计算，避免小类别在监督训练中被多数类覆盖。

### 8.2 Affected-side auxiliary head

患侧辅助头继续保留，但不作为三分类的主要输入：

```text
[signed bilateral direction, signed magnitude gap]
→ left / right logits
```

该辅助 loss 只在具有有效 affected-side 标签的样本上计算，默认：

```text
L = L_three_class + 0.20 * L_affected_side
```

疾病 head 使用左右交换不变特征，患侧 head 使用带符号特征，两个目标的结构语义被明确分开。

### 8.3 Checkpoint 规则

Downstream 仍按 `dev_validation_data` 的 macro-F1 选择 `best_downstream.pt`，patience 默认 10。Checkpoint 保存：

- 冻结或微调后的 Sentence Encoder；
- 健康 `reference_mean/reference_std`；
- DOF、side、subject projections；
- disease 与 affected-side heads；
- optimizer state 和 validation macro-F1。

## 9. 训练执行顺序

完整 downstream 阶段为：

```text
1. 加载 best_vq.pt
2. 加载 best_ssl.pt 中的 Sentence Encoder
3. 构建 HierarchicalGaitDeviationEncoder
4. 默认冻结 Sentence Encoder
5. 遍历 KGKD dev_data 的 Healthy subject
6. 拟合 side×DOF reference_mean/reference_std
7. 使用全部 KGKD dev_data 训练三分类器与患侧辅助头
8. 使用 dev_validation macro-F1 early stopping
9. 加载 best_downstream.pt
10. 在 internal/external split 上评估
```

健康参考拟合由：

```text
fit_healthy_deviation_reference(...)
```

完成，并在 `fit_downstream(...)` 创建 optimizer 之前执行。

## 10. 新增参数

`run.py` 中新增或调整的 downstream 参数：

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `--deviation-dof-dim` | 64 | 每个 DOF 偏移 embedding 的维度 |
| `--deviation-std-floor` | 0.05 | 健康 token 标准差的最小归一化尺度 |
| `--freeze-sentence-encoder` | true | 固定 SSL embedding 坐标，只训练偏移聚合器和分类头 |
| `--no-freeze-sentence-encoder` | false | 可选全量微调对照 |
| `--affected-side-weight` | 0.20 | 患侧辅助 loss 权重 |
| `--classifier-dropout` | 0.20 | DOF/side/subject projection 与分类头 dropout |

其余 downstream epoch、learning rate、weight decay 和 patience 参数保持原有接口。

## 11. 训练日志中的偏移指标

`metrics.jsonl` 的 downstream train/validation 记录新增：

```text
word_deviation_magnitude
dof_deviation_magnitude
side_deviation_magnitude
subject_deviation_magnitude
bilateral_magnitude_gap
```

这些值用于观察不同 split 和训练阶段的总体健康偏离程度。模型 forward 还返回完整的 word/DOF/side/subject direction 与 magnitude tensor，可在后续分析中按类别、DOF、侧别和 subject 导出，不需要重新定义指标。

## 12. 当前实现的定位

当前实现完成了：

1. KGKD train Healthy-only 的无泄漏健康参考；
2. word/DOF/side/subject 四级偏移方向和程度；
3. 对少数异常周期敏感的 RMS、std 和 max 聚合；
4. disease 使用左右交换不变特征；
5. affected-side 使用带符号左右差异；
6. 基于偏移表征的有监督 Healthy/ACLD/KOA 三分类训练。

当前的“方向”定义在 SSL contextual embedding space 中，而不是直接定义在原始关节角或解码波形空间。它适合作为分类与表示分析的第一版 normative deviation；后续若需要更强的临床解释，可进一步把高偏移 word 映射到 VQ prototype waveform，并增加 code frequency surprise、healthy transition surprise、duration/interval normative deviation 等独立指标。

## 13. 首次 Deviation-Aware Downstream 实验结果（2026-08-26）

### 13.1 实验配置

首次完整实验为：

```text
Results/gait_language/dev_exp/dev_exp_0826_2212
stage = all
seed = 42
deviation_dof_dim = 64
deviation_std_floor = 0.05
freeze_sentence_encoder = true
affected_side_weight = 0.20
```

该 run 从 VQ 开始重新训练，随后训练完整新版 SSL task，最后拟合 KGKD train Healthy reference，并训练 deviation-aware classifier。因此它可以用于判断新 downstream 是否能够运行、偏移指标是否合理，以及当前完整 pipeline 的最终表现；但它不是只替换 downstream head 的严格单变量对照。

### 13.2 与旧 downstream 的可比性

最接近的旧 downstream run 为：

```text
Results/gait_language/dev_exp/dev_exp_0826_1542
```

两次实验具有相同数据、seed、VQ/SSL 超参数和 `word_statistics.json`，但存在三项不可忽略的差异：

1. `0826_2212` 重新训练了 VQ 和 SSL；新旧 `best_vq.pt`、`best_ssl.pt` 的 SHA-256 均不相同；
2. 旧模型使用 `shared_embedding + absolute_difference`，新模型使用分层健康偏移特征；
3. 旧模型 `freeze_sentence_encoder=false`，新模型默认冻结 Sentence Encoder。

上游 checkpoint 的主要指标如下：

| 指标 | 旧 run 0826_1542 | 新 run 0826_2212 |
|---|---:|---:|
| VQ best epoch | 86 | 81 |
| VQ validation reconstruction | 0.065385 | **0.064376** |
| VQ validation perplexity | 73.974 | 74.399 |
| SSL best epoch | 47 | 50 |
| SSL validation total loss | **8.604829** | 8.648146 |
| Within exact accuracy | 44.67% | **46.56%** |
| Cross-DOF top-5 accuracy | **38.42%** | 37.01% |
| Duration MAE | 0.0676 s | **0.0628 s** |
| Interval MAE | **0.1478 s** | 0.1535 s |
| Contralateral top-5 accuracy | **22.01%** | 21.68% |
| Bilateral pair accuracy | **84.15%** | 83.17% |

两组上游结果处于相近范围，没有出现 VQ 或 SSL collapse，但并非同一个表示空间。因此后续与旧 downstream 的差值只能作为工程参照，不能全部归因于 deviation-aware design 或 encoder freezing。

### 13.3 Healthy reference 状态

`best_downstream.pt` 中保存的 Healthy reference 统计如下：

| Reference statistic | 数值 |
|---|---:|
| `abs(reference_mean)` 平均值 | 0.4764 |
| `reference_std` 平均值 | 1.1997 |
| `reference_std` 中位数 | 1.0605 |
| `reference_std` 最小值 | 0.4737 |
| `reference_std` 最大值 | 3.1801 |
| 被 `std_floor=0.05` 截断的 coordinate | 0 / 1536 |

左右两侧六个 DOF 的 mean std 均位于约 1.18～1.24，没有某一侧或某一个 DOF 出现异常小尺度。说明首次实验中 Healthy reference 拟合数值稳定，`std_floor` 没有实际改变任何 coordinate；当前结果不是由标准差截断或 standardized residual 爆炸造成的。

### 13.4 Downstream 训练动态

Downstream checkpoint 状态如下：

```text
best epoch = 10
last epoch = 20
actual epochs = 21
checkpoint rule = maximum validation macro-F1
```

最佳 epoch 的指标为：

| Metric | Train | Validation | Gap |
|---|---:|---:|---:|
| Total loss | 0.1205 | 0.4321 | +0.3116 |
| Disease loss | 0.0332 | 0.3399 | +0.3066 |
| Accuracy | 99.13% | 91.46% | -7.67 pp |
| Macro-F1 | 98.84% | 90.38% | -8.46 pp |
| Macro-AUROC | 99.98% | 98.85% | -1.13 pp |
| Affected-side accuracy | 81.86% | 83.78% | +1.92 pp |

分类器学习速度很快：validation macro-F1 在 epoch 4 已达到 90.18%，而 train F1 已达到 95.93%；epoch 6 后 train F1 接近或达到 100%，validation F1 则在约 84%～90% 之间波动。冻结 Sentence Encoder 没有消除过拟合，因为 DOF/side/subject projections 和 disease head 对当前 KGKD train 规模仍具有较强拟合能力。

Early stopping 正确保留了 epoch 10，而不是后期 train F1=100% 的模型。不过最佳 train/validation F1 仍相差 8.46 个百分点，说明下一步的主要问题已经从“破坏健康 encoder”转为“偏移聚合器和分类头容量相对数据量仍偏大”。

### 13.5 偏移程度的 split 分布

最佳 checkpoint 对应的聚合偏移指标如下：

| Split | Word magnitude | DOF magnitude | Side magnitude | Subject magnitude | Bilateral magnitude gap |
|---|---:|---:|---:|---:|---:|
| Downstream train | 1.0495 | 1.0702 | 1.0702 | 1.0702 | 0.06188 |
| Downstream validation | 1.0454 | 1.0690 | 1.0690 | 1.0690 | 0.05919 |
| Internal test | 1.0582 | 1.0794 | 1.0794 | 1.0794 | 0.05488 |
| External test | **1.1059** | **1.1127** | **1.1127** | **1.1127** | 0.05459 |

由于 Sentence Encoder 默认冻结，validation 偏移指标在全部 downstream epoch 中完全不变；train 的极小波动只来自 batch 组成和平均方式。这符合设计预期，证明训练 classifier 没有反向改变 normative coordinate。

External word magnitude 比 train 高约 5.37%，subject magnitude 高约 3.97%，表明 external subject 整体上距离 KGKD-train Healthy reference 更远。与此同时，external bilateral magnitude gap 没有增大，反而比 train 低约 11.8%。该组合更符合“两侧共同发生表示分布偏移”，而不是“external 数据具有更强左右不对称”的现象。

但当前 `evaluation.json` 只记录整个 split 的平均值，没有分别统计 Healthy、ACLD 和 KOA。External magnitude 增大可能同时包含真实疾病程度差异、类别比例差异和采集域差异；在获得 per-class/per-DOF 分解前，不能把它直接解释为更严重的病理偏移。

DOF、side、subject 三个日志均值完全相同是聚合定义造成的：

```text
side mean    = mean over six DOF means
subject mean = mean over two side means
```

这不表示三个层级的 direction tensor 或 learned embedding 相同，但说明当前全局 scalar 日志存在代数冗余。真正比较层级差异时应查看每侧、每 DOF 的 mean/RMS/std/max 和方向向量，而不是只使用这三个全局均值。

### 13.6 Internal 与 external 结果

首次 deviation-aware downstream 的绝对结果为：

| Split | Accuracy | Macro-F1 | Macro-AUROC | Affected-side accuracy | Disease loss |
|---|---:|---:|---:|---:|---:|
| Dev validation | **91.46%** | **90.38%** | **98.85%** | 83.78% | **0.3399** |
| Internal test | 85.93% | 83.54% | 96.44% | 76.19% | 0.5108 |
| External test | 81.70% | 81.65% | 94.62% | **84.16%** | 0.8864 |

Internal disease classification 保持较强，AUROC 为 96.44%，说明 deviation representation 仍包含有效类别排序信息。Affected-side 从 validation 到 external 也较稳定，external accuracy 达到 84.16%，表明带符号 bilateral direction 与 magnitude gap 是有效的患侧特征。

主要问题出现在 external disease generalization：external accuracy/F1 相比 validation 分别下降 9.76/8.73 个百分点，disease loss 从 0.3399 增至 0.8864。AUROC 仍有 94.62%，但 loss 明显增大，说明模型仍具有一定排序能力，却在 external 上出现更多错误或更不合适的置信度。

### 13.7 与旧 pooled downstream 的工程参照

以下比较新 run 与旧 `0826_1542`；由于上游 checkpoint 和冻结策略不同，该表不是严格架构消融：

| Metric | 旧 pooled downstream | 新 deviation-aware | 变化 |
|---|---:|---:|---:|
| Dev-validation macro-F1 | **91.20%** | 90.38% | -0.83 pp |
| Internal accuracy | **87.41%** | 85.93% | -1.48 pp |
| Internal macro-F1 | **85.95%** | 83.54% | -2.41 pp |
| Internal AUROC | 95.87% | **96.44%** | +0.58 pp |
| Internal side accuracy | 66.67% | **76.19%** | +9.52 pp |
| External accuracy | **87.58%** | 81.70% | -5.88 pp |
| External macro-F1 | **86.92%** | 81.65% | -5.28 pp |
| External AUROC | **96.55%** | 94.62% | -1.94 pp |
| External side accuracy | 82.18% | **84.16%** | +1.98 pp |

新结构的积极信号是 internal AUROC 和内外部 affected-side accuracy 提升，尤其 internal side accuracy 增加 9.52 个百分点。这与设计中“疾病使用对称偏移、患侧使用带符号偏移”的结构一致。

但 disease accuracy/F1，特别是 external 指标，没有超过旧 pooled baseline。当前不能认为 deviation-aware classifier 已经完成 downstream 优化，也不能据此断言显式偏移特征无效：本轮同时更换了上游随机 realization、分类输入和 encoder freezing，尚未隔离“冻结导致适应不足”“新 head 过拟合”与“偏移特征本身不足”三种原因。

### 13.8 当前结论与下一步

首次实验支持以下判断：

1. **Healthy normative reference 实现有效。** Reference std 分布正常，没有 floor saturation，四级偏移量数值稳定。
2. **冻结 encoder 保持了解释坐标。** 偏移量在训练 epoch 间不漂移，但冻结本身没有解决分类头过拟合。
3. **Signed bilateral feature 对患侧任务有效。** Internal/external side accuracy 均较旧模型提高。
4. **当前 disease head 尚未改善 external transfer。** External accuracy/F1 分别为 81.70%/81.65%，低于旧工程参照。
5. **External 存在更大的总体 reference deviation。** 但需要按 Healthy/ACLD/KOA 和 DOF 分解后，才能区分病理差异与采集域差异。
6. **全局 DOF/side/subject magnitude 日志冗余。** 后续诊断应报告 per-class、per-side、per-DOF 的 direction strength、mean/RMS/std/max，而不是继续比较相同的全局均值。

下一轮最重要的是使用同一个固定 VQ 和 SSL checkpoint 做严格 downstream 对照：

```text
A. 冻结 encoder + 旧 shared/absolute pooled head
B. 冻结 encoder + 当前 deviation-aware head
C. 冻结 encoder + 更小的线性/单层 deviation classifier
D. 当前 deviation head + 仅解冻最后一个 Sentence block，encoder LR 为 head 的 1/10～1/30
```

这样可以分别回答：

- 收益或下降来自显式 deviation feature，还是来自冻结策略；
- 当前多层 deviation projection 是否相对 KGKD 样本量过大；
- 少量 encoder adaptation 是否能恢复 external disease performance，同时保留健康参考结构。

同时应增加下列只基于 downstream train/validation 的诊断：

```text
Healthy / ACLD / KOA 各自的 word/DOF/side/subject magnitude
每类 subject_deviation_direction centroid
逐 DOF mean/RMS/std/max 与 class effect size
三分类 confusion matrix、per-class recall/F1
external 只做最终泛化审计
```

在完成固定 checkpoint 的对照前，当前 deviation-aware model 应保留为可解释 downstream baseline，但不应直接替换旧 pooled model 作为性能最佳模型。
