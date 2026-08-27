# Healthy Gait-Language Sentence Encoder SSL Task Design

## 1. 目标与边界

本阶段只优化健康人的 `GaitLanguageSentenceEncoder`。VQ gait-word tokenizer、DOF-specific codebook 和 VQ Decoder 已完成结构选择，在 Sentence SSL 阶段作为固定的健康 gait-word 语义系统使用。

训练数据只来自 `ssl_data` 中的健康受试者；`ssl_validation_data` 用于 early stopping 和 SSL 任务选择。KGKD Healthy、ACLD、KOA、internal test 和 external test 不参与 Sentence Encoder 的 SSL 参数更新。

核心输入为：

```text
words     [B, 2, W, 6, T_word]
word_mask [B, 2, W]
timing    [B, 2, W, 4]
```

`timing` 四个通道固定为：

```text
0 cycle duration，单位为秒
1 normalized cycle center position，记录内归一化到 recording length
2 interval to preceding cycle，单位为秒
3 cycle quality score
```

左右两侧是在不同时间先后采集的两条独立记录。因此左右 `timing[..., 1]` 只表示各自记录内部的位置，不能相减解释为真实双侧相位差，也不能作为同步 gait-cycle alignment。

## 2. Token 构造

每个 token 对应：

```text
单侧 × 单周期 × 单 DOF
```

最终 token 明确定义为：

```text
token
= shape_embedding
+ DOF_embedding
+ side_embedding
+ cycle_position_embedding
+ continuous_timing_embedding
+ duration_embedding
+ interval_embedding
+ quality_embedding
```

各部分含义如下：

### 2.1 Shape embedding

```text
shape_embedding = VQ word encoder(cycle waveform)
```

输入是单侧、单周期、单 DOF 的标准化 100 点波形。Sentence Encoder 使用从健康 VQ tokenizer 复制的 word encoder，将连续波形映射到 `word_dim` 维形态表示。

被 masked-word task 选中的位置不使用原始 shape embedding，而使用该 DOF 对应的 learned mask token。

### 2.2 DOF 与 side embedding

```text
DOF_embedding   [6, D]
side_embedding  [2, D]
```

六个 DOF 保留独立身份；左右侧共享健康 codebook，但 token 仍保留 side identity，以便学习同一形态在左右侧语境中的差异。

### 2.3 Cycle position embedding

```text
cycle_position_embedding [max_words, D]
```

它表示单侧记录内部的离散周期顺序 `0...W-1`，只用于记录内 temporal modeling，不表示左右侧之间的同步周期编号。

### 2.4 Continuous timing embedding

```text
continuous_timing_embedding
= MLP(normalized cycle center position)
```

它表达周期在本侧完整记录中的连续位置。该值只在同一侧内部有序，不进入左右相对时间 bias。

### 2.5 Duration、interval 与 quality embedding

三个标量分别使用独立 MLP：

```text
duration_embedding = MLP(cycle duration seconds)
interval_embedding = MLP(preceding interval seconds)
quality_embedding  = MLP(cycle quality score)
```

独立 embedding 避免原先四通道联合 timing MLP 将节律、记录位置与质量不可控地混合，也允许 rhythm task 对指定 timing 分量执行显式 mask。

## 3. Sentence Encoder 主体

每个 block 依次执行：

```text
temporal axis attention
→ DOF axis attention
→ optional content-based contralateral cross-attention
```

### 3.1 Temporal axis

固定 side 和 DOF，沿周期轴 `W` 建模：

```text
[B, 2, W, 6, D]
→ [B × 2 × 6, W, D]
```

该轴学习同一 DOF 的周期序列、形态变化和记录内节律上下文。

### 3.2 DOF axis

固定 side 和 cycle，沿六个 DOF 建模：

```text
[B, 2, W, 6, D]
→ [B × 2 × W, 6, D]
```

该轴学习同一周期内 FE、VV、IE、AP、ML、SI 之间的健康运动学耦合。

### 3.3 双侧处理原则

默认 Sentence Encoder 不执行左右周期级 cross-attention。左右侧分别经过 temporal 和 DOF blocks，随后各自池化为 subject-side representation。

纯对侧预测任务需要从对侧向目标侧传递信息时，显式开启 content-based contralateral cross-attention。该 attention：

- 不使用左右 center time difference；
- 不加入 relative-time bias；
- 不假设左周期 `i` 与右周期 `i` 同步；
- 只根据健康形态和 sentence content 学习软关联。

## 4. SSL Task 1：Within-DOF masked word prediction

### 4.1 Mask

在全部有效 `side × cycle × DOF` 位置中按 `word_mask_ratio` 随机选择 target，并按 `span_length` 扩展输入 mask。

### 4.2 条件信息

主要使用：

```text
同一侧
+ 同一 DOF
+ 前后周期
+ 记录内 timing/rhythm context
```

默认不启用双侧 cross-attention。

### 4.3 目标与损失

Within-DOF 仍预测冻结健康 codebook 的唯一 code ID：

```text
L_within = CE(logits, assigned_code_id)
```

该任务保留严格分类目标，用于衡量同一 DOF temporal context 是否足以恢复被 mask 的健康 gait word。

报告：

```text
within_loss
within_accuracy
```

## 5. SSL Task 2：Whole-DOF / Cross-DOF prediction

### 5.1 Mask

对每个 subject-side 随机选择一个完整 DOF，并 mask 该 DOF 的全部有效周期：

```text
target DOF waveform tokens → DOF-specific mask token
```

模型必须利用同侧其他五个 DOF、周期顺序和 timing 恢复目标 DOF。

### 5.2 Top-k codebook neighborhood soft target

严格唯一 code ID 可能惩罚与目标 prototype 非常接近的健康 code。对真实 assigned code `c*`，首先在同一 DOF codebook 内计算：

```text
s_j = cosine(v_c*, v_j)
```

选择最接近的 `k` 个 code：

```text
N_k(c*) = TopK_j(s_j)
```

再通过温度参数构建距离加权 soft target：

```text
q_j = softmax(s_j / temperature),  j ∈ N_k(c*)
q_j = 0,                           j ∉ N_k(c*)
```

soft discrete loss 为：

```text
L_cross_soft = -Σ_j q_j log p_j
```

### 5.3 Continuous prototype prediction

除 code logits 外，每个 DOF 还有独立的 continuous prototype head：

```text
prototype_hat = PrototypeHead_d(contextual_token)
```

目标为冻结 codebook 中 assigned code 的 normalized prototype：

```text
L_cross_prototype
= 1 - cosine(prototype_hat, v_c*)
```

### 5.4 可选 hard code loss

保留可调 hard CE：

```text
L_cross_hard = CE(logits, c*)
```

默认 hard weight 为 `0.0`，即默认优化 top-k distance-weighted soft classification 与 continuous prototype regression，而不是强迫唯一 code ID。

Cross-DOF 总损失：

```text
L_cross
= hard_weight      × L_cross_hard
+ soft_weight      × L_cross_soft
+ prototype_weight × L_cross_prototype
```

报告 exact accuracy 和 top-k accuracy；top-k accuracy 表示真实 assigned code 是否出现在预测 logits 的前 `k` 名中。

## 6. SSL Task 3：Explicit rhythm prediction

### 6.1 Mask

按 `rhythm_mask_ratio` 选择有效 `side × cycle`。目标周期的下列输入 embedding 被替换为 learned timing mask embedding：

```text
continuous center timing
cycle duration
preceding interval
```

shape、DOF、side、cycle position 和 quality embedding 保留。模型需要利用目标周期波形、同侧相邻周期和整体节律上下文恢复被隐藏的真实 timing。

### 6.2 Cycle-level representation

同一周期六个 DOF 的 contextual tokens 做均值池化：

```text
cycle_embedding = mean_DOF(tokens)
```

Rhythm head 同时输出：

```text
duration_hat
preceding_interval_hat
```

### 6.3 损失

```text
L_duration = SmoothL1(duration_hat, duration_seconds)
L_interval = SmoothL1(interval_hat, preceding_interval_seconds)

L_rhythm
= duration_prediction_weight × L_duration
+ interval_prediction_weight × L_interval
```

报告：

```text
rhythm_loss
duration_loss
interval_loss
duration_mae
interval_mae
```

所有节律目标都只在单侧记录内部定义，不构造左右 phase difference。

## 7. SSL Task 4：Healthy bilateral pair discrimination

### 7.1 任务动机

左右侧不是同步采集，不能进行真实 cycle-to-cycle phase matching。但同一健康受试者的两侧完整 sentence 仍共享个体解剖、稳定运动模式和健康双侧统计结构。

因此双侧任务改为 subject-level 配对判别：

```text
正样本：同一健康受试者的 left sentence + right sentence
负样本：当前受试者的 left sentence + 另一受试者的 right sentence
```

### 7.2 Side representation

左右侧先独立完成 temporal 和 DOF modeling，再分别对全部有效 cycle 和 DOF 池化：

```text
left_embedding  [B,D]
right_embedding [B,D]
```

pair feature 使用对左右顺序不敏感的组合：

```text
pair_feature
= concat(
    abs(left_embedding - right_embedding),
    left_embedding × right_embedding
  )
```

### 7.3 负样本

每个 batch 将 right embeddings 按随机非零 offset 循环移动，保证负样本来自不同受试者：

```text
negative_right = roll(right_embedding, nonzero_shift)
```

每个正样本对应一个负样本。

### 7.4 损失与指标

```text
L_pair = BCEWithLogits(pair_score, same_subject_label)
```

报告：

```text
bilateral_pair_loss
bilateral_pair_accuracy
```

该任务学习真实健康双侧配对与伪配对的差异，不学习不存在的同步相位关系。

## 8. SSL Task 5：Pure contralateral prediction

### 8.1 Mask 与条件

随机选择一个 target side，并 mask 该侧全部 shape tokens。模型通过不含 relative-time bias 的 content cross-attention，从完整 contralateral sentence 向目标侧传递信息。

loss target 只在 target side 中按 `contralateral_mask_ratio` 采样的位置计算。

### 8.2 Relaxed target

Pure contralateral prediction 使用与 Cross-DOF 相同的联合目标：

```text
L_contralateral
= hard_weight      × L_contralateral_hard
+ soft_weight      × L_contralateral_soft
+ prototype_weight × L_contralateral_prototype
```

其中 soft target 是同一 DOF codebook 内 top-k cosine-neighbor 的距离加权分布，continuous target 是 assigned code prototype embedding。

报告：

```text
contralateral_loss
contralateral_hard_loss
contralateral_soft_loss
contralateral_prototype_loss
contralateral_accuracy
contralateral_topk_accuracy
```

由于左右采集不同步，即使使用 relaxed target，该任务仍然是高难度探索任务。它衡量对侧完整健康 sentence 对目标侧形态分布的条件信息，而不是逐周期同步预测能力。

## 9. 默认关闭的旧双侧任务

### 9.1 Bilateral contextual masked-code prediction

旧任务在部分 mask 目标侧 word 后预测唯一 code。它仍保留为可选消融，但默认关闭：

```text
--no-ssl-bilateral-context-task
```

启用后使用 content cross-attention，不再使用 relative-time bias。

### 9.2 Left-right swap consistency

shared/absolute/directional representation 的 swap consistency 仍保留为可选消融，但默认关闭：

```text
--no-ssl-swap-task
```

当前双侧学习的默认主任务是 subject-level healthy pair discrimination。

## 10. 总损失

启用任务集合为 `T` 时：

```text
L_SSL
= within_weight          × L_within
+ cross_dof_weight       × L_cross
+ rhythm_weight          × L_rhythm
+ bilateral_weight       × L_bilateral_context
+ contralateral_weight   × L_contralateral
+ bilateral_pair_weight  × L_pair
+ swap_weight            × L_swap
```

关闭的任务不执行 encoder forward，也不进入总损失。

当前默认启用：

```text
within-DOF masked word
whole-DOF / cross-DOF relaxed prediction
explicit rhythm prediction
pure contralateral relaxed prediction
healthy bilateral pair discrimination
```

当前默认关闭：

```text
bilateral contextual masked-code prediction
left-right swap consistency
```

## 11. `run.py::get_args()` 实验选项

### 11.1 任务开关

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `--ssl-within-task / --no-ssl-within-task` | on | Within-DOF masked word |
| `--ssl-cross-dof-task / --no-ssl-cross-dof-task` | on | Whole-DOF relaxed prediction |
| `--ssl-rhythm-task / --no-ssl-rhythm-task` | on | Duration/interval rhythm prediction |
| `--ssl-bilateral-context-task / --no-ssl-bilateral-context-task` | off | 旧 bilateral contextual masked-code task |
| `--ssl-contralateral-task / --no-ssl-contralateral-task` | on | Pure contralateral relaxed prediction |
| `--ssl-bilateral-pair-task / --no-ssl-bilateral-pair-task` | on | 同受试者/跨受试者双侧配对判别 |
| `--ssl-swap-task / --no-ssl-swap-task` | off | Left-right swap consistency |

### 11.2 Mask 与任务总权重

| 参数 | 默认值 |
|---|---:|
| `--word-mask-ratio` | 0.30 |
| `--span-length` | 2 |
| `--bilateral-mask-ratio` | 0.30 |
| `--contralateral-mask-ratio` | 0.30 |
| `--rhythm-mask-ratio` | 0.30 |
| `--within-weight` | 1.00 |
| `--cross-dof-weight` | 1.00 |
| `--rhythm-weight` | 0.50 |
| `--duration-prediction-weight` | 1.00 |
| `--interval-prediction-weight` | 1.00 |
| `--bilateral-weight` | 1.00 |
| `--contralateral-weight` | 0.50 |
| `--bilateral-pair-weight` | 1.00 |
| `--swap-weight` | 0.10 |

### 11.3 Relaxed conditional prediction

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `--conditional-code-top-k` | 5 | soft target neighborhood 与 top-k accuracy 的 k |
| `--conditional-soft-target-temperature` | 0.10 | codebook cosine soft target 温度 |
| `--cross-dof-hard-code-weight` | 0.00 | Cross-DOF 唯一 code CE 权重 |
| `--cross-dof-soft-code-weight` | 1.00 | Cross-DOF top-k soft CE 权重 |
| `--cross-dof-prototype-weight` | 1.00 | Cross-DOF continuous prototype loss 权重 |
| `--contralateral-hard-code-weight` | 0.00 | Contralateral 唯一 code CE 权重 |
| `--contralateral-soft-code-weight` | 1.00 | Contralateral top-k soft CE 权重 |
| `--contralateral-prototype-weight` | 1.00 | Contralateral continuous prototype loss 权重 |

## 12. 训练与 checkpoint 选择

训练阶段固定冻结：

```text
target VQ tokenizer
copied VQ word encoder used for shape embedding
target DOF-specific codebook
VQ Decoder
```

Sentence context blocks、token identity/timing embeddings 和各 SSL prediction heads 由健康 SSL 数据训练。固定 copied word encoder 可以避免 Sentence SSL 反向改变已经完成选择的 gait-word morphology 坐标系。

由于 token timing 参数和 SSL heads 已改变，旧 `best_ssl.pt` 不与当前 Sentence Encoder 结构兼容。新一轮 Sentence SSL 应从已确定的 `best_vq.pt` 开始训练新的 `best_ssl.pt`，而不是继续加载旧 SSL checkpoint。

Validation 使用固定 `validation_mask_seed`。同一固定随机流同时控制：

```text
masked word positions
target DOF
rhythm mask positions
target side
contralateral target positions
negative bilateral pairing shift
```

这样每个 epoch 的 validation tasks 和负配对保持可比。`best_ssl.pt` 继续按照 validation total SSL loss 选择。

## 13. 建议的消融顺序

第一组先验证新增主任务整体可训练：

```text
within + cross-DOF relaxed + rhythm + contralateral relaxed + pair
```

随后一次只改变一个因素：

```text
1. 移除 rhythm task
2. 移除 bilateral pair task
3. 移除 contralateral task
4. Cross-DOF: hard-only vs soft-only vs soft+prototype
5. Contralateral: hard-only vs soft-only vs soft+prototype
6. top-k ∈ {3, 5, 10}
7. soft-target temperature ∈ {0.05, 0.10, 0.20}
8. 可选重新启用 bilateral contextual 或 swap task
```

任务判断不能只看 total SSL loss。至少分别报告：

```text
Within exact accuracy
Cross-DOF exact/top-k accuracy 与 prototype loss
Duration/interval MAE
Contralateral exact/top-k accuracy 与 prototype loss
Bilateral pair accuracy
冻结健康模型后的 downstream/internal generalization
```

不同 VQ codebook 的 code label space 不同；跨 VQ 实验比较时，应优先使用 top-k、prototype cosine、rhythm MAE、pair accuracy 和 downstream transfer，而不是只比较 raw code CE。

## 14. 首次新版 SSL Task 实验结果（2026-08-26）

### 14.1 实验与可比性

最新完整实验为：

```text
Results/gait_language/dev_exp/dev_exp_0826_1542
seed = 42
VQ residual_energy_weight = 0.01
```

启用任务：

```text
Within-DOF masked word
Cross-DOF top-k soft code + continuous prototype
Explicit duration / preceding-interval rhythm prediction
Pure contralateral top-k soft code + continuous prototype
Healthy bilateral pair discrimination
```

关闭任务：

```text
Bilateral contextual masked-code prediction
Left-right swap consistency
```

训练状态如下；epoch 使用 `metrics.jsonl` 中从 0 开始的编号：

| 阶段 | 最佳 checkpoint epoch | 实际运行 epoch 数 | checkpoint 规则 |
|---|---:|---:|---|
| VQ | 86 | 97 | minimum validation total loss |
| SSL | 47 | 58 | minimum validation total loss |
| Downstream | 8 | 19 | maximum dev-validation macro-F1 |

`dev_exp_0825_1713` 是最接近的旧 SSL 参照：两者均使用 seed=42、相同数据、`residual_energy_weight=0.01` 和相同 VQ 超参数，`word_statistics.json` 也完全一致。但两个 `best_vq.pt` 的 SHA-256 不同，最佳 VQ epoch 分别为 84 和 86；新版 run 的 VQ validation final MSE 为 0.065385，旧 run 为 0.065672，scaled residual RMS 则分别为 0.224811 和 0.208758。因此 0825/0826 downstream 差异不能完全归因于 SSL task，严格任务消融仍需复用同一个固定 VQ checkpoint。

### 14.2 最佳 SSL checkpoint 的总损失分解

`best_ssl.pt` 位于 epoch 47。Validation total loss 为：

```text
L_SSL = 8.604829
```

各任务贡献如下：

| Task | Raw validation loss | 外层权重 | 加权贡献 | Total 占比 |
|---|---:|---:|---:|---:|
| Within-DOF | 1.800016 | 1.0 | 1.800016 | 20.92% |
| Cross-DOF relaxed | 3.999860 | 1.0 | 3.999860 | 46.48% |
| Rhythm | 0.059674 | 0.5 | 0.029837 | 0.35% |
| Contralateral relaxed | 4.757993 | 0.5 | 2.378997 | 27.65% |
| Bilateral pair | 0.396119 | 1.0 | 0.396119 | 4.60% |
| Bilateral contextual / swap | 0 | disabled | 0 | 0% |

Cross-DOF 和 contralateral 合计约占 total loss 的 74.1%，是当前优化的主要梯度来源；rhythm 加权后只占 0.35%。这不表示 rhythm 没有学习，而是说明 raw regression loss 与 code-distribution loss 的自然尺度差异很大，当前任务权重不能直接按数值相等理解。

Cross-DOF loss 内部为：

```text
soft code loss       = 3.716517
prototype loss       = 0.283344
hard code loss       = 3.550388  # 只记录，默认权重为 0
```

Contralateral loss 内部为：

```text
soft code loss       = 4.323176
prototype loss       = 0.434817
hard code loss       = 4.278830  # 只记录，默认权重为 0
```

### 14.3 各 SSL task 的学习结果

下表中的 epoch 0 已经完成一个训练 epoch，不是未训练随机初始化；“全程最佳”用于观察各任务自己的最优点是否与 total-loss checkpoint 一致。

| Metric | Epoch 0 validation | Epoch 47 checkpoint | 全程最佳 |
|---|---:|---:|---:|
| Within exact accuracy | 0.020334 | 0.446673 | 0.464784 @ 54 |
| Cross-DOF exact accuracy | 0.020153 | 0.125317 | 0.129555 @ 57 |
| Cross-DOF top-5 accuracy | 0.080943 | 0.384157 | 0.389129 @ 49 |
| Cross-DOF prototype loss ↓ | 0.529936 | 0.283344 | 0.276989 @ 57 |
| Duration MAE, seconds ↓ | 0.100541 | 0.067604 | 0.061727 @ 53 |
| Preceding interval MAE, seconds ↓ | 0.146322 | 0.147808 | 0.138916 @ 54 |
| Contralateral exact accuracy | 0.017465 | 0.056791 | 0.061487 @ 50 |
| Contralateral top-5 accuracy | 0.072936 | 0.220074 | 0.222434 @ 43 |
| Contralateral prototype loss ↓ | 0.551551 | 0.434817 | 0.433793 @ 44 |
| Bilateral pair loss ↓ | 0.527916 | 0.396119 | 0.347164 @ 28 |
| Bilateral pair accuracy | 0.730488 | 0.841463 | 0.856098 @ 41 |

#### Within-DOF

Exact accuracy 从第一个 epoch 的 2.03% 提升到 44.67%，全程最高为 46.48%。Temporal context 能够稳定恢复同一 DOF 的 masked gait word，任务没有因新增多任务目标而失效。

#### Cross-DOF relaxed prediction

随机 128 类 exact accuracy 约为 0.78%，随机 top-5 accuracy 约为 3.91%。当前 checkpoint 达到 12.53% exact 和 38.42% top-5，明显高于随机水平。Prototype cosine loss 从 0.5299 降到 0.2833，对应平均 cosine similarity 从约 0.470 提升到约 0.717。

这说明其他五个 DOF 对目标 DOF code neighborhood 和连续 prototype 都提供了强条件信息。Exact accuracy 不是默认直接优化目标，因此应优先使用 top-5 accuracy、soft loss 和 prototype loss 判断该任务。

#### Explicit rhythm prediction

Duration MAE 在 checkpoint 降至 0.0676 秒，约为 SSL train 健康周期平均 duration 0.91 秒的 7.4%，全程最低达到 0.0617 秒。Duration 可以由目标周期形态、相邻周期 timing 和记录内节律上下文有效恢复。

Preceding interval MAE 在 epoch 47 为 0.1478 秒，与 epoch 0 的 0.1463 秒接近，全程最低 0.1389 秒出现在 epoch 54。当前实验对 duration 的学习明显强于 interval；interval task 尚未表现出稳定的持续改善。

#### Pure contralateral relaxed prediction

Checkpoint 的 exact accuracy 为 5.68%，top-5 accuracy 为 22.01%，均显著高于随机 0.78%/3.91%。Prototype loss 从 0.5516 降到 0.4348，对应 cosine similarity 从约 0.448 提升到约 0.565。

即使左右侧异步采集，对侧完整 sentence 仍包含目标侧健康 code distribution 的条件信息。但该任务明显比 Cross-DOF 困难：top-5 accuracy 低约 16.4 个百分点，prototype similarity 低约 0.15。当前 `contralateral_weight=0.5` 与其难度相匹配，不建议仅为提高 exact accuracy 恢复同步相位假设。

#### Healthy bilateral pair discrimination

Validation pair accuracy 在 checkpoint 达到 84.15%，全程最高 85.61%，显著高于 50% 的平衡随机基线。这证明不依赖同步相位，仅使用左右完整 sentence 的 subject-level pooled representation，也能学习真实健康双侧配对与跨受试者伪配对的差异。

该任务同时表现出本轮最明显的过拟合：epoch 47 的 train accuracy 为 92.57%，validation accuracy 为 84.15%，相差 8.42 个百分点；validation pair loss 在 epoch 28 已达到最低 0.3472，之后虽然 accuracy 仍有波动提升，但置信度校准开始变差。

### 14.4 多任务 checkpoint 的折中

不同任务的 validation 最佳 epoch 并不一致：

```text
Bilateral pair loss            epoch 28
Bilateral pair accuracy        epoch 41
Contralateral top-5            epoch 43
Contralateral loss/prototype   epoch 44
Total SSL loss                 epoch 47
Cross-DOF top-5                epoch 49
Duration MAE                   epoch 53
Within / interval              epoch 54
Cross-DOF prototype            epoch 57
```

因此 epoch 47 是多任务 total loss 的折中点，不是每个任务的单独最优点。后期 total validation loss 回升主要伴随 pair loss 和 contralateral loss 恶化，而 duration、interval、within 和 Cross-DOF prototype 仍有改善。当前 early stopping 工作正常，但 total loss 被不同自然尺度的任务主导，后续 task-weight 消融需要同时观察分任务指标。

### 14.5 Downstream 与旧任务参照

以下将新版 `dev_exp_0826_1542` 与相同 residual 权重的旧任务 run `dev_exp_0825_1713` 比较：

| 指标 | 旧 SSL tasks | 新 SSL tasks | 变化 |
|---|---:|---:|---:|
| Dev-validation macro-F1 | 0.891113 | **0.912048** | +2.09 pp |
| Internal accuracy | **0.888889** | 0.874074 | -1.48 pp |
| Internal macro-F1 | **0.863501** | 0.859463 | -0.40 pp |
| Internal AUROC | **0.962805** | 0.958653 | -0.42 pp |
| Internal side accuracy | 0.634921 | **0.666667** | +3.17 pp |
| External accuracy | 0.866013 | **0.875817** | +0.98 pp |
| External macro-F1 | 0.859708 | **0.869238** | +0.95 pp |
| External AUROC | 0.954936 | **0.965529** | +1.06 pp |
| External side accuracy | **0.831683** | 0.821782 | -0.99 pp |

新版任务取得更高的 dev-validation macro-F1，并同时改善三项 external disease-classification 指标；internal disease 指标小幅下降，affected-side 在 internal 改善而 external 略降。整体上没有出现迁移能力崩溃，新任务对 external disease representation 显示积极信号。

但该表不是严格的单变量 SSL task 消融，因为两个 run 的 VQ checkpoint 不完全相同，而且 downstream Sentence Encoder 默认继续微调。不能把约 1 个百分点的 external 改善全部归因于 rhythm、soft target 或 pair task 中的某一个组件。

另外，新旧 SSL total loss 不可直接比较：旧 loss 使用 hard code CE、bilateral contextual 和 swap，新 loss 使用 soft/prototype、rhythm 与 pair，loss 组成和尺度已经改变。

### 14.6 当前结论

首次新版 SSL 实验支持以下判断：

1. **新版默认任务组合整体可训练。** 五个启用任务都产生了非平凡学习信号，没有发现 task collapse 或 total loss 数值异常。
2. **Cross-DOF relaxed target 是当前最成熟的新目标。** Top-5 和 prototype prediction 都显著优于随机，并且 train/validation gap 较小。
3. **Pure contralateral relaxed prediction 可保留。** 异步采集限制了上限，但其 top-5 与 prototype 指标证明对侧 sentence 仍包含健康条件信息。
4. **Subject-level bilateral pair 设计成立。** 84%～86% validation accuracy 证明真实配对与伪配对可区分，不需要制造不存在的左右周期同步关系。
5. **Pair task 需要控制过拟合。** 它是 train/validation gap 最大、最早达到最低 validation loss 的任务。
6. **Rhythm task 中 duration 有效，interval 仍弱。** 当前 duration MAE 明显改善，interval MAE 改善小且最佳点较晚。
7. **当前 total loss 权重不平衡。** Cross-DOF 与 contralateral 主导总损失，rhythm 几乎不影响 checkpoint 选择；这应作为下一轮 SSL 消融的主要优化点。

### 14.7 下一步建议

第一优先级是建立严格的 SSL task 对照，而不是立即增加更多任务：

```text
固定同一个 best_vq.pt
固定 subject split、validation mask seed 和训练 seed
旧 SSL tasks vs 新 SSL tasks
```

在固定 VQ 后，建议按以下顺序消融：

1. 保留完整新版任务作为基线；
2. 移除 bilateral pair，确认 external disease improvement 是否依赖配对学习；
3. 移除 rhythm，确认 duration/interval 是否对迁移有实际贡献；
4. 移除 contralateral，只保留 within + cross-DOF + rhythm + pair；
5. 比较 Cross-DOF `soft-only` 与 `soft+prototype`；
6. 比较 Contralateral `soft-only` 与 `soft+prototype`。

针对当前诊断，后续优化方向为：

- 对 duration 和 interval 使用 SSL healthy train 拟合的标准化目标，再比较 `rhythm_weight=0.5/1.0`；否则 raw regression loss 的尺度过小，几乎不参与 checkpoint 选择。
- Pair task 使用更多 batch 内负样本或对比学习形式，而不是继续提高 `bilateral_pair_weight`；当前问题是泛化差距，不是训练信号不足。
- 保留 `contralateral_weight=0.5`，优先观察 top-5 和 prototype loss，不再恢复 relative-time bias。
- 继续将 bilateral contextual 与 swap task 保持默认关闭，除非作为明确消融重新启用。
- 至少重复 3 个训练 seed，并报告每个 SSL task 指标及 downstream 均值/标准差。

External test 只应作为本次完成实验的泛化审计，不应据此继续选择 task weight。正式 task selection 应使用 SSL validation 与 dev-validation；最终确认需要未用于反复调参的测试数据。

## 15. 固定 VQ 的新版 SSL Task 消融结果（2026-08-26）

### 15.1 实验设置与公平性核对

本轮六个实验均固定加载同一个 VQ checkpoint：

```text
Results/gait_language/dev_exp/dev_exp_0826_1542/best_vq.pt
```

实验与消融项的对应关系如下：

| Run | Ablation | 相对完整基线的唯一任务变化 |
|---|---|---|
| `ablation_exp_0826_1655` | `full` | 无；完整新版任务 |
| `ablation_exp_0826_1719` | `no_bilateral_pair` | 关闭 healthy bilateral pair |
| `ablation_exp_0826_1759` | `no_rhythm` | 关闭 duration 与 preceding-interval prediction |
| `ablation_exp_0826_1825` | `no_contralateral` | 关闭 pure contralateral prediction |
| `ablation_exp_0826_1847` | `cross_dof_soft_only` | Cross-DOF prototype weight：1.0 → 0.0 |
| `ablation_exp_0826_1914` | `contralateral_soft_only` | Contralateral prototype weight：1.0 → 0.0 |

逐字段比较 `args.json` 后，除上述目标变量以及 run 名称等输出元数据外，其余参数完全相同。所有实验均使用：

```text
seed = 42
validation_mask_seed = 10042
相同 subject split
相同 SSL/downstream 最大 epoch、patience 与学习率
相同 VQ checkpoint
```

六份 `word_statistics.json` 的 SHA-256 均为：

```text
a0c4dc63e8a451860a39f3d8fa373d9e4ecf041c763b48e21e4ff4794ad27bc3
```

因此，本轮是严格的固定数据、固定 VQ、固定 seed 单变量消融。需要注意：关闭任务或关闭 prototype loss 会改变 SSL total loss 的组成和自然尺度，所以不同变体的 total loss 数值不能横向排名。Early stopping 规则虽然相同，但由于优化目标不同，最佳 epoch 可以不同。

### 15.2 Early stopping 与 checkpoint 状态

epoch 仍使用 `metrics.jsonl` 中从 0 开始的编号；“运行数”是实际完成的 epoch 数量。

| Ablation | SSL 最佳 epoch / 运行数 | Downstream 最佳 epoch / 运行数 | Dev-validation macro-F1 |
|---|---:|---:|---:|
| Full | 47 / 58 | 16 / 27 | 0.893876 |
| No bilateral pair | 79 / 90 | 2 / 13 | 0.889060 |
| No rhythm | 50 / 61 | 18 / 29 | 0.892619 |
| No contralateral | 51 / 62 | 5 / 16 | 0.876380 |
| Cross-DOF soft-only | 45 / 56 | 2 / 13 | 0.895238 |
| Contralateral soft-only | 47 / 58 | 9 / 20 | **0.898878** |

移除 pair 后 SSL 持续到 epoch 79 才取得最低 validation loss，明显晚于完整基线的 epoch 47；但其 downstream 最佳点在 epoch 2，且最终测试结果下降。由此可见，更低或更晚收敛的 SSL objective 并不自动对应更好的疾病迁移表示。

Dev-validation macro-F1 也没有可靠预测 external 排名：两个 soft-only 变体的 dev-validation F1 均略高于完整基线，但 external macro-F1 都更低；`no_rhythm` 的 dev-validation F1 与完整基线只差 0.13 个百分点，external F1 却下降 10.12 个百分点。这表明当前 dev-validation split 对 external domain shift 的代理能力有限，不能仅凭单次 dev 最优值决定任务取舍。

### 15.3 Downstream 绝对结果

下表全部来自各 run 的 `evaluation.json`，以百分比表示：

| Ablation | Internal Acc | Internal F1 | Internal AUROC | Internal Side Acc | External Acc | External F1 | External AUROC | External Side Acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Full | 85.93 | 82.51 | **96.75** | 68.25 | **89.54** | **88.90** | 96.26 | 84.16 |
| No bilateral pair | 82.22 | 80.52 | 95.15 | 61.90 | 87.25 | 86.38 | 96.14 | 76.24 |
| No rhythm | 86.67 | 84.49 | 95.95 | **73.02** | 78.43 | 78.78 | 92.58 | 83.17 |
| No contralateral | 83.70 | 80.08 | 95.90 | 66.67 | 88.24 | 87.46 | **96.79** | **86.14** |
| Cross-DOF soft-only | 82.22 | 80.82 | 95.10 | 69.84 | 85.29 | 84.86 | 96.32 | 81.19 |
| Contralateral soft-only | **87.41** | **85.59** | 96.42 | 65.08 | 86.93 | 86.16 | 96.54 | 85.15 |

完整任务不是每一个 internal 指标的最高点，但它取得了最高的 external accuracy 和 external macro-F1，同时保持最高的 internal AUROC，是当前六个单次实验中最均衡、external disease classification 最强的配置。

相对完整基线的百分点变化如下：

| Ablation | Δ Internal Acc | Δ Internal F1 | Δ External Acc | Δ External F1 | Δ External AUROC | Δ External Side Acc |
|---|---:|---:|---:|---:|---:|---:|
| No bilateral pair | -3.70 | -1.99 | -2.29 | -2.53 | -0.12 | -7.92 |
| No rhythm | +0.74 | +1.98 | **-11.11** | **-10.12** | **-3.68** | -0.99 |
| No contralateral | -2.22 | -2.43 | -1.31 | -1.44 | +0.53 | +1.98 |
| Cross-DOF soft-only | -3.70 | -1.69 | -4.25 | -4.05 | +0.06 | -2.97 |
| Contralateral soft-only | +1.48 | +3.08 | -2.61 | -2.74 | +0.28 | +0.99 |

### 15.4 最佳 SSL checkpoint 的分任务指标

为了避免 total loss 组成不同造成误判，下表只比较各自最佳 SSL checkpoint 的同名 validation 指标。MAE 单位为秒；关闭的任务记为 `—`。

| Ablation | Within exact | Cross-DOF top-5 | Duration MAE | Interval MAE | Contralateral top-5 | Pair accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Full | 45.06% | 36.78% | 0.0586 | 0.1724 | 20.33% | 84.15% |
| No bilateral pair | 48.11% | 41.00% | 0.0536 | 0.1585 | 20.59% | — |
| No rhythm | 46.16% | 37.88% | — | — | 22.80% | 85.61% |
| No contralateral | 46.90% | 37.40% | 0.0624 | 0.1574 | — | 85.24% |
| Cross-DOF soft-only | 45.76% | 36.34% | 0.0611 | 0.1645 | 21.60% | 85.12% |
| Contralateral soft-only | 45.70% | 37.32% | 0.0617 | 0.1828 | 21.18% | 85.85% |

完整基线的分任务结果与上一轮新版 SSL 实验一致：Within、Cross-DOF、rhythm、contralateral 和 pair 均保持非平凡学习信号。不同消融会释放模型容量并改变 early-stopping 位置，因此移除一个任务后其他 SSL 指标小幅上升是正常现象，不能据此认为被移除任务无效。最典型的是 `no_bilateral_pair`：Within 和 Cross-DOF top-5 分别提高约 3.05 和 4.23 个百分点，但内外部 downstream 指标仍同时下降。

### 15.5 各消融的解释

#### 15.5.1 Bilateral pair task

移除 pair 后，internal/external macro-F1 分别下降 1.99/2.53 个百分点，affected-side accuracy 分别下降 6.35/7.92 个百分点；八项 downstream 指标全部下降。与此同时，Within、Cross-DOF 和 rhythm validation 指标反而改善。

这说明 pair task 的价值不是提高局部 code reconstruction，而是为 sentence encoder 增加 subject-level、双侧共享的健康步态表征。它对 affected-side 任务的影响尤其明显，也支持“真实同受试者左右配对 vs 跨受试者伪配对”的设计。当前证据支持保留 pair task，而不是因为其 train/validation gap 较大就直接移除；后续应优化它的泛化方式，例如 batch 内更多负样本或对比学习形式。

#### 15.5.2 Rhythm task

移除 rhythm 后 internal accuracy/F1 分别上升 0.74/1.98 个百分点，但 external accuracy、macro-F1 和 AUROC 分别下降 11.11、10.12 和 3.68 个百分点，是所有消融中最大的 external disease degradation。

该结果表明显式节律监督很可能提供了跨数据域更稳定的生理信息：即使 rhythm 加权贡献在 SSL total loss 中很小，它仍可能约束 sentence representation 使用 duration/interval，而不是只拟合数据集特有的 code 共现模式。Internal 上升而 external 大幅下降也符合“无 rhythm 后更容易贴合开发域、但域外泛化变差”的解释。

本次消融同时关闭 duration 与 preceding interval，不能单独证明二者各自的贡献。结合完整任务中 duration MAE 明显优于 interval 的训练诊断，当前更可能由 duration prediction 提供主要收益，但这仍需通过 `duration-only` 与 `interval-only` 消融确认，不能由本轮结果直接下定论。

#### 15.5.3 Pure contralateral task

移除 contralateral 后，internal/external macro-F1 分别下降 2.43/1.44 个百分点，external accuracy 下降 1.31 个百分点；但 external AUROC 和 side accuracy分别上升 0.53/1.98 个百分点。

因此 contralateral task 对疾病分类 accuracy/F1 有小幅正向作用，但效果远小于 rhythm，并且不同指标方向不完全一致。它没有显示出造成明显负迁移的证据，当前可继续以较低的 `contralateral_weight=0.5` 保留；是否长期保留应由多 seed 的 external F1/accuracy 稳定性决定，而不是只依据 AUROC 或 affected-side 单项结果。

#### 15.5.4 Cross-DOF soft-only vs soft+prototype

关闭 Cross-DOF prototype 后：

```text
Cross-DOF top-5 accuracy      36.78% → 36.34%  (-0.44 pp)
Cross-DOF soft loss           3.7318 → 3.7333   (基本不变)
Cross-DOF prototype loss      0.2857 → 0.3187   (变差；该项仅评估、不参与优化)
External accuracy             89.54% → 85.29%  (-4.25 pp)
External macro-F1             88.90% → 84.86%  (-4.05 pp)
```

Soft-only 几乎保持了离散 neighborhood prediction，却显著损害 continuous prototype recovery 和 downstream transfer。这是本轮最清楚的目标形式证据：Cross-DOF prototype branch 学到的 codebook 几何信息不能被 top-k soft target 完全替代。Cross-DOF 应继续使用 `soft+prototype`。

#### 15.5.5 Contralateral soft-only vs soft+prototype

关闭 Contralateral prototype 后：

```text
Contralateral top-5 accuracy  20.33% → 21.18%  (+0.85 pp)
Contralateral soft loss       4.3228 → 4.3155   (小幅改善)
Contralateral prototype loss  0.4352 → 0.4558   (变差；该项仅评估、不参与优化)
Internal macro-F1             82.51% → 85.59%  (+3.08 pp)
External macro-F1             88.90% → 86.16%  (-2.74 pp)
```

Prototype supervision 没有提高 contralateral 的离散 top-5 指标，但提高了 external accuracy/F1；soft-only 则更偏向 internal 数据。说明 continuous prototype 可能在对侧异步、严格 code ID 不确定时充当几何正则项，牺牲部分开发域拟合以换取域外迁移。若 external disease generalization 是主目标，当前仍应保留 Contralateral `soft+prototype`；其证据强度低于 Cross-DOF prototype，需要多 seed 确认。

### 15.6 本轮结论与后续决策

本轮固定 VQ 消融支持以下结论：

1. **完整新版任务继续作为默认配置。** 它取得最高 external accuracy 和 macro-F1，并在内外部指标间提供最均衡的结果。
2. **Rhythm 是 external transfer 最关键的新增任务。** 移除后 external F1 下降 10.12 个百分点；但 duration 与 interval 的独立贡献尚未拆开。
3. **Bilateral pair 应保留。** 移除后内外部 disease 和 affected-side 指标全面下降，说明它补充了局部 code prediction 不包含的 subject-level 双侧信息。
4. **Cross-DOF 明确使用 `soft+prototype`。** Prototype 对离散 top-5 影响很小，但对 continuous geometry 和 external transfer 有明显贡献。
5. **Contralateral 暂时保留 `soft+prototype` 和 0.5 外层权重。** 其 external F1 方向积极，但效应较小且 internal/external 存在权衡。
6. **不能用 SSL total loss 或单次 dev-validation F1 选择任务。** 不同目标的 loss 尺度不同，且本轮 dev 排名与 external 排名明显不一致。

当前推荐的默认 SSL 组合保持为：

```text
Within hard code prediction
Cross-DOF top-k soft target + prototype
Duration + preceding-interval rhythm prediction
Contralateral top-k soft target + prototype, weight = 0.5
Healthy bilateral pair discrimination
```

上述结论来自一个训练 seed。尤其是 contralateral 与其 prototype 的 1～3 个百分点变化，仍可能受初始化和 downstream 微调方差影响；在把任务组合定为最终版本前，应至少重复 3 个 seed 并报告均值、标准差以及逐 seed 方向一致性。`no_rhythm` 的约 10 个百分点 external 降幅较大，优先级最高，但同样需要复现。

此外，external test 已被用于多轮分析，只适合作为现阶段泛化审计，不应继续承担超参数选择功能。下一轮正式比较应以预先确定的主指标、固定 seed 列表和不再反复查看的最终测试集完成确认。
