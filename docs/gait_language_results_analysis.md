# 步态语言模型 VQ、SSL 与 Downstream 实验结果分析

基于最新两次完整实验的 VQ/SSL 指标、downstream 指标和最终评估结果，本文对当前版本的 VQ、SSL 和 downstream 结果进行分析，并提出后续改进内容。

使用的实验结果目录为：

```text
Results/gait_language/dev_exp/dev_exp_20260823_175415_467270
Results/gait_language/downstream_20260823_181807_720136
```

总体判断是：

> 当前版本已经证明“步态语言 + 健康 codebook”具备可行性。VQ 没有明显坍缩，SSL 能学习局部形状和一定的跨 DOF/双侧关系，downstream 也有较强分类能力。
>
> 目前主要问题不是模型完全学不到，而是：VQ 对节律约束不足、跨 DOF/纯对侧 SSL 偏弱、downstream 严重过拟合且没有充分利用可解释的健康偏离信息。

## 1. VQ 结果

最佳结果出现在最后一个 epoch 49：

| 指标 | Train | Validation |
|---|---:|---:|
| 总损失 | 0.1068 | 0.1163 |
| 重建损失 | 0.1062 | 0.1157 |
| velocity loss | 0.00136 | 0.00144 |
| commitment loss | 0.00107 | 0.00115 |
| active ratio | 75.9% | 76.7% |
| perplexity | 73.3 | 74.1 |

### 正面结果

- 训练集和验证集差距很小，VQ 泛化正常。
- active ratio 约为 77%，perplexity 约为 74/128，没有明显 codebook collapse。
- VQ 可以把单周期 DOF 波形压缩成有区分度的离散词。

### 主要问题

1. 最佳 epoch 是最后一个 epoch，说明 VQ 尚未完全收敛，50 epoch 可能偏少。

2. 总损失几乎完全由波形重建控制：

```text
reconstruction ≈ 99.5%
weighted velocity ≈ 0.25%
weighted commitment ≈ 0.25%
```

因此，当前 VQ 更接近“波形压缩器”，不一定充分保留峰值位置、斜率和局部节律结构。

3. 当前 active ratio 是 batch 级统计，不能单独证明每个 DOF 的 128 个词都被合理使用。仍需检查：

- 每个 DOF 的全局 code usage；
- 高频词和死词；
- 不同词对应的平均波形；
- 同一个词内部波形方差；
- 每个 DOF 的重建误差和相关系数。

4. 周期质量分数中仍存在负值，最低约为 `-0.48`。虽然中位数约为 `0.87`，但少量低质量周期可能会污染 codebook 边缘词。

## 2. SSL 结果

最佳验证结果出现在 epoch 48，并在 epoch 58 early stopping：

| 任务 | Train accuracy | Validation accuracy | Validation loss |
|---|---:|---:|---:|
| 随机词 mask | 51.65% | 47.67% | 1.694 |
| 整个 DOF mask | 20.43% | 12.10% | 3.464 |
| 单侧部分词 mask | 51.47% | 46.03% | 1.726 |
| 整侧 mask，仅由对侧预测 | 8.35% | 5.12% | 4.311 |
| 左右交换约束 | — | — | 0.00025 |

128 类随机准确率只有 `0.78%`，因此四项预测任务都不是随机猜测。

### 已经学到的内容

- 随机词 mask 达到约 48%，说明 encoder 能利用同一 DOF 前后周期和其他上下文恢复局部形状。
- 单侧部分词 mask 达到约 46%，说明模型能够利用同侧及对侧信息补全缺失词。
- 整个 DOF mask 达到约 12%，证明 DOF 之间确实存在可学习的运动学耦合关系。
- 纯对侧预测约 5%，虽然不高，但明显超过随机水平，说明双侧之间存在一定可预测性。

### 当前瓶颈

1. 跨 DOF 和纯对侧任务明显较弱。

跨 DOF 与纯对侧任务合计贡献约 62% 的验证总损失，是当前 SSL 的主要瓶颈。

2. 跨 DOF 泛化差距较大：

```text
train 20.43% → validation 12.10%
```

这说明模型可能记住了部分健康训练波形组合，但对新的受试者泛化不足。

3. 双腿周期对齐不够精确。

数据统计显示左右最近周期中心偏移：

```text
median ≈ 0.258 秒
mean   ≈ 0.344 秒
max    > 5 秒
```

如果直接让左腿预测右腿，而没有先匹配同一步或相邻相位，任务中会混入大量错误对应关系。

4. 时间归一化后的单周期词不包含绝对步频。

VQ 主要学习周期形状；周期时长和间隔通过 timing 输入 sentence encoder。但目前没有直接要求 encoder 重建周期时长、步间隔或左右相位差，因此“节律信息是否真正被使用”尚不能确认。

5. swap loss 极小不一定代表效果非常好。

它也可能说明当前 pooling 结构天然满足左右交换关系，或者 shared/absolute/directional embedding 的方差过小。需要检查：

- embedding 各维方差；
- embedding norm；
- covariance/effective rank；
- 未训练模型的初始 swap loss。

如果未训练模型的 swap loss 同样很小，这项任务几乎没有提供训练信号。

## 3. Downstream 结果

### 总体性能

| 数据集 | Accuracy | Macro-F1 | Macro-AUROC |
|---|---:|---:|---:|
| Internal test | 90.37% | 87.18% | 96.39% |
| External test | 84.64% | 84.32% | 93.93% |

外部数据相对内部数据：

- Accuracy 下降 5.73 个百分点；
- Macro-F1 下降 2.87 个百分点；
- AUROC 下降 2.46 个百分点。

这属于有效的外部泛化结果，并不能算模型整体失效。

### 内部测试混淆矩阵

行是真实类别，列是预测类别：

| 真实类别 | Healthy | ACLD | KOA |
|---|---:|---:|---:|
| Healthy | 47 | 2 | 0 |
| ACLD | 1 | 60 | 2 |
| KOA | 2 | 6 | 15 |

内部测试的主要问题是 KOA：

| 类别 | Precision | Recall | F1 |
|---|---:|---:|---:|
| Healthy | 94.0% | 95.9% | 94.9% |
| ACLD | 88.2% | 95.2% | 91.6% |
| KOA | 88.2% | 65.2% | 75.0% |

23 个 KOA 中有 6 个被预测成 ACLD，说明当前模型难以稳定区分部分 KOA 与 ACLD 的异常模式。

### 外部测试混淆矩阵

| 真实类别 | Healthy | ACLD | KOA |
|---|---:|---:|---:|
| Healthy | 69 | 20 | 0 |
| ACLD | 11 | 87 | 3 |
| KOA | 2 | 11 | 103 |

外部测试中：

| 类别 | Precision | Recall | F1 |
|---|---:|---:|---:|
| Healthy | 84.1% | 77.5% | 80.7% |
| ACLD | 73.7% | 86.1% | 79.5% |
| KOA | 97.2% | 88.8% | 92.8% |

外部数据的主要混淆变成了 Healthy 与 ACLD：

- 20 个 Healthy 被预测为 ACLD；
- 11 个 ACLD 被预测为 Healthy；
- ACLD precision 只有 73.7%。

这说明外部数据中的正常波形、节律或采集分布与 SSL 健康参考存在一定偏移。

### 严重过拟合和过度置信

最佳 downstream epoch 16：

```text
Train Macro-F1      = 99.46%
Validation Macro-F1 = 89.11%

Train disease loss  = 0.0377
Validation loss     = 0.9756
```

虽然验证分类性能较高，但验证 loss 已经明显升高。

重新计算的置信度结果：

| 指标 | Internal | External |
|---|---:|---:|
| NLL | 0.550 | 0.855 |
| ECE | 8.82% | 12.86% |
| 正确样本平均置信度 | 99.64% | 99.16% |
| 错误样本平均置信度 | 95.03% | 88.39% |

模型即使预测错误，也经常给出很高的置信度。这意味着 AUROC 排序能力很好，但概率没有良好校准。

## 4. 与最终研究目标之间的差距

当前 downstream 分类器使用：

```text
shared_embedding + absolute_difference
→ MLP
→ Healthy / ACLD / KOA
```

它取得了不错的分类结果，但没有显式使用：

- word 到健康 codebook 的距离；
- word 偏离了哪个健康词；
- 偏离方向；
- 每个 DOF、每个周期位置的偏离；
- 健康 code transition 的异常程度；
- 节律 deviation；
- 双侧条件 surprise。

因此它目前是一个有效的分类 baseline，但还不能充分回答：

> ACLD 和 KOA 分别在什么 DOF、什么词、什么方向上偏离健康步态？

## 5. 建议的后续改进顺序

### 第一阶段：先诊断 VQ，不立即扩大模型

建议增加以下离线分析：

- 每个 DOF 单独统计 code usage 和 perplexity；
- 绘制每个 code 对应的平均周期波形；
- 统计每个 code 内部波形方差；
- 输出每个 DOF 的 RMSE、相关系数和 velocity error；
- 删除或降低负质量周期的训练权重；
- 检查同一受试者相似周期是否被编码到相近词；
- 检查相近 codebook prototype 是否存在大量重复。

VQ 训练可以延长至 100 epoch，并加入 patience 10，但暂时不建议直接把 K 从 128 增大。

### 第二阶段：强化节律与双侧 SSL

优先增加三类任务。

#### 1. 显式节律任务

- mask 后预测 cycle duration；
- 预测 preceding interval；
- 预测左右周期相位差；
- 判断周期顺序是否被打乱。

#### 2. 相位匹配的双侧任务

先根据周期中心时间或归一化步态相位匹配左右周期，再进行对侧预测。超过合理时间偏差的周期对不参与 loss。

#### 3. 健康双侧配对任务

- 正样本：同一健康受试者、相邻相位的左右周期；
- 负样本：不同受试者的左右周期；
- 学习真实双侧配对与伪配对的差异。

对于跨 DOF 和纯对侧预测，可以考虑从“严格预测唯一 code ID”改为：

- top-k code prediction；
- codebook 距离加权的 soft target；
- 同时预测离散 code 和连续 prototype embedding。

这样预测到相邻健康词不会被视为与完全错误的词同等严重。

固定验证 mask 应继续保留，但建议使用 3～5 组固定 mask，取平均验证损失，避免 early stopping 依赖单一 mask realization。

### 第三阶段：改造 downstream

不建议继续从第一个 epoch 全量微调 sentence encoder。建议比较：

1. 完全冻结 VQ 和 encoder，只训练线性分类器；
2. 先冻结训练分类头，再仅解冻最后一个 Transformer block；
3. encoder 学习率设为分类头的 `1/10～1/30`；
4. 保留当前全量微调作为 baseline。

最重要的是构建显式健康偏离特征：

```text
每个 word
→ 最近健康 prototype 距离
→ 健康 code 频率 surprise
→ 连续残差及偏离方向
→ 健康转移概率异常
→ 周期时长/间隔偏离
→ 左右条件预测 surprise
→ 按 DOF、侧别和时间聚合
→ 三分类器
```

健康参考参数只能使用训练数据中的 Healthy 样本拟合。ACLD/KOA 的疾病方向可以使用 downstream train 拟合，但不能使用 validation/test。

### 第四阶段：解决分类和概率问题

- 使用 class-balanced CE 或 balanced sampler，重点改善内部 KOA recall；
- 使用 validation 做 temperature scaling；
- early stopping 仍以 Macro-F1 为主，但 validation NLL 作为并列选择条件；
- 增加轻量 word dropout、timing jitter，减少全量微调过拟合；
- 至少使用 3～5 个训练随机种子，报告均值和标准差；
- 分别对 `shared`、`absolute difference`、`directional difference`、`rhythm` 和 `codebook deviation` 做消融实验。

## 6. 最推荐的下一组实验

建议下一步只做三个可解释的对照：

| 实验 | Encoder | 分类输入 | 目的 |
|---|---|---|---|
| A | 冻结 | shared + absolute | 当前模型的冻结基线 |
| B | 冻结 | 显式 codebook/rhythm/bilateral deviation | 验证健康偏离特征是否有效 |
| C | 部分解冻 | B 的特征 + sentence embedding | 验证少量微调能否提升性能 |

下一阶段最关键的目标不应只是继续提高总体 accuracy，而应同时观察：

- external Healthy recall；
- internal KOA recall；
- calibration ECE；
- 每个 DOF 的疾病偏离可视化；
- ACLD 与 KOA 偏离方向是否稳定；
- 多随机种子下结论是否一致。

## 7. 结论

当前模型已经形成一个较强 baseline，后续收益最大的方向不是简单加深 encoder 或扩大 codebook，而是改善双侧周期对应关系、显式监督节律、构建健康 codebook 偏离特征，并限制 downstream 对健康表征空间的破坏。
