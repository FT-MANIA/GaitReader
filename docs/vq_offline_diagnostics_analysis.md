# VQ 离线诊断结果分析

## 0. 指标与参数定义

本节统一定义本文、`summary.json` 及各诊断 CSV/NPZ 文件中使用的参数和指标。

### 0.1 基本符号、数据形状与单位

| 符号或参数 | 当前值 | 定义 |
|---|---:|---|
| `DOF` | 6 | 每条膝关节运动学信号包含的自由度数量 |
| `K` | 128 | 每个 DOF 独立 codebook 中的 prototype/code 数量 |
| `T` | 100 | 每个步态周期完成时间归一化后的采样点数 |
| `D` | 128 | 每个 encoder word 和 codebook prototype 的特征维度 |
| `N_d` | 6892 | 本次验证集中每个 DOF 的有效周期数 |
| `x_{i,d}(t)` | — | 第 `i` 个周期、第 `d` 个 DOF 在归一化时间点 `t` 的标准化波形值 |
| `x̂_{i,d}(t)` | — | VQ decoder 对 `x_{i,d}(t)` 的重建结果 |
| `z_{i,d}` | — | word encoder 输出的连续 embedding |
| `e_{d,k}` | — | 第 `d` 个 DOF 的第 `k` 个 codebook prototype |
| `c_{i,d}` | — | 周期 `i` 在 DOF `d` 上被分配的 code 编号 |

当前 RMSE、波形方差和 velocity error 都在训练使用的标准化坐标中计算，不是原始角度或位移物理单位。六个 DOF 的内部固定顺序为：

```text
0 FE = flexion_extension
1 AA = adduction_abduction
2 IE = internal_external_rotation
3 AP = anterior_posterior_translation
4 ML = medial_lateral_translation
5 SI = superior_inferior_translation
```

### 0.2 Assignment count、概率和 Code usage

#### Assignment count

`assignment_count` 表示某个 code 在指定数据划分中被分配到的有效周期数量：

```text
n(d,k) = Σ_i 1[c(i,d) = k]
```

其中 `1[·]` 为指示函数。

#### Assignment probability

`assignment_probability` 是该 code 在对应 DOF 中的经验使用概率：

```text
p(d,k) = n(d,k) / Σ_k n(d,k)
```

#### Active code count

`active_code_count` 是 `assignment_count > 0` 的 code 数量。

#### Active code ratio

```text
active_code_ratio = active_code_count / K
```

该值越接近 1，说明在当前统计范围内被使用的 code 越多。但 active ratio 只反映“是否出现”，不能反映词频是否均衡。

#### Top-10 code 占比

将某个 DOF 的 code 按 `assignment_count` 从高到低排序，前 10 个 code 的样本数之和占该 DOF 全部 assignment 的比例。该指标用于检查是否有少数 code 垄断大量周期。

### 0.3 Perplexity

首先计算 code 使用分布的 Shannon entropy：

```text
H_d = -Σ_k p(d,k) log p(d,k)
```

然后定义：

```text
perplexity_d = exp(H_d)
```

perplexity 可以理解为该 DOF 实际有效使用的等概率 code 数量：

- 如果所有周期都落入一个 code，perplexity 为 1；
- 如果 128 个 code 完全均匀使用，perplexity 为 128；
- perplexity 高不等于 prototype 一定具有临床可解释性，只表示使用分布较分散。

本文同时使用：

```text
perplexity/K = perplexity / 128
```

用于比较有效词表规模占配置词表规模的比例。

### 0.4 Code 内部波形均值与方差

#### Mean waveform

对分配到同一 DOF、同一 code 的周期逐时间点求均值：

```text
μ(d,k,t) = mean{x(i,d,t) | c(i,d) = k}
```

对应 `code_waveform_statistics.npz` 中的 `mean_waveforms`。

#### Pointwise waveform variance

对同一 code 内的波形逐时间点计算总体方差：

```text
var(d,k,t) = mean[(x(i,d,t) - μ(d,k,t))² | c(i,d) = k]
```

对应 `code_waveform_statistics.npz` 中的 `waveform_variances`。当前实现使用总体方差，即分母为 code 内周期数，而不是无偏样本方差的 `n-1`。

#### Mean pointwise waveform variance

```text
mean_pointwise_waveform_variance(d,k)
    = mean_t var(d,k,t)
```

该值越低，说明一个 code 内的周期波形越集中；该值较高可能表示波形簇混杂、周期质量较差或量化边界不稳定。

#### Maximum pointwise waveform variance

```text
maximum_pointwise_waveform_variance(d,k)
    = max_t var(d,k,t)
```

该指标用于发现只在某些步态相位出现明显分散的 code。

#### P10、P50、P90 与最大方差

本文先为每个 code 计算 `mean_pointwise_waveform_variance`，再在一个 DOF 的 128 个 code 之间计算第 10、50、90 百分位数和最大值。P50 即中位数。

#### 加权组内方差

按照每个 code 的 assignment count 对组内方差加权：

```text
weighted_within_variance_d
    = Σ_k n(d,k) × mean_t var(d,k,t) / Σ_k n(d,k)
```

它表示随机抽取一个周期时，其所属 code 内部的平均剩余波形方差。

#### Code assignment 解释的方差

```text
explained_variance_d
    = 1 - weighted_within_variance_d / total_variance_d
```

其中 `total_variance_d` 是该 DOF 全部波形值的总体方差。该指标描述 code assignment 对原始波形聚类的紧致程度，不是 decoder 的重建 `R²`，也不能单独作为预测性能指标。

### 0.5 重建指标

#### RMSE

```text
RMSE_d = sqrt(mean_{i,t}[(x̂(i,d,t) - x(i,d,t))²])
```

RMSE 同时受波形形状、幅值和基线误差影响。当前值使用标准化单位，数值越低越好。

#### Pooled Pearson correlation

`pooled_pearson_correlation` 将某个 DOF 的所有周期和时间点展开为两个长向量，再计算原始值与重建值的 Pearson correlation：

```text
corr(vec(x_d), vec(x̂_d))
```

该指标会利用不同周期之间的幅值和基线差异，可能高估单个周期内部的形状恢复能力。

#### Mean cycle Pearson correlation

先分别计算每个周期原始波形与重建波形在 100 个时间点上的 Pearson correlation，再对全部周期取平均：

```text
mean_cycle_corr_d
    = mean_i corr_t(x(i,d,t), x̂(i,d,t))
```

该指标更关注单周期内部形状是否被恢复，越接近 1 越好。

#### Std cycle Pearson correlation

所有单周期 Pearson correlation 的总体标准差。该值越大，说明 VQ 对不同周期的重建稳定性差异越大。

#### Velocity RMSE

当前 velocity 使用相邻归一化采样点的一阶差分：

```text
Δx(i,d,t) = x(i,d,t+1) - x(i,d,t)
```

```text
velocity_RMSE_d
    = sqrt(mean[(Δx̂(i,d,t) - Δx(i,d,t))²])
```

需要特别注意：当前指标没有除以真实时间间隔，也不是物理意义上的角速度或平移速度。它衡量的是 100 点时间归一化波形的相邻点变化误差。

### 0.6 相似周期与编码一致性

#### Waveform similarity threshold

本次分析配置为：

```text
waveform_similarity_threshold = 0.90
```

两条周期波形在 100 个时间点上的 Pearson correlation 大于或等于 0.90 时，被定义为相似周期。Pearson correlation 对整体平移和正比例缩放不敏感，因此“形状相似”不代表两条波形的幅值和基线相同。

#### Within-side candidate pair

同一受试者、同一侧、同一 DOF 的所有不同周期组合。左右侧分别构造后合并统计，记为 `within_side`。

#### Across-sides candidate pair

同一受试者、同一 DOF 的每个左侧周期与每个右侧周期的组合，记为 `across_sides`。当前定义不要求两个周期来自实际时间相邻或相位匹配的左右步。

#### Similar pair count

candidate pair 中波形 Pearson correlation `≥0.90` 的周期对数量。

#### Same code rate

```text
same_code_rate
    = 相似周期对中 code ID 完全相同的数量 / similar_pair_count
```

code ID 是离散类别，没有自然的数值顺序，因此不能使用 `|code_a-code_b|` 衡量词距离。

#### Prototype cosine similarity

对两个 L2 归一化 prototype embedding 计算：

```text
cos(e_a,e_b) = e_a · e_b
```

越接近 1，表示两个 prototype 在 embedding 空间的方向越相似。

#### Near-code threshold 和 Near code rate

本次分析配置为：

```text
near_code_similarity_threshold = 0.90
```

若两个周期对应 prototype 的 cosine similarity `≥0.90`，则认为它们被分配到了相近词。

```text
near_code_rate
    = 相似周期对中 prototype cosine ≥ 0.90 的数量
      / similar_pair_count
```

相同 code 的 prototype similarity 必然为 1，因此 same-code pair 也包含在 near-code pair 中。

#### Mean assigned prototype similarity

所有相似周期对所对应 prototype cosine similarity 的平均值。该指标保留连续相似程度，不依赖 `0.90` 阈值。

### 0.7 Prototype 距离与重复度

#### Embedding cosine similarity

两个 codebook prototype 在归一化 embedding 空间中的 cosine similarity。

#### Embedding Euclidean distance

prototype 已完成 L2 归一化，因此欧氏距离与 cosine similarity 的关系为：

```text
distance(e_a,e_b) = sqrt(2 - 2 × cosine_similarity)
```

#### Decoded waveform correlation

将两个 prototype 分别输入 VQ decoder，得到两条长度为 100 的标准化波形，再计算它们的 Pearson correlation。该指标主要比较解码形状，对幅值和基线不敏感。

#### Decoded waveform RMSE

两个 prototype 解码波形之间的 RMSE。它同时考虑形状、幅值和基线差异。

#### Embedding-near pair

本次阈值为：

```text
prototype_duplicate_threshold = 0.95
```

prototype embedding cosine similarity `≥0.95` 的 pair 被记为 embedding-near pair。

#### Decoded-waveform-near pair

本次阈值为：

```text
decoded_waveform_duplicate_threshold = 0.98
```

两个 prototype 的解码波形 Pearson correlation `≥0.98` 时，被记为 decoded-waveform-near pair。

#### Joint duplicate pair

同时满足以下两个条件的 prototype pair：

```text
embedding cosine similarity ≥ 0.95
decoded waveform correlation ≥ 0.98
```

`joint_duplicate_pair_ratio` 为 joint duplicate pair 数量占该 DOF 全部 `K(K-1)/2` 个 prototype pair 的比例。

#### Prototype in joint duplicate ratio

至少参与一个 joint duplicate pair 的不同 prototype 数量占 K 的比例。

#### Maximum off-diagonal similarity

一个 DOF 的 prototype cosine similarity 矩阵中，去除自身与自身的对角线后，最大的 pair similarity。

#### Mean nearest prototype similarity

对每个 prototype 找到除自身外 cosine similarity 最高的最近邻，再对 K 个最近邻相似度求平均。

### 0.8 文中建议但当前诊断尚未输出的指标

#### Assignment margin

```text
assignment_margin = top1_similarity - top2_similarity
```

其中 top-1 和 top-2 是 encoder embedding 与最近、次近 prototype 的 cosine similarity。margin 较小说明周期靠近量化边界，code assignment 更容易因微小扰动发生变化。

#### Jensen-Shannon divergence

Jensen-Shannon divergence 用于比较左右侧 code usage 概率分布：

```text
M = (P_left + P_right) / 2
JSD(P_left,P_right)
    = 0.5 × KL(P_left || M) + 0.5 × KL(P_right || M)
```

JSD 越接近 0，表示左右侧 code 使用分布越相似。该指标必须在确认左右坐标方向和 code 语义后解释。

#### Codebook collapse

Codebook collapse 不是单一数值，而是以下现象的组合：

- active code ratio 很低；
- perplexity 远低于 K；
- 少数 code 占据绝大部分 assignment；
- 大量 code 长期没有样本；
- 多个 prototype 收敛为近乎相同的表示。

本文结合全局 active ratio、perplexity、词频长尾和 prototype duplicate ratio 判断是否存在 collapse。

## 1. 分析对象

本文分析以下 VQ 实验及其离线诊断结果：

```text
训练运行：Results/gait_language/dev_exp/dev_exp_0823_1849
VQ checkpoint：Results/gait_language/dev_exp/dev_exp_0823_1849/best_vq.pt
诊断数据：ssl_validation_data
诊断受试者数：410
每个 DOF 的有效周期数：6892
```

离线诊断目录为：

```text
Results/gait_language/dev_exp/dev_exp_0823_1849/
└── vq_analysis_ssl_validation_data/
```

本次 VQ 最多配置为 100 个 epoch，实际训练到 epoch 84 后 early stopping，最佳验证损失出现在 epoch 74：

```text
validation total loss          = 0.110657
validation reconstruction loss = 0.110049
validation velocity loss       = 0.001404
validation commitment loss     = 0.001308
```

因此，之前“最佳结果出现在最后一个 epoch，VQ 可能训练不足”的问题在本次实验中已经得到改善。当前 checkpoint 是已完成 early stopping 选择的最佳模型。

## 2. 总体结论

当前 VQ 的主要结论是：

1. 六个 DOF 的 128 个 code 在完整验证集上全部被使用，没有 codebook collapse。
2. 全局 perplexity 为 113.27～120.31，词表利用率较高，`K=128` 暂时不需要增大。
3. code 使用存在一定长尾，但没有少数 code 垄断大部分样本。
4. FE 的重建效果最好；IE 和 ML 的 RMSE、周期相关性及 velocity error 明显较差，是当前 VQ 的主要重建瓶颈。
5. 大部分 code 内部波形较集中，但 FE、IE、ML 和 SI 各存在少数高方差 code，需要检查是否混入不同形态或低质量周期。
6. 同侧相似周期通常能分配到相同或相近词，但 FE 的稳定性偏低。
7. 左右侧波形即使具有较高相关性，也很少分配到相同或相近 prototype。该现象不能直接解释为 VQ 错误，因为当前“相似周期”只使用 Pearson 相关系数定义，忽略了基线和幅值差异。
8. 严格意义上的重复 prototype 极少，当前没有证据表明 128 个 prototype 大量重复。

综合判断：

> 当前 VQ 已经具备可用的健康步态离散词表。下一步不应优先扩大 codebook，而应改善 IE/ML 的重建能力、检查高方差 code、完善左右侧坐标与相似性定义，并提高相似周期的分词稳定性。

---

## 3. 每个 DOF 的 Code Usage 和 Perplexity

### 3.1 全局统计

| DOF | Active code | Perplexity | Perplexity/K | 单 code 最小/中位/最大样本数 | Top-10 code 占比 |
|---|---:|---:|---:|---:|---:|
| FE | 128/128 | 118.92 | 92.9% | 15 / 54 / 103 | 13.06% |
| AA | 128/128 | 113.43 | 88.6% | 14 / 51.5 / 152 | 17.37% |
| IE | 128/128 | 120.31 | 94.0% | 16 / 50 / 117 | 13.91% |
| AP | 128/128 | 113.52 | 88.7% | 8 / 50 / 165 | 16.66% |
| ML | 128/128 | 115.74 | 90.4% | 13 / 50 / 133 | 15.51% |
| SI | 128/128 | 113.27 | 88.5% | 6 / 50.5 / 132 | 16.21% |

DOF 缩写含义：

```text
FE = flexion_extension
AA = adduction_abduction
IE = internal_external_rotation
AP = anterior_posterior_translation
ML = medial_lateral_translation
SI = superior_inferior_translation
```

### 3.2 结果解释

六个 DOF 在 6892 个验证周期中都使用了全部 128 个 code。perplexity/K 均高于 88%，说明词频分布虽然不是完全均匀，但不存在明显的词表坍缩。

如果 128 个 code 完全均匀，每个 code 约有 53.8 个周期。当前中位数约为 50～54，与均匀分布比较接近。最高频 code 的占比只有约 1.5%～2.4%，Top-10 code 总占比为 13.1%～17.4%，没有少数 code 垄断绝大部分周期。

低频 code 数量也很少：

- FE、AA、IE、ML 没有样本数低于 10 的 code；
- AP 有 2 个 code 低于 10 个样本；
- SI 有 1 个 code 低于 10 个样本。

因此，目前没有必要因为 code 使用不足而减小 K，也没有必要为了增加容量而扩大 K。可以保留 `K=128`，后续仅把 `K=64` 作为稳定性消融实验，而不是默认替代方案。

### 3.3 Batch 指标与全局指标为什么不同

训练日志中的验证 active ratio 约为 0.768，perplexity 约为 74；离线诊断则得到 active ratio 为 1.0，perplexity 为 113～120。两者并不冲突：

- 训练日志先在每个 batch 内计算 active ratio 和 perplexity，再对 batch 求平均；
- 离线诊断将完整验证集的 6892 个周期合并后，再计算全局词频和 perplexity。

某个 code 不一定在每个 batch 中出现，但可能在完整验证集中出现。因此，判断 codebook 是否全局坍缩时，应以本次离线统计为准；训练日志中的 batch 指标更适合监控训练过程中的局部使用状态。

---

## 4. 每个 Code 内部的波形方差

### 4.1 方差分布

`mean_pointwise_waveform_variance` 表示被分配到同一个 code 的周期，在 100 个时间点上的平均组内方差。

| DOF | 方差 P10 | 方差中位数 | 方差 P90 | 最大方差 | 最大方差 code（样本数） |
|---|---:|---:|---:|---:|---:|
| FE | 0.0313 | 0.0465 | 0.0955 | 0.5455 | code 41（21） |
| AA | 0.0354 | 0.0611 | 0.1587 | 0.2693 | code 14（48） |
| IE | 0.0815 | 0.1350 | 0.2638 | 0.6472 | code 114（50） |
| AP | 0.0353 | 0.0679 | 0.1791 | 0.4613 | code 120（37） |
| ML | 0.0735 | 0.1396 | 0.3957 | 0.9690 | code 37（39） |
| SI | 0.0313 | 0.0650 | 0.1672 | 1.1813 | code 103（44） |

按 code 样本数加权后，组内方差占该 DOF 验证集总方差的比例为：

| DOF | 加权组内方差 | 总方差 | 组内/总方差 | Code assignment 解释的方差 |
|---|---:|---:|---:|---:|
| FE | 0.0551 | 1.0130 | 5.44% | 94.56% |
| AA | 0.0699 | 0.9909 | 7.05% | 92.95% |
| IE | 0.1487 | 1.0262 | 14.49% | 85.51% |
| AP | 0.0764 | 0.9284 | 8.23% | 91.77% |
| ML | 0.1602 | 1.0560 | 15.17% | 84.83% |
| SI | 0.0781 | 1.0881 | 7.18% | 92.82% |

这里的“解释方差”描述 code assignment 对原始波形聚类的紧致程度，不等同于 decoder 的重建 `R²`。

### 4.2 结果解释

FE、AA、AP 和 SI 的典型 code 组内方差较低，说明大多数 code 对应比较集中的波形簇。IE 和 ML 的中位组内方差分别为 0.1350 和 0.1396，明显高于其他 DOF，code assignment 解释的方差也只有约 85%。

这与后续重建结果一致：IE 和 ML 的周期形态更难被当前共享 encoder/decoder 表达。

需要重点检查以下高方差 code：

```text
FE code 41
IE code 114
ML code 37
SI code 103
```

这些 code 的样本数分别为 21、50、39、44，并不全部属于极低频 code。因此，高方差不能简单归因于样本过少，可能存在：

- 同一个 code 混入多个形态模式；
- 周期幅值或基线差异较大；
- 少数分周期错误或低质量周期；
- 某些 code 被少数受试者的大量周期主导；
- encoder 对相近形态的判别边界不稳定。

下一步应为这些 code 绘制均值波形、标准差带和单周期叠加图，并统计其受试者来源数、周期质量分数和 assignment margin。

---

## 5. 每个 DOF 的重建结果

| DOF | RMSE | Pooled Pearson | 平均单周期 Pearson | 单周期 Pearson 标准差 | Velocity RMSE |
|---|---:|---:|---:|---:|---:|
| FE | 0.2524 | 0.9682 | 0.9730 | 0.0579 | 0.0274 |
| AA | 0.2823 | 0.9590 | 0.6949 | 0.2805 | 0.0322 |
| IE | 0.4024 | 0.9178 | 0.7581 | 0.2342 | 0.0487 |
| AP | 0.2959 | 0.9517 | 0.8557 | 0.1804 | 0.0293 |
| ML | 0.4203 | 0.9127 | 0.6964 | 0.2599 | 0.0510 |
| SI | 0.3014 | 0.9578 | 0.8191 | 0.2649 | 0.0282 |

### 5.1 FE 重建最好

FE 的 RMSE 最低，平均单周期相关系数达到 0.973，velocity RMSE 也最低之一。当前 VQ 对屈伸周期形状具有较强表达能力。

### 5.2 IE 和 ML 是主要瓶颈

IE 和 ML 的 RMSE 分别为 0.4024 和 0.4203，velocity RMSE 分别为 0.0487 和 0.0510，明显高于其他 DOF。ML 的平均单周期相关系数只有 0.6964，是综合表现最弱的 DOF。

这说明当前 VQ 对 IE/ML 的细节、局部变化速度和周期内形状恢复不足。后续使用 codebook distance 解释疾病偏离时，这两个 DOF 的原始距离可能包含较多 tokenizer 误差，不能与 FE 使用完全相同的可信度。

### 5.3 Pooled correlation 会高估部分 DOF 的重建质量

AA 的 pooled correlation 为 0.959，但平均单周期 correlation 只有 0.695；ML 也从 pooled 0.913 降至单周期 0.696。

Pooled correlation 会同时利用不同周期之间的整体幅值和基线差异，因此即使模型只恢复了总体水平，也可能获得很高的相关性。判断周期形状重建能力时，应优先使用：

- 平均单周期 Pearson correlation；
- 单周期 correlation 分布；
- velocity RMSE；
- 峰值位置误差；
- 关键相位区间误差。

### 5.4 Velocity loss 的训练贡献仍然偏小

最佳 checkpoint 的验证 velocity loss 为 0.001404。在当前 `velocity_weight=0.2` 下，其加权贡献约为 0.000281，只占总损失约 0.25%。

因此，IE/ML 的 velocity error 偏高与训练目标中速度约束过弱是一致的。后续更合理的方案是先按每个 DOF 的目标速度方差归一化 velocity loss，再让其贡献总损失的大约 5%～15%，而不是仅盲目小幅增加当前权重。

---

## 6. 同一受试者相似周期的编码一致性

当前诊断将周期波形 Pearson correlation `≥0.90` 定义为“相似周期”。“相近词”定义为对应 prototype 的余弦相似度 `≥0.90`。

### 6.1 同侧周期

| DOF | 相似周期对数 | 相同 code 比例 | 相同或相近 code 比例 | 平均 prototype 相似度 |
|---|---:|---:|---:|---:|
| FE | 20400 | 39.39% | 42.57% | 0.871 |
| AA | 12975 | 63.91% | 65.39% | 0.929 |
| IE | 11651 | 50.75% | 50.79% | 0.864 |
| AP | 17821 | 59.82% | 61.96% | 0.926 |
| ML | 7842 | 53.09% | 53.09% | 0.877 |
| SI | 17298 | 55.76% | 59.57% | 0.916 |

AA、AP 和 SI 的同侧一致性相对较好。FE 周期的平均波形相关性高达 0.977，但相同 code 比例只有 39.4%，是最明显的不一致项。

这说明 FE codebook 可能在进一步区分：

- 幅值；
- 基线偏移；
- 局部细节；
- Pearson correlation 不敏感的尺度变化。

也可能存在周期位于多个 prototype 决策边界附近，导致形态非常相似但 assignment 不稳定。需要结合波形 RMSE、encoder embedding 距离以及 top-1/top-2 assignment margin 才能区分这两种情况。

### 6.2 左右侧周期

| DOF | 相似周期对数 | 相同 code 比例 | 相同或相近 code 比例 | 平均 prototype 相似度 |
|---|---:|---:|---:|---:|
| FE | 15863 | 12.16% | 15.43% | 0.696 |
| AA | 869 | 8.86% | 8.86% | 0.564 |
| IE | 1461 | 10.61% | 10.61% | 0.565 |
| AP | 4884 | 13.23% | 14.11% | 0.691 |
| ML | 312 | 5.77% | 5.77% | 0.585 |
| SI | 4945 | 11.10% | 12.98% | 0.659 |

左右侧的一致性明显低于同侧。但不能据此直接认定共享 codebook 失败，原因是 Pearson correlation 对以下差异不敏感：

- 波形整体幅值不同；
- 正负方向或尺度不同；
- 基线偏移；
- 左右侧解剖坐标没有做镜像统一；
- 周期虽然形状相似，但并不是实际配对的同一步。

VQ tokenizer 本身没有接收 side label。如果两条输入波形完全相同，其编码结果必然相同。因此，当前跨侧差异来自输入波形在相关系数之外仍存在差别，或者处于不同的量化边界，而不是模型直接根据“左腿/右腿”选择不同词。

在将“左右腿使用不同词”解释为健康双侧差异之前，需要先确认每个 DOF 的左右解剖正方向，并根据物理定义决定是否需要符号翻转或镜像变换。该检查不应通过盲目强制左右相同 code 完成。

---

## 7. Prototype 重复度

每个 DOF 有 128 个 prototype，共有：

```text
128 × 127 / 2 = 8128 个 prototype pair
```

严格重复判据为：

```text
embedding cosine similarity ≥ 0.95
且 decoded waveform correlation ≥ 0.98
```

### 7.1 结果

| DOF | Embedding 相近 pair | 解码波形相关 pair | 同时满足两个条件 | 最大 embedding 相似度 | 平均最近邻相似度 |
|---|---:|---:|---:|---:|---:|
| FE | 0 | 595（7.32%） | 0 | 0.9369 | 0.8665 |
| AA | 1（0.012%） | 85（1.05%） | 0 | 0.9506 | 0.8712 |
| IE | 1（0.012%） | 26（0.32%） | 0 | 0.9670 | 0.8264 |
| AP | 0 | 101（1.24%） | 0 | 0.9270 | 0.8667 |
| ML | 0 | 27（0.33%） | 0 | 0.8929 | 0.8298 |
| SI | 0 | 180（2.21%） | 0 | 0.9498 | 0.8751 |

没有任何 pair 同时满足两个严格重复条件。因此，没有证据表明 codebook 中存在大量完全重复 prototype。

### 7.2 为什么 FE 有较多高相关解码波形

FE 有 595 个 decoded waveform correlation `≥0.98` 的 pair，但这些 pair 的解码 RMSE 中位数为 0.550，embedding cosine 中位数只有 0.686。

这说明这些 prototype 更可能表示：

- 形状相似但幅值不同的 FE 周期；
- 形状相似但基线不同的周期；
- 相似屈伸模式的不同强度或活动范围。

相关系数只关心形状，不关心幅值和基线。因此，这些 pair 不应直接合并，也不能仅凭相关系数判断为冗余词。

### 7.3 需要人工检查的极近 embedding pair

目前只有两个 pair 的 embedding cosine 超过 0.95：

```text
AA：code 72 与 code 75，cosine = 0.9506，decoded RMSE = 0.2541
IE：code 95 与 code 108，cosine = 0.9670，decoded RMSE = 0.1268
```

IE code 95/108 是最值得检查的潜在近重复 pair，但整个词表只有极少数此类情况，不足以说明 `K=128` 过大。

---

## 8. 对当前 VQ 合理性的判断

### 8.1 可以保留的设计

- 六个 DOF 使用独立 codebook是合理的；
- 左右腿共享同一套 DOF 词表仍有研究价值；
- `K=128` 的整体利用率良好；
- EMA codebook 没有明显死词或坍缩；
- 当前 encoder 能够为大部分 DOF 建立紧致波形簇；
- 当前 checkpoint 已经过完整 early stopping，不需要单纯继续增加 epoch。

### 8.2 当前主要结构限制

encoder 包含 DOF-specific adapter，但 decoder 主要使用共享 MLP，只通过 DOF embedding 区分不同自由度。IE 和 ML 的波形复杂度、噪声性质及局部变化可能与 FE/AP 不同，共享 decoder 可能限制它们的重建能力。

同时，当前 hard nearest-neighbor assignment 没有显式约束：

- 相似周期具有稳定 assignment；
- 相近形态映射到相近 prototype；
- augmentation 前后保持 code 一致；
- codebook 在局部形成连续的形态拓扑。

因此，当前 codebook 更接近一组离散类别，而不是具有明确顺序关系的连续词汇空间。这也解释了为什么预测到错误 code 时，不能简单通过 code 编号差判断偏离大小。

---

## 9. 后续改进优先级

### 第一优先级：完善诊断与数据定义

在修改模型前，先完成：

1. 绘制所有 code 的均值波形和标准差带；
2. 重点检查 FE 41、IE 114、ML 37、SI 103；
3. 统计每个 code 的独立受试者数和最大单受试者占比；
4. 将周期质量分数与 code 内方差关联；
5. 输出 top-1/top-2 similarity margin，检查量化边界不稳定性；
6. 为周期相似性同时使用 correlation、RMSE、幅值差和基线差；
7. 分别统计左右侧 code usage，并计算 Jensen-Shannon divergence；
8. 核对六个 DOF 左右腿的解剖正方向及是否需要镜像变换。

其中第 6～8 项完成前，不应强制左右相似周期使用相同 code。

### 第二优先级：改善 IE 和 ML 重建

建议按以下顺序进行小规模消融：

1. 对 velocity loss 按 DOF 的真实速度方差归一化；
2. 让 velocity loss 对总损失贡献约 5%～15%；
3. 为 decoder 增加轻量 DOF-specific adapter 或独立输出 head；
4. 比较共享 decoder 与 DOF-specific decoder 的参数量和重建指标；
5. 以平均单周期 correlation 和 velocity RMSE 选择模型，而不是只看 pooled reconstruction MSE。

重点观察：

```text
IE RMSE、IE velocity RMSE
ML RMSE、ML velocity RMSE
AA/ML 平均单周期 correlation
```

### 第三优先级：提高相似周期的分词稳定性

如果使用 correlation + RMSE 的严格标准后，FE 等 DOF 仍存在大量相似周期被分到远距离词，可考虑加入：

- 同一周期弱增强前后的 code consistency；
- encoder embedding cosine consistency；
- prototype soft assignment；
- 相似健康周期的轻量对比约束；
- top-1/top-2 margin regularization。

不建议直接要求同一受试者所有周期使用相同 code，因为真实步态周期本身存在自然变化。

### 第四优先级：将不确定性传递到 SSL 和 Downstream

后续健康偏离不应只使用到最近 prototype 的原始距离。更合理的偏离量包括：

```text
deviation = (word - healthy prototype) / healthy within-code scale
```

即使用每个 code、每个 DOF 的健康组内方差对偏离进行归一化。这样：

- 低方差健康词发生小变化时，可以得到较高异常分数；
- 高方差健康词不会因为正常变化范围较大而被误判；
- IE/ML 可以根据其较高 tokenizer 不确定性降低置信度；
- downstream 可获得更有统计意义的偏离方向和程度。

可以进一步把以下信息一并传入 downstream：

- 最近 prototype 距离；
- assignment margin；
- code 内健康方差；
- reconstruction error；
- code 健康出现频率；
- 左右侧条件 surprise。

---

## 10. 推荐的下一组 VQ 实验

建议只运行以下三个对照，避免同时改变过多因素：

| 实验 | Codebook | Decoder | Velocity loss | 目的 |
|---|---|---|---|---|
| VQ-A | K=128 | 当前共享 decoder | 当前定义 | 当前基线 |
| VQ-B | K=128 | 当前共享 decoder | 按 DOF 归一化 | 判断速度监督是否改善 IE/ML |
| VQ-C | K=128 | DOF-specific adapter/head | 按 DOF 归一化 | 判断共享 decoder 是否为主要瓶颈 |

每个实验至少比较：

- 每个 DOF 的全局 perplexity；
- 每个 DOF 的 RMSE；
- 平均单周期 correlation；
- velocity RMSE；
- code 内加权方差；
- 相似周期 same/near-code rate；
- prototype duplicate ratio。

只有在 VQ-C 仍出现明显 assignment 不稳定时，再进行 `K=64` 消融。当前结果不支持直接增大到 K=256。

## 11. 最终结论

当前 VQ 的词表容量和使用状态是健康的：所有 DOF 都使用了全部 128 个 code，全局 perplexity 高，严格重复 prototype 几乎不存在。因此，codebook collapse 和词表容量不足不是当前主要问题。

真正需要改进的是：

1. IE 和 ML 的波形及速度重建；
2. 少数高方差 code 的纯度；
3. FE 等 DOF 的相似周期 assignment 稳定性；
4. 左右侧坐标和相似周期定义；
5. 将 code 内健康方差和量化不确定性用于后续偏离评分。

在完成这些工作后，当前 VQ 可以作为健康步态 codebook 的可靠基础，并为后续节律 SSL、双侧差异建模以及 ACLD/KOA 偏离可视化提供更稳定、更可解释的离散词表示。
