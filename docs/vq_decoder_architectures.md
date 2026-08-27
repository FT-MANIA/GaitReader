# VQ 四套波形 Decoder 架构说明

本文档描述 `knee_kinematics/gait_language/vq.py` 中实际实现的四套 VQ 波形重建 Decoder：

1. `DOFWordDecoder`：MLP Decoder；
2. `TemporalTransformerWordDecoder`：单词内部 Temporal Transformer Decoder；
3. `SentenceTransformerWordDecoder`：双侧完整句子 Sentence Transformer Decoder；
4. `LocalContextResidualSentenceDecoder`：local morphology 主干与 sentence contextual residual Decoder。

四套 Decoder 共用相同的 `DOFWordEncoder`、六个 DOF 独立 EMA codebook、量化方法和基础 VQ loss。实验时只切换 Decoder，因此可以直接比较不同重建归纳偏置对 encoder 与 codebook 学习结果的影响。

## 1. 共用的 VQ pipeline

语言 batch 中的主要张量为：

```text
words:     [B, 2, W, 6, T]
word_mask: [B, 2, W]
timing:    [B, 2, W, 4]
```

其中：

- `B`：batch size；
- `2`：左腿和右腿；
- `W`：一个受试者每侧保留的最大步态周期数；
- `6`：六个膝关节 DOF；
- `T`：每个周期时间归一化后的采样点数，即 `--word-length`；
- `timing[..., 0]`：周期持续时间；
- `timing[..., 1]`：周期中心位置；
- `timing[..., 2]`：与前一个周期的间隔；
- `timing[..., 3]`：周期质量分数。

完整计算过程为：

```text
标准化周期波形 words [B,2,W,6,T]
    ↓ DOFWordEncoder
连续 word embedding z_e [B,2,W,6,D]
    ↓ 六个 DOF 独立的 EMA codebook
量化 word embedding z_q [B,2,W,6,D]
    ↓ 可选择的 Decoder
重建周期波形 x_hat [B,2,W,6,T]
```

`D` 为 `--word-dim`。左右腿共享同一套 DOF codebook，即每个 DOF 具有一个大小为 `K × D` 的词表，整个 codebook 的形状为：

```text
[6, K, D]
```

切换 Decoder 不改变以下内容：

- encoder 结构；
- 最近邻 code assignment；
- EMA codebook 更新；
- commitment loss；
- waveform reconstruction loss；
- velocity reconstruction loss；
- code usage 和 perplexity 统计。

总损失仍为：

```text
L_total
= L_reconstruction
+ velocity_weight × L_velocity
+ commitment_weight × L_commitment
```

## 2. MLP Decoder

类名：

```python
DOFWordDecoder
```

运行参数：

```text
--vq-decoder mlp
```

这也是默认选项，用于保持原有实验行为。

### 2.1 输入与输出

```text
输入：[..., 6, D]
输出：[..., 6, T]
```

它既可以接收完整 batch 的 `[B,2,W,6,D]`，也可以接收 codebook prototype 重建时的 `[K,6,D]`。

### 2.2 模块组成

MLP Decoder 包含：

```text
DOF identity embedding: [6,D]
Linear(D, 2D)
GELU
Linear(2D, T)
```

对第 `d` 个 DOF，计算过程为：

```text
h_d = z_q,d + e_dof,d
x_hat_d = Linear_2(GELU(Linear_1(h_d)))
```

六个 DOF 共用两层 MLP 参数，通过不同的 DOF identity embedding 区分屈伸、内外翻、内外旋及三个平移自由度。

### 2.3 能够学习的信息

MLP Decoder 从单个量化词直接回归完整周期波形，主要约束一个词中保留足够的整体波形形态信息。它不显式建模：

- 周期内部不同相位之间的关系；
- 同侧相邻周期之间的关系；
- 六个 DOF 之间的关系；
- 左右腿之间的关系；
- timing 信息。

因此它是四套 Decoder 中结构最简单、计算量最小、最适合作为基线的一套。

## 3. Temporal Transformer Word Decoder

类名：

```python
TemporalTransformerWordDecoder
```

运行参数：

```text
--vq-decoder temporal_transformer
```

该 Decoder 的目标是在每个 DOF 的单个 gait word 内部显式恢复时间相位结构。不同单词彼此独立解码，不进行跨周期、跨 DOF 或跨腿注意力。

### 3.1 输入与输出

```text
输入：[..., 6, D]
输出：[..., 6, T]
```

### 3.2 从一个 word 生成 phase tokens

设 phase token 数量为 `P`：

```text
P = --vq-decoder-phase-tokens
```

每个 phase token 最终负责重建的点数为：

```text
patch_size = ceil(T / P)
```

如果 `P × patch_size` 大于 `T`，拼接后会截取前 `T` 个点。因此该实现不要求 `T` 必须能被 `P` 整除。

对第 `d` 个 DOF 的量化词，首先加入 DOF embedding，并经过该 DOF 独立的 residual adapter：

```text
h_d = z_q,d + e_dof,d
h_d = h_d + Adapter_d(h_d)
```

adapter 的结构为：

```text
Linear(D, max(8, D/4))
GELU
Linear(max(8, D/4), D)
```

随后使用 `P` 个可学习 phase embedding，将一个 word 展开成一组相位 token：

```text
p_d,j = h_d + e_phase,j,  j = 1 ... P
```

### 3.3 Transformer 与波形输出

`P` 个 phase tokens 输入共享的 Transformer Encoder：

```text
TransformerEncoderLayer:
    d_model = D
    nhead = --vq-decoder-heads
    dim_feedforward = --vq-decoder-ff-dim
    dropout = --vq-decoder-dropout
    activation = GELU
    norm_first = True
    batch_first = True

Transformer depth = --vq-decoder-depth
Final normalization = LayerNorm(D)
```

Transformer 在同一个 word 的 phase tokens 之间执行 self-attention，使不同步态相位可以相互交换信息。每个 DOF 使用独立的线性输出头：

```text
Linear(D, patch_size)
```

得到 `P` 段波形 patch 后按时间维拼接并截取为 `T` 点：

```text
[P,D]
  ↓ DOF-specific output head
[P,patch_size]
  ↓ flatten and crop
[T]
```

### 3.4 能够学习的信息

相较 MLP Decoder，它显式引入了：

- 周期内部的有序相位位置；
- 不同相位之间的全局依赖；
- DOF-specific adapter；
- DOF-specific waveform head。

它仍然不利用：

- 同侧其他周期；
- 其他 DOF 的词；
- 对侧腿的词；
- timing 信息。

因此这套 Decoder 最适合回答：仅增强单个词内部的时间建模，是否能使 codebook prototype 更像真实步态周期，并降低 RMSE、velocity error。

## 4. Sentence Transformer Decoder

类名：

```python
SentenceTransformerWordDecoder
```

运行参数：

```text
--vq-decoder sentence_transformer
```

这套 Decoder 不再把每个 gait word 视为独立样本，而是把一个受试者左右腿的全部有效周期和全部 DOF 词组成一条双侧 gait sentence，在完整上下文中联合重建所有波形。

### 4.1 输入与输出

```text
量化词：  [B,2,W,6,D]
有效 mask：[B,2,W]
timing：  [B,2,W,4]
输出波形：[B,2,W,6,T]
```

### 4.2 Token identity 与 timing 注入

每个 token 由五部分相加：

```text
token
= quantized word embedding
+ side embedding
+ cycle-position embedding
+ DOF embedding
+ timing projection
```

对应的可学习模块为：

```text
side_embedding:  [2,D]
cycle_embedding: [max_words,D]
dof_embedding:   [6,D]

timing_projection:
    Linear(4,D)
    GELU
    Linear(D,D)
```

side embedding 区分左右腿，cycle-position embedding 表示该词在当前腿 gait sentence 中的周期位置，DOF embedding 表示自由度类别，timing projection 则注入周期时长、记录位置、相邻间隔和周期质量。

### 4.3 Sentence flatten 与有效 token mask

带有身份和 timing 信息的张量 `[B,2,W,6,D]` 被展平为：

```text
[B, 2 × W × 6, D]
```

`word_mask [B,2,W]` 同时扩展到六个 DOF，形成 Transformer 的 padding mask：

```text
[B, 2 × W × 6]
```

padding 周期不会作为有效 token 参与注意力，Transformer 输出中的 padding 位置也会被清零。VQ reconstruction loss 和 velocity loss 仍只在有效周期位置计算。

### 4.4 Transformer 与 DOF-specific heads

Sentence Transformer 的配置为：

```text
TransformerEncoderLayer:
    d_model = D
    nhead = --vq-decoder-heads
    dim_feedforward = --vq-decoder-ff-dim
    dropout = --vq-decoder-dropout
    activation = GELU
    norm_first = True
    batch_first = True

Transformer depth = --vq-decoder-depth
Final normalization = LayerNorm(D)
```

Transformer 输出恢复为 `[B,2,W,6,D]` 后，每个 DOF 使用独立波形头：

```text
Linear(D,D)
Tanh
Linear(D,T)
```

六个输出头的结果在 DOF 维重新堆叠，最终得到 `[B,2,W,6,T]`。

### 4.5 能够学习的信息

Sentence Transformer 可以在重建过程中利用：

- 单侧相邻周期之间的节律与形态一致性；
- 六个 DOF 的耦合关系；
- 左右腿之间的对应、对称和不对称关系；
- 周期顺序；
- 周期时长、位置、间隔和质量。

它对 VQ encoder 的约束与前两套 Decoder 不同：一个量化词不再必须独立解释完整波形，Decoder 可以借助句子上下文完成重建。因此需要同时观察 reconstruction 指标与 codebook 诊断，避免强 Decoder 通过上下文补全波形，却让单个 code 的 morphology 语义变弱。

### 4.6 Codebook prototype 的离线解码

Sentence Decoder 的真实重建依赖上下文，而离线诊断仍需要把每个 codebook prototype 单独变成一条波形。`decode_codebook()` 因此采用统一的 canonical context：

- side position 固定为左侧，即 `side_embedding[0]`；
- cycle position 固定为第一个周期，即 `cycle_embedding[0]`；
- 不加入 timing；
- 每个 prototype 作为长度为 1 的 sentence 单独通过同一 Transformer；
- 使用相应 DOF 的输出头生成波形。

这样可以继续执行 prototype 重复度和 prototype 波形相似度分析，但该结果表示 canonical context 下的原型波形，不等同于某个具体受试者句子上下文中的重建结果。

## 5. 四套 Decoder 对比

| 属性 | MLP | Temporal Transformer | Sentence Transformer | Local + Context Residual |
|---|---|---|---|---|
| 输入上下文 | 单个 DOF word | 单个 DOF word 的 phase tokens | 左右腿全部周期和全部 DOF words | local 单词与完整 sentence |
| 周期内部相位建模 | 隐式 | 显式 | 由 word embedding 与句子上下文间接建模 | local 整段重建加上下文修正 |
| 跨周期建模 | 否 | 否 | 是 | 仅 residual branch |
| 跨 DOF 建模 | 否 | 否 | 是 | 仅 residual branch |
| 双腿关系建模 | 否 | 否 | 是 | 仅 residual branch |
| 使用 timing | 否 | 否 | 是 | 仅 residual branch |
| DOF-specific 参数 | identity embedding | embedding、adapter、output head | embedding、output head | local embedding 与 residual output head |
| 计算量 | 最低 | 中等 | 高 | 最高 |
| prototype 独立语义压力 | 最强 | 较强 | 较弱，可能依赖上下文 | 由 local loss 显式保持 |
| 主要研究问题 | 基线重建 | 单词内部时序结构 | 节律、DOF 耦合和双侧上下文 | morphology 与上下文信息的平衡 |

## 6. `run.py` 可调参数

### 6.1 Decoder 类型

```text
--vq-decoder {mlp,temporal_transformer,sentence_transformer,local_context_sentence}
```

默认值：

```text
temporal_transformer
```

### 6.2 Transformer 共用参数

```text
--vq-decoder-depth       Transformer Encoder 层数，默认 2
--vq-decoder-heads       multi-head attention 头数，默认 4
--vq-decoder-ff-dim      feed-forward hidden dimension，默认 512
--vq-decoder-dropout     Transformer dropout，默认 0.10
```

这些参数对 `temporal_transformer`、`sentence_transformer` 和 `local_context_sentence` 生效，对 `mlp` 不生效。

### 6.3 Temporal Transformer 专用参数

```text
--vq-decoder-phase-tokens
```

默认值为 `20`，只对 `temporal_transformer` 生效。若 `--word-length 100`，则每个 phase token 输出 `5` 个连续采样点。

### 6.4 Sentence Transformer 相关参数

```text
--max-words
```

该参数同时确定一个 sentence 的最大周期词数及 `cycle_embedding` 的长度，默认值为 `32`，对 `sentence_transformer` 和 `local_context_sentence` 生效。

## 7. 运行示例

只运行 VQ 阶段并使用原始 MLP Decoder：

```powershell
D:\anaconda\python.exe run.py --stage vq --vq-decoder mlp
```

使用 Temporal Transformer Word Decoder：

```powershell
D:\anaconda\python.exe run.py `
  --stage vq `
  --vq-decoder temporal_transformer `
  --vq-decoder-depth 2 `
  --vq-decoder-heads 4 `
  --vq-decoder-ff-dim 512 `
  --vq-decoder-dropout 0.10 `
  --vq-decoder-phase-tokens 20
```

使用 Sentence Transformer Decoder：

```powershell
D:\anaconda\python.exe run.py `
  --stage vq `
  --vq-decoder sentence_transformer `
  --vq-decoder-depth 2 `
  --vq-decoder-heads 4 `
  --vq-decoder-ff-dim 512 `
  --vq-decoder-dropout 0.10 `
  --max-words 32
```

使用 local morphology 与 contextual residual 组合 Decoder：

```powershell
D:\anaconda\python.exe run.py `
  --stage vq `
  --vq-decoder local_context_sentence `
  --vq-context-residual-scale 0.5 `
  --vq-local-reconstruction-weight 1.0 `
  --vq-decoder-depth 2 `
  --vq-decoder-heads 4 `
  --vq-decoder-ff-dim 512 `
  --vq-decoder-dropout 0.10 `
  --max-words 32
```

运行完整 `vq → ssl → downstream` 流程时同样只需指定 `--vq-decoder`。后续阶段会使用本次 VQ 训练得到的 tokenizer 和 checkpoint。

## 8. 对照实验建议

比较四套 Decoder 时，除 Decoder 相关参数外，应固定：

- 数据划分与随机种子；
- `word_length`、`word_dim` 和 encoder hidden dimension；
- codebook size、EMA decay 和 dead-code threshold；
- commitment weight 和 velocity weight；
- batch size、学习率、epoch 与 early stopping；
- 标准化参数与质量控制结果。

至少同时比较以下结果：

1. 总 reconstruction loss；
2. 每个 DOF 的 RMSE、相关系数和 velocity error；
3. 每个 DOF 的 code usage 与 perplexity；
4. code 内波形方差；
5. 相似周期的 code 一致性；
6. prototype 重复度；
7. 后续 masked-word、masked-DOF 与双侧任务的表现。

Sentence Transformer 及 Local + Context Residual 都具备上下文补全能力，因此不能仅凭 reconstruction loss 更低就认定 codebook 更好。最终判断应以“重建质量、词表利用率、prototype 可解释性、健康步态结构表征能力”四类结果共同决定。

## 9. Checkpoint 说明

四套 Decoder 的参数名称和结构不同，Decoder checkpoint 不能在不同类型之间直接互换。旧版本 checkpoint 对应 `mlp`，运行旧实验或加载旧 VQ 权重时应使用：

```text
--vq-decoder mlp
```

每次实验应使用独立输出目录，并在结果中保留本次 `args.json`，从而记录实际采用的 Decoder 类型及全部 Decoder 参数。

旧实验的 `args.json` 不包含 Decoder 字段时，构建函数会按原有行为使用 `mlp` 及当前默认 Decoder 参数。若单独运行 `ssl`、`downstream` 或 `evaluate` 阶段，则命令中的 Decoder 类型和结构参数必须与被加载的 VQ checkpoint 完全一致。

## 10. 四种 Decoder 的实际实验结果

### 10.1 实验对应关系与可比性

本节分析 `Results/gait_language` 中以下四次完整实验。按照实验开始时间从早到晚，对应关系为：

| Decoder | 实验目录 | 实验时间 | `args.json` 中的类型 |
|---|---|---|---|
| MLP | `dev_exp_0823_1849` | 08-23 18:49 | 旧版本未记录该字段，实际为默认 MLP |
| Sentence Transformer | `dev_exp_0824_1718` | 08-24 17:18 | `sentence_transformer` |
| Temporal Transformer | `dev_exp_0824_1802` | 08-24 18:02 | `temporal_transformer` |
| Local + Context Residual | `dev_exp_0824_2014` | 08-24 20:14 | `local_context_sentence` |

四组实验均使用相同的：

- 数据文件与受试者划分；
- `seed=42`；
- `word_dim=128`；
- 每个 DOF 的 `codebook_size=128`；
- VQ 学习率、loss 权重和 early stopping；
- SSL mask、loss 权重和训练配置；
- downstream 配置。

对四个 `args.json` 逐字段比较后，除运行目录和 Decoder 相关字段外，没有发现其他实验参数差异。Sentence、Temporal 和 Local + Context 均使用两层 Transformer、4 个 attention heads、512 维 FFN 和 0.1 dropout；Temporal 使用 20 个 phase tokens。Local + Context 使用 `residual_scale=0.5` 和 `local_reconstruction_weight=1.0`。

因此，本次结果可以作为 Decoder 架构消融实验。不过每个结构目前只有一个随机种子，所有优劣都应视为当前 split 和 seed 下的观察结果，不能直接解释为具有统计显著性的最终结论。

另有两个比较限制需要明确：

1. 四次 VQ 会形成不同的离散词表，因此 SSL 中预测的 class ID 语义并不相同。SSL loss 和 accuracy 可以反映对应词表的可预测性，但不是完全相同 label space 上的严格逐点比较。
2. Sentence 和 Local + Context Decoder 的 prototype 离线波形是在 canonical context 下生成的，而真实重建使用整个双侧 sentence。其单 prototype 解码结果不能完全代表上下文重建行为。

### 10.2 VQ 训练阶段

VQ checkpoint 按最小 validation total loss 保存。四组最佳结果为：

| Decoder | 实际 epoch 数 | 最佳 epoch | Val total loss | Val reconstruction | Val velocity | Val commitment | Batch active ratio | Batch perplexity | VQ checkpoint 大小 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MLP | 85 | 74 | 0.110657 | 0.110049 | 0.001404 | 0.001308 | 0.7681 | 73.961 | 3.26 MB |
| Sentence | 64 | 53 | **0.067468** | 0.066846 | 0.001564 | **0.001235** | 0.7670 | 74.640 | 9.75 MB |
| Temporal | 71 | 60 | 0.108712 | 0.108024 | 0.001677 | 0.001409 | 0.7515 | 72.123 | 8.07 MB |
| Local + Context | 77 | 66 | 0.182793 | **0.066247** | **0.001027** | 0.001252 | **0.7725** | **75.213** | 10.48 MB |

Sentence 的 validation total loss 相比 MLP 降低约 39.0%，主要来自 reconstruction loss 的明显下降。这说明完整句子上下文确实为波形恢复提供了大量有效信息。它也在更少的 epoch 内触发 early stopping。

Temporal 的 validation total loss 只比 MLP 低约 1.8%，没有表现出与模型复杂度相匹配的重建收益；其 validation velocity loss 还是四者中最高。这说明当前 phase-token-to-patch 设计虽然增加了时序建模能力，但没有自动转化为更连续的局部波形。

四组实验的 batch active ratio 和 perplexity 接近，没有训练期 codebook collapse。Sentence 的大幅 loss 优势不能归因于使用更多 code，而主要来自 Decoder 利用上下文的能力。

Local + Context 的 total loss 为 0.182793，不能和前三组直接比较，因为它额外包含 `local_reconstruction_loss=0.116027`。可直接比较的最终 reconstruction 为 0.066247，略优于 Sentence 的 0.066846；velocity loss 则从 Sentence 的 0.001564 降至 0.001027，成为四者最佳。

最佳 checkpoint 的 raw `context_residual_rms=0.459674`。乘以 `alpha=0.5` 后，实际 residual contribution RMS 约为 0.229837。它不是微小扰动，而是最终波形的重要组成部分。Local-only MSE 为 0.116027，比 MLP 的最终 MSE 0.110049高约 5.4%，说明 local branch 尚未恢复到原始 MLP 的独立重建水平。

### 10.3 验证集离线波形重建

离线分析统一使用 `ssl_validation_data` 的 6892 个有效周期。下表为六个 DOF 的宏平均：

| Decoder | Mean RMSE ↓ | Mean pooled corr ↑ | Mean cycle corr ↑ | Mean velocity RMSE ↓ | Mean perplexity ↑ |
|---|---:|---:|---:|---:|---:|
| MLP | 0.325781 | 0.944522 | 0.799534 | 0.036142 | 115.866 |
| Sentence | 0.253133 | 0.967051 | 0.858147 | 0.038450 | **116.906** |
| Temporal | 0.323233 | 0.945799 | 0.787181 | 0.039816 | 114.938 |
| Local + Context | **0.252345** | **0.967172** | **0.873414** | **0.030592** | 116.491 |

Sentence 相比 MLP：

- mean RMSE 降低约 22.3%；
- pooled correlation 从 0.9445 提升到 0.9671；
- mean cycle correlation 从 0.7995 提升到 0.8581；
- velocity RMSE 反而增加约 6.4%。

这表示 Sentence 更擅长恢复整体波形幅值与低频形态，但局部一阶差分并没有同步改善。上下文能够帮助判断“该周期大致应该长什么样”，却不一定能准确恢复所有局部斜率细节。

Temporal 相比 MLP 的 mean RMSE 只降低约 0.8%，mean cycle correlation 和 velocity RMSE 均更差。当前实现让 20 个 phase tokens 各自输出 5 点 patch，再直接拼接为 100 点波形。在 19 个 patch 边界处没有显式连续性约束，这是 velocity error 较高的一个合理结构性解释。

Local + Context 相比原始 Sentence：

- mean RMSE 从 0.253133 小幅降到 0.252345；
- mean cycle correlation 从 0.858147 提升到 0.873414；
- mean velocity RMSE 从 0.038450 降到 0.030592，降低约 20.4%；
- 相比 MLP 的 velocity RMSE 也降低约 15.4%。

这说明 local MLP 主干成功补回了 Sentence Decoder 缺失的局部连续性，同时 contextual residual 保留了整体形态重建优势。这是新结构最明确的成功点。

各 DOF 详细结果如下。每个单元格依次为 `RMSE / mean cycle correlation / velocity RMSE`：

| DOF | MLP | Sentence | Temporal | Local + Context |
|---|---|---|---|---|
| FE 屈伸 | 0.2524 / 0.9730 / 0.0274 | 0.2150 / 0.9783 / 0.0309 | 0.2548 / 0.9697 / 0.0344 | **0.2037 / 0.9812 / 0.0229** |
| AA 内外翻 | 0.2823 / 0.6949 / 0.0322 | 0.2132 / 0.8056 / 0.0327 | 0.2735 / 0.6874 / 0.0345 | **0.2096 / 0.8249 / 0.0254** |
| IE 内外旋 | 0.4024 / 0.7581 / 0.0487 | 0.3108 / 0.8266 / 0.0490 | 0.3978 / 0.7378 / 0.0526 | **0.3044 / 0.8461 / 0.0417** |
| AP 前后平移 | 0.2959 / 0.8557 / 0.0293 | **0.2171** / 0.9039 / 0.0324 | 0.2935 / 0.8470 / 0.0328 | 0.2231 / **0.9148 / 0.0242** |
| ML 内外平移 | 0.4203 / 0.6964 / 0.0510 | 0.3442 / 0.7542 / 0.0538 | 0.4118 / 0.6792 / 0.0536 | **0.3403 / 0.7822 / 0.0461** |
| SI 上下平移 | 0.3014 / 0.8191 / 0.0282 | **0.2184** / 0.8803 / 0.0320 | 0.3081 / 0.8019 / 0.0310 | 0.2330 / **0.8912 / 0.0232** |

Local + Context 在 FE、AA、IE 和 ML 上取得最低 RMSE，Sentence 在 AP 和 SI 上最低；Local + Context 在六个 DOF 上都取得最高 cycle correlation 和最低 velocity RMSE。由此可见，local branch 与 contextual residual 的组合不是只改善某个容易的 DOF，而是系统性改善了波形相关性和局部连续性。

### 10.4 Codebook 利用率与 code 内波形方差

离线全验证集统计与训练日志的 batch 统计口径不同：离线统计会合并全部 6892 个周期，所以 perplexity 明显高于单 batch 日志是正常现象。

| Decoder | Active codes | Mean perplexity | Code 内方差，加权平均 ↓ | 同侧相似周期 same-code ↑ | 同侧 near-code ↑ | 双侧相似周期 same-code ↑ | 双侧 near-code ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| MLP | 768/768 | 115.866 | 0.098073 | 0.5379 | 0.5556 | **0.1029** | **0.1129** |
| Sentence | 767/768 | **116.906** | 0.107435 | 0.5449 | **0.5815** | 0.0805 | 0.0996 |
| Temporal | 768/768 | 114.938 | **0.097018** | **0.5616** | 0.5722 | 0.0947 | 0.1003 |
| Local + Context | 768/768 | 116.491 | 0.103094 | 0.5225 | 0.5419 | 0.0930 | 0.0989 |

四组都使用了几乎全部词表。Sentence 只在 AA 中有一个 code 未使用，不构成 collapse；Local + Context 使用了全部 768 个 code。

Sentence 的 assignment-count 加权 code 内波形方差为 0.1074，比 MLP 高约 9.5%，而且六个 DOF 的方差都比 MLP 高。其各 DOF 加权方差为：

```text
FE 0.0653, AA 0.0736, IE 0.1727,
AP 0.0869, ML 0.1671, SI 0.0790
```

这与 Sentence 的强上下文重建能力形成关键对照：重建结果显著更好，但分到同一个 code 的原始波形反而更加分散。合理解释是 Decoder 可以从其他周期、其他 DOF 和对侧腿补充信息，因而 encoder/codebook 不再需要让单个 code 独立携带全部 morphology 信息。这是一种潜在的 contextual shortcut。

Temporal 的加权 code 内方差最低，并在六个 DOF 中有五个取得最高的同侧相似周期 exact same-code rate；只有 SI 略低于 Sentence。说明 Temporal Decoder 虽然重建 loss 改善不明显，却使量化边界对同侧相似 morphology 更稳定。这一点对“健康 gait word 词表”的目标有实际价值。

MLP 的双侧相似周期 same-code 与 near-code 宏平均最高。Sentence 和 Temporal 都没有提升左右腿相似波形的共同词映射。因此，当前 Decoder 改动不能替代后续专门的 bilateral SSL objective。

Local + Context 的加权 code 内方差为 0.103094，相比 Sentence 的 0.107435降低约 4.0%，说明 local loss 确实部分收紧了 code 内 morphology；但它仍比 MLP 高约 5.1%，也高于 Temporal。分 DOF 看，Local + Context 相比 Sentence 改善了 FE、AA、IE 和 AP，却在 ML 与 SI 上略差。

更值得注意的是，Local + Context 的同侧相似周期 same-code rate 为 0.5225，低于 MLP、Sentence 和 Temporal；near-code rate 也只有 0.5419。这说明 local reconstruction 虽然改善了 code 内方差，却没有让相似周期的量化边界更稳定。当前 `local_weight=1.0` 对 code assignment consistency 的约束仍不充分。

其双侧 same-code rate 从 Sentence 的 0.0805回升到 0.0930，但仍低于 MLP 的 0.1029。新结构部分恢复了双侧共享词映射，却尚未超过简单 MLP。

### 10.5 Prototype 重复与 Decoder aliasing

四套实验均未发现同时满足 embedding 相似阈值和 decoded-waveform 相似阈值的 joint duplicate pair：

```text
MLP joint duplicates      = 0
Sentence joint duplicates = 0
Temporal joint duplicates = 0
Local + Context joint duplicates = 0
```

因此，codebook embedding 本身没有大规模原型坍缩。但只看 decoded waveform，相近 prototype pair 的比例为：

| Decoder | Decoded-near pairs | 全部 prototype pairs | Decoded-near ratio ↓ |
|---|---:|---:|---:|
| MLP | 1014 | 48768 | 2.079% |
| Sentence | 989 | 48768 | 2.028% |
| Temporal | 1495 | 48768 | 3.066% |
| Local + Context | **688** | 48768 | **1.411%** |

Temporal 的 decoded-near ratio 最高，其中 FE 有 740 对 decoded-near prototype，占 FE 全部 prototype pairs 的 9.10%。这说明部分 latent code 虽然在 embedding 空间中不同，经过当前 Temporal Decoder 后却生成了高度相似的波形，即出现 Decoder aliasing。

这再次说明 Temporal 的问题主要可能位于输出结构，而不一定是 codebook 已经坍缩。独立 phase patch head 可能把多个不同 latent pattern 映射成相似的局部 patch 组合。

Local + Context 将 decoded-near ratio 从 Sentence 的 2.028% 降到 1.411%，比 MLP 也低约 32.1%，是四者最佳。这说明 local anchor 不仅改善了速度连续性，也降低了不同 latent code 被 Decoder 映射成相似波形的概率。

Sentence 和 Local + Context 都使用 canonical context 做 prototype 解码，因此该结果不能完全代表真实完整上下文中的重复率。不过在相同离线口径下，新结构相对 Sentence 的改善仍然成立。

### 10.6 后续 SSL 结果

SSL checkpoint 按固定验证 mask 下的最小 validation loss 保存：

| Decoder 对应词表 | 最佳 epoch | Val loss ↓ | Within acc ↑ | Cross-DOF acc ↑ | Bilateral acc ↑ | Contralateral acc ↑ | Swap loss ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| MLP | 45 | **9.0008** | 0.4820 | 0.1219 | 0.4630 | **0.0574** | **0.000352** |
| Sentence | 41 | 9.1793 | 0.4821 | 0.1170 | 0.4666 | 0.0520 | 0.000582 |
| Temporal | 37 | 9.0511 | **0.4843** | 0.1074 | **0.4801** | 0.0477 | 0.001224 |
| Local + Context | 43 | 9.0714 | 0.4696 | **0.1404** | 0.4547 | 0.0555 | 0.000507 |

Sentence 在 VQ reconstruction 上的巨大优势没有传递为更好的 SSL total loss。它的 within accuracy 与 MLP 基本相同，cross-DOF、contralateral 和 total loss 均更差。这进一步支持“强上下文 Decoder 改善了重建，但没有同比增强单个离散词语义”的判断。

Temporal 的 within accuracy 和 bilateral accuracy 最高，与其更高的同侧相似周期 code 一致性相符；但 cross-DOF 和 contralateral accuracy 最低。当前 Temporal 结构主要强化的是单个 DOF 周期内部 morphology，而不是 DOF 之间或左右腿之间的关系。

MLP 的 cross-DOF、contralateral、swap 和 total loss 最好。简单 Decoder 对单个 word 施加了更强的信息瓶颈，可能迫使 encoder 将更多可供其他任务预测的信息保留在 code ID 中。

Local + Context 的 cross-DOF accuracy 达到 0.1404，明显高于此前最佳 MLP 的 0.1219；contralateral accuracy 也回升到 0.0555，接近 MLP 的 0.0574。这说明 local anchor 确实让 code ID 恢复了更多跨 DOF 可预测信息。

但其 within accuracy 为 0.4696、bilateral accuracy 为 0.4547，都是四者最低。这与离线同侧 same-code consistency 下降一致：新词表携带了更强的跨 DOF 关系，却没有形成最稳定的单 DOF morphology token 或 bilateral token。

由于四个词表具有不同的 code 分布和语义，这一表格更适合用于解释趋势，不宜把很小的 accuracy 差异当成严格显著提升。

### 10.7 Downstream 与最终测试集

Downstream checkpoint 按 validation macro-F1 保存。结果为：

| Decoder | Val macro-F1 | Internal acc | Internal macro-F1 | Internal AUROC | Internal side acc | External acc | External macro-F1 | External AUROC | External side acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MLP | 0.8912 | 0.8667 | 0.8335 | **0.9699** | 0.6190 | 0.8203 | 0.8175 | 0.9384 | 0.8119 |
| Sentence | **0.9273** | 0.8444 | 0.8186 | 0.9531 | 0.7460 | 0.8627 | **0.8566** | 0.9612 | 0.8218 |
| Temporal | 0.9203 | 0.8370 | 0.8241 | 0.9412 | **0.7619** | **0.8660** | 0.8536 | 0.9565 | 0.8119 |
| Local + Context | 0.9106 | **0.8741** | **0.8483** | 0.9307 | 0.7460 | 0.8562 | 0.8540 | **0.9629** | **0.8317** |

Sentence 的 validation macro-F1 和 external macro-F1 最高。Temporal 的 external accuracy 最高，macro-F1 与 Sentence 非常接近，同时 internal affected-side accuracy 最高。三种包含 Transformer 的 Decoder 都明显改善了 MLP 的 external disease classification。

前三组的 internal test 曾呈现与 external 不同的排序：MLP 的 accuracy、macro-F1 和 AUROC 都最高。加入 Local + Context 后，internal accuracy 与 macro-F1 被新结构超过，但 MLP 仍保留最高 internal AUROC。不同指标的排序仍说明当前 downstream 数据规模下存在较明显的 split 方差或模型选择方差。

加入新结构后，Local + Context 取得最高 internal accuracy 和 macro-F1，同时获得最高 external AUROC 和 affected-side accuracy；external macro-F1 为 0.8540，与 Sentence 的 0.8566 和 Temporal 的 0.8536 基本相当。它是目前 internal/external 综合表现最均衡的一组。

需要注意 Local + Context 的 internal AUROC 只有 0.9307，是四者最低，但 macro-F1 却最高。这说明其当前分类阈值下的离散预测较好，概率排序或类别间置信度仍不理想，后续应检查逐类 AUROC、混淆矩阵与 calibration，而不能只看 macro-F1。

本项目的首要目标是健康 codebook 和 gait-language encoder，而不是只优化当前三分类结果。因此 downstream 结果应作为表示迁移能力的辅助证据，不能覆盖 code 内方差、prototype 可解释性和相似周期一致性等核心诊断。

### 10.8 综合判断

四套 Decoder 没有一个在全部目标上占优，但 Local + Context 已经取得最均衡的总体结果：

#### MLP

优势：

- 在不使用上下文的结构中具有最好的局部速度重建；
- SSL total loss、contralateral 和 swap 指标最好；
- 双侧相似周期映射到同一或相近 code 的比例最高；
- 参数最少，单个 code 必须独立承担波形重建。

不足：

- 波形 RMSE 和整体 correlation 明显落后于 Sentence 与 Local + Context；
- external downstream 表现最弱；
- 没有显式相位建模。

#### Sentence Transformer

优势：

- 在只有单一 reconstruction loss 的结构中，VQ validation loss 显著最低；
- external macro-F1 和 downstream validation macro-F1 最好；
- 能够利用跨周期、跨 DOF、timing 和双侧上下文。

不足：

- code 内波形方差最高，单个 code 的 morphology 聚合程度下降；
- VQ 的重建优势没有转化成 SSL total loss 优势；
- 双侧相似波形的 same-code 一致性低于 MLP；
- 存在 Decoder 借助上下文绕过单词信息瓶颈的风险。

#### Temporal Transformer

优势：

- code 内波形方差最低；
- 同侧相似周期 exact same-code 一致性最高；
- SSL within 和 bilateral accuracy 最高；
- external downstream 接近 Sentence。

不足：

- 相比 MLP 的 reconstruction 改善很小；
- velocity error 最高；
- cycle correlation 低于 MLP；
- decoded-near prototype 比例最高，存在明显 Decoder aliasing；
- 当前 patch 拼接缺乏边界连续性。

#### Local + Context Residual

优势：

- mean RMSE、mean cycle correlation 和 velocity RMSE 综合最好；
- 六个 DOF 的 cycle correlation 与 velocity RMSE 全部最好；
- decoded prototype alias ratio 最低；
- code 内方差相比 Sentence 降低约 4.0%；
- SSL cross-DOF accuracy 最高；
- internal macro-F1、external AUROC 和 external affected-side accuracy 最高。

不足：

- local-only validation MSE 仍比 MLP 高约 5.4%；
- code 内方差仍高于 MLP 和 Temporal；
- 同侧相似周期 same-code 与 near-code consistency 最低；
- SSL within 和 bilateral accuracy 最低；
- residual 缩放后的 RMS 约为 0.230，context branch 仍承担较大重建责任；
- internal AUROC 最低，分类概率排序仍需检查。

### 10.9 建议的后续改进顺序

根据当前项目“建立健康 gait word codebook，并学习节律、DOF 关系和双侧差异”的目标，建议按以下顺序继续：

1. **将 Local + Context 作为当前主候选，继续保留 MLP 作为固定基线。** 新结构已经同时保留 Sentence 的整体重建能力并显著改善速度连续性和 prototype aliasing，因此不建议退回原始 Sentence 作为主线。
2. **提高 local morphology 权重的方向已经得到支持。** `local_reconstruction_weight=2.0` 的实验使 local-only MSE、code 内方差和同侧 same-code consistency 同时改善，并取得更好的 external test；因此后续 Local + Context 实验建议以 `2.0` 作为当前基准。
3. **显式约束 residual 能量。** 固定 `alpha=0.5` 只能缩放输出和梯度，网络最终可以通过放大 residual 权重抵消该缩放。更直接的方法是增加：

   ```text
   L_residual = lambda_residual × mean((alpha × r_context)²)
   ```

   或对 sentence context 做随机 dropout，使 local branch 在部分 batch 中必须独立完成重建。当前 scaled residual RMS 约为 0.230，可作为后续实验的基准监控值。
4. **再考虑修正 Temporal 的输出连续性。** Temporal 的 code 内方差和同侧一致性仍然最好，可作为另一条研究分支；但当前 Local + Context 已在实际波形、prototype alias 和 downstream 综合结果上更占优，优先级应低于 local/context 平衡调参。
5. **不要仅按 total loss 或 final reconstruction 选择 Decoder。** Local + Context 的 total loss 额外包含 local loss，与其他结构不可直接横向比较。建议采用组合选择指标：

   ```text
   reconstruction RMSE
   + velocity RMSE
   + code 内波形方差
   + 相似周期 code 一致性
   + decoded prototype alias ratio
   + SSL masked-task validation
   ```

6. **在确定 local weight 与 residual regularization 后重复 3～5 个随机种子。** 固定相同 subject split，报告均值和标准差，尤其关注 internal AUROC 与 macro-F1 的反向排序是否稳定。

当前最值得继续发展的结构已经从方案变成了有实测支持的主候选：

```text
Local MLP morphology anchor
+
Sentence contextual residual
+
更强的 local loss 或 residual-energy regularization
```

如果下一阶段只进行一项实验，建议保留 `vq_local_reconstruction_weight=2.0`，增加轻量 residual-energy regularization。因为提高 local weight 已经强化 local branch，但没有降低 residual RMS，下一步需要直接约束 context branch 的输出能量。

### 10.10 `local_reconstruction_weight=2.0` 消融结果

实验目录：

```text
dev_exp_0824_2100
```

该实验只将 `vq_local_reconstruction_weight` 从 `1.0` 提高到 `2.0`，其余关键参数保持不变，包括 `residual_scale=0.5` 和 `seed=42`。

#### VQ 与离线诊断

| 指标 | Weight 1.0 | Weight 2.0 | 变化 |
|---|---:|---:|---:|
| Final validation MSE ↓ | 0.066247 | **0.065596** | -1.0% |
| Local-only validation MSE ↓ | 0.116027 | **0.113775** | -1.9% |
| Raw residual RMS ↓ | **0.459674** | 0.461701 | +0.4% |
| Validation velocity loss ↓ | 0.001027 | **0.001004** | -2.2% |
| Offline mean RMSE ↓ | 0.252345 | **0.250651** | -0.7% |
| Offline mean cycle corr ↑ | 0.873414 | **0.879339** | +0.0059 |
| Offline velocity RMSE ↓ | 0.030592 | **0.030216** | -1.2% |
| Code 内加权方差 ↓ | 0.103094 | **0.100847** | -2.2% |
| 同侧 same-code ↑ | 0.522482 | **0.538337** | +0.0159 |
| 同侧 near-code ↑ | 0.541914 | **0.545368** | +0.0035 |
| Decoded-near ratio ↓ | **1.411%** | 1.509% | +0.098 percentage point |

提高 local weight 后，local-only reconstruction、最终 reconstruction、code 内方差和同侧相似周期一致性全部向预期方向变化。特别是同侧 same-code rate 从 0.5225 提升到 0.5383，已经略高于 MLP 的 0.5379，但仍低于 Sentence 的 0.5449 和 Temporal 的 0.5616。

Code 内方差从 0.1031 降到 0.1008，进一步接近 MLP 的 0.0981，但仍未达到 MLP 或 Temporal。说明 `weight=2.0` 有效加强了 morphology anchor，但尚未完全恢复最紧致的量化聚类。

Raw residual RMS 从 0.4597 微升到 0.4617，经过 `alpha=0.5` 后仍约为 0.2309。也就是说，提高 local loss 并没有减少模型对 contextual residual 的实际使用，只是让 local branch 与 context branch 同时变强。若目标是进一步抑制 contextual shortcut，应直接正则化 scaled residual，而不是继续无限提高 local weight。

Decoded-near ratio 从 1.411% 小幅回升到 1.509%，但仍明显优于 MLP、Sentence 和 Temporal，不构成新的 prototype alias 问题。

本次 VQ 共运行 100 个 epoch，最佳 checkpoint 出现在 epoch 95，没有触发 patience early stopping。因此它与 `weight=1.0` 的 77 个 epoch 结果相比，还包含更长优化时间的影响；后续多随机种子比较应保持相同的最大 epoch 与停止规则。

#### SSL 与 downstream

| 指标 | Weight 1.0 | Weight 2.0 |
|---|---:|---:|
| SSL validation loss ↓ | 9.0714 | **8.9986** |
| Within accuracy ↑ | **0.4696** | 0.4684 |
| Cross-DOF accuracy ↑ | **0.1404** | 0.1385 |
| Bilateral accuracy ↑ | 0.4547 | **0.4725** |
| Contralateral accuracy ↑ | 0.0555 | **0.0610** |
| Swap loss ↓ | 0.000507 | **0.000202** |
| Downstream validation macro-F1 ↑ | **0.9106** | 0.8891 |
| Internal macro-F1 ↑ | **0.8483** | 0.8398 |
| External accuracy ↑ | 0.8562 | **0.8758** |
| External macro-F1 ↑ | 0.8540 | **0.8654** |
| External AUROC ↑ | 0.9629 | **0.9686** |
| External affected-side accuracy ↑ | 0.8317 | **0.8713** |

SSL total loss、bilateral、contralateral 和 swap 均改善，cross-DOF 虽从 0.1404 小幅降到 0.1385，但仍高于 MLP、Sentence 和 Temporal。说明加强 local branch 没有破坏跨 DOF 表征，并改善了双侧相关任务。

Downstream validation macro-F1 和 internal macro-F1 略有下降，但 external accuracy、macro-F1、AUROC 和 affected-side accuracy 全部提升，而且成为当前所有实验中的最佳 external 结果。这表明 `weight=2.0` 更有利于外部泛化，但这一判断仍需多随机种子复现。

综合来看，`local_reconstruction_weight=2.0` 优于 `1.0`，可以作为新的默认候选配置；下一步不建议继续单独增大 local weight，而应尝试小权重 residual-energy regularization，在保持 external 泛化和波形重建的同时进一步降低 context 依赖。

## 11. Local morphology + contextual residual 实现

类名：

```python
LocalContextResidualSentenceDecoder
```

运行选项：

```text
--vq-decoder local_context_sentence
```

### 11.1 两条重建分支

local branch 使用原始 `DOFWordDecoder`，每个量化词只依赖自身 `z_q` 和 DOF identity embedding：

```text
x_local = LocalDecoder(z_q)
```

context residual branch 使用完整 `SentenceTransformerWordDecoder` 的 side、cycle、DOF 和 timing embedding，以及双侧完整 sentence self-attention，但它的输出被解释为波形修正量而不是完整波形：

```text
r_context = SentenceResidualDecoder(z_q, word_mask, timing)
```

最终波形为：

```text
x_hat = x_local + alpha × r_context
```

其中 `alpha` 由以下参数控制：

```text
--vq-context-residual-scale
```

默认值为 `0.5`。它是固定缩放系数，不参与学习，用于限制 contextual branch 对最终波形的直接控制强度。

### 11.2 Residual 零初始化

context residual branch 的每个 DOF 最后一层输出权重和 bias 均初始化为零。因此训练开始时：

```text
r_context = 0
x_hat = x_local
```

模型从稳定的 local reconstruction 起点开始训练，随后逐步学习哪些信息需要由跨周期、跨 DOF、timing 或对侧上下文进行修正。

### 11.3 Loss

最终重建和 local-only 重建分别计算 masked MSE：

```text
L_contextual = MSE(x_hat, x)
L_local = MSE(x_local, x)
```

实际总损失为：

```text
L_total
= L_contextual
+ local_reconstruction_weight × L_local
+ velocity_weight × L_velocity(x_hat, x)
+ commitment_weight × L_commitment
+ residual_energy_weight × L_residual_energy
```

其中 residual-energy 项只统计有效 word：

```text
scaled_residual = alpha × r_context
L_residual_energy = mean(scaled_residual² on valid words)
```

两个结构约束权重分别由以下参数控制：

```text
--vq-local-reconstruction-weight
--vq-residual-energy-weight
```

当前默认值分别为 `2.0` 和 `0.0`。local loss 直接阻止 contextual branch 完全替代单个 code 的 morphology 信息；residual-energy 默认关闭以保持已有实验可复现，消融实验必须显式提供正权重。固定减小 `alpha` 不能替代能量约束，因为 residual branch 可以放大未缩放输出抵消它。

### 11.4 新增训练指标

VQ 的 `metrics.jsonl` 增加：

```text
train/local_reconstruction_loss
validation/local_reconstruction_loss
train/residual_energy_loss
validation/residual_energy_loss
train/raw_context_residual_rms
validation/raw_context_residual_rms
train/scaled_context_residual_rms
validation/scaled_context_residual_rms
train/local_to_final_improvement
validation/local_to_final_improvement
train/context_residual_rms
validation/context_residual_rms
```

其中：

- `reconstruction_loss`：最终组合波形 `x_hat` 的重建误差；
- `local_reconstruction_loss`：只使用单个量化词时的重建误差；
- `residual_energy_loss`：未乘正则权重的 scaled residual 均方；
- `raw_context_residual_rms`：Sentence branch 原始输出残差的 RMS；
- `scaled_context_residual_rms`：实际加入 local 波形的 `alpha × residual` RMS；
- `local_to_final_improvement`：`L_local - L_contextual`，正值表示 contextual residual 改善最终重建；
- `context_residual_rms`：为兼容旧实验日志而保留的 `raw_context_residual_rms` 别名。

分析新实验时应同时观察：

1. `reconstruction_loss` 是否接近原始 Sentence；
2. `local_reconstruction_loss` 是否接近或优于 MLP；
3. `context_residual_rms` 是否稳定在合理范围，而不是持续扩大；
4. code 内波形方差能否从 Sentence 的 0.1074 降回 MLP 的约 0.0981；
5. 同侧与双侧相似周期 code 一致性是否改善；
6. external downstream 优势是否得到保留。

### 11.5 Prototype 离线解码

`decode_codebook()` 会分别生成 local prototype 和 canonical-context residual，再按照相同的 `alpha` 合并：

```text
prototype_hat
= local_prototype
+ alpha × canonical_context_residual
```

因此现有 `analyze_vq.py` 可以直接分析这种新 Decoder，不需要单独的离线分析入口。

## 12. Residual-energy 权重消融（2026-08-25）

### 12.1 实验设计与公平性审计

本轮在 `LocalContextResidualSentenceDecoder`、`local_reconstruction_weight=2.0`、`residual_scale=0.5` 的基础上，只改变 residual-energy 权重：

| 实验 | `residual_energy_weight` | seed |
|---|---:|---:|
| `dev_exp_0824_2100` | 0.00 | 42 |
| `dev_exp_0825_1713` | 0.01 | 42 |
| `dev_exp_0825_2043` | 0.05 | 42 |
| `dev_exp_0825_2234` | 0.10 | 42 |

`dev_exp_0824_2100` 早于新参数加入代码，其 `args.json` 中没有 `vq_residual_energy_weight` 字段；根据当时的 loss 实现，它严格等价于 `0.0` 基线。逐字段比较四个 `args.json` 后，除 `run_dir` 和 residual-energy 权重外，其余参数完全相同，包括 subject split seed、batch size、VQ/SSL/downstream 最大 epoch、patience、学习率和数据划分比例。四个 `word_statistics.json` 的 SHA-256 也完全一致，说明数据处理与统计结果一致。

所有实验使用相同的最大 epoch 和 early-stopping 规则，但实际停止 epoch 可以不同。VQ checkpoint 状态如下；epoch 是 checkpoint 中保存的从 0 开始的编号：

| 权重 | 最佳 VQ epoch | 实际运行 epoch 数 | 停止方式 |
|---:|---:|---:|---|
| 0.00 | 95 | 100 | 跑满最大 epoch |
| 0.01 | 84 | 95 | patience early stopping |
| 0.05 | 59 | 70 | patience early stopping |
| 0.10 | 66 | 77 | patience early stopping |

因此，这四次运行构成同一 seed 下公平的 residual-energy 权重消融，但不是 3～5 个 seed 的稳定性实验。当前结果只能支持 seed=42 下的结构判断，不能报告均值、标准差或统计显著性。

### 12.2 VQ validation loss 分解

下表均取各自 `best_vq.pt` 对应的 `ssl_validation_data` 指标。基线的 scaled residual RMS 根据 `0.5 × raw RMS` 计算，residual energy 为其平方。

| 权重 | Final MSE ↓ | Local-only MSE ↓ | Scaled residual RMS ↓ | Residual energy ↓ | Local→final improvement ↑ | Velocity loss ↓ |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.065596 | 0.113775 | 0.230851 | 0.053292 | 0.048179 | **0.001004** |
| 0.01 | 0.065672 | **0.112230** | 0.208758 | 0.043580 | 0.046558 | 0.001017 |
| 0.05 | **0.065119** | 0.114906 | 0.208906 | 0.043642 | **0.049787** | 0.001044 |
| 0.10 | 0.066573 | 0.113830 | **0.207363** | **0.043000** | 0.047257 | 0.001037 |

主要现象：

1. residual-energy regularization 在机制上有效。相对 `0.0`，三个正权重分别将 scaled residual RMS 降低 9.57%、9.51% 和 10.17%，residual energy 降低 18.22%、18.11% 和 19.31%。
2. 大部分能量下降在 `0.01` 已经完成。`0.01 → 0.05 → 0.10` 没有清晰的单调剂量响应，scaled RMS 基本停留在约 0.208；因此继续增加权重的边际收益很小。
3. `0.01` 的 local-only MSE 比基线改善 1.36%，final MSE 仅增加 0.12%，是 local morphology 与 residual 抑制之间最平衡的结果。
4. `0.05` 取得最低 final MSE，但 local-only MSE 比基线高 0.99%，velocity loss 高 4.0e-5；它改善的是最终组合重建，而不是更强的 local 独立重建。
5. `0.10` 相比基线只比 `0.01` 多降低约 0.6% 的 scaled RMS，却使 final MSE 增加 1.49%，已经出现过强约束的迹象。

不同权重的 VQ total loss 包含不同大小的 `weight × residual_energy_loss`，不能直接用 total loss 排名。应以上表的未加权分量和离线诊断进行判断。

### 12.3 离线 reconstruction、codebook 与 assignment 诊断

以下指标均由各自 `best_vq.pt` 在同一个 `ssl_validation_data` 上离线计算：

| 权重 | Mean RMSE ↓ | Mean cycle corr ↑ | Velocity RMSE ↓ | Code 内方差 ↓ | 同侧 same-code ↑ | 双侧 same-code ↑ | Decoded-near ratio ↓ | Mean perplexity |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.250651 | **0.879339** | **0.030216** | 0.100847 | 0.538337 | 0.093806 | **1.509%** | 116.647 |
| 0.01 | 0.250939 | 0.876021 | 0.030394 | **0.099532** | **0.540500** | 0.094185 | 1.969% | 116.428 |
| 0.05 | **0.249950** | 0.872675 | 0.030857 | 0.100814 | 0.536382 | 0.093155 | 1.720% | 117.347 |
| 0.10 | 0.252626 | 0.875930 | 0.030717 | 0.101120 | 0.535198 | **0.100411** | 1.710% | 116.509 |

解释如下：

- `0.01` 将 code 内方差降低 1.30%，并取得最高同侧 same-code consistency；这与更低的 local-only MSE一致，是支持轻量正则的最重要 codebook 证据。
- `0.05` 的平均 RMSE 最低，但平均 cycle correlation 和 velocity RMSE 最差。其 RMSE 优势主要来自 ML 与 SI，FE、AA、IE 和 AP 并未一致改善，因此不是六个 DOF 上的统一收益。
- `0.10` 的双侧 same-code 最高，但同侧 same-code、code variance 和平均 RMSE 都劣于基线，不能仅凭双侧指标选择它。
- 四个实验的平均 perplexity 都在 116～117，全部 6×128 个 code 在离线验证集中均被使用；residual regularization 没有引起 code usage collapse。
- 四个实验都没有 embedding-near pair 或 joint duplicate，但所有正权重的 decoded-near ratio 都高于基线。也就是说，降低 residual 能量没有改善 canonical decoded prototype alias，`0.01` 的 decoded-near ratio 反而最高。

综合 VQ 与 codebook 指标，`0.01` 是正则化候选中最均衡的一档；`0.05` 只在平均 RMSE 上占优，`0.10` 已经开始损害重建和同侧语义一致性。

### 12.4 Sentence SSL 结果

每个实验使用自己的 VQ codebook，因此不同实验的 code ID label space 不同。下表可用于观察训练链路是否整体退化，但 raw CE/accuracy 不是完全相同类别空间上的严格配对指标。

| 权重 | SSL val loss ↓ | Within acc ↑ | Cross-DOF acc ↑ | Bilateral acc ↑ | Contralateral acc ↑ |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 8.998605 | 0.468445 | **0.138542** | 0.472469 | 0.061028 |
| 0.01 | **8.950421** | 0.470574 | 0.128729 | **0.480634** | **0.062016** |
| 0.05 | 9.051404 | 0.474579 | 0.121596 | 0.474653 | 0.059046 |
| 0.10 | 8.990196 | **0.479984** | 0.131870 | 0.470374 | 0.053752 |

`0.01` 取得最低 SSL validation loss、最高 bilateral accuracy 和最高 contralateral accuracy，但 cross-DOF accuracy 低于基线。`0.10` 的 within-DOF accuracy 最高，却同时得到最低 contralateral accuracy。没有任何正权重在全部 SSL task 上占优，说明 residual 正则改变了 codebook 的类别分布和任务难度，而不是简单地让所有 sentence task 同时变容易。

### 12.5 Downstream 与泛化结果

Downstream checkpoint 按 `dev_validation` macro-F1 选择。内部测试结果为：

| 权重 | Dev val F1 ↑ | Internal acc ↑ | Internal macro-F1 ↑ | Internal AUROC ↑ | Internal side acc ↑ |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.889060 | 0.866667 | 0.839814 | 0.948281 | 0.698413 |
| 0.01 | 0.891113 | 0.888889 | 0.863501 | **0.962805** | 0.634921 |
| 0.05 | **0.910623** | **0.896296** | **0.883793** | 0.960726 | **0.777778** |
| 0.10 | 0.903843 | 0.859259 | 0.850253 | 0.954122 | 0.666667 |

`0.05` 在 dev validation macro-F1、internal accuracy、macro-F1 和 affected-side accuracy 上最好；`0.01` 的 internal AUROC 最好。因此，如果只观察开发集和内部测试，轻到中等 residual regularization 具有潜在收益。

External test 结果为：

| 权重 | External acc ↑ | External macro-F1 ↑ | External AUROC ↑ | External side acc ↑ |
|---:|---:|---:|---:|---:|
| 0.00 | **0.875817** | **0.865371** | **0.968639** | **0.871287** |
| 0.01 | 0.866013 | 0.859708 | 0.954936 | 0.831683 |
| 0.05 | 0.843137 | 0.839935 | 0.967266 | 0.811881 |
| 0.10 | 0.833333 | 0.825308 | 0.959943 | 0.801980 |

在这个单 seed 结果中，`0.0` 基线仍然保持四项 external 指标全部最佳。正则权重增加时，external accuracy、macro-F1 和 affected-side accuracy 总体下降；`0.10` 的下降最明显。`0.05` 的 external AUROC 接近基线，但 accuracy、macro-F1 和 side accuracy 均明显较低。

internal 与 external 排名不一致，尤其是 `0.05` 的 internal 最优没有转化为 external 最优。这可能来自单 seed 波动、不同 codebook label space、VQ early-stopping epoch 差异以及 downstream 优化随机性，不能据此断言 residual regularization 必然损害外部泛化。

### 12.6 当前结论与模型选择

本轮实验支持以下结论：

1. **正则项在工程目标上有效。** 任一正权重都能把 scaled contextual residual RMS 降低约 10%，证明直接约束 residual energy 比单独调低固定 `alpha` 更有效。
2. **收益在 `0.01` 基本饱和。** 更大的 `0.05/0.10` 没有继续显著降低 residual RMS，却带来局部重建、correlation、velocity、assignment consistency 或 downstream 泛化之间的额外权衡。
3. **`0.01` 是最合理的正则化候选。** 它同时得到最低 local-only MSE、最低 code 内方差、最高同侧 same-code、最佳 SSL validation loss，以及几乎不变的 final reconstruction。
4. **`0.05` 是 internal downstream 候选而不是明确的 VQ 候选。** 它取得最低离线 RMSE 和最佳 dev/internal disease classification，但 local-only、cycle correlation、velocity 和 external 指标不支持把它直接设为默认值。
5. **`0.10` 不建议继续。** 它没有相对 `0.01` 带来有意义的 residual 下降，同时重建和 external performance 最弱。
6. **当前默认仍应保留 `residual_energy_weight=0.0`。** 单 seed 下没有证据表明正权重能稳定保留或提升 external 泛化；在完成多随机种子验证前，不应替换现有主基线。

下一轮不必继续扩展大权重网格。建议只保留：

```text
residual_energy_weight ∈ {0.0, 0.01}
training seed ∈ 3～5 个预先固定值
subject split seed 固定不变
```

如果希望验证 `0.05` 的 internal 优势是否稳定，可以将其作为第三个候选，但不建议继续保留 `0.10`。正式多 seed 实验必须把 subject split seed 与训练 seed 分离，或复用固定 split manifest；否则改变 seed 会同时改变受试者划分，失去配对消融意义。

External test 在本轮分析中只应作为已经完成实验的泛化审计，不应用于继续调节 residual 权重。由于同一个 external holdout 已被多轮 Decoder 消融反复查看，最终确认性研究最好保留新的未见测试集，或明确把当前 external 结果标记为 exploratory，而不是完全未触碰的最终验证。
