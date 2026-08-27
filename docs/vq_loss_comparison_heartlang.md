# 当前 VQ Reconstruction Loss 与 HeartLang 的对比

核心结论是：

> 单独比较 `reconstruction_loss`，当前项目与 HeartLang 基本相同，都是重建波形的逐点 MSE。真正的区别主要发生在完整 VQ loss、重建上下文和权重分配上。

## 1. 两者的损失公式

### HeartLang 论文

HeartLang 论文将 VQ-HBR loss 写成三部分：

\[
\mathcal L_{\text{HeartLang}}
=
\mathcal L_{\text{rec}}
+\mathcal L_{\text{codebook}}
+\mathcal L_{\text{commit}}
\]

其中：

\[
\mathcal L_{\text{rec}}
=
\|\hat{x}_i-x_i\|_2^2
\]

\[
\mathcal L_{\text{codebook}}
=
\left\|
sg(\ell_2(p_i))-\ell_2(v_{z_i})
\right\|_2^2
\]

\[
\mathcal L_{\text{commit}}
=
\left\|
\ell_2(p_i)-sg(\ell_2(v_{z_i}))
\right\|_2^2
\]

也就是：

1. 重建原始 ECG 波形；
2. 将 codebook prototype 拉向 encoder embedding；
3. 将 encoder embedding 拉向选中的 prototype。

论文公式中三个部分没有显式给出不同权重。[HeartLang 论文第 3.3 节](https://arxiv.org/html/2502.10707#S3.SS3)

### HeartLang 官方代码

官方实现实际上使用：

```python
rec_loss = mse(reconstructed, target)
quant_loss = beta * mse(quantized.detach(), encoded)
total_loss = rec_loss + quant_loss
```

其中 `beta=1.0`。Codebook 不通过显式的 `codebook loss` 反向传播更新，而是通过 EMA 更新。因此，官方代码相当于：

\[
\mathcal L_{\text{HeartLang-code}}
=
\mathcal L_{\text{rec}}
+
1.0\,\mathcal L_{\text{commit}}
\]

显式 codebook attraction term 被 EMA 更新替代。[HeartLang VQ-HBR 模型](https://github.com/PKUDigitalHealth/HeartLang/blob/main/modeling_vqhbr.py)、[EMA quantizer](https://github.com/PKUDigitalHealth/HeartLang/blob/main/utils/norm_ema_quantizer.py)

### 当前项目

当前实现为：

\[
\mathcal L_{\text{current}}
=
\mathcal L_{\text{rec}}
+
0.2\,\mathcal L_{\text{velocity}}
+
0.25\,\mathcal L_{\text{commit}}
\]

其中：

\[
\mathcal L_{\text{rec}}
=
\frac{1}{N}
\sum(\hat{x}-x)^2
\]

\[
\mathcal L_{\text{velocity}}
=
\frac{1}{N'}
\sum(\Delta\hat{x}-\Delta x)^2
\]

\[
\mathcal L_{\text{commit}}
=
\frac{1}{ND}
\sum
\left\|
\ell_2(z)-sg(e_k)
\right\|_2^2
\]

实现位置见 [vq.py](/D:/python_file/Knee_Joint_Disorders_work/SSL/knee_kinematics/gait_language/vq.py:264)。

## 2. 主要差异

| 对比项 | HeartLang 论文 | HeartLang 官方代码 | 当前项目 |
|---|---|---|---|
| 波形重建 | MSE | MSE | MSE |
| Velocity loss | 无 | 无 | 有，权重 0.2 |
| 显式 codebook loss | 有 | 无，由 EMA 替代 | 无，由 EMA 替代 |
| Commitment 权重 | 公式中为 1 | `beta=1.0` | `0.25` |
| Codebook 更新 | 论文同时描述 EMA | EMA | EMA |
| 重建上下文 | 完整 ECG sentence | Transformer 编解码 | 每个 DOF 周期独立重建 |
| Padding | 论文使用零填充 ECG words | loss 处没有显式 mask | 无效周期明确通过 mask 排除 |
| 词表 | 单个 8192-word ECG 词表 | 单个词表 | 六个独立的 128-word DOF 词表 |
| Decoder | Transformer decoder | 2 层 Transformer | 共享 MLP + DOF embedding |

## 3. 重建对象不同

### HeartLang

HeartLang 输入为：

```text
[B, 256 ECG words, 96 samples]
```

它先把多导联、多个心搏组成完整 ECG sentence，然后通过 ST-ECGFormer 编码。量化后的所有 ECG words 再共同输入 Transformer decoder。

因此 HeartLang 的重建不仅依赖当前心搏 prototype，还能利用：

- 其他心搏上下文；
- 导联信息；
- 时间位置；
- 空间位置；
- Transformer token interaction。

论文强调通过完整 ECG sentence 学习跨受试者的心搏形态语义。[HeartLang 方法说明](https://arxiv.org/html/2502.10707#S3.SS3)

### 当前项目

当前 VQ 输入虽然组织为：

```text
[B, 2, W, 6, 100]
```

但 `DOFWordEncoder` 实际上将每个周期、每个 DOF 独立编码：

```text
单条 DOF 周期 [100]
→ Conv1d encoder
→ 一个 128 维 word
→ 对应 DOF codebook
→ MLP decoder
→ 重建 [100]
```

VQ 阶段不会利用：

- 同一受试者的其他周期；
- 其他 DOF；
- 对侧腿；
- 周期顺序；
- timing 信息。

这些关系是在后续 sentence-level SSL 中学习的。

所以即使都叫 reconstruction loss，两者优化对象不同：

- HeartLang VQ 同时包含一定的句子上下文；
- 当前 VQ 更纯粹地学习“单个 DOF 周期形状词”。

## 4. 当前 Velocity loss 是针对步态任务的额外改动

HeartLang 只使用逐点 MSE，没有显式约束相邻点变化。

当前项目额外使用：

\[
\mathcal L_{\text{velocity}}
=
MSE(\Delta\hat{x},\Delta x)
\]

它的目的在于保留：

- 波形上升和下降速度；
- 局部斜率；
- 峰值附近变化；
- 周期内部动态形态。

这是针对膝关节运动学的合理适配，因为两条波形可能逐点误差不大，但峰值位置、变化方向或局部斜率不同。

不过当前最佳 checkpoint 中：

```text
reconstruction loss = 0.110049
velocity loss       = 0.001404
commitment loss     = 0.001308
```

代入权重后：

```text
reconstruction contribution = 0.110049  ≈ 99.45%
velocity contribution       = 0.000281  ≈ 0.25%
commitment contribution     = 0.000327  ≈ 0.30%
```

所以尽管当前项目增加了 velocity loss，它实际上几乎没有改变总优化方向。

当前 VQ 本质上仍然非常接近：

```text
纯波形 MSE reconstruction
```

这与 HeartLang 官方实现的主要方向相似。

## 5. Commitment 强度比 HeartLang 更弱

HeartLang 官方实现：

```text
beta = 1.0
```

当前项目：

```text
commitment_weight = 0.25
```

这意味着当前 encoder embedding 被拉向选中 prototype 的约束更弱。

可能的影响包括：

- encoder embedding 可以保留更多连续形态信息；
- 量化边界附近的 assignment 可能更不稳定；
- 相似周期可能被分到不同 code；
- code ID 一致性可能低于连续 embedding 相似性。

这与当前离线诊断具有一定一致性。例如 FE 中波形高度相似的同侧周期，相同 code 比例只有约 39.4%。

但不能简单把 commitment 从 0.25 改成 1.0，因为更强 commitment 也可能：

- 限制 encoder 表达能力；
- 让复杂 IE/ML 波形过早贴近 prototype；
- 降低重建精度；
- 加重错误 assignment 的锁定效应。

更合理的是做 `0.25 / 0.5 / 1.0` 消融，同时观察 reconstruction、assignment margin 和周期一致性。

## 6. Codebook 更新方式实际上相近

HeartLang 和当前项目都：

- 对 encoder embedding 做 L2 normalization；
- 使用 cosine/归一化欧氏距离选择最近 prototype；
- 使用 straight-through estimator；
- 使用 EMA 更新 codebook。

因此，当前实现总体上更接近 HeartLang 的官方代码，而不是严格照搬论文三项公式。

区别是：

- HeartLang 使用 K-means 初始化；
- 当前项目从有效 encoder embedding 中抽样初始化；
- 当前项目额外检测并替换低于阈值的 dead code；
- 当前项目为六个 DOF 分别维护 codebook。

## 7. 两者的 loss 数值不能直接比较

即使 HeartLang 报告的 reconstruction loss 与当前约为 `0.11` 接近，也不能说明重建能力相同，因为两者存在以下差别：

- ECG 与膝关节运动学的信号分布不同；
- HeartLang 每个 word 长度为 96，当前为 100；
- HeartLang 包含零填充 heartbeat patches；
- 当前明确排除了无效 padding cycles；
- 两者标准化方式不同；
- HeartLang 使用完整句子 Transformer decoder；
- 当前使用独立周期 MLP decoder；
- HeartLang 是一个 8192 code 词表，当前是六个 128 code 词表；
- 论文公式写成求和，而官方 PyTorch 实际实现使用 mean reduction。

应该比较的是归一化后的重建质量，例如：

- 每个 DOF 的 RMSE；
- 单周期 correlation；
- velocity RMSE；
- 峰值位置误差；
- code 内方差；
- assignment 稳定性；
- downstream 表征效果。

## 8. 对当前项目的建议

当前 loss 不需要直接改成 HeartLang 的形式。更合适的方向是保留步态任务特有的 velocity 约束，但重新调整尺度：

\[
\mathcal L
=
\mathcal L_{\text{rec}}
+
\lambda_v
\frac{\mathcal L_{\text{velocity}}}
{\operatorname{Var}(\Delta x_d)}
+
\lambda_c\mathcal L_{\text{commit}}
\]

建议优先比较：

| 实验 | Velocity | Commitment |
|---|---:|---:|
| A 当前基线 | 原始 × 0.2 | 0.25 |
| B | 按 DOF 方差归一化 | 0.25 |
| C | 按 DOF 方差归一化 | 0.5 |
| D | 按 DOF 方差归一化 | 1.0 |

目标不是让 velocity loss 数值变大，而是让它实际贡献约 5%～15% 的总损失或相近比例的有效梯度。

最终来说：

> 当前 reconstruction term 与 HeartLang 一样，核心都是逐点 MSE；当前项目额外加入了步态速度约束，但权重过小，几乎没有发挥作用。相比 HeartLang，当前更大的差别其实不是 MSE 公式，而是独立 DOF 周期重建、六套 codebook、较弱 commitment 以及简单 MLP decoder。
