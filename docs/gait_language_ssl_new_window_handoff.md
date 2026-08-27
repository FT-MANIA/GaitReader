# Gait-Language SSL 新窗口接续文档

更新时间：2026-08-25

工作区：

```text
D:\python_file\Knee_Joint_Disorders_work\SSL
```

本文档用于在新的 Codex 窗口中恢复项目上下文。它整合了研究目标、数据语义、当前代码状态、VQ Decoder 消融结果、已确定的工程约束和下一步任务。更完整的理论方案见 `docs/gait_language_ssl_design.md`，Decoder 细节与全部实验分析见 `docs/vq_decoder_architectures.md`。

## 1. 项目的核心目标

项目不是以“将完整原始序列直接送入黑盒分类器”为最终目标，而是先建立健康人的膝关节 6-DOF gait language，再根据患者相对健康语言模型的结构化偏离进行疾病判断与医学解释。

核心语言定义：

```text
一个 gait word
= 单侧 + 单个 DOF + 一个完整运动学周期的标准化波形

一个 bilateral gait sentence
= 同一受试者左右腿全部有效 gait words
+ 周期顺序、持续时间、间隔、质量和绝对时间信息
```

最终目标链路：

```text
健康 SSL 数据
→ 自适应周期划分、时间归一化和质量控制
→ 六个 DOF 独立、左右共享的健康 codebook
→ 健康 gait-language sentence encoder
→ 健康 normative reference
→ KGKD Healthy / ACLD / KOA 结构化偏离
→ 可解释三分类器
→ ACLD / KOA 异常侧别、DOF、周期和波形方向可视化
```

最终系统应回答：

1. 受试者相对健康参考偏离多大；
2. 偏离发生在哪一侧、哪个 DOF、哪些周期和哪些相位；
3. 偏离属于形态、节律、跨 DOF 关系还是双侧协调异常；
4. 偏离模式更接近 Healthy、ACLD 还是 KOA。

## 2. 不可改变的数据语义

### 2.1 原始输入

单个受试者的双侧记录：

```text
X ∈ R[2, 6, T_raw]
T_raw 通常为 600
sampling_rate = 60 Hz
```

当前 CSV 不是预先拼接好的 100/200 点周期，而是完整原始记录。代码必须先自适应划分周期，再进行时间归一化。

### 2.2 DOF 顺序

原始 CSV 每侧顺序：

```text
[VV, IE, FE, AP, SI, ML]
```

模型内部 PMD 顺序：

```text
[FE, VV, IE, AP, ML, SI]
```

含义：

```text
0 flexion_extension
1 adduction_abduction / varus_valgus
2 internal_external_rotation
3 anterior_posterior_translation
4 medial_lateral_translation
5 superior_inferior_translation
```

`reference_dof_index=2` 指原始顺序中的 FE，而不是重排后的索引 2。分周期发生在原始顺序阶段。

### 2.3 周期边界

左右腿分别基于 FE 代理轨迹检测周期边界；同一侧六个 DOF 必须共享完全相同的边界：

```text
FE 检测边界
→ 同一组边界切分该侧全部 6 DOF
→ 每个周期插值为 word_length=100 点
```

该边界是运动学代理事件，不应在未与足底压力或力台验证前称为严格 heel-strike 周期。

### 2.4 受试者与 `source_file`

数据划分必须按真实受试者 ID 隔离。同一受试者不得跨 train/validation/test。

`source_file` 只表示样本来自哪个社区、医院或采集来源：

- 不代表受试者；
- 不作为分组 ID；
- 不进入模型；
- 仅作为审计元数据。

不要再根据 `source_file` 数量推断 downstream 只有几个受试者。

### 2.5 数据用途

```text
ssl_data
  健康 VQ 和健康 gait-language SSL 训练

ssl_validation_data
  VQ/SSL early stopping 和超参数选择

dev_data
  downstream 训练

dev_validation_data
  downstream checkpoint 选择

dev_test_data
  internal test

ext_test_data
  external holdout，只用于最终评估
```

### 2.6 统一标准化坐标系

标准化参数只能由经过分周期和质量控制的 `SSL healthy train` 拟合。VQ、SSL、KGKD Healthy、ACLD、KOA、internal test 和 external test 必须复用同一套参数。

禁止：

- downstream 重新拟合均值/标准差；
- validation/test 参与标准化；
- 先用极端 AP/ML 轨迹拟合标准化，再做质量控制。

当前流程已按“分周期与质量控制后拟合健康标准化参数”组织，并参考了 PACENet `advanced_quality_control` 的鲁棒离群处理思路。

## 3. 当前数据处理 pipeline

```text
CSV 读取与 feature 解析
→ 原始 DOF 顺序确认
→ 左右 FE 自适应周期检测
→ 周期长度和自相关约束
→ 周期间相似性过滤
→ 六个 DOF 共享边界切分
→ 每周期插值为 100 点
→ 受试者级/轨迹级质量控制
→ 只用 SSL healthy train 拟合标准化参数
→ 构建 bilateral language batch
```

语言 batch：

```text
words     [B, 2, W, 6, T_word]
word_mask [B, 2, W]
timing    [B, 2, W, 4]
```

`timing` 四个通道：

```text
0 duration
1 normalized center position
2 interval to previous cycle
3 quality score
```

左右有效周期数可以不同，padding 必须由 `word_mask=False` 标记，不能复制另一侧周期进行伪对齐。

## 4. 当前代码结构

```text
run.py
  唯一一键实验入口；get_args() 集中管理参数；main() 串联全部阶段

analyze_vq.py
  VQ 离线诊断；扫描所有实验；已有 summary.json 的实验自动跳过

knee_kinematics/data/loading.py
  CSV 与 features 解析

knee_kinematics/data/builders.py
  数据仓库、split、标准化与 DataLoader 构建

knee_kinematics/data/collate.py
  变长周期 batch 与 mask

knee_kinematics/data/transforms/gait_cycle.py
  自适应周期划分和时间归一化

knee_kinematics/data/transforms/quality_control.py
  鲁棒质量控制与异常轨迹过滤

knee_kinematics/gait_language/data.py
  words、word_mask、timing 整理

knee_kinematics/gait_language/vq.py
  Word encoder、六个 EMA codebook、四套 Decoder、VQ loss

knee_kinematics/gait_language/sentence.py
  temporal、DOF 和 bilateral sentence modeling

knee_kinematics/gait_language/models.py
  GaitLanguageSSLModel 与基础 downstream model

knee_kinematics/gait_language/trainer.py
  VQ、SSL、downstream、evaluation 循环和 checkpoint
```

当前仓库已清理旧入口，根目录只保留 `run.py` 和 `analyze_vq.py` 两个 Python 入口。

## 5. VQ gait-word tokenizer

### 5.1 Word encoder

```text
word waveform [...,6,T]
→ shared temporal Conv1d stem
→ DOF identity embedding
→ 六个 DOF-specific residual adapters
→ continuous word embedding [...,6,D]
```

默认：

```text
T = 100
D = 128
hidden_dim = 64
```

Word encoder 只编码单周期形态，不直接负责跨周期节律或双腿关系。

### 5.2 DOF-specific codebook

```text
V ∈ R[6,K,D]
K = 128（当前默认）
```

六个 DOF 使用独立 EMA vocabulary；左右腿同一 DOF 共享词表。禁止改为左右独立 `[2,6,K,D]`，否则左右 code distance 失去共同语义。

量化采用归一化 embedding 的 cosine nearest prototype。Code ID 是无序类别编号，不能用 code ID 相减表示偏离。

### 5.3 当前四套 Decoder

```text
mlp
  单个 z_q 通过共享 MLP 重建完整周期

temporal_transformer
  单个 z_q 展开为 phase tokens，通过 Transformer 输出多个 waveform patches

sentence_transformer
  左右腿、全部周期和全部 DOF tokens 联合重建

local_context_sentence
  x_local = MLPDecoder(z_q)
  r_context = SentenceResidual(z_q, word_mask, timing)
  x_hat = x_local + alpha × r_context
```

当前 `run.py` 默认：

```text
vq_decoder = local_context_sentence
vq_context_residual_scale = 0.5
vq_local_reconstruction_weight = 2.0
```

Local + Context 总损失：

```text
L_total
= L_final_reconstruction
+ local_weight × L_local_reconstruction
+ velocity_weight × L_velocity(final)
+ commitment_weight × L_commitment
```

其 total loss 因包含额外 local loss，不能与 MLP/Sentence/Temporal 的 total loss直接横向比较。应比较 final reconstruction、local reconstruction、velocity、code variance 和 assignment consistency。

## 6. Gait-language sentence SSL

### 6.1 Sentence encoder

输入：

```text
[B,2,W,6,D]
```

主要建模轴：

1. Temporal axis：固定 side 与 DOF，沿 W 建模周期序列和节律；
2. DOF axis：固定 side 与周期，沿 6 DOF 建模运动学耦合；
3. Bilateral axis：左右腿通过 relative-time cross-attention 建模。

左右周期不能按同一数组下标强制配对。双侧 attention 使用周期中心相对时间：

```text
delta_t = center_left[i] - center_right[j]
attention_logit += relative_time_bias(delta_t)
```

### 6.2 当前 SSL tasks

```text
Within-DOF masked word prediction
  mask 随机 word/span，根据同 DOF 时序上下文预测 code

Whole-DOF / Cross-DOF prediction
  mask 某侧一个完整 DOF，利用其他 DOF 预测目标 code

Bilateral contextual prediction
  mask部分目标侧 word，利用同侧剩余信息和对侧信息预测

Contralateral-only prediction
  屏蔽目标侧上下文，只根据对侧 sentence 与 timing 预测

Left-right swap consistency
  约束 shared/absolute/directional bilateral representation
```

总损失：

```text
L_SSL
= within_weight × L_within
+ cross_dof_weight × L_cross_dof
+ bilateral_weight × L_bilateral
+ contralateral_weight × L_contralateral
+ swap_weight × L_swap
```

SSL validation mask 使用固定 seed，避免每个 epoch 的随机 mask 扰动 early stopping。默认 `validation_mask_seed = seed + 10000`。

注意：不同 VQ 实验产生不同 codebook，因此 SSL 的 class ID、类别分布和难度不同。不同 codebook 之间的 raw CE/accuracy 不是完全相同 label space 上的严格比较。

## 7. 当前 downstream 的定位

当前实现是基础 supervised downstream：

```text
健康预训练 sentence encoder
→ subject representation
→ Healthy / ACLD / KOA classifier
+ affected-side head
```

`freeze_sentence_encoder` 默认是 `False`，因此当前完整实验会微调 sentence encoder。

这只是表示迁移能力基线，不是最终研究目标中的“基于健康偏离的可解释分类器”。最终 deviation pipeline 建立时，应冻结健康 standardizer、word encoder、codebook 和 sentence encoder，只允许 KGKD label 训练偏离分类器。

## 8. Decoder 实验记录与当前结论

实验顺序：

```text
dev_exp_0823_1849  MLP
dev_exp_0824_1718  Sentence Transformer
dev_exp_0824_1802  Temporal Transformer
dev_exp_0824_2014  Local + Context, local_weight=1.0
dev_exp_0824_2100  Local + Context, local_weight=2.0
```

所有实验使用相同数据 split、`seed=42` 和主要训练超参数；差异集中在 Decoder 及 local weight。

### 8.1 VQ 离线诊断汇总

均在 `ssl_validation_data` 上统计：

| 实验 | Mean RMSE ↓ | Mean cycle corr ↑ | Velocity RMSE ↓ | Code 内方差 ↓ | 同侧 same-code ↑ | 双侧 same-code ↑ | Decoded-near ratio ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| MLP | 0.325781 | 0.799534 | 0.036142 | 0.098073 | 0.537872 | **0.102881** | 2.079% |
| Sentence | 0.253133 | 0.858147 | 0.038450 | 0.107435 | 0.544935 | 0.080507 | 2.028% |
| Temporal | 0.323233 | 0.787181 | 0.039816 | **0.097018** | **0.561587** | 0.094679 | 3.066% |
| Local + Context, w=1 | 0.252345 | 0.873414 | 0.030592 | 0.103094 | 0.522482 | 0.093026 | **1.411%** |
| Local + Context, w=2 | **0.250651** | **0.879339** | **0.030216** | 0.100847 | 0.538337 | 0.093806 | 1.509% |

### 8.2 SSL 与 external downstream

| 实验 | SSL val loss ↓ | Cross-DOF acc ↑ | Bilateral acc ↑ | Contralateral acc ↑ | External acc ↑ | External macro-F1 ↑ | External AUROC ↑ | External side acc ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MLP | 9.0008 | 0.1219 | 0.4630 | 0.0574 | 0.8203 | 0.8175 | 0.9384 | 0.8119 |
| Sentence | 9.1793 | 0.1170 | 0.4666 | 0.0520 | 0.8627 | 0.8566 | 0.9612 | 0.8218 |
| Temporal | 9.0511 | 0.1074 | **0.4801** | 0.0477 | 0.8660 | 0.8536 | 0.9565 | 0.8119 |
| Local + Context, w=1 | 9.0714 | **0.1404** | 0.4547 | 0.0555 | 0.8562 | 0.8540 | 0.9629 | 0.8317 |
| Local + Context, w=2 | **8.9986** | 0.1385 | 0.4725 | **0.0610** | **0.8758** | **0.8654** | **0.9686** | **0.8713** |

### 8.3 `local_weight=2.0` 的关键状态

```text
best VQ epoch                  = 95 / 100
final validation MSE          = 0.065596
local-only validation MSE     = 0.113775
raw context residual RMS      = 0.461701
scaled residual RMS           = 0.230851  (alpha=0.5)
validation velocity loss      = 0.001004
```

提高 local weight 从 1 到 2 后：

- local-only MSE 下降约 1.9%；
- code 内方差下降约 2.2%；
- 同侧 same-code 从 0.5225 提升到 0.5383；
- final reconstruction、correlation 和 velocity 继续小幅改善；
- external 四项指标全部改善并成为当前最佳；
- residual RMS 没有下降，说明 context branch 仍承担较大责任；
- decoded-near ratio略升，但仍显著优于原始三个 Decoder；
- 训练跑满 100 epoch，最佳在 epoch 95，尚未由 patience 提前停止。

### 8.4 当前模型选择判断

当前主候选：

```text
LocalContextResidualSentenceDecoder
local_reconstruction_weight = 2.0
residual_scale = 0.5
```

理由：

- 保留 Sentence 的整体重建优势；
- 修复 Sentence 的局部 velocity error；
- code variance 比 Sentence 更低；
- prototype alias 很低；
- SSL 跨 DOF/双侧结果更加均衡；
- external 泛化最佳。

仍未解决：

- local-only MSE 仍略差于原始 MLP；
- code variance 仍未达到 MLP/Temporal；
- residual RMS 并未随 local weight 增大而降低；
- 同侧 assignment consistency 尚未达到 Temporal；
- 只有单随机种子，尚无统计稳定性结论。

## 9. VQ 离线分析

直接运行：

```powershell
D:\anaconda\python.exe analyze_vq.py
```

默认行为：

- 递归扫描 `Results/gait_language` 中所有包含 `best_vq.pt` 的实验；
- 使用各实验自己的 `args.json` 和 checkpoint；
- 输出到 `<run>/vq_analysis_ssl_validation_data/`；
- 如果目标目录已有 `summary.json`，自动跳过；
- 最终打印 discovered、analyzed 和 skipped 数量。

单独分析指定实验：

```powershell
D:\anaconda\python.exe analyze_vq.py `
  --run-dir Results/gait_language/dev_exp/dev_exp_0824_2100
```

离线输出包括：

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

## 10. 如何运行实验

当前唯一主入口：

```powershell
D:\anaconda\python.exe run.py
```

默认 `stage=all`：

```text
加载/分周期/质控/标准化
→ VQ
→ gait-language SSL
→ downstream
→ internal test
→ external test
```

当前默认 VQ 配置已是：

```text
--vq-decoder local_context_sentence
--vq-context-residual-scale 0.5
--vq-local-reconstruction-weight 2.0
--vq-residual-energy-weight 0.0
```

显式运行：

```powershell
D:\anaconda\python.exe run.py `
  --stage all `
  --vq-decoder local_context_sentence `
  --vq-context-residual-scale 0.5 `
  --vq-local-reconstruction-weight 2.0 `
  --vq-residual-energy-weight 0.0 `
  --device cuda
```

只运行 VQ：

```powershell
D:\anaconda\python.exe run.py --stage vq --device cuda
```

分阶段加载 checkpoint 时，Decoder 类型和结构参数必须与 VQ checkpoint 完全一致：

```powershell
D:\anaconda\python.exe run.py `
  --stage ssl `
  --vq-checkpoint Results/gait_language/<run>/best_vq.pt `
  --vq-decoder local_context_sentence `
  --vq-context-residual-scale 0.5 `
  --vq-local-reconstruction-weight 2.0 `
  --vq-residual-energy-weight 0.0 `
  --device cuda
```

每次运行创建独立目录：

```text
Results/gait_language/<stage>_MMDD_HHMM/
```

目录时间精确到分钟，不包含年份和末尾序号。已有目录不会被复用，因此 `metrics.jsonl` 不会混合多次实验。

## 11. 已实现与未实现

### 11.1 已实现

- 原始 CSV 读取和 feature 解析；
- 原始 600 点记录的自适应周期划分；
- 每周期 100 点时间归一化；
- 鲁棒质量控制和异常 AP/ML 等轨迹过滤；
- 训练集健康标准化及 downstream 复用；
- DOF-specific、左右共享的 EMA VQ codebook；
- 四套 VQ Decoder；
- fixed validation mask；
- temporal、cross-DOF、bilateral sentence encoder；
- within、whole-DOF、bilateral、contralateral 和 swap SSL；
- VQ/SSL early stopping；
- 基础 supervised Healthy/ACLD/KOA 和 affected-side downstream；
- internal/external evaluation；
- 所有 VQ 实验批量离线诊断；
- 每次实验独立输出目录。

### 11.2 尚未实现或尚未完成

设计文档 Stage 4 以后的最终偏离建模仍未落地：

1. `HealthyNormativeReference`；
2. per-code/per-DOF 健康距离分布；
3. deterministic within/cross-DOF/contralateral surprise 推理；
4. rhythm deviation；
5. bilateral absolute/directional deviation；
6. word/DOF/side/subject 四级 deviation extractor；
7. patient waveform 与健康 prototype 的 signed residual；
8. 只使用偏离特征的多项逻辑回归基线；
9. deviation-aware classifier；
10. ACLD/KOA 群体偏离谱和可解释可视化；
11. probability calibration；
12. 多随机种子稳定性验证。

当前 downstream 结果不能代替上述 normative deviation pipeline。

## 12. 下一步建议顺序

### Priority 1：Residual-energy regularization（代码已实现，实验待运行）

当前最明确的问题是 `local_weight=2.0` 改善了 local branch，却没有降低 residual RMS。当前代码已在默认结构中加入可配置项：

```text
--vq-residual-energy-weight
```

损失建议：

```text
scaled_residual = alpha × r_context
L_residual = lambda_residual × mean(scaled_residual² on valid words)

L_total
= L_final
+ 2.0 × L_local
+ L_velocity
+ L_commitment
+ L_residual
```

必须新增日志：

```text
residual_energy_loss
raw_context_residual_rms
scaled_context_residual_rms
local_to_final_improvement
```

不要只继续降低 `alpha`：网络可以通过放大 residual branch 权重抵消固定缩放。

实现约定：

- `--vq-residual-energy-weight` 默认是 `0.0`，因此既有实验和旧 checkpoint 保持可复现；
- `residual_energy_loss` 记录未乘 `lambda_residual` 的 scaled-residual 均方，总损失再显式乘权重；
- `context_residual_rms` 作为旧日志兼容别名保留，等于 `raw_context_residual_rms`；
- `local_to_final_improvement = local_reconstruction_loss - reconstruction_loss`，正值表示 context 改善最终重建；
- 下一步应以 `0.0` 为基线，对轻量正权重（例如 `0.01/0.05/0.1`）做相同协议的消融，尚不能声称正则已改善真实数据指标。

### Priority 2：公平消融与多随机种子

- 保持相同 subject split；
- 保持相同最大 epoch 和 early-stopping 规则；
- 至少运行 3～5 个 seed；
- 报告均值和标准差；
- 同时比较 VQ、codebook、SSL、internal 和 external；
- 注意 `w=2` 当前跑满 100 epoch，而 `w=1` 在 77 epoch 停止。

### Priority 3：进入健康偏离建模

在 VQ/SSL 主结构稳定后，按以下顺序实现：

```text
HealthyNormativeReference
→ deviation feature extractor
→ logistic-regression deviation baseline
→ deviation-aware classifier
→ ACLD/KOA visualization
```

健康模型冻结规则：

```text
freeze standardizer
freeze word encoder
freeze codebook
freeze decoder
freeze sentence encoder
```

KGKD label 只训练偏离分类器，不能反向污染健康 vocabulary。

## 13. 最终偏离表征应包含什么

对每个患者 word、DOF、侧别和受试者保留：

```text
nearest healthy code distance
assigned-code reconstruction residual
signed waveform residual
within-DOF prediction surprise
cross-DOF prediction surprise
contralateral prediction surprise
duration deviation
interval/rhythm deviation
bilateral absolute deviation
bilateral directional deviation
```

不要只输出一个 global deviation scalar。需要保留：

```text
word-level map
side × DOF aggregation
subject-level robust summary
p50 / p90 / max / abnormal ratio
```

Code ID 不能相减。偏离应使用 embedding distance、reconstruction residual、概率 surprise 和波形 signed difference。

## 14. 结果选择原则

不要只按 VQ total loss 或 reconstruction loss 选模型。至少联合判断：

```text
final RMSE
local-only RMSE
velocity RMSE
per-DOF code usage/perplexity
code 内波形方差
同一受试者相似周期 code consistency
prototype embedding duplicate
decoded prototype alias ratio
within / cross-DOF / bilateral / contralateral SSL
external generalization
```

特别注意：

- Local + Context 的 total loss 包含额外 local loss，不可与其他 Decoder 直接比较；
- Sentence 类 Decoder 的 prototype 是 canonical context 下解码，不能完全代表真实上下文；
- 低 reconstruction loss 可能来自 contextual shortcut；
- 高 code usage 不等于 codebook 语义良好；
- 单 seed 的 downstream 排序不能证明稳定优越性。

## 15. 关键文档

```text
docs/gait_language_ssl_design.md
  完整 Stage 0～10 研究方案和 normative deviation 设计

docs/vq_decoder_architectures.md
  四套 Decoder 结构、五次实验结果和详细分析

docs/vq_offline_diagnostics_analysis.md
  VQ 离线指标定义与早期诊断解释

docs/vq_loss_comparison_heartlang.md
  当前 VQ reconstruction 与 HeartLang 的差异

docs/gait_language_results_analysis.md
  早期 VQ/SSL/downstream 结果分析

README.md
  安装、运行和参数说明
```

## 16. 新窗口建议开场提示

可以在新窗口中附上本文件并使用：

```text
请先阅读 docs/gait_language_ssl_new_window_handoff.md，理解当前 gait-language SSL 项目的数据语义、代码状态、Decoder 消融结果和后续优先级。

当前主候选是 LocalContextResidualSentenceDecoder，配置为 local_reconstruction_weight=2.0、residual_scale=0.5。下一步优先考虑 residual-energy regularization，但在修改前请先检查当前代码与文档是否一致。不要改变 DOF 顺序、数据 split、健康标准化坐标系、codebook 左右共享语义或现有 loss，除非任务明确要求。完成修改后需说明它如何影响 final reconstruction、local reconstruction、residual RMS、code variance 和相似周期 consistency。
```

## 17. 一句话项目状态

当前已经完成从原始双侧 6-DOF 记录到健康 gait words、健康 codebook、sentence SSL 和基础分类的完整可运行 pipeline，并通过五次 Decoder 消融确定 `Local + Context, local_weight=2.0` 为当前主候选；下一阶段应先直接约束 contextual residual，再进入冻结健康模型下的 normative deviation 与可解释疾病分类。
