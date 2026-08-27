# Knee 6-DOF Gait Language

本项目将左右膝 6-DOF 原始运动学序列转换为 gait-language：每个自由度的一个自适应步态周期视为一个 word，单侧完整记录视为 sentence。训练流程依次完成 DOF 专属向量量化词表、掩码自监督预训练、下游分类与内外部测试评估。

当前唯一实验入口是根目录的 `run.py`，所有参数集中在 `get_args()` 中。

## 目录结构

```text
SSL/
├── run.py                         # 一键运行入口与全部命令行参数
├── pyproject.toml                 # 安装信息和运行依赖
├── Dataset/                       # 原始 CSV 数据
├── docs/
│   └── gait_language_ssl_design.md
├── knee_kinematics/
│   ├── data/                      # CSV 读取、分周期、质控、标准化、DataLoader
│   │   └── transforms/
│   │       ├── gait_cycle.py      # 自适应周期定位与时间归一化
│   │       └── quality_control.py # 异常轨迹和低质量受试者过滤
│   └── gait_language/
│       ├── data.py                # 周期 word/sentence 整理与统计
│       ├── vq.py                  # DOF 专属编码器和 6 个 codebook
│       ├── sentence.py            # sentence Transformer
│       ├── models.py              # SSL 与 downstream 模型
│       └── trainer.py             # VQ、SSL、微调和评估循环
└── Results/
    └── gait_language/             # 默认输出根目录
        └── all_MMDD_HHMM/              # 每次运行的独立目录
```

## 环境与数据

在项目根目录执行：

```powershell
D:\anaconda\python.exe -m pip install -e .
```

默认读取：

```text
Dataset/SSL_Healthy/ssl_healthy_dataset.csv
Dataset/KGKD/dev_dataset.csv
Dataset/KGKD/test_dataset.csv
```

数据用途固定为：

- `ssl_data`：VQ 词表学习和 SSL 预训练；
- `dev_data`：下游训练；
- `dev_test_data`：内部测试；
- `ext_test_data`：外部测试。

原始单侧 DOF 顺序应为 `[VV, IE, FE, AP, SI, ML]`。数据适配器会重排为模型内部顺序 `[FE, VV, IE, AP, ML, SI]`。`--reference-dof-index 2` 指的是重排前原始数据中的 FE 索引。

输入可以是约 600 点的完整原始记录。程序先依据 FE 参考轨迹自适应定位周期，再把每个周期插值为 `--word-length` 点；不要求 CSV 已经按周期切分。

## 运行完整实验

最简单的方式是在编辑器中打开 `run.py`，点击 **Run Python File**。也可以运行：

```powershell
D:\anaconda\python.exe run.py
```

每次启动都会创建独立目录，例如：

```text
Results/gait_language/dev_exp/dev_exp_0823_1815/
```

控制台会打印实际的 `run_dir`。已有目录不会被复用，因此不同实验的
`metrics.jsonl` 和 checkpoint 不会混合。

默认 `--stage all`，执行顺序为：

```text
加载与质控数据
→ 自适应周期划分与时间归一化
→ 训练 VQ tokenizer/codebook
→ 掩码 gait-language SSL
→ downstream 微调
→ dev_test 内部评估
→ ext_test 外部评估
```

## 分阶段运行

```powershell
# 只训练 VQ 词表
D:\anaconda\python.exe run.py --stage vq

# 读取已有 run 的 best_vq.pt，训练 SSL
D:\anaconda\python.exe run.py --stage ssl `
  --vq-checkpoint Results/gait_language/<vq_run>/best_vq.pt

# 读取已有 VQ 和 SSL checkpoint，训练 downstream 并评估
D:\anaconda\python.exe run.py --stage downstream `
  --vq-checkpoint Results/gait_language/<vq_run>/best_vq.pt `
  --ssl-checkpoint Results/gait_language/<ssl_run>/best_ssl.pt

# 只加载全部 checkpoint 并评估
D:\anaconda\python.exe run.py --stage evaluate `
  --vq-checkpoint Results/gait_language/<vq_run>/best_vq.pt `
  --ssl-checkpoint Results/gait_language/<ssl_run>/best_ssl.pt `
  --downstream-checkpoint Results/gait_language/<downstream_run>/best_downstream.pt
```

`ssl` 阶段不会继续训练 downstream；`vq` 阶段训练结束后立即退出。`downstream` 和 `evaluate` 最终都会输出内部与外部测试结果。

## 修改参数

有两种方式：

1. 临时实验：在命令行追加参数，最适合对比实验；
2. 固定默认值：直接修改 `run.py` 中 `get_args()` 对应参数的 `default`。

查看完整参数：

```powershell
D:\anaconda\python.exe run.py --help
```

常用示例：

```powershell
# 减少显存和 DataLoader 开销
D:\anaconda\python.exe run.py --batch-size 8 --num-workers 0

# 禁用混合精度，用于排查 CUDA 数值问题
D:\anaconda\python.exe run.py --no-mixed-precision

# 缩短三阶段训练并分别设置 early stopping
D:\anaconda\python.exe run.py `
  --vq-epochs 20 --vq-patience 5 `
  --ssl-epochs 40 --ssl-patience 10 `
  --downstream-epochs 30 --downstream-patience 8

# 修改词向量、每个 DOF 的词表大小和 Transformer
D:\anaconda\python.exe run.py `
  --word-dim 192 --codebook-size 256 `
  --sentence-depth 4 --num-heads 6

# 指定输出根目录和可读的唯一实验名
D:\anaconda\python.exe run.py `
  --output-dir Results/gait_language/dev_exp `
  --run-name k128_d128_seed42
```

布尔参数采用成对形式，例如：

```text
--mixed-precision / --no-mixed-precision
--quality-control / --no-quality-control
--similarity-filter / --no-similarity-filter
--freeze-sentence-encoder / --no-freeze-sentence-encoder
```

### 实验与运行参数

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `--stage` | `all` | 运行 `all/vq/ssl/ssl_downstream/downstream/evaluate` 中的一个阶段；`ssl_downstream` 固定读取 VQ 后训练 SSL、下游模型并评估 |
| `--output-dir` | `Results/gait_language/dev_exp` | `run.py` 开发实验的输出根目录 |
| `--run-name` | 自动生成 | run 子目录名；默认格式为 `dev_exp_MMDD_HHMM`，指定名称必须尚不存在 |
| `--device` | `auto` | 自动选 CUDA/CPU，也可显式设为 `cuda` 或 `cpu` |
| `--seed` | `42` | Python、NumPy、PyTorch 随机种子 |
| `--batch-size` | `32` | 受试者级 batch size |
| `--num-workers` | `0` | DataLoader 子进程数；Windows 建议先使用 0 |
| `--mixed-precision` | 开启 | CUDA AMP；CPU 下自动关闭 |

### 数据与周期参数

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `--ssl-csv` | SSL Healthy CSV | 自监督健康数据路径 |
| `--dev-csv` | dev CSV | downstream 开发数据路径 |
| `--ext-test-csv` | test CSV | 外部测试数据路径 |
| `--recording-length` | `600` | 原始记录预期长度，用于训练批处理整理 |
| `--sampling-rate-hz` | `60` | 原始采样率 |
| `--word-length` | `100` | 每个周期时间归一化后的采样点数，也是一个 word 的长度 |
| `--reference-dof-index` | `2` | 原始 `[VV, IE, FE, AP, SI, ML]` 中用于分周期的 FE |
| `--min-cycle-seconds` | `0.4` | 合法周期的最短时长 |
| `--max-cycle-seconds` | `4.0` | 合法周期的最长时长 |
| `--smoothing-window-seconds` | `0.15` | 峰值检测前的平滑窗口 |
| `--peak-prominence-fraction` | `0.15` | 峰显著性相对幅值阈值 |
| `--peak-distance-fraction` | `0.55` | 相邻峰最小距离相对估计周期的比例 |
| `--period-min-correlation` | `0.05` | 周期估计的最低自相关阈值 |
| `--period-relative-min/max` | `0.55/1.60` | 候选周期相对估计周期的合法范围 |
| `--min-cycles` | `1` | 分割结果最低周期数 |
| `--similarity-filter` | 开启 | 使用周期间相似性过滤异常周期 |
| `--min-cycle-similarity` | `-1.0` | 固定最低相似性；-1 表示主要由 MAD 自适应阈值决定 |
| `--similarity-mad-scale` | `3.0` | 相似性鲁棒阈值的 MAD 倍数 |

周期过少时，先核对采样率、原始 FE 索引和周期时长范围，不建议直接关闭全部过滤。

### 质量控制参数

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `--quality-control` | 开启 | 在拟合标准化统计量前剔除异常受试者/轨迹 |
| `--min-cycles-per-side` | `2` | 每侧至少保留的有效周期数 |
| `--robust-z-threshold` | `6.0` | 基于中位数/MAD 的异常程度阈值 |
| `--min-upper-scale-factor` | `3.0` | 防止参考尺度过小的下界系数 |
| `--min-reference-subjects` | `20` | 拟合鲁棒质控参考所需最少受试者数 |

标准化参数只使用 `SSL healthy train` 中完成分周期且通过质量控制的周期拟合。该参数随后冻结并统一应用于 SSL validation、downstream train/validation、internal test 和 external test；KGKD 与测试数据都不会重新拟合标准化坐标系。

### VQ 与 codebook 参数

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `--word-dim` | `128` | word embedding 和 code vector 维度 D |
| `--word-hidden-dim` | `64` | 单 DOF 周期编码器隐藏维度 |
| `--codebook-size` | `128` | 每个 DOF 的 vocabulary 大小 K |
| `--codebook-decay` | `0.99` | EMA codebook 更新衰减率 |
| `--dead-code-threshold` | `1.0` | 低使用率 code 的重置阈值 |
| `--commitment-weight` | `0.25` | VQ commitment loss 权重 |
| `--velocity-weight` | `0.20` | 重建速度/差分损失权重 |
| `--vq-decoder` | `local_context_sentence` | VQ 波形 Decoder；当前主候选为 local morphology + contextual residual |
| `--vq-context-residual-scale` | `0.5` | contextual residual 加到 local 波形前的固定缩放系数 |
| `--vq-local-reconstruction-weight` | `2.0` | local-only reconstruction loss 权重 |
| `--vq-residual-energy-weight` | `0.0` | 有效 word 上 scaled contextual residual 均方能量的权重；`0` 保持既有基线 |
| `--vq-epochs/lr/weight-decay` | `100/3e-4/1e-4` | VQ 优化参数 |
| `--vq-patience` | `10` | VQ 验证损失 early stopping patience |

最终 codebook 逻辑形状为 `[6, K, D]`；六个 DOF 使用独立 vocabulary。

Residual-energy 消融应显式开启正则，例如：

```powershell
D:\anaconda\python.exe run.py --stage vq `
  --vq-decoder local_context_sentence `
  --vq-local-reconstruction-weight 2.0 `
  --vq-context-residual-scale 0.5 `
  --vq-residual-energy-weight 0.05
```

`residual_energy_loss` 记录未乘权重的 scaled-residual 均方，实际加入总损失的项为 `vq_residual_energy_weight × residual_energy_loss`。

### Sentence 与 SSL 参数

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `--max-words` | `32` | 单侧、单 DOF sentence 支持的最大 word 数 |
| `--sentence-depth` | `2` | Transformer encoder 层数 |
| `--num-heads` | `4` | 多头注意力头数；必须整除 `word-dim` |
| `--dropout` | `0.10` | sentence encoder dropout |
| `--word-mask-ratio` | `0.30` | 同一 DOF 内随机/span mask 比例 |
| `--bilateral-mask-ratio` | `0.30` | 双腿预测任务的 mask 比例 |
| `--span-length` | `2` | 连续 mask 的 word 数 |
| `--within-weight` | `1.0` | 同 DOF masked-word 任务权重 |
| `--cross-dof-weight` | `1.0` | mask 整个 DOF、由其他 DOF 预测的权重 |
| `--bilateral-weight` | `1.0` | 双腿关系总权重 |
| `--contralateral-weight` | `0.50` | 对侧 word 预测子任务权重 |
| `--swap-weight` | `0.10` | 左右交换一致性子任务权重 |
| `--ssl-epochs/lr/weight-decay` | `100/3e-4/1e-2` | SSL 优化参数 |
| `--ssl-patience` | `10` | SSL 验证损失 early stopping patience |
| `--validation-mask-seed` | `seed + 10000` | 每个验证 epoch 重新生成同一组 SSL mask，使 early stopping 可比较 |

### Downstream 参数

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `--downstream-epochs` | `50` | 最大微调 epoch |
| `--downstream-learning-rate` | `3e-4` | 微调学习率 |
| `--downstream-weight-decay` | `1e-2` | 微调权重衰减 |
| `--downstream-patience` | `10` | downstream early stopping patience |
| `--affected-side-weight` | `0.20` | 患侧辅助分类 loss 权重 |
| `--classifier-dropout` | `0.20` | 分类头 dropout |
| `--deviation-dof-dim` | `64` | 分层健康偏移编码器的 per-DOF embedding 维度 |
| `--deviation-std-floor` | `0.05` | KGKD train Healthy token 标准差的最小归一化尺度 |
| `--freeze-sentence-encoder` | 开启 | 固定 SSL embedding 坐标并训练偏移聚合器与分类头；可用 `--no-freeze-sentence-encoder` 关闭 |
| `--gradient-clip` | `1.0` | 三阶段统一的梯度范数裁剪阈值 |

## 输出文件

默认写入 `Results/gait_language/<run_name>/`：

| 文件 | 内容 |
|---|---|
| `args.json` | 本次实际参数 |
| `word_statistics.json` | SSL/dev 数据的 word 与 sentence 统计 |
| `best_vq.pt` | 最优 VQ tokenizer 与 codebook |
| `best_ssl.pt` | 最优 gait-language SSL 模型 |
| `best_downstream.pt` | 最优下游模型 |
| `metrics.jsonl` | 当前实验各 epoch 的结构化指标 |
| `evaluation.json` | 内部和外部测试结果 |

`--output-dir` 可以在多次实验中保持不变，因为程序会为每次调用创建新的 run 子目录。`run.py` 默认写入 `Results/gait_language/dev_exp/dev_exp_MMDD_HHMM`；`run_exp.py` 默认写入 `Results/gait_language/ablation_exp/ablation_exp_MMDD_HHMM`。同一分钟内重复使用相同目录名，或者手动指定已经存在的 `--run-name` 时，程序会直接报错，不会覆盖或继续写入已有实验。

## 设计说明

算法设计、三类掩码任务、DOF 专属 codebook 和双腿差异建模的详细理由见 [docs/gait_language_ssl_design.md](docs/gait_language_ssl_design.md)。
