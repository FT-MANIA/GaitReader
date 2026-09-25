# GaitReader

Healthy-only self-supervised learning for knee disease classification with GaitParser, VQ-Gait, and GaitFormer.

## Code structure

```text
run.py                       # Experiment entry point
configs/
  paper.json                 # Model and training settings
  label_efficiency.json      # Label-efficiency settings
  comparison.json            # Comparison model settings
gaitreader/
  config.py                  # Arguments and experiment definitions
  pipeline.py                # Pretraining and integrated five-fold evaluation
  factory.py                 # Model construction
  data/                      # GaitParser, normalization, and data loading
  models/                    # VQ-Gait, GaitFormer, and pretraining objectives
  comparisons/               # Adapters for comparison models
  training.py                # Training loops
  evaluation.py              # Metrics and summaries
  utils.py                   # Checkpoints, seeds, and shared utilities
benchmark_sources.json       # Pinned upstream comparison repositories
.benchmark_sources/          # Bundled comparison source snapshots and licenses
THIRD_PARTY_NOTICES.md        # Third-party attribution
```

## Environment

- Python 3.10 or later.
- PyTorch; a CUDA-capable GPU is recommended for training.
- NumPy, pandas, SciPy, scikit-learn, and Matplotlib.
- Comparison models additionally require einops, reformer-pytorch, and psutil.

Install a PyTorch build compatible with your CUDA environment, then run from the repository root:

```powershell
python -m pip install -e .
python -m pip install -e ".[comparisons]"
```

## Running experiments

All commands below run from the repository root. Supply authorized CSV files at:

```text
data/ssl_healthy_dataset.csv
data/dev_dataset.csv
data/test_dataset.csv
```

Alternatively, append `--ssl-csv PATH --dev-csv PATH --ext-test-csv PATH` to a command.

### Full model

```powershell
python run.py --suite full --seed 42
```

This runs VQ training, SSL pretraining, and downstream five-fold evaluation in one pipeline.

### Pretraining and label efficiency

Fine-tuning, training from scratch, and linear probing at 10%, 25%, 50%, and 100% labeled training data:

```powershell
python run.py --suite pretraining --config configs/label_efficiency.json --seed 42
```

### Ablations

```powershell
# Core method ablations, including GaitParser
python run.py --suite ablations --seed 42

# VQ and SSL module ablations
python run.py --suite modules --seed 42

# Input embedding ablations
python run.py --suite embeddings --seed 42

# All core, module, and embedding ablations
python run.py --suite full_ablations --seed 42
```

The `full_ablations` suite excludes codebook and masking-ratio sweeps.

### Codebook and masking ratio

```powershell
# Five (K, D) settings: (128,128), (64,128), (256,128), (128,32), (128,64)
python run.py --suite codebook --seed 42

# Codebook size K only, or code dimension D only
python run.py --suite codebook_k --seed 42
python run.py --suite codebook_d --seed 42

# Masking ratios: 10%, 15%, 30%, 50%, 75%
python run.py --suite mask_ratio --seed 42
```

### Comparison models

Pinned comparison source snapshots are bundled; no separate download is needed:

```powershell
python run.py --suite comparison --seed 42
```

Available methods: TimesNet, PatchTST, iTransformer, TS-TCC, TS2Vec, T-Rep, VQShape, and HeartLang.

### Selection, configuration, and resume

```powershell
# Select methods within a suite
python run.py --suite ablations --methods full without_waveform --seed 42
python run.py --suite comparison --methods timesnet patchtst --seed 42

# Override configuration values
python run.py --suite full --set batch_size=32 --set downstream_epochs=200

# Run every suite
python run.py --suite all --seed 42

# Resume an interrupted run in its original directory
python run.py --resume-dir results/gaitreader_TIMESTAMP
```

Results are saved under `results/` by default; change this with `--output-dir PATH`.
Resuming uses saved settings and skips completed folds. Use the same seed across
methods for paired comparisons; `--seed` also controls data and fold splits.

## Evaluation protocol

- Keep the internal and external test sets fixed across methods and folds.
- Split the development training-plus-validation pool into five class-stratified, subject-grouped folds. Both legs and all recordings of a subject stay together; provide `--subject-groups-csv PATH` when repeated recordings use different identifiers.
- Each fold starts from the method's fixed pretrained checkpoint, or a fresh initialization for scratch/supervised baselines. Fine-tuning updates the encoder; linear probing freezes it.
- Select the downstream checkpoint by validation macro-F1, with a default maximum of 200 epochs and early-stopping patience of 10. Test sets are not used for checkpoint selection.
- Report the mean of five models' scores on the same internal and external test sets, not an ensemble score. Label-efficiency experiments share matched labeled subsets across methods.
- `summary_brief.json` contains internal/external accuracy and macro-F1 means; `summary.json` contains detailed metrics and fold completion counts. Compare completed five-fold runs.

## Reproducibility and release status

During double-blind review, only the code is publicly available. The dataset will be made publicly available after the work is formally published.
