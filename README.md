# GaitReader

Healthy-only self-supervised learning for subject-level knee disease recognition.
The paper implementation consists of **GaitParser**, **VQ-Gait**, and **GaitFormer**.
This directory is self-contained: it does not import the development repository,
its entrypoints, or its result directories. The development repository is unchanged.

## Scope

- GaitParser: adaptive cycle anchors, shared six-DOF boundaries, resampling, and healthy-training normalization.
- VQ-Gait: independent DOF shape CNNs, shared code projection, six EMA vocabularies, and independent MLP shape decoders.
- GaitFormer: shape/attribute tokens, DOF/timing embeddings, optional cycle embeddings, within-side self-attention, and bilateral cross-attention.
- Pretraining: masked code ID, continuous attributes, and vocabulary-mediated waveform reconstruction from one masked backbone pass.
- Classification: mean of the two side CLS embeddings followed by one linear classifier.
- Integrated five-fold evaluation, scratch/linear-probe/label-efficiency experiments, paper ablations, and eight official-source comparison adapters.

There is **no rhythm-prediction loss**. Duration/interval input embeddings are retained.
New runs default to **`ssl_use_cycle_embedding=false`**. The cycle parameter and
its initialization remain in place, but its contribution is omitted in both SSL
and downstream. This is a new experimental baseline, not a relabeling of archived
all-on Full results. Other embeddings and SSL losses are unchanged. Restore the
old input configuration with `--set ssl_use_cycle_embedding=true`.
VQ is unchanged and can be reused; SSL cache keys distinguish cycle on/off.
Start a new run (no `--resume-dir`) to try this default, and do not import an old
cycle-on SSL checkpoint. Resuming an old run preserves its saved cycle setting.
HCR, deviation fusion, alternative contextual VQ encoders, CNN decoders, top-k code losses,
cross-DOF prediction and bilateral auxiliary tasks are not included.

## Install

Use Python 3.10+ and install PyTorch appropriate to your CUDA environment first.
From this directory:

```powershell
python -m pip install -e .
```

Runs record their software environment; cross-version or cross-GPU bitwise
reproducibility is not guaranteed.
For comparison models also install `python -m pip install -e ".[comparisons]"`.

## Data

The clinical recordings are **not bundled**. Supply the three authorized CSVs:

```text
data/ssl_healthy_dataset.csv
data/dev_dataset.csv
data/test_dataset.csv
```

Each CSV has paired left/right rows per `person_id`, with `leg`, `label`, `features`,
`gender`, `age`, and `bmi` columns. `features` is a serialized `[600,6]` matrix.
Raw channel order is `[VV, IE, FE, AP, SI, ML]`; the model order is
`[FE, VV, IE, AP, ML, SI]`. Labels are Healthy=0, ACLD=1, KOA=2.
Demographic columns are read for input compatibility, **not used as model inputs**.
There must be one row per side per recording identifier. For repeated recordings,
give each paired recording a unique `person_id` and provide a CSV with
`subject_id,group_id` to keep aliases from the same physical subject together.
The pipeline rejects group overlap between development and fixed test cohorts.
Do not publish private CSVs, fold manifests, per-subject predictions, or result
folders without the appropriate data-release authorization.

You can supply authorized data paths without copying the files:

```powershell
python run.py --suite full --ssl-csv data/ssl_healthy_dataset.csv --dev-csv data/dev_dataset.csv --ext-test-csv data/test_dataset.csv
```

## One command, including five folds

```powershell
python run.py --suite full
python run.py --suite ablations
python run.py --suite pretraining --config configs/label_efficiency.json
```

`full` trains VQ-Gait and GaitFormer once, then immediately fine-tunes five fresh
downstream models. No separate CV entrypoint is needed. `ablations` runs the
following five core comparisons only (including Full):

| Name | Intervention |
| --- | --- |
| `full` | Code + attributes + vocabulary-mediated waveform |
| `without_vocabulary` | No VQ; direct masked waveform MSE |
| `without_waveform` | Code + attributes |
| `without_attributes_waveform` | Code only; input attributes remain |
| `without_gaitparser` | Non-overlapping 100-frame windows, with the same eligible cohort and scaler |

Use `--suite full_ablations` to run all core, module and embedding ablations
together: **24 methods with Full included once**, excluding codebook K/D and mask-ratio sweeps,
label-efficiency experiments and comparison models. Results share one timestamped
run directory and its `summary.json` / `summary_brief.json`.

| Suite | Included methods |
| --- | --- |
| `ablations` | 5 core methods, including Full |
| `modules` | Full + 7 module ablations |
| `embeddings` | Full + 12 embedding ablations |
| `full_ablations` | Union of the three above, Full only once; no codebook sweep |
| `codebook` | 5 K/D combinations only |
| `mask_ratio` | 5 SSL masking probabilities only |
| `all` | All ablations, codebook/mask-ratio sweeps, label-efficiency and comparisons |

```powershell
python run.py --suite full_ablations
```

This changes newly created experiment plans only. Resuming an existing plan
continues its saved method list; existing results are not moved or deleted.

### Detailed module ablations

`--suite modules` runs `full` and the following seven leave-one-out variants.
They are also included in `--suite full_ablations`, not `--suite ablations`. Names distinguish **VQ losses**
from **SSL losses**; `without_waveform` above removes only the SSL waveform loss.

| Name | Changed setting | Retrained pretraining stages |
| --- | --- | --- |
| `without_vq_attribute_separation` | `vq_v2_separate_shape=false` | VQ + SSL |
| `without_vq_geometry` | `vq_geometry_weight=0` | VQ + SSL |
| `without_vq_shape_reconstruction` | `vq_v2_shape_loss_weight=0` | VQ + SSL |
| `without_vq_waveform_reconstruction` | `vq_v2_waveform_loss_weight=0` | VQ + SSL |
| `without_ssl_attribute_loss` | `ssl_attribute_weight=0` | SSL, shared VQ |
| `without_ssl_code_loss` | `masked_top1_weight=0` | SSL, shared VQ |
| `without_bilateral_context` | `bilateral_depth=0` | SSL, shared VQ |

Each method then fine-tunes five newly initialized downstream models from its
own fixed SSL checkpoint. The data split, subject folds, normalization, masks,
other loss weights and training budgets remain controlled. Removing geometry
also removes its random neighborhood sampling; shared seeds do not imply
identical subsequent random streams between different objectives/architectures.

The no-separation VQ arm sends the dataset-normalized **raw word** into the CNN
and directly decodes the quantized feature into a raw waveform, without the
ground-truth mean/scale bypass. Its shape loss still compares standardized
decoded/input shapes; its waveform loss compares raw decoded/input waveforms.
The geometry target remains standardized shape, isolating the representation
factorization instead of changing the geometry definition simultaneously.
Unlike the old development no-separation implementation, the two reconstruction
losses do not become duplicate raw-waveform losses. In SSL the frozen decoded
templates are still standardized before combining with predicted attributes;
SSL input attributes and all Full SSL objectives remain enabled.

`without_ssl_attribute_loss` disables explicit attribute supervision, not the
attribute head or input embeddings: the head still receives waveform gradients.
Similarly, `without_ssl_code_loss` removes cross-entropy, not the code head or
codebook: soft code probabilities still support waveform reconstruction.
`without_bilateral_context` removes bilateral cross-attention, not either leg;
both within-side encodings still contribute to the subject representation.

```powershell
python run.py --suite modules
python run.py --suite modules --methods full without_vq_attribute_separation without_vq_geometry
```

### Input embedding ablations

`--suite embeddings` runs Full plus twelve variants. These switches affect
GaitFormer inputs in **both SSL and downstream**, not VQ, SSL targets or loss
weights. Each group shares its VQ checkpoint and trains distinct SSL configurations,
followed by the same fixed-test five-fold protocol. These arms are also included
in `--suite full_ablations` and `--suite all`, not `--suite ablations`.

The leave-one-out/progressive-addition definitions below describe the original
**all-on control**: use `--set ssl_use_cycle_embedding=true` to reproduce that
suite. Under the new cycle-off default, `full` and `without_cycle_embedding`
have identical settings, and the other leave-one-out arms inherit cycle-off.

| Leave-one-out method | Input removed |
| --- | --- |
| `without_shape_embedding` | Shape CNN output (attributes/metadata remain) |
| `without_attribute_embedding` | Entire mean/log-scale projection, including bias |
| `without_mean_embedding` | Word mean input to the attribute projection |
| `without_scale_embedding` | Word log-standard-deviation input to the attribute projection |
| `without_dof_embedding` | Learned additive DOF embedding |
| `without_cycle_embedding` | Learned cycle-index embedding |
| `without_timing_embedding` | Entire duration/interval MLP output, including bias |
| `without_duration_embedding` | Duration input to the timing MLP |
| `without_interval_embedding` | Preceding center-to-center interval input to the timing MLP |

For progressive addition, compare the following sequence (all are selectable
with `--methods`):

| Method | Active word input embeddings |
| --- | --- |
| `embedding_shape_only` | Shape |
| `embedding_shape_attributes` | Shape + mean/log-scale |
| `embedding_shape_attributes_dof` | Shape + mean/log-scale + DOF |
| `without_timing_embedding` | Shape + mean/log-scale + DOF + cycle |
| `full` | Shape + mean/log-scale + DOF + cycle + duration/interval |

The fourth step reuses `without_timing_embedding` instead of running the same
configuration under another name. Shape/attribute embeddings are replaced by
the same mask token at masked positions; DOF/cycle/timing are added afterwards,
as in Full. CLS, mask tokens and bilateral attention are unchanged.

All embedding parameters are constructed in the original order, even when
their contributions are disabled. Switching embeddings does not itself add or
remove random initialization draws. Disabled whole branches do not enter the
token sum. For a single mean/log-scale or duration/interval component, its input
is set to zero while retaining the shared projection and its bias. Full's default
arithmetic and state-dict structure are unchanged. Trainable models still diverge
as intended when their inputs differ; identical final metrics are not expected.

Removing the DOF embedding does **not** remove DOF-specific CNNs or attribute
projections; it tests the extra additive identifier, not all channel identity.
Likewise shape-only still uses DOF-specific CNNs, padding masks and two-side
processing. This release has no separate side, quality, continuous cycle-center,
or flattened-token positional embedding; no fictitious ablations are added for them.

```powershell
python run.py --suite embeddings --set ssl_use_cycle_embedding=true
python run.py --suite embeddings --set ssl_use_cycle_embedding=true --methods full without_attribute_embedding without_dof_embedding without_cycle_embedding without_timing_embedding
python run.py --suite embeddings --set ssl_use_cycle_embedding=true --methods embedding_shape_only embedding_shape_attributes embedding_shape_attributes_dof without_timing_embedding full
```

Individual switches are also available through `--set`:
`ssl_use_shape_embedding`, `ssl_use_mean_embedding`, `ssl_use_scale_embedding`,
`ssl_use_dof_embedding`, `ssl_use_cycle_embedding`,
`ssl_use_duration_embedding`, and `ssl_use_interval_embedding` (all default to
`true` except cycle, which defaults to `false`). For example, `--set ssl_use_duration_embedding=false` removes duration
input without disabling waveform/attribute supervision or changing VQ training.
Named experiment overrides take precedence over general config values.

### Codebook size and dimension

`--suite codebook` reproduces the five legacy **one-factor** combinations, not a
full Cartesian grid. CNN capacity, shared projection, MLP decoder, SSL encoder
width and the Full objectives remain unchanged.

| Method | Codes per DOF (K) | Code dimension (D) |
| --- | ---: | ---: |
| `codebook_k128_d128` | 128 | 128 |
| `codebook_k64_d128` | 64 | 128 |
| `codebook_k256_d128` | 256 | 128 |
| `codebook_k128_d32` | 128 | 32 |
| `codebook_k128_d64` | 128 | 64 |

New Full runs default to **K=128, D=128** (cycle embedding remains off).
Archived K=64 results retain their original configuration; they are not results
for this new default. The historical sweep anchor is also K=128, D=128.
K is per DOF, so the codebook tensor is `[6,K,D]`. Start a new VQ -> SSL ->
five-fold run for this change, without importing K=64 checkpoints or resuming
an old K=64 run. Named K/D sweep arms still use their explicit combinations.
Each distinct VQ configuration trains its own VQ -> SSL -> five-fold pipeline.
Stage caches within the same run share only matching configurations (and SSL's
VQ checkpoint hash); compatible cached pretraining is not trained twice.

```powershell
python run.py --suite codebook
python run.py --suite codebook_k   # K=64/128/256, D=128
python run.py --suite codebook_d   # D=32/64/128, K=128
```

All commands above run from `GaitReader/`; provide the same three CSV options as
in the Data section. `--suite all` also includes these new experiments. Method
overrides are saved in `plan.json`, so resume preserves the planned variants.
An old saved plan is not expanded by resuming it: start a new run for new arms.
Do not supply Full VQ/SSL checkpoints to a variant that changes that stage.

### SSL mask-ratio sweep

`--suite mask_ratio` changes only `random_mask_ratio` relative to the selected
Full config. Current defaults are cycle embedding **off**, K=128, D=128 and
code + attribute + waveform SSL losses. All five arms share the same VQ inside
the run, train SSL separately, and then execute five-fold downstream fine-tuning
with the same subject folds, seeds, label fraction and training budgets.

| Method | Masking probability |
| --- | ---: |
| `mask_ratio_010` | 0.10 |
| `mask_ratio_015` | 0.15 (current Full baseline) |
| `mask_ratio_030` | 0.30 |
| `mask_ratio_050` | 0.50 |
| `mask_ratio_075` | 0.75 |

Masking remains independent Bernoulli sampling over valid `[B,2,W,6]` word
positions, not a fixed count, contiguous window or whole-cycle mask. Thus the
realized fraction varies by batch. Code, attribute and waveform losses share
the same masked positions. Metadata embeddings follow the existing Full rules;
downstream evaluation does not apply the SSL training mask. Loss reduction and
mask generation are unchanged; 0% is not included because the masked objective
would have no prediction targets.

The probability is already part of the SSL cache key, but not the VQ key.
Changing it therefore shares VQ without reusing a different-ratio SSL result.
`mask_ratio_015` has the same effective config as Full at the default ratio;
there is no extra `full` arm in this suite. Overrides are stored in `plan.json`
for resume. This sweep is separate from `ablations` / `full_ablations`, but
included in `all`. Other config overrides are inherited; to compare ratios,
keep all non-ratio settings identical.

From `GaitReader/` (with the usual CSV options):

```powershell
python run.py --suite mask_ratio
python run.py --suite mask_ratio --methods mask_ratio_015 mask_ratio_030 mask_ratio_050
```

Example with explicit data and output paths:

```powershell
python .\run.py `
  --suite mask_ratio --seed 42 `
  --ssl-csv .\data\ssl_healthy_dataset.csv `
  --dev-csv .\data\dev_dataset.csv `
  --ext-test-csv .\data\test_dataset.csv `
  --set ssl_use_cycle_embedding=false --set codebook_size=128 `
  --output-dir .\results
```

To reuse a trusted existing **K=128 VQ** checkpoint, provide the same `vq` path
under each selected `mask_ratio_XXX` entry in `--checkpoint-map` and omit `ssl`
to retrain SSL. A `full` checkpoint-map entry does not apply to these variant
names. Do not substitute a K=64 VQ or import a different masking-ratio SSL.
New runs save the five-fold means to `summary_brief.json` in one run directory;
resume that directory with `--resume-dir` if interrupted.

`pretraining` compares fine-tuning, scratch, and linear probing at 10%, 25%, 50%,
and 100% labels. Subsets are nested, class-stratified, subject-grouped, and shared
between methods. All suites now default to a **200-epoch** downstream ceiling,
including the main and label-efficiency configurations. Checkpoint selection uses
validation macro-F1 with patience 10, so training may stop earlier. Historical
main/ablation/comparison results used a 50-epoch ceiling. Match the training budget
of the experiment being reproduced rather than assuming current defaults match it.

Select a subset without editing code:

```powershell
python run.py --suite ablations --methods full without_waveform
python run.py --suite pretraining --methods finetune_labels010 scratch_labels010 --config configs/label_efficiency.json
```

Settings live in `configs/paper.json`. Override existing fields with, for example,
`--set batch_size=32 --set downstream_epochs=200`. `--seed 43` sets the split,
fold, pretraining, downstream, validation-mask and label-subset seeds together.
Changing the seed therefore changes the population partition as in the original
implementation; use the same seed/manifest for a paired method comparison.

## Evaluation protocol

1. Preserve the original internal and external test subjects.
2. Pool the original post-QC development training and validation subjects (549 in the paper data).
3. Stratify **unique subject groups** into five train/validation folds (439/110 or 440/109).
4. Reload the same fixed SSL checkpoint at every fold; scratch starts from initialization.
5. Train on that fold's labeled subset, early-stop on its full validation set, and evaluate both fixed tests.

The reported result is the mean of five model scores on the **same** test sets,
not an ensemble and not five independent held-out-test folds. No fold or checkpoint
is selected by test performance. For the fixed-window ablation, adaptive processing
still defines the reference cohort and scaler: only the model's word construction
changes. The experiment is not an end-to-end removal of all adaptive preprocessing.

```text
results/gaitreader_TIMESTAMP/
  plan.json                 # settings, source/data hashes and environment
  fold_manifest.json        # shared subject groups, folds and label subsets
  pretraining/vq/KEY/        # one vocabulary per relevant configuration
  pretraining/ssl/KEY/       # one SSL model per objective configuration
  full/
    checkpoints.json
    fold_01/...fold_05/      # logs, best/last checkpoints, evaluation.json
  summary.json              # means, sample SDs, per-class metrics, fold completion
  summary_brief.json        # internal/external accuracy and macro-F1 means
```

Only compare complete five-fold summaries. Interim summaries include completed
folds only; completion counts are in `summary.json`.

## Resume

```powershell
python run.py --resume-dir results/gaitreader_TIMESTAMP
```

Saved settings override current CLI defaults. Completed folds are skipped;
unfinished stages restore optimizer, AMP scaler, RNG, DataLoader generators,
best checkpoint, and early-stopping counters from the last committed epoch.
An interrupted partial epoch is replayed. Each new fold still starts from the
fixed pretrained checkpoint. Do not run two writers against the same output directory.

## Existing paper checkpoints

Trusted historical VQ/SSL model checkpoints can be supplied without retraining:

```json
{
  "full": {"vq": "checkpoints/best_vq.pt", "ssl": "checkpoints/best_ssl.pt"},
  "without_vocabulary": {"ssl": "checkpoints/waveform_ssl.pt"}
}
```

```powershell
python run.py --suite full --checkpoint-map checkpoint_map.json
```

Paths in this map resolve against the working directory. A `full` entry also serves
fine-tuning/probing label-fraction variants. Models must match the selected config.
Only obsolete disabled rhythm/timing-mask parameters are removed at load time;
all remaining state keys are checked strictly. Load only trusted checkpoints.

Locally, `--fold-manifest PATH` can
reuse a prior manifest after validating cohort IDs, ordering and normalization.

## Comparison methods

```powershell
python run.py --fetch-sources
python run.py --suite comparison
```

Official repositories are fetched only on explicit request and pinned by
`benchmark_sources.json`. No upstream source is vendored here. Adapters cover
TimesNet, PatchTST, iTransformer, TS-TCC, TS2Vec, T-Rep, VQShape, and HeartLang.
Pretrained methods train once on healthy subjects before five-fold downstream
evaluation; supervised methods initialize anew per fold. Comparison adapters use
FP32 as in the reference runs. Their upstream licensing terms still apply.
These are gait adaptations with dataset-sized configurations, not the original
papers' unmodified experimental protocols.

## Layout

```text
run.py
configs/
gaitreader/
  config.py, factory.py, pipeline.py
  data/                     # GaitParser, cohort construction, batches
  models/                   # tokens, VQ-Gait, GaitFormer, masked objectives
  training.py, evaluation.py, utils.py
  comparisons/              # isolated official-source adapters
```

## Reproducibility and release status

This release provides model, training, evaluation, and experiment configuration
code. Manuscripts, historical results, and local utility scripts are not bundled.
The clinical data and pretrained weights are also not bundled; reproducing
the reported numerical results requires authorized data access and matching
experiment configurations and splits. Publishing code alone does not provide
independent end-to-end numerical reproducibility. A repository license must be selected by the authors before
calling this a licensed public open-source release.
