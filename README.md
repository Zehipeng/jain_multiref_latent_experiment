# Jain + Robust Multi-Reference Latent Prototype

This folder contains the stage-1 Tree-Ring forgery experiment. It keeps Jain's
simplified Tree-Ring configuration (`channel=0`, `radius=16`, Jain ring-key
construction) and compares:

- `baseline`: Jain optimization toward the first reference latent.
- `simple_average`: optimization toward the mean of all five reference latents.
- `full`: the same optimization toward a robust 5-to-4 reference prototype.

The attack objective and optimizer are identical between the methods. Only
the target latent changes.

## 1. Repository layout

```text
configs/tree_ring_stage1.yaml     experiment configuration
prompts/reference_prompts.txt     five fixed reference prompts
rmlp/                             reusable method code
prepare_references.py             same-key reference generation
run_forgery.py                    baseline/full image optimization
evaluate.py                       Tree-Ring p-value and quality metrics
tests/                             model-free unit tests
```

Generated images, downloaded data, and checkpoints are ignored by Git. Keep the
source code in GitHub and keep large experiment outputs on AutoDL or download
them separately.

## 2. GitHub to AutoDL workflow

On the local machine, commit this folder to the GitHub repository. On AutoDL:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <YOUR_REPOSITORY>/jain_multiref_latent_experiment
```

For later updates:

```bash
git pull
```

The repository contains helper scripts for the exact smoke and core command
sequences. Run them from this project directory with `bash scripts/autodl_smoke.sh`
or `bash scripts/autodl_core.sh` after covers and dependencies are ready.

Use an AutoDL PyTorch image with CUDA. Do not reinstall PyTorch unless the
installed build is incompatible with the GPU. Install the remaining packages:

```bash
python -m pip install -U pip
python -m pip install -r requirements.txt
```

The core Hugging Face packages are compatibility-locked for the AutoDL
PyTorch 2.1.2 + CUDA 12.1 image. Do not independently upgrade Transformers;
newer releases use a pytree API that is unavailable in PyTorch 2.1.2.

If Hugging Face requests authentication:

```bash
huggingface-cli login
```

## 3. Prepare cover images

Upload MS-COCO under the configured dataset root, for example:

```text
data/MS-COCO/
  train2017/
  val2017/
  annotations/
```

The loader recursively scans the dataset root, ignores non-image files, and uses
a deterministic directory/name order. It stops as soon as `--limit` images have
been found, so the two-image smoke does not enumerate the entire dataset. Images
are resized and center-cropped to 512 x 512; baseline and full always receive the
same cover list. To select one COCO split explicitly, pass for example
`--cover-dir data/MS-COCO/val2017`.

## 4. Generate the five-reference bank

All five references use `w_seed=0` and the same `w_key`. Their prompts and
generation seeds differ. The expensive `--verify` option runs Jain's DDIM
detector on every reference.

```bash
python prepare_references.py \
  --config configs/tree_ring_stage1.yaml \
  --verify
```

Outputs:

```text
outputs/tree_ring_stage1/references/
  metadata.json
  watermark_key.pt
  ref_00_gseed_0.png
  ...
```

If rerunning intentionally, add `--overwrite`.

## 5. Cheap smoke test

Run two covers for 200 optimization steps:

```bash
python run_forgery.py \
  --config configs/tree_ring_stage1.yaml \
  --mode both \
  --limit 2 \
  --iterations 200 \
  --run-name smoke_2x200
```

Evaluate without LPIPS first to avoid its optional weight download:

```bash
python evaluate.py \
  --config configs/tree_ring_stage1.yaml \
  --run-dir outputs/tree_ring_stage1/attacks/smoke_2x200 \
  --no-lpips
```

The smoke test checks execution only; it is not method evidence.

## 6. Stage-1 core experiment

Ten covers, 3,000 steps, paired baseline/full comparison:

```bash
python run_forgery.py \
  --config configs/tree_ring_stage1.yaml \
  --mode both \
  --limit 10 \
  --iterations 3000 \
  --run-name core_10x3000
```

Then evaluate:

```bash
python evaluate.py \
  --config configs/tree_ring_stage1.yaml \
  --run-dir outputs/tree_ring_stage1/attacks/core_10x3000
```

Important outputs:

- `manifest.json`: exact attack inputs and outputs.
- `prototype_diagnostics.json`: reference distances and rejected index.
- `logs/<method>/*.csv`: optimization curves.
- `metrics.csv`: per-image paired metrics.
- `summary.json`: ASR, p-value, PSNR, SSIM, and LPIPS averages.

`eligible_asr` excludes covers that were already detector false positives before
the attack. Both raw `asr` and `eligible_asr` are retained. Runtime and CUDA peak
allocated memory are also recorded per image. A partially interrupted attack can
be resumed with the same arguments plus `--skip-existing`.

## 7. Confirmation run

Only after the core run shows a useful trend:

```bash
python run_forgery.py \
  --config configs/tree_ring_stage1.yaml \
  --mode both \
  --limit 20 \
  --iterations 7500 \
  --run-name confirm_20x7500
```

## 7.1 Cross-model smoke with five detector-positive references

The cross-model configuration follows Jain's released example: Tree-Ring
generation and detection use SD2-base, while image optimization uses the VAE
from `CompVis/stable-diffusion-v1-4`. Because the original Stability AI Hub
repository was deprecated, the configuration uses the public archival mirror
`sd2-community/stable-diffusion-2-base` pinned to revision
`f5bc1bd97485577aa0b946fa8a9004e2ec147402`. All accepted
references share `w_seed=0`. Candidate generation continues until five
references satisfy `p_value <= 0.05`; rejected candidates are recorded in
`metadata.json` but are not saved into the attack reference bank.

```bash
python prepare_references.py \
  --config configs/tree_ring_cross_model_smoke.yaml \
  --verify \
  --overwrite

python run_forgery.py \
  --config configs/tree_ring_cross_model_smoke.yaml \
  --mode all \
  --limit 2 \
  --iterations 200 \
  --run-name cross_model_smoke_2x200

python evaluate.py \
  --config configs/tree_ring_cross_model_smoke.yaml \
  --run-dir outputs/tree_ring_cross_model_smoke/attacks/cross_model_smoke_2x200 \
  --no-lpips
```

Before running the attack, inspect `outputs/tree_ring_cross_model_smoke/references/metadata.json`
and confirm `accepted_count=5` and that every accepted reference has
`p_value <= 0.05`.

The corrected smoke also requires every reference-latent statistic and every
prototype distance in `prototype_diagnostics.json` to be finite. Prototype
distances, median centering, and aggregation are computed in fp32; a non-finite
latent now stops the run instead of silently influencing reference rejection.

## 7.2 Cross-model 10 x 3,000 formal pretest

`configs/tree_ring_cross_model_formal.yaml` fixes the exact SD2 target revision,
the exact SD1.4 proxy-VAE revision, MS-COCO 2017 `val2017`, five detector-positive
same-key references, 3,000 optimization steps, and all three methods. It saves
snapshots at steps 1,000/2,000/3,000 for only the first five covers and computes
LPIPS during evaluation.

```bash
python run_forgery.py \
  --config configs/tree_ring_cross_model_formal.yaml \
  --mode all \
  --limit 10 \
  --iterations 3000 \
  --run-name cross_model_pretest_10x3000

python evaluate.py \
  --config configs/tree_ring_cross_model_formal.yaml \
  --run-dir outputs/tree_ring_cross_model_formal/attacks/cross_model_pretest_10x3000
```

Each run records a config snapshot and checksum, Git commit/status, package
versions, target/proxy model IDs and revisions, reference metadata/key checksum,
and a SHA-256 cover manifest. Evaluation adds `evaluation_manifest.json` with
the detector settings and result-file checksums.

## 8. Configuration notes

- `lambda_pixel=2` and `alpha=5/255` follow Jain's hard-coded defaults.
- The first run uses SD 1.4 for both the target and proxy VAE to obtain a quick
  feasibility result. Cross-model transfer is a later experiment.
- The detector is never queried during optimization. It is used only in the
  separate evaluation step.
- A clean image already satisfying `p <= 0.05` is a detector false positive and
  must be flagged when interpreting ASR.
- This code does not yet include two-stage optimization, confidence weighting,
  LPIPS loss, removal, or multi-key evaluation.

## 9. Local checks without model downloads

```bash
python -m compileall rmlp prepare_references.py run_forgery.py evaluate.py
python -m pytest -q
```
