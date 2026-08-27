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

Upload MS-COCO under the configured dataset root. The current AutoDL layout
stores the image files directly in:

```text
data/MS-COCO/
```

The loader recursively scans the dataset root, ignores non-image files, and uses
a deterministic directory/name order. It stops as soon as `--limit` images have
been found, so the two-image smoke does not enumerate the entire dataset. Images
are resized and center-cropped to 512 x 512; baseline and full always receive the
same cover list. Both cross-model configs therefore use
`cover_dir: data/MS-COCO`. Each run manifest records the selected relative paths
and SHA-256 hashes. Confirm the COCO split label separately from the uploaded
file count before reporting it in the paper.

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
the exact SD1.4 proxy-VAE revision, the uploaded MS-COCO cover root, five detector-positive
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

## 8.1 Simple Average + Jain lambda smoke with earliest-success stopping

`configs/tree_ring_simple_average_lambda1e4_smoke.yaml` uses all five shared-key
references directly (no robust reference rejection), the official Jain Tree-Ring
README value `lambda_pixel=1e4`, and a maximum budget of 3000 optimization steps.
The SD2-base target detector runs every 100 steps. The first checkpoint with
`p_value <= 0.05` is retained as the final output and optimization stops, avoiding
unnecessary later distortion. Each sample records its full periodic detector trace
under `detections/simple_average/` and its earliest success in `manifest.json`.

```bash
python -u run_forgery.py \
  --config configs/tree_ring_simple_average_lambda1e4_smoke.yaml \
  --mode simple_average \
  --limit 2 \
  --iterations 3000 \
  --lambda-pixel 10000 \
  --detection-every 100 \
  --early-stop-on-success \
  --run-name simple_average_lambda1e4_smoke_2x3000
```
- The first run uses SD 1.4 for both the target and proxy VAE to obtain a quick
  feasibility result. Cross-model transfer is a later experiment.
- Legacy configurations query the detector only during separate evaluation.
  The Section 8.1 smoke explicitly enables periodic detection for earliest-success
  stopping.
- A clean image already satisfying `p <= 0.05` is a detector false positive and
  must be flagged when interpreting ASR.
- This code does not include two-stage optimization, confidence weighting,
  LPIPS loss, or removal. Multi-key paired evaluation is provided in Section 10.

## 9. Local checks without model downloads

```bash
python -m compileall rmlp prepare_references.py prepare_multikey_references.py \
  run_forgery.py run_multikey_forgery.py evaluate.py evaluate_multikey.py
python -m pytest -q
```

## 10. Paired 10-key experiment: baseline vs five-reference average

`configs/tree_ring_multikey_paired_10x15000.yaml` defines ten one-to-one
key-cover pairs. Key `i` attacks only cover `i`, so each method runs ten attacks
(twenty total), not a 10 x 10 Cartesian product. Baseline uses reference index
zero for the key; Simple Average uses all five references. Both methods use
`lambda_pixel=1e4`, `alpha=5/255`, target detection every 100 steps, earliest
success stopping, and a maximum budget of 15,000 steps.

```bash
python -u prepare_multikey_references.py \
  --config configs/tree_ring_multikey_paired_10x15000.yaml \
  --verify \
  --skip-existing

python -u run_multikey_forgery.py \
  --config configs/tree_ring_multikey_paired_10x15000.yaml \
  --pair-count 10 \
  --iterations 15000 \
  --lambda-pixel 10000 \
  --detection-every 100 \
  --run-name paired_10keys_10covers_15000

python -u evaluate_multikey.py \
  --config configs/tree_ring_multikey_paired_10x15000.yaml \
  --run-dir outputs/tree_ring_multikey_paired/attacks/paired_10keys_10covers_15000
```

Evaluation produces `metrics.csv`, `paired_metrics.csv`,
`asr_by_iteration.csv`, and `summary.json`. The ASR curve is cumulative by
first-success step, allowing a fair comparison under every shared iteration
budget while still stopping successful attacks before later distortion.

## 11. Single proposed-method forgery visualization

`run_forgery_visualization.py` is a separate, detector-free attack-stage entry
for one qualitative trajectory. It regenerates and verifies five Tree-Ring
references for `w_seed=0`, unloads the target pipeline, selects the 1,314th
deterministically ordered COCO image (Python index 1,313), and then runs only
the five-reference FP32 `simple_average` attack for 3,000 fixed iterations.
The attack saves images and loss logs every 500 steps and performs no detection
or quality evaluation during optimization.

The four-channel proxy-VAE latents are retained exactly as `.pt` files. For
single-panel viewing, one PCA mapping is fitted jointly to the clean latent,
the five reference latents, and their mean, then reused to project every latent
from four channels to one RGB PNG. These PCA colors are descriptive and are not
a decoded natural image or an invertible representation.

```bash
python -u run_forgery_visualization.py \
  --config configs/tree_ring_forgery_visualization_1x3000.yaml \
  --run-name forgery_visualization_key0_coco1314_3000
```

The run is written under
`outputs/tree_ring_forgery_visualization/<run-name>/`. Run names are immutable:
the script refuses to overwrite an existing directory.

## 12. Single proposed-method removal visualization

`run_removal_visualization.py` generates six detector-positive Tree-Ring images
for `w_seed=52`. The first five form the FP32 watermarked-latent mean; the sixth
is an independent held-out removal target and is never included in that mean.
Deterministically ordered COCO positions 1,314--1,318 form the five unpaired
clean priors. Only the proposed `mean_shift` target with `beta=1` is optimized.

The run always executes 3,000 steps, saves an image and loss record every 500
steps, and measures the target-key p-value at step 0 and every 500 steps. These
detector calls are diagnostic only: they do not affect the loss, select an
output, or trigger early stopping. Consequently this is a fixed-budget run with
online detector monitoring, not a strict zero-query black-box protocol. The
final output is always step 3,000. No quality metrics or baseline are computed.

```bash
python -u run_removal_visualization.py \
  --config configs/tree_ring_removal_visualization_1x3000.yaml \
  --run-name removal_visualization_key52_3000
```

Exact latent tensors are saved as `.pt` files. Fifteen single-panel PNGs use one
shared descriptive PCA mapping: five reference latents, the held-out target,
five clean priors, both means, the estimated direction, and the removal target.
They are not VAE-decoded natural images.

After a complete run, create the three package artifacts under the fixed AutoDL
directory `/root/autodl-tmp/experiment_packages`:

```bash
bash scripts/package_removal_visualization.sh removal_visualization_key52_3000
```
