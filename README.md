# ECE285 Deep Generative Models - HW1

## Chest X-ray generation with a variational autoencoder

A PyTorch implementation and a reproducible experiment on the Kaggle
[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
dataset. The VAE is implemented from scratch; pretrained Inception-v3 is used only
for the requested evaluation metrics.

## Repository contents

This repository includes source code, tests, pinned dependencies, editable report
sources and figures, final PDF deliverables, and compact results from the original
experiment. Raw datasets, model checkpoints, generated sample arrays, Inception
feature caches, per-image manifests, training logs, intermediate epoch images,
archives, virtual environments, and temporary files are excluded by `.gitignore`.
These excluded files remain local; download the data and train a new checkpoint
to run evaluation or sampling after cloning.

## Gradescope submission

- Gradescope accepts PDF only. Upload `output/pdf/submission_report_and_code.pdf`:
  pages 1-2 are the report (including references; official NeurIPS 2025 preprint
  style), and pages 3-11 contain the source code with file names and line numbers.
- Separate PDFs are also available: `output/pdf/report.pdf` and
  `output/pdf/code.pdf` if report and code are assigned to separate upload fields.
- The code appendix includes the VAE, data preparation, training, evaluation,
  sampling, dataset download/recovery scripts, tests, and package requirements.
  It contains no model weights, datasets, caches, or experiment logs.
- `output/submission.zip` is an optional local reproducibility archive, not the
  PDF-only Gradescope deliverable; it is excluded from Git.

## Setup and reproduce

Python 3.12 was used. From this directory:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash scripts/download_data.sh
python -m chest_vae.train --output runs/repeat
TORCH_HOME="$PWD/.cache/torch" python -m chest_vae.evaluate --run runs/repeat
python -m pytest -q
```

The public Kaggle download endpoint normally works without credentials. If Kaggle
requires authentication, download the dataset from its page and save the ZIP at
`data/chest-xray-pneumonia.zip`, then run `python -m chest_vae.data`.
The first evaluation downloads official torchvision Inception weights (~104 MB).
The report requires a LaTeX engine; the official NeurIPS 2025 style is included.
To compile the included reference report without retraining:

```bash
mkdir -p output/pdf
tectonic -o output/pdf report/report.tex
```

Training refuses to overwrite an existing history. To repeat an experiment, use
`--output runs/repeat` and pass `--run runs/repeat` to evaluation and reporting.
The report builder also reads `runs/<name>/discussion.txt`, which must contain an
honest interpretation of that run's curves and generated images. Review and update
that text after rerunning; qualitative conclusions are not inferred automatically.
The included report template describes the default architecture and training
settings: edit its prose when changing hyperparameters. Seeds are fixed, but
results may vary across hardware and PyTorch versions.

After writing `runs/repeat/discussion.txt` for your new results, generate a separate
report with:

```bash
MPLCONFIGDIR="$PWD/.cache/matplotlib" python -m chest_vae.report --run runs/repeat --output report/repeat --author 'Your Name'
cp report/neurips_2025.sty report/repeat/
mkdir -p output/repeat
tectonic -o output/repeat report/repeat/report.tex
```

## Data and split policy

- Only one of the archive's repeated directory trees is used.
- Byte-identical files are deduplicated by SHA-256, prioritizing official test data.
- Patient-like IDs are parsed from `personNNN_...`, `IM-NNNN-...`, and
  `NORMAL2-IM-NNNN-...` filenames. Any official training group found in test is
  excluded before splitting training groups 90/10 (seed 42).
- The 16-image official validation partition is unused. Test images never guide
  checkpoint selection.
- Images become grayscale, are directly resized to 64 x 64 by PIL bilinear
  interpolation, and scaled to [0,1]. There is no augmentation or class balancing.
- `data/prepared/manifest.json` records every path, hash, group, label, and split.
  Filename grouping cannot establish clinical patient identity or detect all
  near-duplicates.

For the supplied run: 4,269 training, 468 validation, and 618 test images;
32 byte-identical files removed, 453 training images excluded for filename-group
overlap with test, and 16 official validation images unused.

The selected checkpoint is epoch 30. Test negative ELBO is 2,391.059 nats/image,
FID (pool2048) is 210.965, and full-sample IS is 1.697. Ten-split IS is
1.677 +/- 0.080. `runs/vae/metrics.json` retains full precision and the exact
checkpoint hash. `requirements-lock.txt` records the full installed environment.

The original download interrupted after a complete nested copy had arrived.
`scripts/recover_dataset.py` recovered that copy, checked every image's CRC32 and
uncompressed length, and required the complete original split counts of
5,216 / 16 / 624 before accepting it. Preparation then used
`python -m chest_vae.data --archive data/chest-xray-recovered.zip`.
This is an optional recovery path, not a subset of the dataset.

## Model and training

Four stride-2 convolution blocks encode a 64 x 64 image into a diagonal Gaussian
with 32 latent dimensions. A linear projection and four transposed-convolution
blocks decode it. SiLU is used in hidden layers; the output is a Bernoulli logit.
The objective is pixel-summed binary cross-entropy plus latent-summed KL,
averaged per image. The KL coefficient increases from 0.2 to 1 over five epochs.

Defaults: 30 epochs, Adam lr 0.001, batch size 64, gradient norm limit 10,
seed 42. The device is automatically CUDA, Metal/MPS, or CPU; `--device` overrides
this choice. The checkpoint with the lowest validation BCE + KL is selected.
Validation uses fixed reparameterization noise; training and validation curves
report BCE + KL at coefficient 1 throughout warm-up for comparison.

## Evaluation details

Adapted from the assigned
[PyTorch-Ignite GAN evaluation tutorial](https://pytorch-ignite.ai/blog/gan-evaluation-with-fid-and-is/).
The evaluator uses Ignite's `Engine`, `FID`, and `InceptionScore`, with one shared
Inception pass supplying explicit features/probabilities to identity extractors.

- **FID, pool2048:** the primary score uses 2,048-dimensional pooled torchvision
  Inception-v3 features.
- **FID, probability1000:** a supplemental score uses the 1,000-class softmax
  representation used by Ignite's default extractor. The magnitudes are different
  and these scores must not be conflated.
- **IS:** computed over all generated samples. Also reports mean and population
  standard deviation across ten nearly equal contiguous subsets. This is not a
  confidence interval or variability across independently trained models.
- Both real and generated 64 x 64 images are quantized to uint8, replicated to RGB,
  PIL-bilinearly resized to 299 x 299, and ImageNet-normalized. This normalization
  is explicit, unlike the blog example.
- The primary score uses torchvision ImageNet weights, **not** the TensorFlow
  Inception weights used by `pytorch-fid`. Do not compare scores across those
  implementations or different sample counts/preprocessing.
- All 618 test images are compared with 618 independent prior samples (seed 123).
  The test count is below feature dimension, so empirical covariance is rank
  deficient and FID is affected by finite-sample bias. ImageNet metrics are not
  validated measures of medical accuracy.

`generated.png` shows the first 32 prior samples, without selection or individual
contrast normalization. `reconstructions.png` shows eight test inputs above their
posterior-mean reconstructions. These two outputs serve different purposes.

## Files

- `chest_vae/`: preparation, VAE, training, evaluation, sampling, and report builder.
- `tests/`: loss reductions, reparameterization gradients, group parsing, metric checks.
- `runs/vae/config.json`, `history.csv`, `metrics.json`: settings and actual results.
- `runs/vae/best.pt`: local selected model state and configuration (excluded from Git).
- `report/report.tex`, `neurips_2025.sty`, and figures: editable report sources.
- `output/pdf/report.pdf`: final two-page report.

After training, generate new images from your checkpoint without loading the dataset:

```bash
python -m chest_vae.sample --checkpoint runs/repeat/best.pt --count 32 --seed 2026 --output samples.png
```

The NeurIPS style comes from the
[official 2025 template archive](https://media.neurips.cc/Conferences/NeurIPS2025/Styles.zip).
The model follows [Kingma and Welling, Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114).
