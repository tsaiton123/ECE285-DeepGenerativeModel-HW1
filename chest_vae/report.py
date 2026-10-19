"""Build figures and a two-page NeurIPS LaTeX report from measured results."""
import argparse
import csv
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="runs/vae")
    p.add_argument("--data", default="data/prepared")
    p.add_argument("--output", default="report")
    p.add_argument("--author", default="Deep Generative Models Homework")
    a = p.parse_args()
    run, out = Path(a.run), Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    c = json.loads((run / "config.json").read_text())
    m = json.loads((run / "metrics.json").read_text())
    manifest = json.loads((Path(a.data) / "manifest.json").read_text())
    timing = json.loads((run / "timing.json").read_text())
    rows = list(csv.DictReader((run / "history.csv").open()))
    h = {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
    selected = rows[m["checkpoint_epoch"]-1]
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 1.55), layout="constrained")
    for ax, train_key, val_key, title in zip(axes,
            ["train_elbo_loss", "train_bce", "train_kl"],
            ["val_elbo_loss", "val_bce", "val_kl"],
            ["Negative ELBO (beta = 1)", "Reconstruction BCE", "KL divergence"]):
        ax.plot(h["epoch"], h[train_key], color="#245a81", label="Train")
        ax.plot(h["epoch"], h[val_key], color="#b04a31", label="Validation")
        ax.axvline(m["checkpoint_epoch"], color="0.55", linestyle=":", linewidth=1)
        ax.set(title=title, xlabel="Epoch", ylabel="Nats / image")
        ax.grid(alpha=.15)
    axes[0].legend(frameon=False, fontsize=7)
    fig.savefig(out / "loss_curves.pdf"); plt.close(fig)
    for name in ["generated.png", "reconstructions.png"]:
        shutil.copyfile(run / name, out / name)
    n = {k: sum(v.values()) for k,v in manifest["counts"].items()}
    duplicates = len(manifest["duplicate_files_removed"])
    params = c["parameters"]
    text = r"""\documentclass{article}
\PassOptionsToPackage{numbers}{natbib}
\usepackage[preprint]{neurips_2025}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{amsmath,amssymb,booktabs,graphicx,url}
\usepackage[hidelinks]{hyperref}
\title{Generating Chest X-Rays with a Convolutional VAE}
\author{@AUTHOR@}
\begin{document}
\maketitle
\begin{abstract}
A PyTorch variational autoencoder (VAE) generates $64\times64$ chest radiographs.
After @EPOCHS@ epochs, Inception Score is @IS@ and torchvision-feature FID is @FID@.
Samples capture coarse chest structure but lack fine detail.
\end{abstract}

\section{Data and model}
Normal and pneumonia images from Kaggle [1] are pooled. SHA-256 deduplication removes
@DUPLICATES@ files, retaining @NTEST@ official test images. Training excludes 453 images
whose filename-derived patient IDs occur in test; 10\% of remaining shuffled groups
form validation (seed 42): @NTRAIN@ training and @NVAL@ validation images.
The 16 official validation images are unused. Preprocessing uses grayscale conversion, PIL bilinear
resize to $64\times64$, and scaling to $[0,1]$, without augmentation; aspect ratios can change.

The encoder uses four $4\times4$ convolutions (stride 2, padding 1; channels
$1,32,64,128,256$; SiLU), then linear heads for 32-dimensional $\mu$ and
$\log\sigma^2$. Reparameterization $z=\mu+\sigma\odot\epsilon$,
$\epsilon\sim\mathcal N(0,I)$, enables backpropagation [2]. A linear projection and
four transposed convolutions reverse these dimensions; sigmoid outputs intensities.
The model has @PARAMS@ parameters, without batch normalization.

\section{Training protocol and learning curves}
BCE sums over 4,096 pixels; KL sums over latent dimensions. Minibatch means minimize:
\vspace{-3pt}
\begin{equation}
\mathcal L_\beta=\mathbb E_{q_\phi(z|x)}[-\log p_\theta(x|z)]
+\frac{\beta}{2}\sum_{j=1}^{32}(\mu_j^2+\sigma_j^2-1-\log\sigma_j^2).
\end{equation}
BCE is a soft-target surrogate for continuous intensities. Settings: Adam learning
rate $10^{-3}$, moments $(0.9,0.999)$, batch size 64, zero weight decay, gradient-norm
limit 10, @EPOCHS@ epochs, seed 42, and warm-up $\beta=\min(1,e/5)$ at epoch $e$.
Validation uses fixed posterior noise and $\beta=1$; its minimum selects epoch
@BEST@ (@VALLOSS@ nats/image).
\begin{figure}[h]
\centering\includegraphics[width=\linewidth]{loss_curves.pdf}
\caption{Losses per image; the left panel uses $\beta=1$ even during warm-up.
Training values average across updates; dotted lines mark the selected checkpoint.}
\end{figure}
\newpage
\section{Generation and quantitative evaluation}
The selected decoder generates @NTEST@ prior samples from $z\sim\mathcal N(0,I)$
(seed 123), compared with all @NTEST@ test images using Ignite's FID/IS evaluator [3].
Both sets receive 8-bit conversion, RGB replication, PIL bilinear resize to
$299\times299$, and ImageNet normalization (an addition to the tutorial).
Frozen torchvision Inception-v3 (\texttt{IMAGENET1K\_V1}) supplies 2,048 pooled
features for the primary FID and 1,000 softmax probabilities for IS and the
Ignite-default FID representation. Neither FID is comparable to TensorFlow-weight
\texttt{pytorch-fid} scores. Generated images are sigmoid decoder means, with no visual selection.

\begin{table}[h]
\centering
\caption{Test results at epoch @BEST@. Losses: nats/image. Split IS reports mean
and standard deviation over ten subsets, not independently trained models.}
\begin{tabular}{lr@{\qquad}lr}
\toprule
Metric & Value & Metric & Value \\
\midrule
FID (2,048 pool features)$\downarrow$ & @FID@ & Test negative ELBO$\downarrow$ & @TESTLOSS@ \\
FID (1,000 probabilities)$\downarrow$ & @FIDPROB@ & Test reconstruction BCE$\downarrow$ & @TESTBCE@ \\
IS (all samples)$\uparrow$ & @IS@ & Test KL & @TESTKL@ \\
IS (10 splits)$\uparrow$ & @ISSPLIT@ & Real / generated count & @NTEST@ / @NTEST@ \\
\bottomrule
\end{tabular}
\end{table}
\begin{figure}[h]
\centering
\includegraphics[width=.76\linewidth]{generated.png}
\caption{The first 32 generated images from evaluation seed 123, in order.
All panels use the same $[0,1]$ display range.}
\end{figure}

\section{Discussion and limitations}
@DISCUSSION@
Filename grouping cannot verify clinical identity or near-duplicates.
With only @NTEST@ test images, the 2,048-feature covariance is rank deficient and
FID is affected by finite-sample bias. ImageNet-based IS cannot establish pathology
coverage or anatomical correctness. Only one training and sampling seed are evaluated.
Higher resolution, repeated runs, and radiograph-specific evaluation are useful next steps.

\small
\begin{thebibliography}{3}
\setlength{\itemsep}{0pt}
\bibitem{data} P. Mooney. \href{https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia}{\emph{Chest X-Ray Images (Pneumonia)}}. Kaggle.
\bibitem{vae} D. P. Kingma and M. Welling. Auto-Encoding Variational Bayes. ICLR, 2014.
\href{https://arxiv.org/abs/1312.6114}{arXiv:1312.6114}.
\bibitem{ignite} PyTorch-Ignite. \href{https://pytorch-ignite.ai/blog/gan-evaluation-with-fid-and-is/}{GAN evaluation using FID and IS}.
\end{thebibliography}
\end{document}
"""
    discussion = (run / "discussion.txt").read_text().strip()
    author = a.author.replace("\\", r"\textbackslash{}").replace("&", r"\&").replace("_", r"\_").replace("%", r"\%")
    values = {"AUTHOR": author, "EPOCHS": c["epochs"], "NTEST": n["test"], "NTRAIN": n["train"], "NVAL": n["val"],
              "DUPLICATES": duplicates, "PARAMS": f"{params:,}", "BEST": m["checkpoint_epoch"],
              "VALLOSS": f"{float(selected['val_elbo_loss']):.2f}",
              "MINUTES": f"{timing['training_seconds']/60:.1f}", "TORCH": c["torch"],
              "FID": f"{m['fid_pool2048']:.2f}", "FIDPROB": f"{m['fid_probability1000']:.5f}",
              "IS": f"{m['inception_score']:.3f}",
              "ISSPLIT": f"${m['is_10split_mean']:.3f}\\pm{m['is_10split_std']:.3f}$",
              "TESTLOSS": f"{m['test_elbo_loss']:.2f}", "TESTBCE": f"{m['test_bce']:.2f}",
              "TESTKL": f"{m['test_kl']:.2f}", "DISCUSSION": discussion}
    for key, value in values.items():
        text = text.replace("@"+key+"@", str(value))
    (out / "report.tex").write_text(text)
    print(out / "report.tex")


if __name__ == "__main__":
    main()
