"""Ignite FID/IS, adapted from the assignment's GAN evaluation tutorial.

Uses explicit 2048-dimensional torchvision pool features for the main FID.
Also reports the tutorial-default 1000-probability representation for comparison.
Neither number should be mixed with TensorFlow/pytorch-fid benchmark scores.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from ignite.engine import Engine
from ignite.metrics import FID, InceptionScore
from ignite.metrics.gan.utils import InceptionModel
from PIL import Image
from torchvision.transforms import functional as TF
from torchvision.utils import save_image

from .data import load_split
from .model import VAE, loss_terms
from .train import device_for


def inception_input(batch):
    # PIL resizing follows the supplied tutorial. Identical processing for both sets.
    images = []
    for x in batch.cpu():
        im = TF.to_pil_image(x.clamp(0, 1)).convert("RGB")
        im = im.resize((299, 299), Image.Resampling.BILINEAR)
        images.append(TF.normalize(TF.to_tensor(im), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    return torch.stack(images)


def compute_metrics(fake_features, real_features, fake_probs, real_probs):
    # Identity feature extractors allow one shared Inception forward pass.
    evaluator = Engine(lambda engine, batch: batch)
    FID(num_features=2048, output_transform=lambda o: (o[0], o[1])).attach(evaluator, "fid_pool2048")
    FID(num_features=1000, output_transform=lambda o: (o[2], o[3])).attach(evaluator, "fid_probability1000")
    InceptionScore(num_features=1000, output_transform=lambda o: o[2]).attach(evaluator, "inception_score")
    batches = list(zip(fake_features.split(32), real_features.split(32), fake_probs.split(32), real_probs.split(32)))
    result = dict(evaluator.run(batches).metrics)
    scores = []
    for probs in torch.tensor_split(fake_probs, 10):
        metric = InceptionScore(num_features=1000)
        metric.update(probs)
        scores.append(metric.compute())
    result.update(is_10split_mean=float(np.mean(scores)), is_10split_std=float(np.std(scores)),
                  is_10split_values=scores)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/prepared")
    p.add_argument("--run", default="runs/vae")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--device", default="auto")
    a = p.parse_args()
    torch.set_num_threads(6)
    out = Path(a.run); device = device_for(a.device)
    ckpt = torch.load(out / "best.pt", map_location="cpu", weights_only=True)
    config = ckpt["config"]
    model = VAE(config["latent_dim"], config["base_channels"]).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()
    real = load_split(a.data, "test")
    rng = torch.Generator().manual_seed(a.seed)
    z = torch.randn(len(real), config["latent_dim"], generator=rng)
    t = time.perf_counter()
    with torch.inference_mode():
        fake = torch.cat([model.decode(b.to(device)).sigmoid().cpu() for b in z.split(a.batch_size)])
        save_image(fake[:32], out / "generated.png", nrow=8, padding=2)
        mu, _ = model.encode(real[:8].to(device))
        recon = model.decode(mu).sigmoid().cpu()
        save_image(torch.cat([real[:8], recon]), out / "reconstructions.png", nrow=8, padding=2)
        terms = np.zeros(3)
        eps_rng = torch.Generator().manual_seed(a.seed+1)
        for x in real.split(a.batch_size):
            x = x.to(device)
            eps = torch.randn(len(x), config["latent_dim"], generator=eps_rng).to(device)
            logits, mu, logvar = model(x, eps)
            terms += np.array([v.item() for v in loss_terms(logits, x, mu, logvar)])*len(x)
        terms /= len(real)
    np.savez_compressed(out / "generated_samples.npz", images=fake.numpy(), z=z.numpy())
    print(f"Evaluating {len(real)} real and {len(fake)} generated images on {device}", flush=True)
    net = InceptionModel(return_features=False, device=str(device))
    pooled = {}
    hook = net.model.avgpool.register_forward_hook(lambda module, inputs, output: pooled.update(value=output.flatten(1)))
    all_features, all_probs = [], []
    with torch.inference_mode():
        for name, images in [("real", real), ("generated", fake)]:
            fs, ps = [], []
            for i, batch in enumerate(images.split(a.batch_size)):
                probs = net(inception_input(batch).to(device))
                fs.append(pooled["value"].cpu()); ps.append(probs.cpu())
                if i % 10 == 0:
                    print(f"{name}: {min((i+1)*a.batch_size,len(images))}/{len(images)}", flush=True)
            all_features.append(torch.cat(fs)); all_probs.append(torch.cat(ps))
    hook.remove()
    # Save intermediate tensors so expensive feature extraction need not be repeated.
    torch.save(dict(real_features=all_features[0], fake_features=all_features[1],
                    real_probs=all_probs[0], fake_probs=all_probs[1]), out / "inception_features.pt")
    result = compute_metrics(all_features[1], all_features[0], all_probs[1], all_probs[0])
    result.update(n_real=len(real), n_generated=len(fake), seed=a.seed, checkpoint_epoch=ckpt["epoch"],
                  checkpoint_sha256=hashlib.sha256((out / "best.pt").read_bytes()).hexdigest(),
                  test_elbo_loss=float(terms[0]), test_bce=float(terms[1]), test_kl=float(terms[2]),
                  feature_model="torchvision Inception_V3_Weights.IMAGENET1K_V1",
                  preprocessing="64x64 grayscale -> uint8 RGB -> PIL bilinear 299x299 -> ImageNet mean/std",
                  fid_note="Ignite FID with 2048 pool features; not TensorFlow/pytorch-fid weights",
                  small_sample_note="Test count is below feature dimension; covariance is rank deficient and FID is sample-size biased",
                  seconds=time.perf_counter()-t)
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
