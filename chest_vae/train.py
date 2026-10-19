import argparse
import csv
import json
import platform
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from .data import load_split
from .model import VAE, loss_terms


def device_for(name):
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/prepared")
    p.add_argument("--output", default="runs/vae")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--latent-dim", type=int, default=32)
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    a = p.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    torch.set_num_threads(6)
    device = device_for(a.device)
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    if (out / "history.csv").exists():
        raise FileExistsError("Choose a new --output to preserve existing experiment results")
    train, val = load_split(a.data, "train"), load_split(a.data, "val")
    loader = DataLoader(train, batch_size=a.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(a.seed), num_workers=0)
    model = VAE(a.latent_dim, a.base_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=a.lr)
    config = vars(a) | {"device_used": str(device), "torch": str(torch.__version__),
                        "python": platform.python_version(), "platform": platform.platform(),
                        "parameters": sum(p.numel() for p in model.parameters()),
                        "n_train": len(train), "n_val": len(val)}
    (out / "config.json").write_text(json.dumps(config, indent=2))
    print(json.dumps(config), flush=True)
    fixed_z = torch.randn(32, a.latent_dim, generator=torch.Generator().manual_seed(2026)).to(device)
    best = float("inf"); started = time.perf_counter()
    fields = ["epoch", "beta", "train_objective", "train_bce", "train_kl", "train_elbo_loss",
              "val_elbo_loss", "val_bce", "val_kl", "seconds"]
    with (out / "history.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for epoch in range(1, a.epochs+1):
            t = time.perf_counter()
            beta = min(1.0, epoch / max(1, a.warmup))
            model.train(); sums = np.zeros(3)
            for x in loader:
                x = x.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits, mu, logvar = model(x)
                loss, bce, kl = loss_terms(logits, x, mu, logvar, beta=beta)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training objective")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step()
                sums += np.array([loss.item(), bce.item(), kl.item()])*len(x)
            sums /= len(train)
            model.eval(); vs = np.zeros(3)
            # Fixed Monte Carlo noise makes validation curves comparable across epochs.
            eps_rng = torch.Generator().manual_seed(1000+a.seed)
            with torch.no_grad():
                for x in val.split(a.batch_size):
                    x = x.to(device)
                    eps = torch.randn(len(x), a.latent_dim, generator=eps_rng).to(device)
                    logits, mu, logvar = model(x, eps)
                    values = loss_terms(logits, x, mu, logvar)
                    vs += np.array([v.item() for v in values])*len(x)
            vs /= len(val)
            row = dict(zip(fields, [epoch, beta, *sums, sums[1]+sums[2], *vs, time.perf_counter()-t]))
            writer.writerow(row); f.flush()
            print(json.dumps(row), flush=True)
            checkpoint = {"model": {k: v.detach().cpu() for k,v in model.state_dict().items()},
                          "config": config, "epoch": epoch, "val_elbo_loss": float(vs[0])}
            if vs[0] < best:
                best = vs[0]; torch.save(checkpoint, out / "best.pt")
            if epoch == a.epochs:
                torch.save(checkpoint, out / "last.pt")
            if epoch == 1 or epoch % 5 == 0:
                with torch.no_grad():
                    save_image(model.decode(fixed_z).sigmoid().cpu(), out / f"samples_epoch_{epoch:03d}.png", nrow=8)
    (out / "timing.json").write_text(json.dumps({"training_seconds": time.perf_counter()-started}))


if __name__ == "__main__":
    main()
