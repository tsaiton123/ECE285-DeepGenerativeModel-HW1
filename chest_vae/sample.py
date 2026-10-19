"""Generate new prior samples from a saved VAE, without the training dataset."""
import argparse
from pathlib import Path
import torch
from torchvision.utils import save_image
from .model import VAE
from .train import device_for


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="runs/vae/best.pt")
    p.add_argument("--output", default="samples.png")
    p.add_argument("--count", type=int, default=32)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--device", default="auto")
    a = p.parse_args()
    if a.count < 1:
        p.error("--count must be positive")
    ckpt = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
    c = ckpt["config"]; device = device_for(a.device)
    model = VAE(c["latent_dim"], c["base_channels"]).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()
    z = torch.randn(a.count, c["latent_dim"], generator=torch.Generator().manual_seed(a.seed))
    with torch.inference_mode():
        images = torch.cat([model.decode(b.to(device)).sigmoid().cpu() for b in z.split(64)])
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    save_image(images, a.output, nrow=8, padding=2)


if __name__ == "__main__":
    main()
