import torch
from torch import nn
from torch.nn import functional as F


class VAE(nn.Module):
    """64 x 64 grayscale VAE; decoder returns Bernoulli logits."""

    def __init__(self, latent_dim=32, base_channels=32):
        super().__init__()
        self.latent_dim = latent_dim
        b = base_channels
        layers = []
        for cin, cout in zip([1, b, 2*b, 4*b], [b, 2*b, 4*b, 8*b]):
            layers += [nn.Conv2d(cin, cout, 4, 2, 1), nn.SiLU()]
        self.encoder = nn.Sequential(*layers, nn.Flatten())
        self.mu = nn.Linear(8*b*4*4, latent_dim)
        self.logvar = nn.Linear(8*b*4*4, latent_dim)
        self.project = nn.Linear(latent_dim, 8*b*4*4)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (8*b, 4, 4)),
            nn.ConvTranspose2d(8*b, 4*b, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(4*b, 2*b, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(2*b, b, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(b, 1, 4, 2, 1),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)

    def decode(self, z):
        return self.decoder(self.project(z))

    def forward(self, x, eps=None):
        mu, logvar = self.encode(x)
        if eps is None:
            eps = torch.randn_like(mu)
        z = mu + torch.exp(0.5 * logvar) * eps
        return self.decode(z), mu, logvar


def loss_terms(logits, x, mu, logvar, beta=1.0):
    """Sum over pixels/latent dimensions, then mean over examples."""
    recon = F.binary_cross_entropy_with_logits(logits, x, reduction="none").flatten(1).sum(1)
    kl = 0.5 * (mu.square() + logvar.exp() - 1 - logvar).sum(1)
    return (recon + beta*kl).mean(), recon.mean(), kl.mean()
