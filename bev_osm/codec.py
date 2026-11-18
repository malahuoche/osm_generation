"""Continuous map AE; no GT skip paths or VQ codebook."""

import torch
from torch import nn
from torch.nn import functional as F


def layer(cin, cout, kernel=3, stride=1, transpose=False):
    conv = nn.ConvTranspose2d if transpose else nn.Conv2d
    return nn.Sequential(
        conv(cin, cout, kernel, stride=stride, padding=1),
        nn.GroupNorm(4, cout),
        nn.SiLU(),
    )


class MapAutoencoder(nn.Module):
    def __init__(self, classes=6, latent_channels=12):
        super().__init__()
        self.encoder = nn.Sequential(
            layer(classes, 16),
            layer(16, 32, 4, 2),
            nn.Conv2d(32, latent_channels, 3, padding=1),
        )
        self.decoder = nn.Sequential(
            layer(latent_channels, 32),
            layer(32, 16, 4, 2, True),
            layer(16, 16),
            nn.Conv2d(16, classes, 1),
        )

    def encode(self, maps):
        return self.encoder(maps)

    def decode(self, latent):
        return self.decoder(latent)

    def forward(self, maps):
        return self.decode(self.encode(maps))

    def encode_normalized(self, maps, mean, std):
        return (self.encode(maps) - mean[None, :, None, None]) / std[
            None, :, None, None
        ]

    def decode_normalized(self, latent, mean, std):
        return self.decode(
            latent * std[None, :, None, None] + mean[None, :, None, None]
        )


def reconstruction_loss(logits, target):
    prob = logits.sigmoid()
    bce = F.binary_cross_entropy_with_logits(logits, target)
    dice = 1 - (2 * (prob * target).sum((0, 2, 3)) + 1) / (
        (prob + target).sum((0, 2, 3)) + 1
    )
    # Finite differences are differentiable; no hard threshold or argmax in loss.
    edge = 0.5 * (
        F.l1_loss(
            prob[:, :, 1:] - prob[:, :, :-1], target[:, :, 1:] - target[:, :, :-1]
        )
        + F.l1_loss(
            prob[:, :, :, 1:] - prob[:, :, :, :-1],
            target[:, :, :, 1:] - target[:, :, :, :-1],
        )
    )
    return bce + 0.5 * dice.mean() + 0.1 * edge
