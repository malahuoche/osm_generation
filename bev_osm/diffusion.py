import hashlib, math
import torch
from torch import nn
from torch.nn import functional as F
from .model import FeatureGenerator, boundary_loss, ddim_update
from .codec import MapAutoencoder


def schedule():
    f = torch.cos(
        ((torch.arange(1001, dtype=torch.float64) / 1000 + 0.008) / 1.008) * math.pi / 2
    ).square()
    f = f / f[0]
    return (1 - (1 - f[1:] / f[:-1]).clamp(0, 0.999)).cumprod(0).float()


class Generator(nn.Module):
    def __init__(self, kind, codec=None):
        super().__init__()
        self.kind = kind
        self.net = FeatureGenerator(128, 12 if kind == "latent_x0" else 6)
        self.codec = codec
        if codec is not None:
            codec.requires_grad_(False).eval()
        self.register_buffer("latent_mean", torch.zeros(12))
        self.register_buffer("latent_std", torch.ones(12))

    def denoise(self, z, t, condition):
        if self.kind == "latent_x0":
            return F.avg_pool2d(
                self.net.denoise(
                    F.interpolate(
                        z, size=(100, 100), mode="bilinear", align_corners=False
                    ),
                    t,
                    condition,
                ),
                2,
            )
        return self.net.denoise(z, t, condition)

    def decode(self, z):
        if self.kind == "latent_x0":
            return self.codec.decode_normalized(
                z, self.latent_mean, self.latent_std
            ).sigmoid()
        return (z + 1) / 2

    def loss(self, b, boundary=0.1):
        y = b["target"]
        x = b["bev"]
        s = b["support"]
        with torch.no_grad():
            target = (
                self.codec.encode_normalized(y, self.latent_mean, self.latent_std)
                if self.kind == "latent_x0"
                else 2 * y - 1
            )
        noise = torch.randn_like(target)
        n = len(x)
        if self.kind == "fm":
            t = torch.rand(n, device=x.device)
            a = t[:, None, None, None]
            z = (1 - a) * noise + a * target
            truth = target - noise
            weight = t.square()
        else:
            ti = torch.randint(1000, (n,), device=x.device)
            t = ti.float() / 999
            a = schedule().to(x.device)[ti, None, None, None]
            z = a.sqrt() * target + (1 - a).sqrt() * noise
            truth = target
            weight = a.flatten()
        pred = self.denoise(z, t, self.net.encode_condition(x, s))
        gen = F.mse_loss(pred, truth)
        endpoint = z + (1 - a) * pred if self.kind == "fm" else pred
        signed = 2 * self.decode(endpoint) - 1 if self.kind == "latent_x0" else endpoint
        edge = boundary_loss(
            signed[:, [0, 2, 3]], (2 * y - 1)[:, [0, 2, 3]], b["eval_mask"], weight
        )
        return gen + boundary * edge

    @torch.no_grad()
    def sample(self, b, nfe=50, seed=9137):
        if not 1 <= nfe <= 1000:
            raise ValueError("NFE must be1..1000")
        x = b["bev"]
        shape = (12, 50, 50) if self.kind == "latent_x0" else (6, 100, 100)
        # Stable per-ID noise: independent of batch size/order and Python hash seed.
        seeds = [
            int.from_bytes(
                hashlib.sha256(f"{seed}:{sid}".encode()).digest()[:8], "little"
            )
            % (2**63 - 1)
            for sid in b["id"]
        ]
        z = torch.stack(
            [
                torch.randn(shape, generator=torch.Generator().manual_seed(v))
                for v in seeds
            ]
        ).to(x.device)
        c = self.net.encode_condition(x, b["support"])
        if self.kind == "fm":
            for j in range(nfe):
                z = z + self.denoise(z, x.new_full((len(x),), j / nfe), c) / nfe
        else:
            abar = schedule().to(x.device)
            steps = torch.linspace(999, 0, nfe).round().long().tolist()
            for j, step in enumerate(steps):
                clean = self.denoise(z, x.new_full((len(x),), step / 999), c)
                a = abar[step]
                prev = abar[steps[j + 1]] if j + 1 < len(steps) else z.new_tensor(1.0)
                if self.kind == "latent_x0":
                    eps = (z - a.sqrt() * clean) / (1 - a).sqrt().clamp_min(1e-6)
                    z = prev.sqrt() * clean + (1 - prev).sqrt() * eps
                else:
                    z = ddim_update(z, clean, a, prev)
        return self.decode(z)
