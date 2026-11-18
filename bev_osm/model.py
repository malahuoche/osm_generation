"""No-aux feature generator. Original non-aux initialization is preserved."""

import math
import torch
from torch import nn
from torch.nn import functional as F


def coordinates(x):
    # Actual cached MIA pixel centers, normalized by half-width / forward range.
    h, w = x.shape[-2:]
    if (h, w) != (100, 100):
        raise ValueError("This version preserves the audited100x100 metric grid")
    row, col = torch.meshgrid(
        torch.arange(h, device=x.device, dtype=x.dtype),
        torch.arange(w, device=x.device, dtype=x.dtype),
        indexing="ij",
    )
    return torch.stack(((0.5 * col - 25) / 25, (50 - 0.5 * row) / 50))[None].expand(
        len(x), -1, -1, -1
    )


class TimeEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer(
            "freq", torch.exp(-math.log(10000) * torch.arange(16) / 15)
        )
        self.mlp = nn.Sequential(nn.Linear(32, 64), nn.SiLU(), nn.Linear(64, 64))

    def forward(self, t):
        phase = t[:, None] * 1000 * self.freq[None]
        return self.mlp(torch.cat((phase.sin(), phase.cos()), -1))


class Residual(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.n1 = nn.GroupNorm(4, cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.n2 = nn.GroupNorm(4, cout)
        self.film = nn.Linear(64, cout * 2)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, time):
        a = self.n1(self.c1(x))
        scale, shift = self.film(time).chunk(2, 1)
        a = F.silu(a * (1 + scale[:, :, None, None]) + shift[:, :, None, None])
        return F.silu((self.n2(self.c2(a)) + self.skip(x)) / math.sqrt(2))


class SpatialAttention(nn.Module):
    """Explicit contiguous projections avoid MHA's strided linear MPS path."""

    def __init__(self):
        super().__init__()
        self.q = nn.Linear(32, 32)
        self.k = nn.Linear(32, 32)
        self.v = nn.Linear(32, 32)
        self.out = nn.Linear(32, 32)

    def forward(self, query, keys, padding):
        batch, n, _ = query.shape
        count = keys.shape[1]
        q = (
            self.q(query.contiguous())
            .reshape(batch, n, 4, 8)
            .transpose(1, 2)
            .contiguous()
        )
        k = (
            self.k(keys.contiguous())
            .reshape(batch, count, 4, 8)
            .transpose(1, 2)
            .contiguous()
        )
        v = (
            self.v(keys.contiguous())
            .reshape(batch, count, 4, 8)
            .transpose(1, 2)
            .contiguous()
        )
        mask = (~padding)[:, None, None, :]
        attended = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0
        )
        return self.out(attended.transpose(1, 2).contiguous().reshape(batch, n, 32))


class FeatureGenerator(nn.Module):
    """BEV pyramid, FiLM time blocks and spatial bottleneck cross-attention.

    No auxiliary head is retained. All skips come from image/state paths.
    """

    def __init__(self, channels=128, classes=6):
        super().__init__()
        self.classes = classes
        self.project = nn.Sequential(nn.Conv2d(channels, 16, 1), nn.SiLU())
        self.cond0 = nn.Sequential(
            nn.Conv2d(19, 16, 3, padding=1), nn.GroupNorm(4, 16), nn.SiLU()
        )
        self.cond1 = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(16, 24, 3, padding=1),
            nn.GroupNorm(4, 24),
            nn.SiLU(),
        )
        self.cond2 = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(24, 32, 3, padding=1),
            nn.GroupNorm(4, 32),
            nn.SiLU(),
        )
        self.aux = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1), nn.SiLU(), nn.Conv2d(16, classes, 1)
        )
        self.time = TimeEmbedding()
        self.a = Residual(classes + 19, 16)
        self.b = Residual(40, 24)
        self.c = Residual(56, 32)
        self.query_pos = nn.Conv2d(2, 32, 1)
        self.key_pos = nn.Conv2d(2, 32, 1)
        self.attn = SpatialAttention()
        self.query_norm = nn.LayerNorm(32)
        self.key_norm = nn.LayerNorm(32)
        self.up1 = Residual(80, 24)
        self.up0 = Residual(56, 16)
        self.out = nn.Conv2d(16, classes, 3, padding=1)
        # Preserve the original RNG draw order, then remove the unused head.
        # No auxiliary module, parameter, forward operation or loss remains.
        del self.aux

    def encode_condition(self, bev, support):
        xy = coordinates(bev)
        c0 = self.cond0(torch.cat((self.project(bev), support, xy), 1))
        c1 = self.cond1(c0)
        c2 = self.cond2(c1)
        # Keep25x25keys: exactposition match and avoids nondivisibleMPS pooling.
        pos = F.adaptive_avg_pool2d(xy, c2.shape[-2:])
        keys = c2 + self.key_pos(pos)
        keyvalid = F.adaptive_avg_pool2d(support, c2.shape[-2:]) > 0
        padding = ~keyvalid.flatten(1)
        keys = keys.flatten(2).transpose(1, 2)
        # An entirely unsupported image gets one zero token, not NaN attention.
        empty = padding.all(1)
        if empty.any():
            keys = keys.clone()
            padding = padding.clone()
            keys[empty] = 0
            padding[empty, 0] = False
        return dict(
            c0=c0,
            c1=c1,
            c2=c2,
            xy=xy,
            support=support,
            keys=keys,
            padding=padding,
            pos=pos,
        )

    def denoise(self, state, t, condition):
        q = condition
        time = self.time(t)
        a = self.a(torch.cat((state, q["c0"], q["support"], q["xy"]), 1), time)
        b = self.b(torch.cat((F.avg_pool2d(a, 2), q["c1"]), 1), time)
        c = self.c(torch.cat((F.avg_pool2d(b, 2), q["c2"]), 1), time)
        query = (c + self.query_pos(q["pos"])).flatten(2).transpose(1, 2)
        kv = self.key_norm(q["keys"])
        attended = self.attn(self.query_norm(query), kv, q["padding"])
        c = c + attended.transpose(1, 2).reshape_as(c)
        b = self.up1(
            torch.cat(
                (
                    F.interpolate(
                        c, size=b.shape[-2:], mode="bilinear", align_corners=False
                    ),
                    b,
                    q["c1"],
                ),
                1,
            ),
            time,
        )
        a = self.up0(
            torch.cat(
                (
                    F.interpolate(
                        b, size=a.shape[-2:], mode="bilinear", align_corners=False
                    ),
                    a,
                    q["c0"],
                ),
                1,
            ),
            time,
        )
        return self.out(a)

    def forward(self, bev, support, state, t):
        c = self.encode_condition(bev, support)
        return self.denoise(state, t, c)


def semantic_loss(logits, y, support):
    weight = support.expand_as(y)
    bce = (
        F.binary_cross_entropy_with_logits(logits, y, reduction="none") * weight
    ).sum() / weight.sum().clamp_min(1)
    p = logits.sigmoid()
    dice = 1 - (2 * (p * y * weight).sum((0, 2, 3)) + 1) / (
        (p * weight + y * weight).sum((0, 2, 3)) + 1
    )
    return bce + 0.5 * dice.mean()


def boundary_loss(endpoint, target, support, time_weight):
    """Differentiable finite-gradient error of signed continuous map, per meter.

    No argmax, threshold, output clamp or GT feature enters the forward model.
    Erode a1m support band and require BOTH adjacent pixels to be valid.
    """
    m = support
    m = 1 - F.max_pool2d(F.pad(1 - m, (2, 2, 2, 2), value=1), 5, stride=1)
    terms = []
    for dim in (-2, -1):
        dp = torch.diff(endpoint, dim=dim) / 0.5
        dg = torch.diff(target, dim=dim) / 0.5
        pair = (
            m[..., 1:, :] * m[..., :-1, :]
            if dim == -2
            else m[..., :, 1:] * m[..., :, :-1]
        )
        weight = pair.expand_as(dp)
        err = F.smooth_l1_loss(dp, dg, reduction="none", beta=0.5)
        per_sample = (err * weight).sum((1, 2, 3)) / weight.sum((1, 2, 3)).clamp_min(1)
        terms.append((per_sample * time_weight).mean())
    return sum(terms) / 2


def ddim_update(state, clean, alpha, previous_alpha):
    # Recompute epsilon from the very same bounded clean estimate.
    clean = clean.clamp(-1, 1)
    eps = (state - alpha.sqrt() * clean) / (1 - alpha).sqrt().clamp_min(1e-6)
    return previous_alpha.sqrt() * clean + (1 - previous_alpha).sqrt() * eps
