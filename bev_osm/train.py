"""Single-device streaming training, validation-only selection and resumable updates."""

import argparse, json, random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from .data import RasterDataset, normalization, save_json, sha, CLASSES
from .codec import MapAutoencoder, reconstruction_loss
from .diffusion import Generator
from .metrics import Metrics, geometry_score


def move(b, device):
    return {
        k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
        for k, v in b.items()
    }


def loader(ds, batch, workers=0):
    return DataLoader(
        ds,
        batch_size=batch,
        shuffle=False,
        num_workers=workers,
        generator=torch.Generator().manual_seed(876),
        pin_memory=torch.cuda.is_available(),
    )


class StepBatches:
    def __init__(self, n, batch, start, steps, seed):
        self.n = n
        self.batch = batch
        self.start = start
        self.steps = steps
        self.seed = seed

    def __iter__(self):
        for step in range(self.start, self.steps):
            yield np.random.default_rng(self.seed + step).integers(
                self.n, size=self.batch
            ).tolist()

    def __len__(self):
        return self.steps - self.start


def make(kind, ae=None):
    if kind == "ae":
        return MapAutoencoder()
    net = Generator(kind, MapAutoencoder() if kind == "latent_x0" else None)
    if ae is not None:
        ck = torch.load(ae, map_location="cpu", weights_only=False)
        if ck["kind"] != "ae" or "latent_mean" not in ck:
            raise ValueError(
                "Expected completed AE best.pt with train-only latent statistics"
            )
        if not ck["ae_gate_pass"]:
            raise ValueError(
                "AE failed road/sidewalk/building IoU and boundary F1>=.95; improve reconstruction first"
            )
        net.codec.load_state_dict(ck["model"])
        net.latent_mean.copy_(ck["latent_mean"])
        net.latent_std.copy_(ck["latent_std"])
    return net


@torch.no_grad()
def validate(net, kind, ds, device, batch, workers, nfe):
    net.eval()
    m = Metrics(CLASSES)
    for b in loader(ds, batch, workers):
        b = move(b, device)
        p = net(b["target"]).sigmoid() if kind == "ae" else net.sample(b, nfe, 19317)
        mask = torch.ones_like(b["support"]) if kind == "ae" else b["eval_mask"]
        m.update(p.cpu().numpy(), b["target"].cpu().numpy(), mask.cpu().numpy())
    return m.result()


def atomic_checkpoint(path, payload):
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--kind", choices=["ae", "x0", "fm", "latent_x0"], required=True)
    ap.add_argument("--ae-checkpoint", type=Path)
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--boundary", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--validate-every", type=int, default=1000)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--val-nfe", type=int, default=20)
    ap.add_argument("--allow-external-codec", action="store_true")
    ap.add_argument(
        "--val-limit",
        type=int,
        default=0,
        help="Fixed evenly spaced validation subset;0=full validation",
    )
    ap.add_argument("--resume", type=Path)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    if min(args.steps, args.batch, args.validate_every, args.save_every) < 1:
        raise ValueError("Positive steps/batch/intervals required")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    manifest_hash = sha(args.manifest)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    ck = (
        torch.load(args.resume, map_location="cpu", weights_only=False)
        if args.resume
        else None
    )
    if ck:
        for key in ("kind", "batch", "lr", "boundary", "seed", "val_nfe", "val_limit"):
            if ck["config"][key] != config[key]:
                raise ValueError("Resume mismatch: " + key)
        if ck["manifest_sha256"] != manifest_hash:
            raise ValueError("Resume dataset manifest changed")
        if args.out.resolve() != args.resume.parent.resolve():
            raise ValueError("Resume in same run directory; best.pt stays paired")
    elif (args.out / "last.pt").exists() or (args.out / "config.json").exists():
        raise ValueError("Use a new output directory or --resume")
    if args.kind == "latent_x0" and not (args.ae_checkpoint or ck):
        raise ValueError("--ae-checkpoint required")
    if args.kind == "latent_x0" and not ck:
        ae_meta = torch.load(args.ae_checkpoint, map_location="cpu", weights_only=False)
        if (
            ae_meta.get("manifest_sha256") != manifest_hash
            and not args.allow_external_codec
        ):
            raise ValueError(
                "AE was not trained with this manifest; use --allow-external-codec only after auditing pretraining overlap"
            )
    args.out.mkdir(parents=True, exist_ok=True)
    norm = (
        ck["normalization"]
        if ck
        else (None if args.kind == "ae" else normalization(args.manifest))
    )
    train = RasterDataset(args.manifest, "train", norm, args.kind == "ae")
    val = RasterDataset(args.manifest, "val", norm, args.kind == "ae")
    if args.val_limit < 0:
        raise ValueError("val-limit must be nonnegative")
    if args.val_limit and args.val_limit < len(val):
        val = Subset(
            val, np.linspace(0, len(val) - 1, args.val_limit).astype(int).tolist()
        )
    net = make(args.kind, args.ae_checkpoint if not ck else None).to(device)
    opt = torch.optim.AdamW(
        (p for p in net.parameters() if p.requires_grad),
        lr=args.lr,
        weight_decay=0.0001,
    )
    start = 0
    best = -1.0
    selected = 0
    if ck:
        net.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        start = ck["step"]
        best = ck["best_score"]
        selected = ck["selected_step"]
        torch.set_rng_state(ck["rng"])
        if device.type == "cuda" and ck.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(ck["cuda_rng"])
        if device.type == "mps" and ck.get("mps_rng") is not None:
            torch.mps.set_rng_state(ck["mps_rng"])
    if args.steps <= start:
        raise ValueError("--steps must exceed checkpoint step")
    save_json(
        args.out / "config.json",
        {
            **config,
            "manifest_sha256": manifest_hash,
            "normalization": norm,
            "ae_sha256": (
                sha(args.ae_checkpoint)
                if args.ae_checkpoint
                else (ck.get("ae_sha256") if ck else None)
            ),
            "sampling": "uniform samples with replacement; differs from historical region-balanced pilot",
            "training_scope": "frozen cached MIA BEV; no auxiliary head; single device FP32",
            "validation_samples": len(val),
        },
    )
    dl = DataLoader(
        train,
        batch_sampler=StepBatches(len(train), args.batch, start, args.steps, args.seed),
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(987),
    )

    def payload(step):
        return dict(
            kind=args.kind,
            model={k: v.detach().cpu() for k, v in net.state_dict().items()},
            optimizer=opt.state_dict(),
            step=step,
            best_score=best,
            selected_step=selected,
            normalization=norm,
            manifest_sha256=manifest_hash,
            config=config,
            rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
            mps_rng=torch.mps.get_rng_state() if device.type == "mps" else None,
            ae_sha256=(
                sha(args.ae_checkpoint)
                if args.ae_checkpoint
                else (ck.get("ae_sha256") if ck else None)
            ),
        )

    for step, b in enumerate(dl, start + 1):
        net.train()
        b = move(b, device)
        opt.zero_grad(set_to_none=True)
        loss = (
            reconstruction_loss(net(b["target"]), b["target"])
            if args.kind == "ae"
            else net.loss(b, args.boundary)
        )
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5)
        opt.step()
        row = dict(step=step, loss=float(loss.detach().cpu()))
        if step % args.validate_every == 0 or step == args.steps:
            rows = validate(
                net, args.kind, val, device, args.batch, args.workers, args.val_nfe
            )
            score = geometry_score(rows)
            row.update(val_score=score, val_metrics=rows)
            if score > best:
                best = score
                selected = step
                p = payload(step)
                p["val_metrics"] = rows
                atomic_checkpoint(args.out / "best.pt", p)
        if step % args.save_every == 0 or step == args.steps:
            atomic_checkpoint(args.out / "last.pt", payload(step))
        if step == 1 or step % 100 == 0 or "val_score" in row:
            with (args.out / "history.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(
                json.dumps({k: v for k, v in row.items() if k != "val_metrics"}),
                flush=True,
            )
    if args.kind == "ae":
        bestck = torch.load(
            args.out / "best.pt", map_location="cpu", weights_only=False
        )
        net.load_state_dict(bestck["model"])
        net.eval()
        total = torch.zeros(12, dtype=torch.float64)
        squares = total.clone()
        count = 0
        with torch.no_grad():
            for b in loader(train, args.batch, args.workers):
                z = net.encode(b["target"].to(device)).cpu().double()
                total += z.sum((0, 2, 3))
                squares += z.square().sum((0, 2, 3))
                count += len(z) * z.shape[-2] * z.shape[-1]
        mean = total / count
        std = (squares / count - mean.square()).clamp_min(1e-6).sqrt()
        core = [
            r
            for r in bestck["val_metrics"]
            if r["class_name"] in ("road", "sidewalk", "building")
        ]
        passed = all(
            r["iou"] is not None
            and r["boundary_f1_0.5m"] is not None
            and r["iou"] >= 0.95
            and r["boundary_f1_0.5m"] >= 0.95
            for r in core
        )
        bestck.update(
            latent_mean=mean.float(), latent_std=std.float(), ae_gate_pass=passed
        )
        atomic_checkpoint(args.out / "best.pt", bestck)
        save_json(
            args.out / "ae_gate.json",
            dict(passed=passed, validation_metrics=core, selected_step=selected),
        )
    print(
        f'Finished. Validation-selected checkpoint: {args.out / "best.pt"}', flush=True
    )


if __name__ == "__main__":
    main()
