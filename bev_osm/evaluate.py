"""Independent test entry; GT never enters a generator's inference path."""

import argparse, hashlib, json
from pathlib import Path
import numpy as np
import torch
from .data import RasterDataset, CLASSES, save_json, sha
from .train import make, move, loader
from .metrics import Metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--nfe", type=int, default=50)
    ap.add_argument("--seed", type=int, default=9137)
    ap.add_argument("--mia-mapper", action="store_true")
    ap.add_argument("--save-predictions", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    if (args.out / "metrics.json").exists():
        raise ValueError("Output exists; use new directory")
    if args.mia_mapper:
        from .mia_runtime import load_model

        net, provenance = load_model(args.checkpoint, args.threads)
        net.to(args.device).eval()
        kind = "mia_mapper"
        norm = None
    else:
        ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        kind = ck["kind"]
        norm = ck["normalization"]
        net = make(kind)
        net.load_state_dict(ck["model"])
        net.to(args.device).eval()
        provenance = {
            "training_manifest_sha256": ck["manifest_sha256"],
            "evaluation_manifest_sha256": sha(args.manifest),
            "different_manifest": ck["manifest_sha256"] != sha(args.manifest),
        }
    ds = RasterDataset(args.manifest, args.split, norm, kind == "ae")
    m = Metrics(CLASSES)
    args.out.mkdir(parents=True, exist_ok=True)
    ids = []
    if args.save_predictions:
        (args.out / "predictions").mkdir(exist_ok=True)
    with torch.inference_mode():
        for b in loader(ds, args.batch, args.workers):
            b = move(b, args.device)
            if kind == "ae":
                p = net(b["target"]).sigmoid()
                mask = torch.ones_like(b["support"])
            elif kind == "mia_mapper":
                p = net.decoder(b["bev"]).sigmoid()
                mask = b["eval_mask"]
            else:
                # Explicitly omit target/eval_mask from inference input.
                p = net.sample(
                    {k: b[k] for k in ("bev", "support", "id")}, args.nfe, args.seed
                )
                mask = b["eval_mask"]
            pp = p.cpu().numpy()
            m.update(pp, b["target"].cpu().numpy(), mask.cpu().numpy())
            ids += b["id"]
            if args.save_predictions:
                for sid, v in zip(b["id"], pp):
                    np.savez_compressed(
                        args.out
                        / "predictions"
                        / (hashlib.sha256(sid.encode()).hexdigest() + ".npz"),
                        id=sid,
                        continuous=v,
                        decoded=v >= 0.5,
                    )
    result = dict(
        kind=kind,
        scope=(
            "GT reconstruction diagnostic, NOT image generation"
            if kind == "ae"
            else "image-conditioned map evaluation"
        ),
        samples=len(ids),
        split=args.split,
        ids=ids,
        metrics=m.result(),
        checkpoint_sha256=sha(args.checkpoint),
        provenance=provenance,
        nfe=1 if kind in ("ae", "mia_mapper") else args.nfe,
        seed=args.seed,
        threshold=0.5,
    )
    save_json(args.out / "metrics.json", result)
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
