"""Prepare portable frozen-BEV sample files from calibrated RGB + native map pairs."""

import argparse, hashlib, json, os
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from .data import GRID, save_json, sha, load_sample


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--mia-checkpoint", required=True, type=Path)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--flood-eval-mask", action="store_true")
    args = ap.parse_args()
    pairs = [json.loads(s) for s in args.pairs.read_text().splitlines() if s.strip()]
    regions = {k: set() for k in ("train", "val", "test")}
    ids = set()
    for row in pairs:
        if row["id"] in ids:
            raise ValueError("Duplicate sample id")
        ids.add(row["id"])
        regions[row["split"]].add(row["region"])
    if any(not v for v in regions.values()):
        raise ValueError("Require nonempty train/val/test before extracting features")
    if any(
        regions[a] & regions[b]
        for a, b in [("train", "val"), ("train", "test"), ("val", "test")]
    ):
        raise ValueError("Region overlap in pairs")
    from .mia_runtime import load_model, preprocess, read_array, GT_CHANNELS

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "samples").mkdir(exist_ok=True)
    config = dict(
        pairs_sha256=sha(args.pairs),
        checkpoint_sha256=sha(args.mia_checkpoint),
        flood_eval_mask=args.flood_eval_mask,
    )
    done = out / "prepare.json"
    if done.exists() and json.loads(done.read_text()) != config:
        raise ValueError("Preparation settings changed; use fresh output")
    if (out / "manifest.jsonl").exists() and not args.resume:
        raise ValueError("Existing manifest; use --resume")
    save_json(done, config)
    net, prov = load_model(args.mia_checkpoint, args.threads)
    net.to(args.device).eval().requires_grad_(False)
    seen = set()
    rows = []
    resolve = lambda p: (args.pairs.resolve().parent / p).resolve()
    for i, r in enumerate(pairs):
        if r["id"] in seen:
            raise ValueError("Duplicate sample id")
        seen.add(r["id"])
        key = hashlib.sha256(r["id"].encode()).hexdigest()
        file = out / "samples" / f"{key}.npz"
        if not (args.resume and file.exists()):
            image = np.array(Image.open(resolve(r["image"])).convert("RGB"))
            batch = preprocess(image, r["camera"], r["roll_pitch_degrees"])
            batch = {k: v.to(args.device) for k, v in batch.items()}
            with torch.inference_mode():
                f, cam = net.image_encoder(batch)
                scales = net.scale_classifier(f.moveaxis(1, -1))
                polar = net.projection_polar(f, scales, cam)
                bev, valid, _ = net.projection_bev(polar.float(), None, cam.float())
            support = valid[0, :, :-1].cpu().numpy()[None].astype(bool)
            raw = read_array(resolve(r["target"]))
            if r["target_format"] == "mia_native":
                if raw.ndim != 3 or raw.shape[-1] < 8:
                    raise ValueError("Expected native HWC>=8 mask")
                cy, cx = raw.shape[0] // 2, raw.shape[1] // 2
                target = raw[cy - 100 : cy, cx - 50 : cx + 50][
                    :, :, GT_CHANNELS
                ].transpose(2, 0, 1)
            elif r["target_format"] == "six_chw":
                target = raw
            else:
                raise ValueError("Unknown target_format")
            evaluation = support.copy()
            if args.flood_eval_mask:
                flood = read_array(resolve(r["flood"]))
                cy, cx = np.array(flood.shape) // 2
                crop = flood[cy - 100 : cy, cx - 50 : cx + 50]
                evaluation &= (crop == crop.min())[None]
            tmp = file.with_suffix(".tmp")
            with tmp.open("wb") as stream:
                np.savez_compressed(
                    stream,
                    bev=bev[0, :, :, :-1].cpu().numpy().astype(np.float16),
                    target=target,
                    support=support,
                    eval_mask=evaluation,
                )
            load_sample(tmp)
            tmp.replace(file)
        else:
            load_sample(file)
        rows.append(
            dict(
                id=r["id"],
                split=r["split"],
                region=r["region"],
                file=str(file.relative_to(out)),
                file_sha256=sha(file),
            )
        )
        if (i + 1) % 100 == 0:
            print(f"Prepared {i+1}/{len(pairs)}", flush=True)
    manifest = out / "manifest.jsonl"
    temp = manifest.with_suffix(".tmp")
    temp.write_text("".join(json.dumps(r) + "\n" for r in rows))
    temp.replace(manifest)
    save_json(
        manifest.with_suffix(".meta.json"),
        dict(
            grid=GRID,
            source=prov,
            prepare=config,
            eval_mask=(
                "projection support intersected with GT flood mask (evaluation only)"
                if args.flood_eval_mask
                else "projection support, NOT physical visibility"
            ),
        ),
    )
    from .data import records

    records(manifest)
    print(manifest)


if __name__ == "__main__":
    main()
