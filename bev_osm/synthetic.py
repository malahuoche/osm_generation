"""Tiny synthetic data ONLY for installation/CLI tests; not scientific results."""

import argparse, json
from pathlib import Path
import numpy as np
from .data import GRID, save_json, sha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    if (args.out / "manifest.jsonl").exists():
        raise ValueError("Output exists")
    rng = np.random.default_rng(7)
    for i, split in enumerate(["train"] * 4 + ["val"] * 2 + ["test"] * 2):
        y = np.zeros((6, 100, 100), bool)
        y[0, :, 40:60] = 1
        y[3, 15:55, 65:85] = 1
        y[2, :, 35:40] = 1
        bev = rng.normal(0, 0.2, (128, 100, 100)).astype(np.float16)
        bev[:6] += y.astype(np.float16)
        s = np.ones((1, 100, 100), bool)
        file = args.out / f"synthetic_{i}.npz"
        np.savez_compressed(file, bev=bev, target=y, support=s, eval_mask=s)
        rows.append(
            dict(
                id=f"SYNTHETIC_ONLY_{i}",
                split=split,
                region=split,
                file=file.name,
                file_sha256=sha(file),
            )
        )
    p = args.out / "manifest.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    save_json(
        p.with_suffix(".meta.json"),
        dict(grid=GRID, source="SYNTHETIC SOFTWARE TEST ONLY"),
    )


if __name__ == "__main__":
    main()
