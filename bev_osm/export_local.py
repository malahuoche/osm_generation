"""Export existing native KITTI pilot to portable per-frame files (no model needed)."""

import argparse, hashlib, json
from pathlib import Path
import numpy as np
from .data import GRID, save_json, sha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument(
        "--features", type=Path, help="Override old machine-specific cache path"
    )
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    meta = json.loads((args.source / "metadata.json").read_text())
    a = np.load(args.source / "targets.npz")
    p = args.features or Path(meta["raw_feature_cache_path"])
    raw = np.load(p, mmap_mode="r")
    if (args.out / "manifest.jsonl").exists():
        raise ValueError("Output already exists")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "samples").mkdir(exist_ok=True)
    splits = {int(i): k for k, indices in meta["split"].items() for i in indices}
    rows = []
    for i, sid in enumerate(a["ids"].astype(str)):
        key = hashlib.sha256(sid.encode()).hexdigest()
        f = args.out / "samples" / (key + ".npz")
        np.savez_compressed(
            f,
            bev=raw[meta["raw_cache_indices"][i]],
            target=a["osm_full"][i].astype(bool),
            support=a["projection_support"][i].astype(bool),
            eval_mask=a["evaluation_visibility"][i].astype(bool),
        )
        rows.append(
            dict(
                id=sid,
                region=sid.split("/")[0],
                split=splits[i],
                file=str(f.relative_to(args.out)),
                file_sha256=sha(f),
            )
        )
    m = args.out / "manifest.jsonl"
    m.write_text("".join(json.dumps(r) + "\n" for r in rows))
    save_json(
        m.with_suffix(".meta.json"),
        dict(
            grid=GRID,
            source="historical native-MIA KITTI pilot",
            targets_sha256=sha(args.source / "targets.npz"),
            metadata_sha256=sha(args.source / "metadata.json"),
            note="No images or labels are committed to Git. Geographic separation was audited in original pilot.",
        ),
    )
    print(m)


if __name__ == "__main__":
    main()
