"""Lazy, one-sample-per-file dataset. Relative paths resolve against manifest."""

import hashlib, json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

CLASSES = ["road", "crossing", "sidewalk", "building", "terrain", "parking"]
GRID = {
    "shape": [100, 100],
    "meters_per_pixel": 0.5,
    "camera_rc": [100, 50],
    "forward": "up",
    "classes": CLASSES,
}


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    tmp.replace(path)


def records(path):
    path = Path(path).resolve()
    rows = [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
    if not rows:
        raise ValueError("Empty manifest")
    ids = set()
    regions = {k: set() for k in ("train", "val", "test")}
    for row in rows:
        if row["id"] in ids:
            raise ValueError("Duplicate id: " + row["id"])
        ids.add(row["id"])
        regions[row["split"]].add(row["region"])
        row["file"] = str((path.parent / row["file"]).resolve())
    if any(not v for v in regions.values()):
        raise ValueError("Require nonempty train/val/test")
    if any(
        regions[a] & regions[b]
        for a, b in [("train", "val"), ("train", "test"), ("val", "test")]
    ):
        raise ValueError("Region leakage across splits")
    meta = json.loads(path.with_suffix(".meta.json").read_text())
    if meta.get("prepare", {}).get("qc_only", False):
        raise ValueError("QC-only preparation cannot be used as a training/evaluation benchmark")
    if meta["grid"] != GRID:
        raise ValueError("Unsupported grid/classes; do not silently resize maps")
    return rows


def load_sample(path, with_bev=True):
    with np.load(path, allow_pickle=False) as a:
        keys = ["target", "support", "eval_mask"] + (["bev"] if with_bev else [])
        out = {k: np.array(a[k], dtype=np.float32) for k in keys}
    for k in ("target", "support", "eval_mask"):
        shape = (6, 100, 100) if k == "target" else (1, 100, 100)
        if out[k].shape != shape or not np.isin(out[k], [0, 1]).all():
            raise ValueError(f"Invalid {k}: {path}")
    if (out["eval_mask"] > out["support"]).any():
        raise ValueError("Eval mask must be subset of projection support")
    if with_bev and (
        out["bev"].shape != (128, 100, 100) or not np.isfinite(out["bev"]).all()
    ):
        raise ValueError("Invalid BEV features")
    return out


class RasterDataset(Dataset):
    def __init__(self, manifest, split, normalization=None, maps_only=False):
        self.rows = [r for r in records(manifest) if r["split"] == split]
        self.norm = normalization
        self.maps_only = maps_only

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        a = load_sample(r["file"], not self.maps_only)
        if not self.maps_only and self.norm is not None:
            mu = np.array(self.norm["mean"], np.float32)[:, None, None]
            sd = np.array(self.norm["std"], np.float32)[:, None, None]
            a["bev"] = np.clip((a["bev"] - mu) / sd, -8, 8) * a["support"]
        return {**{k: torch.from_numpy(v) for k, v in a.items()}, "id": r["id"]}


def normalization(manifest):
    total = np.zeros(128, np.float64)
    squares = total.copy()
    count = 0
    for r in records(manifest):
        if r["split"] != "train":
            continue
        a = load_sample(r["file"])
        x = a["bev"].astype(np.float64)
        v = a["support"]
        total += (x * v).sum((1, 2))
        squares += (x * x * v).sum((1, 2))
        count += v.sum()
    if count == 0:
        raise ValueError("No training support")
    mu = total / count
    sd = np.sqrt(np.maximum(squares / count - mu * mu, 0.01))
    return dict(mean=mu.tolist(), std=sd.tolist(), manifest_sha256=sha(manifest))
