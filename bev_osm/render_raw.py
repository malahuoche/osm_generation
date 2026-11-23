"""Render calibrated camera geometry with the native MIA renderer, not a custom style."""

import argparse, hashlib, importlib, json, os, sys
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from pyproj import CRS, Transformer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--geometry", type=Path, required=True)
    p.add_argument("--osm", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--region", required=True)
    p.add_argument("--split", choices=["train", "val", "test"], default="train")
    a = p.parse_args()
    from .native_renderer import render_at_points

    root = ET.parse(a.osm).getroot()
    if root.tag != "osm" or root.find("remark") is not None:
        raise ValueError("Invalid OSM XML")
    nodes = {e.get("id") for e in root.findall("node")}
    missing = [
        n.get("ref")
        for w in root.findall("way")
        for n in w.findall("nd")
        if n.get("ref") not in nodes
    ]
    if missing:
        raise ValueError("OSM contains incomplete way geometry")
    elements = {(e.tag, e.get("id")) for e in root}
    for rel in root.findall("relation"):
        if any(
            t.get("k") == "type" and t.get("v") == "multipolygon"
            for t in rel.findall("tag")
        ):
            if any(
                (m.get("type"), m.get("ref")) not in elements
                for m in rel.findall("member")
            ):
                raise ValueError(
                    "Incomplete multipolygon; fetch full relation before rendering"
                )
    geometry = json.loads(a.geometry.read_text())
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "maps").mkdir(exist_ok=True)
    rows = []
    reports = []
    for sample in geometry["samples"]:
        lat, lon = sample["latitude"], sample["longitude"]
        # Per-frame true ENU tangent plane; avoids map-projection convergence/scale assumptions.
        pipeline = f'+proj=pipeline +step +proj=cart +ellps=WGS84 +step +proj=topocentric +ellps=WGS84 +lat_0={lat} +lon_0={lon} +h_0={sample["altitude"]}'
        enu = Transformer.from_pipeline(pipeline)
        rr, cc = np.indices((100, 100))
        xz = np.stack([cc * 0.5 - 25, 50 - rr * 0.5], -1)
        level = np.array(sample["R_world_level_camera"])
        offset = np.array(sample["camera_offset_enu_m"])
        world = offset + xz[..., 0, None] * level[:, 0] + xz[..., 1, None] * level[:, 2]
        llh = enu.transform(
            world[..., 0], world[..., 1], world[..., 2], direction="INVERSE"
        )
        points = np.stack([llh[1], llh[0]], -1)
        corners = offset + np.array(
            [[-100, -100, 0], [-100, 100, 0], [100, -100, 0], [100, 100, 0]]
        )
        cl = enu.transform(*corners.T, direction="INVERSE")
        bbox = [min(cl[0]), min(cl[1]), max(cl[0]), max(cl[1])]
        result = render_at_points(
            a.osm, bbox, sample["heading_degrees"], points, mpp=0.5
        )
        key = sample["id"].replace("/", "__")
        target = a.out / "maps" / f"{key}.npz"
        np.savez_compressed(target, result["layers"][[0, 1, 2, 4, 6, 7]].astype(bool))
        rows.append(
            dict(
                id=sample["id"],
                region=a.region,
                split=a.split,
                image=sample["image"],
                target=str(target.resolve()),
                target_format="six_chw",
                camera=sample["camera"],
                roll_pitch_degrees=sample["roll_pitch_degrees"],
            )
        )
        reports.append(
            dict(
                id=sample["id"],
                renderer=result["metadata"],
                positive_pixels=result["layers"].sum((1, 2)).tolist(),
            )
        )
        print(f'Rendered {len(rows)}/{len(geometry["samples"])}', flush=True)
    (a.out / "pairs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (a.out / "render-audit.json").write_text(json.dumps(reports, indent=2) + "\n")


if __name__ == "__main__":
    main()
