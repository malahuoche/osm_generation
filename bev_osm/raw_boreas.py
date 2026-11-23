"""Boreas rectified camera + camera_poses + GPS geographic anchors; no sensor point clouds."""

import argparse, csv, json
from pathlib import Path
import numpy as np
from PIL import Image
from .raw_kitti import level_camera


def enu_basis(lat, lon):
    s, c = np.sin, np.cos
    return np.array(
        [
            [-s(lon), -s(lat) * c(lon), c(lat) * c(lon)],
            [c(lon), -s(lat) * s(lon), c(lat) * s(lon)],
            [0, c(lat), s(lat)],
        ]
    )


def boreas_rotation(r, p, y):
    # pyboreas uses passive elementary rotations: roll(r) @ pitch(p) @ yaw(y).
    from scipy.spatial.transform import Rotation

    return Rotation.from_euler("XYZ", [-r, -p, -y]).as_matrix()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sequence-dir", type=Path, required=True)
    p.add_argument("--gps", type=Path, required=True)
    p.add_argument(
        "--reference-gps",
        type=Path,
        required=True,
        help="GPS file of boreas-2020-11-26-13-58; first row anchors shared ENU orientation",
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    seq = a.sequence_dir.resolve()

    def read(path):
        with path.open() as f:
            return list(csv.DictReader(f))

    with a.reference_gps.open() as f:
        ref = next(csv.DictReader(f))
    with a.gps.open() as f:
        header = next(csv.reader(f))
    required = ["GPSTime", "easting", "northing", "altitude", "latitude", "longitude"]
    values = np.loadtxt(
        a.gps,
        delimiter=",",
        skiprows=1,
        usecols=[header.index(k) for k in required],
        ndmin=2,
    )
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite GPS fields")
    # GPS lat/lon are radians, GPSTime is seconds; camera filenames are integer microseconds.
    t = values[:, 0]
    xyz = values[:, 1:4]
    ll = values[:, 4:6]
    if (
        np.any(np.diff(t) <= 0)
        or np.max(abs(ll[:, 0])) > np.pi / 2
        or np.max(abs(ll[:, 1])) > np.pi
    ):
        raise ValueError("Invalid GPS time or radian lat/lon")
    if np.linalg.norm([float(ref["easting"]), float(ref["northing"])]) > 0.1:
        raise ValueError("Reference must begin at the original shared ENU origin")
    basis0 = enu_basis(float(ref["latitude"]), float(ref["longitude"]))
    poses = {r["GPSTime"]: r for r in read(seq / "applanix/camera_poses.csv")}
    P = np.loadtxt(seq / "calib/P_camera.txt")
    if P.shape == (4, 4):
        np.testing.assert_allclose(P[3], [0, 0, 0, 1])
        P = P[:3]
    K = P[:3, :3]
    if P.shape != (3, 4) or np.max(abs(P[:, 3])) > 1e-8:
        raise ValueError("Unexpected noncentral camera P")
    rows = []
    for image in sorted((seq / "camera").glob("*.png")):
        if a.limit and len(rows) >= a.limit:
            break
        if image.stem not in poses:
            raise ValueError("No exact camera timestamp match")
        r = poses[image.stem]
        tm = int(image.stem) / 1e6
        idx = np.searchsorted(t, tm)
        if idx == 0 or idx == len(t) or t[idx] - t[idx - 1] > 0.02:
            raise ValueError("GPS does not bracket camera time within 20 ms")
        w = (tm - t[idx - 1]) / (t[idx] - t[idx - 1])
        position = xyz[idx - 1] * (1 - w) + xyz[idx] * w
        latlon = ll[idx - 1] * (1 - w) + ll[idx] * w
        local = enu_basis(*latlon).T @ basis0
        camera_xyz = np.array(
            [float(r[k]) for k in ["easting", "northing", "altitude"]]
        )
        offset = local @ (camera_xyz - position)
        R = local @ boreas_rotation(
            *[float(r[k]) for k in ["roll", "pitch", "heading"]]
        )
        level, rp, error = level_camera(R)
        width, height = Image.open(image).size
        rows.append(
            dict(
                id=f"{seq.name}/{image.stem}",
                image=str(image),
                latitude=float(np.degrees(latlon[0])),
                longitude=float(np.degrees(latlon[1])),
                altitude=0.0,
                camera_offset_enu_m=offset.tolist(),
                R_world_camera=R.tolist(),
                R_world_level_camera=level.tolist(),
                roll_pitch_degrees=rp.tolist(),
                heading_degrees=float(
                    np.degrees(np.arctan2(level[0, 2], level[1, 2])) % 360
                ),
                camera=dict(
                    model="PINHOLE",
                    width=width,
                    height=height,
                    params=[K[0, 0], K[1, 1], K[0, 2] + 0.5, K[1, 2] + 0.5],
                ),
                K_opencv=K.tolist(),
                gps_bracket_seconds=float(t[idx] - t[idx - 1]),
                rotation_residual=error,
            )
        )
    if not rows:
        raise ValueError("No matching images")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(
        json.dumps(
            dict(
                sequence=seq.name,
                samples=rows,
                no_lidar_measurements=True,
                georeference="GPS latitude/longitude interpolation, shared ENU converted to local tangent orientation; altitude zero is a mapping datum assumption, not measured absolute elevation",
                camera_convention="P_camera rectified; no distortion correction; +0.5 compensates Camera.from_dict",
            ),
            indent=2,
        )
        + "\n"
    )
    print(f"{len(rows)} camera poses: {a.out}")


if __name__ == "__main__":
    main()
