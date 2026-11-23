"""KITTI sync image_02/OXTS -> geographic camera geometry, no point clouds."""

import argparse, json, hashlib
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation


def calibration(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        try:
            out[k] = np.array([float(x) for x in v.split()])
        except ValueError:
            pass
    return out


def rigid(c):
    t = np.eye(4)
    t[:3, :3] = c["R"].reshape(3, 3)
    t[:3, 3] = c["T"]
    return t


def camera_geometry(root, date):
    c = calibration(root / date / "calib_cam_to_cam.txt")
    p = c["P_rect_02"].reshape(3, 4)
    k = p[:, :3]
    rect = np.eye(4)
    rect[:3, :3] = c["R_rect_00"].reshape(3, 3)
    shift = np.eye(4)
    shift[:3, 3] = np.linalg.solve(k, p[:, 3])
    chain = (
        shift
        @ rect
        @ rigid(calibration(root / date / "calib_velo_to_cam.txt"))
        @ rigid(calibration(root / date / "calib_imu_to_velo.txt"))
    )
    return k, chain, c["S_rect_02"].astype(int)


def level_camera(R):
    forward = R[:, 2].copy()
    forward[2] = 0
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 0, 1])
    level = np.column_stack([right, [0, 0, -1], forward])
    correction = R.T @ level
    roll = np.arctan2(correction[1, 0], correction[0, 0])
    pitch = np.arctan2(correction[2, 1], correction[2, 2])
    reconstructed = Rotation.from_euler("ZX", [roll, pitch]).as_matrix()
    # Published calibration matrices have finite decimal precision.
    if np.max(abs(correction - reconstructed)) > 1e-5:
        raise ValueError("Gravity rotation cannot be represented by MIA ZX convention")
    return (
        level,
        np.rad2deg([roll, pitch]),
        float(np.max(abs(correction - reconstructed))),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--sequence", default="2011_09_26_drive_0005_sync")
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    if not a.sequence.endswith("_sync"):
        raise ValueError("Only synced+rectified KITTI is supported")
    date = a.sequence[:10]
    root = a.root.resolve()
    seq = root / date / a.sequence
    k, cam_imu, size = camera_geometry(root, date)
    image_times = (seq / "image_02/timestamps.txt").read_text().splitlines()
    pose_times = (seq / "oxts/timestamps.txt").read_text().splitlines()
    rows = []
    for image in sorted((seq / "image_02/data").glob("*.png")):
        i = int(image.stem)
        packet = np.loadtxt(seq / "oxts/data" / f"{image.stem}.txt")
        if packet.size != 30:
            raise ValueError("Expected KITTI 30-field OXTS packet")
        if Image.open(image).size != tuple(size):
            raise ValueError("Image dimensions disagree with S_rect_02")
        lat, lon, alt, roll, pitch, yaw = packet[:6]
        rwimu = Rotation.from_euler("ZYX", [yaw, pitch, roll]).as_matrix()
        rwcam = rwimu @ cam_imu[:3, :3].T
        offset = -rwcam @ cam_imu[:3, 3]
        level, rp, error = level_camera(rwcam)
        dt = float(
            abs(
                np.datetime64(image_times[i], "ns") - np.datetime64(pose_times[i], "ns")
            )
            / np.timedelta64(1, "s")
        )
        if dt > 0.05:
            raise ValueError("Camera/OXTS sync difference exceeds 50 ms")
        rows.append(
            dict(
                id=f"{a.sequence}/{image.stem}",
                image=str(image),
                latitude=float(lat),
                longitude=float(lon),
                altitude=float(alt),
                camera_offset_enu_m=offset.tolist(),
                R_world_camera=rwcam.tolist(),
                R_world_level_camera=level.tolist(),
                roll_pitch_degrees=rp.tolist(),
                heading_degrees=float(
                    np.degrees(np.arctan2(level[0, 2], level[1, 2])) % 360
                ),
                camera=dict(
                    model="PINHOLE",
                    width=int(size[0]),
                    height=int(size[1]),
                    params=[
                        float(k[0, 0]),
                        float(k[1, 1]),
                        float(k[0, 2] + 0.5),
                        float(k[1, 2] + 0.5),
                    ],
                ),
                K_opencv=k.tolist(),
                timestamp_image=image_times[i],
                timestamp_pose=pose_times[i],
                sync_delta_seconds=dt,
                rotation_residual=error,
                oxts_position_accuracy_m=float(packet[23]),
                oxts_navstat=int(packet[25]),
            )
        )
    if not rows:
        raise ValueError("No images found")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    inputs = [
        root / date / n
        for n in [
            "calib_cam_to_cam.txt",
            "calib_imu_to_velo.txt",
            "calib_velo_to_cam.txt",
        ]
    ] + [seq / "image_02/timestamps.txt", seq / "oxts/timestamps.txt"]
    inputs += list((seq / "oxts/data").glob("*.txt")) + list(
        (seq / "image_02/data").glob("*.png")
    )
    source_hashes = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in inputs
    }
    a.out.write_text(
        json.dumps(
            dict(
                sequence=a.sequence,
                samples=rows,
                source_hashes=source_hashes,
                T_camera_imu=cam_imu.tolist(),
                camera_convention="P_rect_02 is zero-based pixel-center K; +0.5 compensates Camera.from_dict COLMAP conversion exactly once",
                gravity="Camera calibration + OXTS; corrected image and GT share horizontal forward frame",
                no_lidar_measurements=True,
            ),
            indent=2,
        )
        + "\n"
    )
    print(f"{len(rows)} camera poses: {a.out}")


if __name__ == "__main__":
    main()
