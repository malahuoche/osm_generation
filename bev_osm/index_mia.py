"""Index released MIA city folders using explicit city-level split assignments."""

import argparse, json, os
from pathlib import Path
from .data import save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--splits", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    split = json.loads(args.splits.read_text())
    seen = set()
    count = 0
    missing = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise ValueError("Output exists; choose a fresh pairs file")
    with args.out.open("w") as stream:
        for city, part in split.items():
            if part not in ("train", "val", "test"):
                raise ValueError("Split must be train/val/test")
            directory = args.root / city
            dump = json.loads((directory / "dump.json").read_text())
            for seq, item in dump.items():
                for name, view in item["views"].items():
                    sid = name.removesuffix("_undistorted")
                    images = [
                        directory / "images" / (name + ext)
                        for ext in (".jpg", ".png", ".jpeg")
                    ]
                    image = next((p for p in images if p.exists()), None)
                    target = directory / "semantic_masks" / (sid + ".npz")
                    flood = directory / "flood_fill" / (sid + ".npz")
                    if image is None or not target.exists():
                        missing += 1
                        continue
                    if sid in seen:
                        raise ValueError(
                            "Duplicate global MIA id; check copied city/sample folders"
                        )
                    seen.add(sid)
                    camera = item["cameras"][str(view["camera_id"])]
                    if camera["model"] != "PINHOLE":
                        raise ValueError("Require undistorted PINHOLE images")
                    r = dict(
                        id=sid,
                        region=city,
                        split=part,
                        image=os.path.relpath(
                            image.resolve(), args.out.parent.resolve()
                        ),
                        target=os.path.relpath(
                            target.resolve(), args.out.parent.resolve()
                        ),
                        target_format="mia_native",
                        camera=camera,
                        roll_pitch_degrees=view["roll_pitch_yaw"][:2],
                    )
                    if flood.exists():
                        r["flood"] = os.path.relpath(
                            flood.resolve(), args.out.parent.resolve()
                        )
                    stream.write(json.dumps(r) + "\n")
                    count += 1
    save_json(
        args.out.with_suffix(".audit.json"),
        dict(
            indexed=count,
            missing_image_or_map=missing,
            split_assignments=split,
            note="City-level custom split, not claimed official benchmark split. Inspect actual city geography and completeness.",
        ),
    )
    print(
        f'{count} pairs; {missing} dump records lack downloaded files. Audit: {args.out.with_suffix(".audit.json")}'
    )


if __name__ == "__main__":
    main()
