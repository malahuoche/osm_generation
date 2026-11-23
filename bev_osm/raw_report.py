"""Visual and numeric QC for a raw-data preparation run (not a model evaluation)."""

import argparse, html, json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from .data import load_sample, sha
from .mia_runtime import preprocess


def paint(y, support=None):
    colors = np.array(
        [
            [68, 68, 68],
            [244, 162, 97],
            [233, 196, 106],
            [231, 111, 81],
            [42, 157, 143],
            [204, 204, 204],
        ],
        dtype=np.uint8,
    )
    rgb = np.full((100, 100, 3), 255, dtype=np.uint8)
    for i in [4, 5, 0, 2, 1, 3]:
        rgb[y[i].astype(bool)] = colors[i]
    if support is not None:
        rgb[~support[0].astype(bool)] = [30, 43, 54]
    im = Image.fromarray(rgb).resize((300, 300), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(im)
    draw.line((150, 298, 150, 275), fill=(0, 140, 220), width=3)
    return im


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--geometry", type=Path, required=True)
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    samples = json.loads(a.geometry.read_text())["samples"]
    lookup = {s["id"]: s for s in samples}
    rows = [
        json.loads(x) for x in (a.prepared / "manifest.jsonl").read_text().splitlines()
    ]
    items = []
    errors = []
    for i, r in enumerate(rows):
        file = a.prepared / r["file"]
        assert sha(file) == r["file_sha256"]
        data = load_sample(file)
        s = lookup[r["id"]]
        raw = np.array(Image.open(s["image"]).convert("RGB"))
        batch = preprocess(raw, s["camera"], s["roll_pitch_degrees"])
        # from_dict must recover native pixel-center intrinsics before resize.
        from mapper.utils.wrappers import Camera

        cam = Camera.from_dict(s["camera"])
        K = np.array(s["K_opencv"])
        np.testing.assert_allclose(cam.c.numpy(), K[:2, 2], atol=1e-4)
        rect = (
            (batch["image"][0].numpy().transpose(1, 2, 0) * 255)
            .clip(0, 255)
            .astype("uint8")
        )
        canvas = Image.new("RGB", (1280, 350), "#f3f5f7")
        draw = ImageDraw.Draw(canvas)
        rawim = Image.fromarray(raw)
        rawim.thumbnail((340, 280))
        canvas.paste(rawim, (0, 40))
        canvas.paste(Image.fromarray(rect).resize((280, 280)), (350, 40))
        canvas.paste(paint(data["target"]), (650, 35))
        canvas.paste(paint(data["target"], data["support"]), (970, 35))
        for x, label in [
            (0, "Original RGB"),
            (350, "Actual MIA input: gravity + resize"),
            (650, "Native MIA OSM labels"),
            (970, "Labels in projection support"),
        ]:
            draw.text((x, 8), label, fill="black")
        draw.text(
            (650, 335), "0.5 m/cell | forward up | camera at bottom", fill="black"
        )
        name = f"{i:03d}.png"
        canvas.save(a.out / name)
        items.append(
            f'<h3>{html.escape(r["id"])}</h3><p>roll/pitch={s["roll_pitch_degrees"]}; heading={s["heading_degrees"]:.3f} deg</p><img src="{name}" width="1280">'
        )
    summary = dict(
        sample_count=len(rows),
        schema_and_hashes_pass=True,
        intrinsics_half_pixel_roundtrip_pass=True,
        maximum_rotation_residual=max(s["rotation_residual"] for s in samples),
        maximum_sync_delta_seconds=max(s.get("sync_delta_seconds", 0) for s in samples),
        limits=[
            "Not a training or generalization test",
            "Current OSM is temporally mismatched weak supervision",
            "Projection support is not occlusion visibility",
            "Gravity rectification uses provided pose: not a strict RGB-only gravity estimate",
            "Visual agreement does not verify surveyed absolute map accuracy",
        ],
    )
    (a.out / "audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    (a.out / "index.html").write_text(
        '<meta charset="utf-8"><title>Raw-data preprocessing QC</title><style>body{font:16px system-ui;margin:30px;background:#f3f5f7}img{max-width:100%}p{color:#445}</style><h1>原始数据预处理检查 · 非模型预测</h1><p>左：原图；中左：实际重力校正输入；中右：完整 OSM 弱标签；右：投影支持区标签。深色是支持区外，白色是六类均未标注。</p><pre>'
        + html.escape(json.dumps(summary, ensure_ascii=False, indent=2))
        + "</pre>"
        + "".join(items)
    )
    print(a.out / "index.html")


if __name__ == "__main__":
    main()
