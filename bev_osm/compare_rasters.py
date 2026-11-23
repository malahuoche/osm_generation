"""Compare MIA and native OrienterNet raster rules on identical OSM and metric crops."""

import argparse
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .data import load_sample, sha
from .orienternet_raster import render
from .raw_report import paint


def common_rgb(result):
    """Display-only colors. Categorical raster is saved unchanged."""
    a, w, _ = result["raster"]
    g = result["groups"]
    rgb = result["native_rgb"].copy()
    rgb[(a == 0) & (w == 0)] = 255
    for group, color in [
        ("building", [231, 111, 81]),
        ("parking", [204, 204, 204]),
        ("grass", [42, 157, 143]),
        ("park", [42, 157, 143]),
        ("forest", [42, 157, 143]),
    ]:
        rgb[(a == g["areas"].index(group) + 1) & (w == 0)] = color
    rgb[w == g["ways"].index("road") + 1] = [68, 68, 68]
    rgb[w == g["ways"].index("building_outline") + 1] = [231, 111, 81]
    return rgb


def map_picture(rgb, support=None):
    rgb = rgb.copy()
    if support is not None:
        rgb[~support[0].astype(bool)] = [30, 43, 54]
    im = Image.fromarray(rgb).resize((300, 300), Image.Resampling.NEAREST)
    d = ImageDraw.Draw(im)
    d.line((150, 299, 150, 279), fill=(0, 140, 220), width=3)
    return im


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ["geometry", "osm", "prepared", "mia-audit", "out"]:
        p.add_argument("--" + key, required=True, type=Path)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    geometry = json.loads(a.geometry.read_text())
    samples = {s["id"]: s for s in geometry["samples"]}
    rows = [
        json.loads(x) for x in (a.prepared / "manifest.jsonl").read_text().splitlines()
    ]
    mia = {s["id"]: s for s in json.loads(a.mia_audit.read_text())}
    osm_hash = sha(a.osm)
    output = []
    cards = []
    for i, row in enumerate(rows):
        s = samples[row["id"]]
        assert (
            mia[row["id"]]["renderer"]["osm_sha256"] == osm_hash
        ), "MIA and OrienterNet must use identical OSM XML"
        path = a.prepared / row["file"]
        assert sha(path) == row["file_sha256"]
        data = load_sample(path)
        result = render(a.osm, s)
        common = common_rgb(result)
        name = f"orienternet_{i:03d}"
        np.savez_compressed(
            a.out / (name + ".npz"),
            raster=result["raster"],
            support=data["support"].astype(bool),
        )
        canvas = Image.new("RGB", (1310, 735), "#f3f5f7")
        d = ImageDraw.Draw(canvas)
        rgb = Image.open(s["image"]).convert("RGB")
        rgb.thumbnail((330, 285))
        canvas.paste(rgb, (0, 40))
        for x, label in [
            (0, "Input RGB"),
            (350, "MIA semantic areas"),
            (670, "OrienterNet: native colors"),
            (990, "OrienterNet: common colors"),
        ]:
            d.text((x, 10), label, fill="black")
        canvas.paste(paint(data["target"]), (350, 35))
        canvas.paste(map_picture(result["native_rgb"]), (670, 35))
        canvas.paste(map_picture(common), (990, 35))
        canvas.paste(paint(data["target"], data["support"]), (350, 390))
        canvas.paste(map_picture(result["native_rgb"], data["support"]), (670, 390))
        canvas.paste(map_picture(common, data["support"]), (990, 390))
        d.text(
            (350, 365),
            "Same image-derived projection support for all three maps",
            fill="black",
        )
        d.text(
            (350, 710),
            "0.5 m/cell | 50 m forward | camera bottom center | no model predictions",
            fill="black",
        )
        node = result["raster"][2]
        node_ids = np.unique(node[node > 0])
        node_labels = [result["groups"]["nodes"][int(v) - 1] for v in node_ids]
        node_view = map_picture(result["native_rgb"])
        nd = ImageDraw.Draw(node_view)
        for rr, cc in np.argwhere(node > 0):
            nd.ellipse((3 * cc - 2, 3 * rr - 2, 3 * cc + 3, 3 * rr + 3), fill="black")
        node_view.save(a.out / (name + "_nodes.png"))
        canvas.save(a.out / (name + ".png"))
        bidx = result["groups"]["areas"].index("building") + 1
        building = result["raster"][0] == bidx
        rec = dict(
            id=row["id"],
            **result["metadata"],
            groups=result["groups"],
            mia_render=mia[row["id"]]["renderer"],
            building_pixels=dict(
                mia=int(data["target"][3].sum()), orienternet=int(building.sum())
            ),
            node_categories=node_labels,
        )
        output.append(rec)
        cards.append(
            f'<section><h2>{html.escape(row["id"])}</h2><img class="comparison" src="{name}.png"><details><summary>点要素层（原版默认配色不显示）</summary><img src="{name}_nodes.png"><p>{html.escape(str(node_labels))}</p></details><p>建筑像素：MIA {rec["building_pixels"]["mia"]} / OrienterNet {rec["building_pixels"]["orienternet"]}。这是渲染差异，不是预测准确率。</p></section>'
        )
        if len(rows) == 1:
            context = render(a.osm, s, context=True)
            im = Image.fromarray(common_rgb(context)).resize(
                (600, 480), Image.Resampling.NEAREST
            )
            dr = ImageDraw.Draw(im)
            # Crop edges in context canvas; continuous edge coordinates, same half-cell convention.
            dr.rectangle((200, 280, 400, 480), outline=(0, 140, 220), width=3)
            for b in context["buildings"]:
                x0, z0, x1, z1 = b["bounds_right_forward_m"]
                if x1 < -75 or x0 > 75 or z1 < 0 or z0 > 120:
                    continue
                px = int(((x0 + x1) / 2 + 75) * 4)
                py = int((120 - (z0 + z1) / 2) * 4)
                dr.text(
                    (max(0, min(px, 480)), max(0, min(py, 460))),
                    str(b["id"]),
                    fill="black",
                )
            im.save(a.out / "orienternet_context.png")
            (a.out / "orienternet_buildings.json").write_text(
                json.dumps(context["buildings"], indent=2) + "\n"
            )
            cards.append(
                '<section><h2>更大范围：前方120m、左右75m</h2><p>蓝框为上方50m×50m的评估范围；只用于解释图像中远处物体为何不出现在局部地图。</p><img src="orienternet_context.png"></section>'
            )
    (a.out / "orienternet-audit.json").write_text(json.dumps(output, indent=2) + "\n")
    # Preserve the original QC page and images. Comparison images use distinct names.
    index = a.out / "index.html"
    backup = a.out / "raw-qc.html"
    if index.exists() and not backup.exists():
        backup.write_text(index.read_text())
    intro = """<h1>MIA 与 OrienterNet 原生 Raster 对照</h1>
<p>同一份 OSM XML、同一相机位置/方向、同一0.5米格点。上排完整范围，下排相同投影支持区。此处比较标签表示，未训练新模型。</p>
<p>MIA：六个二值语义层，以道路面等形式渲染。OrienterNet：面、线、点三个类别编号层；普通道路为原生1像素细线，显式道路面另依原始标签处理。未给细线补道路宽度。</p>
<p>原生配色：建筑浅蓝、建筑轮廓蓝、道路红。统一配色图仅用于显示：建筑橙、道路灰，原始类别数据保存在NPZ。深色为投影支持区外；空白不等于已确认自由空间。</p>
<p>OrienterNet 的原分类和栅格函数保持不变；在栅格化前把矢量转到车辆坐标，避免旋转已栅格化细线带来二次量化。当前是原生渲染规则的车辆朝向适配，不是作者原始north-up缓存截图。</p>"""
    index.write_text(
        '<meta charset="utf-8"><title>MIA / OrienterNet raster comparison</title><style>body{font:16px system-ui;margin:30px;color:#243744;background:#f3f5f7}p{max-width:1100px;line-height:1.6}.comparison{width:1310px;max-width:100%}section{margin:35px 0}summary{cursor:pointer}</style>'
        + intro
        + ('<a href="raw-qc.html">原预处理核验页面</a>' if backup.exists() else "")
        + "".join(cards)
    )
    print(index)


if __name__ == "__main__":
    main()
