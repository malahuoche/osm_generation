"""Use the unchanged OrienterNet parser/rasterizer on an ego-aligned metric grid.

Only the input vector coordinates and canvas extent are adapted. No road-width
buffering, class remapping, image rotation, or learned model is applied.
"""

import ast
import hashlib
import logging
import os
import sys
import types
from pathlib import Path

import numpy as np

COMMIT = "0280b6e248046dc7d714801d7d0651a162fd5a63"


def load_upstream():
    root = Path(os.environ.get("ORIENTERNET_REPO", "third_party/OrienterNet")).resolve()
    if not (root / "maploc/osm/raster.py").exists():
        raise FileNotFoundError("Set ORIENTERNET_REPO to the pinned official checkout")
    # Rasterization does not require the Lightning training package imported by maploc.__init__.
    if "maploc" not in sys.modules:
        package = types.ModuleType("maploc")
        package.__path__ = [str(root / "maploc")]
        package.logger = logging.getLogger("maploc")
        sys.modules["maploc"] = package
    else:
        if (
            Path(next(iter(sys.modules["maploc"].__path__))).resolve()
            != root / "maploc"
        ):
            raise ValueError("Another OrienterNet checkout is already imported")
    from maploc.osm.reader import OSMData
    from maploc.osm.data import MapData
    from maploc.osm.parser import Groups
    from maploc.osm.raster import Canvas, render_raster_masks, render_raster_map
    from maploc.utils.geo import BoundaryBox, Projection

    # Load official palette/class verbatim, avoiding optional Plotly dependencies.
    tree = ast.parse((root / "maploc/osm/viz.py").read_text())
    definitions = [
        n
        for n in tree.body
        if (
            isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "map_colors" for t in n.targets)
        )
        or (isinstance(n, ast.ClassDef) and n.name == "Colormap")
    ]
    namespace = {"np": np, "Groups": Groups}
    exec(
        compile(
            ast.Module(body=definitions, type_ignores=[]),
            "official-orienternet-colormap",
            "exec",
        ),
        namespace,
    )
    files = [
        "osm/reader.py",
        "osm/data.py",
        "osm/parser.py",
        "osm/raster.py",
        "osm/viz.py",
        "utils/geo.py",
        "utils/geo_opensfm.py",
    ]
    hashes = {
        f: hashlib.sha256((root / "maploc" / f).read_bytes()).hexdigest() for f in files
    }
    return types.SimpleNamespace(**locals())


def canvas_bounds(height=100, width=100, mpp=0.5, left=-25.0, far=50.0):
    """Edges giving exactly x=left+c*mpp, z=far-r*mpp at pixel centers."""
    return [left - mpp / 2, far - (height - 0.5) * mpp], [
        left + (width - 0.5) * mpp,
        far + mpp / 2,
    ]


def render(osm_path, sample, *, context=False):
    u = load_upstream()
    osm = u.OSMData.from_file(Path(osm_path))
    projection = u.Projection(
        sample["latitude"], sample["longitude"], sample["altitude"]
    )
    # Use full 3D ENU then horizontal axes; height is only a common datum here.
    level = np.asarray(sample["R_world_level_camera"])
    offset = np.asarray(sample["camera_offset_enu_m"])
    nodes = list(osm.nodes.values())
    xyz = projection.project(np.stack([n.geo for n in nodes]), return_z=True)
    local = (xyz - offset) @ level[:, [0, 2]]
    for node, xy in zip(nodes, local):
        node.xy = xy
    data = u.MapData.from_osm(osm)
    shape = (240, 300) if context else (100, 100)
    left, far = (-75.0, 120.0) if context else (-25.0, 50.0)
    lo, hi = canvas_bounds(*shape, left=left, far=far)
    canvas = u.Canvas(u.BoundaryBox(lo, hi), ppm=2)
    rr, cc = np.indices(shape)
    expected = np.stack([left + 0.5 * cc, far - 0.5 * rr], -1)
    uv = np.stack([cc, rr], -1).astype(float)
    np.testing.assert_allclose(canvas.to_xy(uv.copy()), expected, atol=1e-12)
    masks = u.render_raster_masks(
        list(data.nodes.values()),
        list(data.lines.values()),
        list(data.areas.values()),
        canvas,
    )
    raster = u.render_raster_map(masks).astype(np.uint8)
    groups = {key: list(getattr(u.Groups, key)) for key in ["areas", "ways", "nodes"]}
    native_rgb = np.rint(u.namespace["Colormap"].apply(raster) * 255).astype(np.uint8)
    buildings = []
    for area in data.areas.values():
        if area.group != "building":
            continue
        xy = np.concatenate(area.outers + area.inners)
        buildings.append(
            dict(
                id=area.id_,
                tags=area.tags,
                bounds_right_forward_m=[*xy.min(0).tolist(), *xy.max(0).tolist()],
            )
        )
    return dict(
        raster=raster,
        masks=masks,
        native_rgb=native_rgb,
        groups=groups,
        buildings=buildings,
        metadata=dict(
            expected_upstream_commit=COMMIT,
            source_hashes=u.hashes,
            osm_sha256=hashlib.sha256(Path(osm_path).read_bytes()).hexdigest(),
            shape=list(shape),
            mpp=0.5,
            canvas_min=lo,
            canvas_max=hi,
            camera_rc=[far / 0.5, -left / 0.5],
            convention="Native parser and rasterizer; geographic vectors transformed to camera right/forward BEFORE rasterization. No raster rotation.",
            grid_center_roundtrip_pass=True,
            native_palette="Official Colormap.apply: areas and ways only; nodes shown separately",
            line_width="Native 1-pixel ordinary ways; explicitly mapped areas may also populate way groups",
        ),
    )
