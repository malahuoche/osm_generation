"""Pinned original MIA -> MapMachine -> CairoSVG renderer, isolated runtime.

Only named official Python AST definitions are loaded, without editing them, to
avoid unrelated network/dataset/plotting imports. Rasterization uses actual
MapMachine constructors and CairoSVG. Coordinate resampling is an explicit
nearest-source-pixel adapter, not a replacement renderer or exact registration.
"""

from pathlib import Path
import ast, ctypes, hashlib, io, json, logging, math, os, sys, traceback
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image

ROOT = Path(
    os.environ.get("BEV_OSM_RENDER_CACHE", str(Path.home() / ".cache/bev-osm-render"))
).resolve()
ROOT.mkdir(parents=True, exist_ok=True)
MIA = (
    Path(os.environ.get("MIA_REPO", "third_party/MapItAnywhere")).resolve()
    / "mia/bev/get_bev.py"
)
STYLE = MIA.parent / "styles/mia.yml"
MAP_MACHINE = Path(
    os.environ.get("MAP_MACHINE_REPO", "third_party/map-machine")
).resolve()
sys.path.insert(0, str(MAP_MACHINE))
if os.environ.get("BEV_OSM_CAIRO_LIBRARY"):
    ctypes.CDLL(os.environ["BEV_OSM_CAIRO_LIBRARY"])
import cairo, cairosvg, svgwrite, colour

# Upstream workspace constructor creates a relative output directory on import.
# Confine that documented side effect to the isolated runtime directory.
_previous_cwd = Path.cwd()
try:
    os.chdir(ROOT)
    from map_machine.map_configuration import MapConfiguration
    from map_machine.scheme import Scheme
    from map_machine.geometry.boundary_box import BoundaryBox
    from map_machine.osm.osm_reader import OSMData
    from map_machine.geometry.flinger import MercatorFlinger
    from map_machine.pictogram.icon import ShapeExtractor
    from map_machine.workspace import workspace
    from map_machine.mapper import Map
    from map_machine.constructor import Constructor
finally:
    os.chdir(_previous_cwd)
workspace.output_path = ROOT / "out"

logger = logging.getLogger("native_mia")
_names = {"COLORS", "checkColor", "draw_bev"}
_tree = ast.parse(MIA.read_text())
_nodes = [
    n
    for n in _tree.body
    if (isinstance(n, ast.FunctionDef) and n.name in _names)
    or (
        isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id in _names for t in n.targets)
    )
]
assert len(_nodes) == 3
exec(compile(ast.Module(body=_nodes, type_ignores=[]), str(MIA), "exec"), globals())


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def provenance():
    import subprocess

    commit = (
        subprocess.check_output(
            ["git", "-C", str(MAP_MACHINE), "rev-parse", "HEAD"], text=True
        ).strip()
        if (MAP_MACHINE / ".git").exists()
        else "unverified-source-tree"
    )
    return {
        "MIA_source": str(MIA),
        "MIA_sha256": digest(MIA),
        "MIA_style": str(STYLE),
        "MIA_style_sha256": digest(STYLE),
        "map_machine_commit": commit,
        "native_wrapper_sha256": digest(__file__),
        "cairo_version": cairo.cairo_version_string(),
        "pycairo_version": cairo.version,
        "cairosvg_version": cairosvg.__version__,
        "ast_definitions": ["COLORS", "checkColor", "draw_bev"],
        "upstream_definitions_unmodified": True,
        "class_order": list(COLORS),
    }


def render_at_points(
    osm_xml, bbox_lbrt_lonlat, heading_degrees_east_of_north, points_latlon, mpp=0.5
):
    """Return native9 layers sampled at requested [H,W,2] geographic centers.

    osm_xml: XML str/bytes or filesystemPath. bbox: [left,bottom,right,top] lonlat.
    points_latlon: lastdim[latitude,longitude]; usually camera-forward metricgrid.
    heading: clockwisefromnorth, passed unchangedtoofficialMIA draw_bev.
    Output layers[9,H,W], raw_layers[9,Ph,Pw], svg[str], metadata[dict].
    Raw alpha may contain boundary antialias support, matching official alpha!=0.
    """
    if isinstance(osm_xml, Path):
        payload = osm_xml.read_bytes()
    elif isinstance(osm_xml, bytes):
        payload = osm_xml
    elif isinstance(osm_xml, str) and osm_xml.lstrip().startswith("<"):
        payload = osm_xml.encode()
    else:
        payload = Path(osm_xml).read_bytes()
    pts = np.asarray(points_latlon, dtype=np.float64)
    assert pts.ndim >= 2 and pts.shape[-1] == 2 and np.isfinite(pts).all()
    assert mpp > 0 and len(bbox_lbrt_lonlat) == 4
    bbox = BoundaryBox(*map(float, bbox_lbrt_lonlat))
    assert bbox.left < bbox.right and bbox.bottom < bbox.top
    osm = OSMData()
    osm.parse_osm(
        ET.fromstring(payload), parse_nodes=True, parse_relations=True, parse_ways=True
    )
    scheme = Scheme.from_file(STYLE)
    assert scheme is not None
    configuration = MapConfiguration(scheme)
    heading = float(heading_degrees_east_of_north) % 360.0
    imgs, svg = draw_bev(bbox, osm, configuration, float(mpp), heading)
    if imgs is None:
        raise RuntimeError(
            "Official MIA draw_bev failed; see logged original exception"
        )
    assert len(imgs) == 9
    raw_alpha = np.stack([np.asarray(im)[..., -1] != 0 for im in imgs])
    # Exact native get_bev_from_bbox convention before optional maxpool (factor1).
    raw = raw_alpha.copy()
    raw[2][raw[0]] = 0
    lat = float(bbox.center()[0])
    zoom = np.log2(abs(osm.equator_length * np.cos(np.deg2rad(lat)) / mpp / 256))
    flinger = MercatorFlinger(bbox, zoom, osm.equator_length)
    flat = pts.reshape(-1, 2)
    xy = np.stack([flinger.fling(p) for p in flat])
    theta = np.deg2rad(360 - heading)
    rotation = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    center = flinger.size / 2
    xy = (xy - center) @ rotation.T + center
    ph, pw = raw.shape[-2:]
    scale = np.array([pw, ph]) / flinger.size
    xy *= scale
    # Pixel centers at(i+.5,j+.5); floor chooses a nearestcenter, ties upward.
    indices = np.floor(xy).astype(np.int64)
    valid = (
        (indices[:, 0] >= 0)
        & (indices[:, 0] < pw)
        & (indices[:, 1] >= 0)
        & (indices[:, 1] < ph)
    )
    if not valid.all():
        raise ValueError(
            f"{int((~valid).sum())} target centers fall outside native rendered canvas; enlarge bbox padding"
        )
    layers = raw[:, indices[:, 1], indices[:, 0]].reshape((9,) + pts.shape[:-1])
    displacement = (indices + 0.5 - xy) / scale
    center_errors = np.linalg.norm(displacement, axis=-1) * mpp
    metadata = {
        **provenance(),
        "osm_sha256": hashlib.sha256(payload).hexdigest(),
        "bbox_lbrt_lonlat": list(map(float, bbox_lbrt_lonlat)),
        "heading_degrees_east_of_north": heading,
        "meters_per_pixel_at_bbox_center": mpp,
        "zoom_level": float(zoom),
        "flinger_size_wh": flinger.size.tolist(),
        "png_size_wh": [pw, ph],
        "svg_rotation_degrees": 360 - heading,
        "svg_rotation_center_xy": center.tolist(),
        "svg_to_png_scale_xy": scale.tolist(),
        "sample_policy": "nearest native pixelcenter; floor(svg_to_png_xy), ties upward; no bilinear mask interpolation",
        "target_centers_valid_fraction": float(valid.mean()),
        "target_to_native_center_max_m": float(center_errors.max()),
        "target_to_native_center_rms_m": float(np.sqrt(np.mean(center_errors**2))),
        "quantization_limit_m_at_bbox_center": float(np.sqrt(2) * mpp / 2),
        "registration_claim": "Known analytic transform, with explicitly quantified nearest-pixel quantization; not independently surveyed geographic truth",
        "road_exclusion": "explicit_pedestrian[road]=False per nativeMIA get_bev_from_bbox",
        "full_raw_alpha_preserved": True,
        "mpp_local_variation_note": "pixel distance conversion uses bbox-center mpp; localMercatorscale varies with latitude",
    }
    return {
        "layers": layers,
        "raw_layers": raw,
        "raw_alpha_layers": raw_alpha,
        "svg": svg.tostring(),
        "metadata": metadata,
        "target_source_xy": xy.reshape(pts.shape),
        "target_source_pixel_indices_xy": indices.reshape(pts.shape),
    }
