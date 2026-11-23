"""Fetch a bounded OSM XML map for raw camera geometry; preserve acquisition provenance."""

import argparse, datetime, hashlib, json, urllib.parse, urllib.request, xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from pyproj import Geod


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--geometry", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--margin-m", type=float, default=180)
    p.add_argument("--endpoint", default="https://overpass-api.de/api/interpreter")
    p.add_argument("--provider", choices=["overpass", "osm-api"], default="osm-api")
    a = p.parse_args()
    samples = json.loads(a.geometry.read_text())["samples"]
    g = Geod(ellps="WGS84")
    points = []
    for s in samples:
        # Offset may be a few meters from GNSS; included conservatively.
        radius = a.margin_m + float(np.linalg.norm(s["camera_offset_enu_m"][:2]))
        points.extend(
            g.fwd(s["longitude"], s["latitude"], az, radius)[:2]
            for az in [0, 90, 180, 270]
        )
    xy = np.array(points)
    w, s = xy.min(0)
    e, n = xy.max(0)
    query = f"[out:xml][timeout:120];(node({s},{w},{n},{e});way({s},{w},{n},{e});relation[type=multipolygon]({s},{w},{n},{e}););(._;>>;);out meta;"
    url = (
        a.endpoint + "?" + urllib.parse.urlencode({"data": query})
        if a.provider == "overpass"
        else "https://api.openstreetmap.org/api/0.6/map?bbox="
        + ",".join(map(str, [w, s, e, n]))
    )
    if a.out.exists():
        raise ValueError(
            "Output exists; use a fresh file to preserve snapshot provenance"
        )
    with urllib.request.urlopen(
        urllib.request.Request(
            url, headers={"User-Agent": "bev-osm-research-preprocessing/0.1"}
        ),
        timeout=150,
    ) as response:
        data = response.read()
    root = ET.fromstring(data)
    if root.tag != "osm" or root.find("remark") is not None:
        raise ValueError("OSM server returned error/incomplete response")
    requests = [url]
    if a.provider == "osm-api":
        known = {(x.tag, x.get("id")) for x in root}
        # The map endpoint may omit relation members outside the requested bbox.
        for relation in list(root.findall("relation")):
            if not any(
                t.get("k") == "type" and t.get("v") == "multipolygon"
                for t in relation.findall("tag")
            ):
                continue
            if any(
                (m.get("type"), m.get("ref")) not in known
                for m in relation.findall("member")
            ):
                extra = (
                    "https://api.openstreetmap.org/api/0.6/relation/"
                    + relation.get("id")
                    + "/full"
                )
                with urllib.request.urlopen(
                    urllib.request.Request(
                        extra,
                        headers={"User-Agent": "bev-osm-research-preprocessing/0.1"},
                    ),
                    timeout=120,
                ) as response:
                    other = ET.fromstring(response.read())
                requests.append(extra)
                for element in other:
                    key = (element.tag, element.get("id"))
                    if key not in known:
                        root.append(element)
                        known.add(key)
        data = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_bytes(data)
    a.out.with_suffix(".source.json").write_text(
        json.dumps(
            dict(
                requests=requests,
                provider=a.provider,
                query=query,
                retrieved_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                sha256=hashlib.sha256(data).hexdigest(),
                note="Current OSM weak labels; not contemporaneous surveyed GT",
                bbox=[w, s, e, n],
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
