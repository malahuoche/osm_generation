import argparse, json
from collections import Counter
from pathlib import Path
from .data import records, load_sample, sha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--verify-hashes", action="store_true")
    args = ap.parse_args()
    rows = records(args.manifest)
    for r in rows:
        load_sample(r["file"])
        if args.verify_hashes and r.get("file_sha256") != sha(r["file"]):
            raise ValueError("Sample content changed: " + r["id"])
    print(
        json.dumps(
            dict(
                samples=len(rows),
                splits=dict(Counter(r["split"] for r in rows)),
                status="passed",
                note="Schema/disjoint-region check only. Independently inspect metric alignment and geographic separation.",
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
