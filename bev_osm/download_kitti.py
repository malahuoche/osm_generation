"""Fetch only selected image_02/OXTS members from the official sync ZIP using ranges."""

import argparse, hashlib, io, json, re, struct, urllib.request, zipfile, zlib
from pathlib import Path


class RemoteZip(io.RawIOBase):
    def __init__(self, url):
        self.url = url
        self.pos = 0
        self.bytes = 0
        with urllib.request.urlopen(
            urllib.request.Request(url, method="HEAD"), timeout=60
        ) as r:
            self.size = int(r.headers["Content-Length"])
            self.etag = r.headers.get("ETag")

    def seek(self, offset, whence=0):
        self.pos = (
            offset
            if whence == 0
            else self.pos + offset if whence == 1 else self.size + offset
        )
        return self.pos

    def tell(self):
        return self.pos

    def read(self, n=-1):
        n = min(self.size - self.pos, n if n >= 0 else self.size - self.pos)
        if n <= 0:
            return b""
        start = self.pos
        headers = {"Range": f"bytes={start}-{start+n-1}", "If-Match": self.etag}
        with urllib.request.urlopen(
            urllib.request.Request(self.url, headers=headers), timeout=120
        ) as r:
            if r.status != 206 or not r.headers.get("Content-Range", "").startswith(
                f"bytes {start}-"
            ):
                raise RuntimeError(
                    "Server did not honor Range; refusing full archive download"
                )
            data = r.read(n + 1)
        if len(data) != n:
            raise ValueError("Incomplete range")
        self.pos += n
        self.bytes += n
        return data

    def member(self, info):
        self.seek(info.header_offset)
        head = self.read(30)
        h = struct.unpack("<4s5H3I2H", head)
        if h[0] != b"PK\x03\x04":
            raise ValueError("Invalid ZIP header")
        self.seek(info.header_offset + 30 + h[-2] + h[-1])
        data = self.read(info.compress_size)
        data = (
            zlib.decompress(data, -15)
            if info.compress_type == 8
            else data if info.compress_type == 0 else None
        )
        if data is None or len(data) != info.file_size or zlib.crc32(data) != info.CRC:
            raise ValueError("ZIP checksum mismatch")
        return data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--date", default="2011_09_26")
    p.add_argument("--drive", default="0005")
    p.add_argument(
        "--frames",
        type=int,
        default=20,
        help="Evenly spaced frames; 0 selects entire sequence",
    )
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if (
        a.frames < 0
        or not re.fullmatch(r"\d{4}_\d{2}_\d{2}", a.date)
        or not re.fullmatch(r"\d{4}", a.drive)
    ):
        p.error("Invalid date, drive or frame count")
    seq = f"{a.date}_drive_{a.drive}_sync"
    base = "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data"
    sources = []
    files = []
    urls = [f"{base}/{a.date}_calib.zip", f"{base}/{a.date}_drive_{a.drive}/{seq}.zip"]
    for url in urls:
        remote = RemoteZip(url)
        with zipfile.ZipFile(remote) as z:
            infos = z.infolist()
        frames = sorted(
            i.filename
            for i in infos
            if "/image_02/data/" in i.filename and i.filename.endswith(".png")
        )
        if frames and a.frames:
            n = min(a.frames, len(frames))
            frames = [
                frames[round(j * (len(frames) - 1) / max(1, n - 1))] for j in range(n)
            ]
        stems = {Path(f).stem for f in frames}
        for info in infos:
            name = info.filename
            take = (
                name in frames
                or ("/oxts/data/" in name and Path(name).stem in stems)
                or name.endswith(("image_02/timestamps.txt", "oxts/timestamps.txt"))
                or Path(name).name
                in {
                    "calib_cam_to_cam.txt",
                    "calib_velo_to_cam.txt",
                    "calib_imu_to_velo.txt",
                }
            )
            if not take:
                continue
            dest = (a.out / name).resolve()
            if not dest.is_relative_to(a.out.resolve()):
                raise ValueError("Unsafe ZIP path")
            data = remote.member(info)
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists() and dest.read_bytes() != data:
                raise ValueError(f"Conflicting existing file: {dest}")
            dest.write_bytes(data)
            files.append(
                dict(
                    path=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()
                )
            )
            print(name, flush=True)
        sources.append(
            dict(
                url=url,
                archive_bytes=remote.size,
                downloaded_bytes=remote.bytes,
                etag=remote.etag,
            )
        )
    (a.out / "download.json").write_text(
        json.dumps(
            dict(sequence=seq, requested_frames=a.frames, sources=sources, files=files),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
