"""Download RSVQA-HR from Zenodo (record 6344367, ~14.4 GB) and extract the images
(the record ships them as Images.tar).

Meant for Colab (needs open internet and ~30 GB of local disk):
    python scripts/download_rsvqa_hr.py /content/rsvqa_hr_raw
Resumable: files already downloaded at full size are skipped. Small JSON files
come first, so inspection can start before the 13.5 GB image archive finishes.
"""
from __future__ import annotations

import json
import subprocess
import sys
import urllib.request
from pathlib import Path

RECORD_URL = "https://zenodo.org/api/records/6344367"


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "data/raw/rsvqa_hr")
    out.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(RECORD_URL) as resp:
        files = json.load(resp)["files"]
    for f in files:
        print(f"{f['key']:45s} {f['size'] / 1e9:6.2f} GB")
    for f in sorted(files, key=lambda f: f["size"]):
        dest = out / f["key"]
        if dest.exists() and dest.stat().st_size == f["size"]:
            continue
        print("downloading", f["key"], flush=True)
        subprocess.run(["wget", "-q", "-c", "-O", str(dest), f["links"]["self"]], check=True)
    for archive in sorted(out.glob("*.zip")):
        print("unzipping", archive.name, flush=True)
        subprocess.run(["unzip", "-q", "-n", str(archive), "-d", str(out)], check=True)
    for archive in sorted(out.glob("*.tar")):
        print("extracting", archive.name, flush=True)
        subprocess.run(["tar", "-xf", str(archive), "-C", str(out), "--skip-old-files"], check=True)
    print("done:", out)


if __name__ == "__main__":
    main()
