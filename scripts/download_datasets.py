"""Download GIRAFE and BAGLS datasets from Zenodo.

Uses the Zenodo REST API (no account required for public records).
If the API is unavailable, the script prints the manual download URLs.

Usage
-----
python scripts/download_datasets.py --girafe --output-dir .
python scripts/download_datasets.py --bagls --output-dir .
python scripts/download_datasets.py --girafe --bagls
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
import urllib.error
import urllib.request
from pathlib import Path

# Zenodo record IDs (from paper / dataset DOIs)
GIRAFE_RECORD_ID = "13773163"   # https://zenodo.org/records/13773163
BAGLS_RECORD_ID = "3377544"     # https://doi.org/10.5281/zenodo.3377544 (BAGLS, Gómez et al. Scientific Data 2020)

ZENODO_API = "https://zenodo.org/api/records"


def _get_record(record_id: str) -> dict | None:
    url = f"{ZENODO_API}/{record_id}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.load(resp)
    except (urllib.error.URLError, urllib.error.HTTPError, OSError):
        return None


def _download_file(url: str, dest: Path, timeout: int = 3600) -> bool:
    # Zenodo returns 406 if Accept is too strict; use */* for file content
    req = urllib.request.Request(url, headers={"Accept": "*/*"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            total = int(resp.headers.get("Content-Length", 0)) or None
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as f:
                if total:
                    try:
                        from tqdm import tqdm
                        with tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as pbar:
                            while True:
                                chunk = resp.read(1 << 20)
                                if not chunk:
                                    break
                                f.write(chunk)
                                pbar.update(len(chunk))
                    except ImportError:
                        while True:
                            chunk = resp.read(1 << 20)
                            if not chunk:
                                break
                            f.write(chunk)
                else:
                    while True:
                        chunk = resp.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
        return True
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
        print(f"  Error: {e}")
        return False


def download_record(
    record_id: str,
    name: str,
    output_dir: Path,
    only_keys: tuple[str, ...] | None = None,
) -> bool:
    record = _get_record(record_id)
    if not record or "files" not in record:
        print(f"{name}: API failed or no files. Manual download:")
        print(f"  https://zenodo.org/record/{record_id}")
        return False
    files = record.get("files", [])
    if not files:
        print(f"{name}: record has no files. Try: https://zenodo.org/record/{record_id}")
        return False
    ok = True
    for fi in files:
        key = fi.get("key", fi.get("filename", "file"))
        if only_keys is not None and key not in only_keys:
            continue
        link = (fi.get("links") or {}).get("self")
        if not link:
            continue
        dest = output_dir / key
        if dest.exists():
            print(f"  {key} already exists, skip")
            continue
        print(f"  Downloading {key} ...")
        # Large zips need long timeout (e.g. training.zip ~3.5 GB)
        timeout = 3600 if key.endswith(".zip") else 120
        if not _download_file(link, dest, timeout=timeout):
            print(f"  Failed: {key}")
            rec_id = record.get("id", record_id)
            print(f"  Manual: https://zenodo.org/record/{rec_id} (download {key} from the record page)")
            ok = False
    return ok


def main() -> None:
    p = argparse.ArgumentParser(
        description="Download GIRAFE and/or BAGLS from Zenodo.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--girafe", action="store_true", help="Download GIRAFE dataset")
    p.add_argument("--bagls", action="store_true", help="Download BAGLS dataset")
    p.add_argument("--output-dir", type=Path, default=Path("."),
                   help="Directory to save files into")
    args = p.parse_args()
    if not (args.girafe or args.bagls):
        p.print_help()
        sys.exit(0)
    args.output_dir = args.output_dir.resolve()
    if args.girafe:
        print("GIRAFE:")
        download_record(GIRAFE_RECORD_ID, "GIRAFE", args.output_dir)
    if args.bagls:
        print("BAGLS:")
        bagls_dir = args.output_dir / "BAGLS"
        bagls_dir.mkdir(parents=True, exist_ok=True)
        download_record(
            BAGLS_RECORD_ID,
            "BAGLS",
            bagls_dir,
            only_keys=("training.zip", "test.zip", "_readme.md"),
        )
        # Extract so BAGLS/training/ and BAGLS/test/ exist (README paths)
        for zname in ("training.zip", "test.zip"):
            zpath = bagls_dir / zname
            if zpath.exists():
                print(f"  Extracting {zname} ...")
                with zipfile.ZipFile(zpath, "r") as z:
                    z.extractall(bagls_dir)
                print(f"  → {bagls_dir / zname.replace('.zip', '')}/")


if __name__ == "__main__":
    main()
