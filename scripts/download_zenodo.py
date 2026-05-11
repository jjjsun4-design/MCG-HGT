from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, dest.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download MCG-HGT Zenodo files.")
    parser.add_argument("--record", required=True, help="Zenodo record id or DOI suffix, for example 12345678.")
    parser.add_argument("--out", default="data", help="Output directory.")
    parser.add_argument("--dry-run", action="store_true", help="List files without downloading.")
    args = parser.parse_args(argv)

    record = args.record.rsplit(".", 1)[-1] if "zenodo." in args.record else args.record
    api_url = f"https://zenodo.org/api/records/{record}"
    with urllib.request.urlopen(api_url) as response:
        meta = json.load(response)

    out = Path(args.out)
    files = meta.get("files", [])
    if not files:
        print(f"No files found in Zenodo record {record}", file=sys.stderr)
        return 1

    for item in files:
        key = item["key"]
        size = item.get("size")
        checksum = item.get("checksum", "")
        url = item["links"]["self"]
        dest = out / key
        print(f"{key}\t{size}\t{checksum}")
        if args.dry_run:
            continue
        download(url, dest)
        if checksum.startswith("md5:"):
            # Zenodo commonly reports MD5. Keep SHA256 output for local manifests.
            print(f"downloaded {dest} sha256={sha256_file(dest)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
