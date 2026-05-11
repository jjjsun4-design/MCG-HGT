from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path


API = "https://api.github.com/repos/jjjsun4-design/MCG-HGT/releases/tags/{tag}"


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "MCG-HGT-release-downloader"})
    with urllib.request.urlopen(request) as response, dest.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download MCG-HGT GitHub Release assets.")
    parser.add_argument("--tag", default="v1.0.0", help="GitHub Release tag.")
    parser.add_argument("--out", default=".", help="Output directory.")
    parser.add_argument("--dry-run", action="store_true", help="List assets without downloading.")
    args = parser.parse_args(argv)

    request = urllib.request.Request(API.format(tag=args.tag), headers={"User-Agent": "MCG-HGT-release-downloader"})
    try:
        with urllib.request.urlopen(request) as response:
            release = json.load(response)
    except Exception as exc:
        print(f"Could not read release {args.tag}: {exc}", file=sys.stderr)
        return 1

    assets = release.get("assets", [])
    if not assets:
        print(f"No release assets found for {args.tag}", file=sys.stderr)
        return 1

    out = Path(args.out)
    for asset in assets:
        name = asset["name"]
        size = asset.get("size", 0)
        url = asset["browser_download_url"]
        print(f"{name}\t{size}\t{url}")
        if not args.dry_run:
            download(url, out / name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
