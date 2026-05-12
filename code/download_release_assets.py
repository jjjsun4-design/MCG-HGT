from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_tarball(path: Path, dest: Path) -> None:
    with tarfile.open(path, "r:gz") as tar:
        tar.extractall(dest)


def load_checksums(root: Path) -> dict[str, str]:
    checksum_path = root / "manifests" / "checksums.txt"
    if not checksum_path.exists():
        raise FileNotFoundError(f"Checksum file not found: {checksum_path}")
    checksums: dict[str, str] = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, rel = line.split(None, 1)
        checksums[rel.strip()] = digest
    return checksums


def verify_checksums(root: Path) -> None:
    checksums = load_checksums(root)
    missing: list[str] = []
    mismatched: list[str] = []
    for rel, expected in checksums.items():
        path = root / rel
        if not path.exists():
            missing.append(rel)
            continue
        observed = sha256_file(path)
        if observed.lower() != expected.lower():
            mismatched.append(rel)
    if missing or mismatched:
        details = []
        if missing:
            details.append(f"missing: {', '.join(missing[:10])}")
        if mismatched:
            details.append(f"checksum mismatch: {', '.join(mismatched[:10])}")
        raise RuntimeError("; ".join(details))
    print(f"Verified {len(checksums)} files from manifests/checksums.txt")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download MCG-HGT GitHub Release assets.")
    parser.add_argument("--tag", default="v1.0.0", help="GitHub Release tag.")
    parser.add_argument("--out", default=".", help="Output directory.")
    parser.add_argument("--dry-run", action="store_true", help="List assets without downloading.")
    parser.add_argument("--extract", action="store_true", help="Extract downloaded .tar.gz assets into --out.")
    parser.add_argument("--verify", action="store_true", help="Verify extracted files with manifests/checksums.txt.")
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
            dest = out / name
            download(url, dest)
            if args.extract and name.endswith(".tar.gz"):
                extract_tarball(dest, out)
    if args.verify and not args.dry_run:
        try:
            verify_checksums(out)
        except Exception as exc:
            print(f"Release asset verification failed: {exc}", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
