from __future__ import annotations

import argparse
import hashlib
import tarfile
from pathlib import Path


MAX_ASSET_BYTES = 1900 * 1024 * 1024


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_checksums(source: Path) -> None:
    checksums = []
    for path in sorted(p for p in source.rglob("*") if p.is_file()):
        rel = path.relative_to(source).as_posix()
        if rel == "manifests/checksums.txt":
            continue
        checksums.append(f"{sha256_file(path)}  {rel}\n")
    manifest_dir = source / "manifests"
    manifest_dir.mkdir(exist_ok=True)
    (manifest_dir / "checksums.txt").write_text("".join(checksums), encoding="utf-8")


def add_tree(tar: tarfile.TarFile, path: Path, arc_prefix: str) -> None:
    for item in sorted(path.rglob("*")):
        tar.add(item, arcname=f"{arc_prefix}/{item.relative_to(path).as_posix()}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create GitHub Release assets for MCG-HGT.")
    parser.add_argument("--source", required=True, help="Directory containing weights/, preprocessed/, manifests/.")
    parser.add_argument("--output-dir", default="release-assets")
    args = parser.parse_args(argv)

    source = Path(args.source).resolve()
    output = Path(args.output_dir).resolve()
    if not source.is_dir():
        raise SystemExit(f"Source directory not found: {source}")

    missing = [name for name in ["weights", "preprocessed"] if not (source / name).exists()]
    if missing:
        raise SystemExit(f"Missing required directories: {', '.join(missing)}")

    write_checksums(source)
    output.mkdir(parents=True, exist_ok=True)

    assets = {
        "MCG-HGT-HIT-preprocessed-inputs.tar.gz": ["preprocessed", "manifests", "README_RELEASE_ASSETS.md"],
        "MCG-HGT-HIT-CVS1-weights.tar.gz": ["weights"],
    }
    for asset_name, entries in assets.items():
        out_path = output / asset_name
        with tarfile.open(out_path, "w:gz") as tar:
            for entry in entries:
                p = source / entry
                if p.is_dir():
                    add_tree(tar, p, entry)
                elif p.is_file():
                    tar.add(p, arcname=entry)
        size = out_path.stat().st_size
        print(f"{out_path}\t{size}")
        if size > MAX_ASSET_BYTES:
            print(f"WARNING: {out_path.name} is larger than the conservative 1.9 GiB asset target.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
