from __future__ import annotations

import argparse
import hashlib
import tarfile
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create a Zenodo-ready MCG-HGT artifact bundle.")
    parser.add_argument("--source", required=True, help="Directory containing weights/, preprocessed/, manifests/.")
    parser.add_argument("--output", default="MCG-HGT-zenodo-artifacts.tar.gz")
    args = parser.parse_args(argv)

    source = Path(args.source).resolve()
    if not source.is_dir():
        raise SystemExit(f"Source directory not found: {source}")

    required = ["weights", "preprocessed"]
    missing = [name for name in required if not (source / name).exists()]
    if missing:
        raise SystemExit(f"Missing required directories: {', '.join(missing)}")

    checksums = []
    for path in sorted(p for p in source.rglob("*") if p.is_file()):
        rel = path.relative_to(source).as_posix()
        checksums.append(f"{sha256_file(path)}  {rel}\n")
    manifest_dir = source / "manifests"
    manifest_dir.mkdir(exist_ok=True)
    (manifest_dir / "checksums.txt").write_text("".join(checksums), encoding="utf-8")

    output = Path(args.output).resolve()
    with tarfile.open(output, "w:gz") as tar:
        for path in sorted(source.rglob("*")):
            tar.add(path, arcname=path.relative_to(source.parent))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
