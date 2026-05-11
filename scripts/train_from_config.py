from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _simple_yaml(path: Path) -> dict[str, object]:
    data: dict[str, object] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        value = value.strip()
        if value.lower() in {"true", "false"}:
            data[key.strip()] = value.lower() == "true"
        else:
            try:
                if "." in value or "e-" in value.lower():
                    data[key.strip()] = float(value)
                else:
                    data[key.strip()] = int(value)
            except ValueError:
                data[key.strip()] = value
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run MCG-HGT training from a small YAML config.")
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs=argparse.REMAINDER, help="Additional arguments passed to training.")
    args = parser.parse_args(argv)

    config = _simple_yaml(Path(args.config))
    cmd = [sys.executable, "-m", "mcg_hgt.train"]
    for key, value in config.items():
        if key == "dataset":
            continue
        flag = "--" + key
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
        else:
            cmd.extend([flag, str(value)])
    if args.overrides:
        cmd.extend(args.overrides[1:] if args.overrides[0] == "--" else args.overrides)
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
