from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


def _write_embeddings(path: Path, n_rows: int, dim: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["node_id"] + [f"f{i}" for i in range(dim)])
        for i in range(n_rows):
            writer.writerow([i] + [round(((i + 1) * (j + 1)) / 100.0, 6) for j in range(dim)])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a tiny MCG-HGT smoke test.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workdir", default="examples/smoke")
    parser.add_argument("--train", action="store_true", help="Run one training epoch if torch and dgl are installed.")
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[1]
    workdir = (root / args.workdir).resolve()
    ligand = workdir / "ligand_embeddings.csv"
    target = workdir / "target_embeddings.csv"
    _write_embeddings(ligand, n_rows=5, dim=8)
    _write_embeddings(target, n_rows=4, dim=8)

    help_cmds = [
        [sys.executable, "-m", "mcg_hgt.train", "--help"],
        [sys.executable, "-m", "mcg_hgt.inference", "--help"],
        [sys.executable, "-m", "mcg_hgt.evaluate", "--help"],
    ]
    for cmd in help_cmds:
        subprocess.run(cmd, cwd=root, check=True, stdout=subprocess.DEVNULL)

    if not args.train:
        print("Smoke help checks passed. Use --train in an environment with torch and dgl to run 1 epoch.")
        return 0

    try:
        import torch  # noqa: F401
        import dgl  # noqa: F401
    except Exception as exc:
        print(f"Cannot run training smoke test because a dependency is missing: {exc}", file=sys.stderr)
        return 2

    cmd = [
        sys.executable,
        "-m",
        "mcg_hgt.train",
        "--device",
        args.device,
        "--k_fold",
        "2",
        "--batch_size",
        "2",
        "--num_epochs",
        str(args.epochs),
        "--val_every",
        "1",
        "--data_dir",
        str(workdir),
        "--ligand_embed",
        str(ligand),
        "--target_embed",
        str(target),
        "--in_dim",
        "8",
        "--h_dim",
        "8",
        "--out_dim",
        "8",
        "--hgt_heads",
        "2",
        "--num_layers",
        "1",
        "--fanout",
        "2",
        "--neg_k",
        "1",
        "--checkpoint_dir",
        str(workdir / "checkpoints"),
        "--input_gate_type",
        "none",
        "--score_gate",
        "gmu",
        "--semantic_gate",
        "none",
        "--cv1_eval",
        "strict",
    ]
    subprocess.run(cmd, cwd=root, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
