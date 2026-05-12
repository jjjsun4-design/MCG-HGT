from __future__ import annotations

import argparse
import csv
import tempfile
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained MCG-HGT checkpoint on a labeled pair CSV."
    )
    parser.add_argument("--labels", required=True, help="CSV with source,target,label columns.")
    parser.add_argument("--output", required=True, help="Metrics output JSON path.")
    parser.add_argument("--scores_csv", default=None, help="Optional path for per-pair scores.")
    parser.add_argument("inference_args", nargs=argparse.REMAINDER, help="Arguments forwarded to mcg_hgt.inference.")
    return parser


def _read_labels(path: str):
    rows = []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"source", "target", "label"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns in labels CSV: {sorted(missing)}")
        for row in reader:
            rows.append((int(row["source"]), int(row["target"]), int(row["label"])))
    if not rows:
        raise ValueError(f"No labeled pairs found in {path}")
    return rows


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.inference_args and args.inference_args[0] == "--":
        args.inference_args = args.inference_args[1:]

    import json
    import numpy as np
    from sklearn.metrics import average_precision_score, roc_auc_score

    from inference import main as inference_main

    labels = _read_labels(args.labels)
    score_path = Path(args.scores_csv) if args.scores_csv else Path(tempfile.mkstemp(suffix=".csv")[1])
    pairs_path = Path(tempfile.mkstemp(suffix=".csv")[1])
    with pairs_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source", "target"])
        for src, dst, _ in labels:
            writer.writerow([src, dst])

    forwarded = list(args.inference_args) + ["--pairs", str(pairs_path), "--output", str(score_path)]
    rc = inference_main(forwarded)
    if rc != 0:
        return rc

    y_true = np.array([label for _, _, label in labels], dtype=int)
    y_score = []
    with score_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            y_score.append(float(row["logit"]))
    y_score = np.asarray(y_score)

    metrics = {
        "auroc": float(roc_auc_score(y_true, y_score)) if len(set(y_true.tolist())) > 1 else None,
        "auprc": float(average_precision_score(y_true, y_score)),
        "n_pairs": int(len(y_true)),
        "n_positive": int(y_true.sum()),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
