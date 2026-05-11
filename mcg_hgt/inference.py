from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score compound-target pairs with a trained MCG-HGT checkpoint."
    )
    parser.add_argument("--data_dir", default="data/HIT", help="Directory containing edges and similarity tables.")
    parser.add_argument("--ligand_embed", default="data/HIT/ingredients_embeddings.csv", help="Ligand embedding file: CSV, NPY, NPZ, PT, or PTH.")
    parser.add_argument("--target_embed", default="data/HIT/protein_embeddings.csv", help="Target embedding file: CSV, NPY, NPZ, PT, or PTH.")
    parser.add_argument("--checkpoint", default="checkpoints/HIT/CVS1_MCG-HGT/best_fold1_auprc.pt", help="Trained model checkpoint.")
    parser.add_argument("--pairs", default="examples/smoke/pairs.csv", help="CSV with source,target columns or two unnamed columns.")
    parser.add_argument("--output", default="outputs/hit_scores.csv", help="Output CSV path.")
    parser.add_argument("--device", default="cuda:0", help="Torch device, for example cuda:0 or cpu.")
    parser.add_argument("--ligand_id_key", default="node_id")
    parser.add_argument("--target_id_key", default="node_id")
    parser.add_argument("--graph_struct", type=int, default=3)
    parser.add_argument("--method", type=int, default=5)
    parser.add_argument("--in_dim", type=int, default=512)
    parser.add_argument("--h_dim", type=int, default=1024)
    parser.add_argument("--out_dim", type=int, default=512)
    parser.add_argument("--hgt_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--input_gate_type", choices=["none", "se", "glu", "etype"], default="glu")
    parser.add_argument("--input_gate_reduce", type=int, default=4)
    parser.add_argument("--residual_gate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--score_gate", choices=["none", "gmu", "film"], default="gmu")
    parser.add_argument("--film_condition", choices=["src", "dst", "both"], default="src")
    parser.add_argument("--semantic_gate", choices=["none", "etype"], default="etype")
    parser.add_argument("--sem_gate_bias", type=float, default=0.8)
    parser.add_argument("--head_gate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--proj_hidden_mult", type=int, default=4)
    parser.add_argument("--proj_dropout", type=float, default=0.2)
    return parser


def _load_pairs(path: str):
    with open(path, newline="", encoding="utf-8") as handle:
        sample = handle.read(2048)
        handle.seek(0)
        has_header = csv.Sniffer().has_header(sample)
        reader = csv.DictReader(handle) if has_header else csv.reader(handle)
        pairs = []
        if has_header:
            for row in reader:
                src = row.get("source", row.get("ingredient", row.get("compound")))
                dst = row.get("target", row.get("protein"))
                if src is None or dst is None:
                    raise ValueError("Pairs CSV must contain source,target columns.")
                pairs.append((int(src), int(dst)))
        else:
            for row in reader:
                if len(row) < 2:
                    continue
                pairs.append((int(row[0]), int(row[1])))
    if not pairs:
        raise ValueError(f"No pairs found in {path}")
    return pairs


def _load_checkpoint_state(torch, checkpoint_path: str):
    try:
        obj = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        obj = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(obj, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
    return obj


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    import torch
    import dgl

    from .data import build_graph, process_data
    from .model import HGTModel

    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    edges, is_edges, ts_edges, initial_features = process_data(args)
    graph, rel_list = build_graph(args, edges, is_edges, ts_edges, initial_features, device)

    model = HGTModel(args, graph, rel_list).to(device)
    state = _load_checkpoint_state(torch, args.checkpoint)
    model.load_state_dict(state, strict=True)
    model.eval()

    pairs = _load_pairs(args.pairs)
    src = torch.tensor([p[0] for p in pairs], dtype=torch.long, device=device)
    dst = torch.tensor([p[1] for p in pairs], dtype=torch.long, device=device)
    score_graph = dgl.heterograph(
        {("ingredient", "it", "target"): (src, dst)},
        num_nodes_dict={
            "ingredient": graph.num_nodes("ingredient"),
            "target": graph.num_nodes("target"),
        },
    ).to(device)

    with torch.no_grad():
        z = model.encoder()
        scores = model.pred(score_graph, z)[("ingredient", "it", "target")]
        probs = torch.sigmoid(scores).detach().cpu().numpy()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source", "target", "logit", "probability"])
        for (src_id, dst_id), logit, prob in zip(pairs, scores.detach().cpu().numpy(), probs):
            writer.writerow([src_id, dst_id, float(logit), float(prob)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
