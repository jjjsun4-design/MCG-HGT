from __future__ import annotations

import argparse
import csv
from pathlib import Path

from checkpoint_contract import (
    MODEL_CONFIG_FIELDS,
    resolve_checkpoint_contract,
    validate_graph_scope,
)


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
    parser.add_argument("--encoder_type", choices=["hgt", "feature_only"], default=None)
    parser.add_argument("--in_dim", type=int, default=None)
    parser.add_argument("--h_dim", type=int, default=None)
    parser.add_argument("--out_dim", type=int, default=None)
    parser.add_argument("--hgt_heads", type=int, default=None)
    parser.add_argument("--num_layers", type=int, default=None)
    parser.add_argument("--share_hgt_layers", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--input_gate_type", choices=["none", "se", "glu", "etype"], default=None)
    parser.add_argument("--input_gate_reduce", type=int, default=None)
    parser.add_argument("--residual_gate", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--residual_message_prior", type=float, default=None)
    parser.add_argument("--feature_fusion", choices=["none", "type_scalar", "type_dynamic"], default=None)
    parser.add_argument("--feature_graph_prior", type=float, default=None)
    parser.add_argument("--score_gate", choices=["none", "gmu", "film"], default=None)
    parser.add_argument("--score_fusion", choices=["none", "feature_residual"], default=None)
    parser.add_argument("--score_graph_prior", type=float, default=None)
    parser.add_argument("--film_condition", choices=["src", "dst", "both"], default=None)
    parser.add_argument("--semantic_gate", choices=["none", "etype"], default=None)
    parser.add_argument("--sem_gate_bias", type=float, default=None)
    parser.add_argument("--head_gate", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--proj_hidden_mult", type=int, default=None)
    parser.add_argument("--proj_dropout", type=float, default=None)
    parser.add_argument(
        "--allow_legacy_unknown_graph_scope",
        action="store_true",
        help=(
            "Explicitly accept a legacy checkpoint whose graph scope is missing "
            "or unknown. Known fold-specific scopes remain blocked."
        ),
    )
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


def _load_checkpoint(torch, checkpoint_path: str):
    try:
        obj = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        obj = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(obj, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key], obj
    raise ValueError(
        "Portable inference requires a checkpoint payload with a state_dict mapping"
    )


_ARCH_DEFAULTS = {
    "encoder_type": "hgt",
    "in_dim": 512,
    "h_dim": 2048,
    "out_dim": 512,
    "hgt_heads": 8,
    "num_layers": 3,
    "share_hgt_layers": False,
    "dropout": 0.2,
    "input_gate_type": "glu",
    "input_gate_reduce": 4,
    "residual_gate": True,
    "residual_message_prior": None,
    "feature_fusion": "none",
    "feature_graph_prior": 0.1,
    "score_gate": "gmu",
    "score_fusion": "none",
    "score_graph_prior": 0.1,
    "film_condition": "src",
    "semantic_gate": "etype",
    "sem_gate_bias": 0.8,
    "head_gate": True,
    "proj_hidden_mult": 4,
    "proj_dropout": 0.2,
}


def _state_architecture(state):
    keys = [str(key) for key in state]
    shared = any("encoder.shared_hgt." in key for key in keys)
    unshared = any("encoder.layers." in key for key in keys)
    if not shared and not unshared:
        return {"encoder_type": "feature_only"}
    if shared and unshared:
        raise ValueError("Checkpoint does not identify exactly one HGT parameter layout")
    prefix = "module." if any(key.startswith("module.encoder.") for key in keys) else ""
    result = {"encoder_type": "hgt", "share_hgt_layers": shared}
    if shared:
        norm_ids = {
            int(key.split("encoder.norms.", 1)[1].split(".", 1)[0])
            for key in keys if "encoder.norms." in key
        }
        result["num_layers"] = len(norm_ids)
        for name, field in (
            ("encoder.shared_input_adapter.weight", "h_dim"),
            ("encoder.shared_output_norm.weight", "out_dim"),
        ):
            tensor = state.get(prefix + name)
            if tensor is not None:
                result[field] = int(tensor.shape[0])
    else:
        layer_ids = {
            int(key.split("encoder.layers.", 1)[1].split(".", 1)[0])
            for key in keys if "encoder.layers." in key
        }
        result["num_layers"] = len(layer_ids)
        first_norm = state.get(prefix + "encoder.norms.0.weight")
        last_norm = state.get(prefix + f"encoder.norms.{len(layer_ids) - 1}.weight")
        if first_norm is not None:
            result["h_dim"] = int(first_norm.shape[0])
        if last_norm is not None:
            result["out_dim"] = int(last_norm.shape[0])
    for key, tensor in state.items():
        if "encoder.input_proj." in str(key) and str(key).endswith(".3.weight"):
            result["in_dim"] = int(tensor.shape[0])
            hidden = int(tensor.shape[1])
            result["proj_hidden_mult"] = max(1, hidden // result["in_dim"])
            break
    return result


def _resolve_architecture(cli_args, payload, state):
    cli_values = {
        field: getattr(cli_args, field, None)
        for field in MODEL_CONFIG_FIELDS
        if field != "schema_version"
    }
    normalized_state, model_config = resolve_checkpoint_contract(
        payload, state, cli_values
    )
    for name, value in model_config.items():
        if name != "schema_version":
            setattr(cli_args, name, value)
    return cli_args, normalized_state


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    import torch
    import dgl

    from data import build_graph, process_data
    from model import HGTModel

    state, checkpoint_payload = _load_checkpoint(torch, args.checkpoint)
    validate_graph_scope(
        checkpoint_payload,
        allow_legacy_unknown=args.allow_legacy_unknown_graph_scope,
    )
    args, state = _resolve_architecture(args, checkpoint_payload, state)

    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    edges, is_edges, ts_edges, initial_features = process_data(args)
    graph, rel_list = build_graph(args, edges, is_edges, ts_edges, initial_features, device)

    model = HGTModel(args, graph, rel_list).to(device)
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
        scores = model.score_pairs(score_graph)[("ingredient", "it", "target")]
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
