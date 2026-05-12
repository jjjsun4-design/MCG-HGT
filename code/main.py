from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train MCG-HGT for ingredient-target interaction prediction.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=410)
    parser.add_argument("--k_fold", type=int, default=10)
    parser.add_argument("--cv_mode", choices=["CVS1", "CVS2", "CVS3", "cv1", "cv2", "cv3"], default="CVS1")
    parser.add_argument("--strict_cold_start", action="store_true")
    parser.add_argument("--strict_unseen", action="store_true")
    parser.add_argument("--no_exclude_cv1", action="store_true")
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--fanout", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--neg_k", type=int, default=5)
    parser.add_argument("--data_dir", type=str, default="data/HIT")
    parser.add_argument("--ligand_embed", type=str, default="data/HIT/ingredients_embeddings.csv")
    parser.add_argument("--ligand_id_key", type=str, default="node_id")
    parser.add_argument("--target_embed", type=str, default="data/HIT/protein_embeddings.csv")
    parser.add_argument("--target_id_key", type=str, default="node_id")
    parser.add_argument("--graph_struct", type=int, default=3)
    parser.add_argument("--method", type=int, default=5)
    parser.add_argument("--in_dim", type=int, default=512)
    parser.add_argument("--h_dim", type=int, default=1024)
    parser.add_argument("--out_dim", type=int, default=512)
    parser.add_argument("--hgt_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--input_gate_type", choices=["none", "se", "glu", "etype"], default="glu")
    parser.add_argument("--input_gate_reduce", type=int, default=4)
    parser.add_argument("--residual_gate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--score_gate", choices=["none", "gmu", "film"], default="gmu")
    parser.add_argument("--film_condition", choices=["src", "dst", "both"], default="src")
    parser.add_argument("--semantic_gate", choices=["none", "etype"], default="etype")
    parser.add_argument("--sem_gate_bias", type=float, default=0.8)
    parser.add_argument("--head_gate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ablate_ligand_llm", action="store_true")
    parser.add_argument("--ablate_target_llm", action="store_true")
    parser.add_argument("--llm_ablation_mode", choices=["zero", "random", "shuffle"], default="zero")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--num_epochs", type=int, default=250)
    parser.add_argument("--lr_period", type=int, default=30)
    parser.add_argument("--lr_decay", type=float, default=0.5)
    parser.add_argument("--use_cosine", action="store_true")
    parser.add_argument("--cosine_T0", type=int, default=10)
    parser.add_argument("--cosine_Tmult", type=int, default=2)
    parser.add_argument("--log_every", type=int, default=5)
    parser.add_argument("--val_every", type=int, default=3)
    parser.add_argument("--tau", type=float, default=0.07)
    parser.add_argument("--top_m", type=int, default=0)
    parser.add_argument("--proj_hidden_mult", type=int, default=4)
    parser.add_argument("--proj_dropout", type=float, default=0.2)
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/HIT/CVS1_MCG-HGT")
    parser.add_argument("--monitor_metric", choices=["auprc", "auroc"], default="auprc")
    parser.add_argument("--print_model", action="store_true")
    parser.add_argument("--cv1_eval", choices=["strict", "keep_reverse", "legacy"], default="keep_reverse")
    parser.add_argument("--augment_sim_loops", action="store_true")
    parser.add_argument("--augment_cv1", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    import torch

    from data import build_graph, process_data
    from training import _augment_similarity_graph_fallback, train

    try:
        from data import augment_similarity_graph
    except Exception:
        augment_similarity_graph = None

    if args.weight_decay is not None:
        args.wd = args.weight_decay
    if args.strict_cold_start:
        args.strict_unseen = True

    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    edges, is_edges, ts_edges, initial_features = process_data(args)
    hetero_graph, rel_list = build_graph(args, edges, is_edges, ts_edges, initial_features, device)

    do_aug = False
    if getattr(args, "augment_sim_loops", False):
        if args.cv_mode.upper() != "CVS1" or getattr(args, "augment_cv1", False):
            do_aug = True
    if do_aug:
        if augment_similarity_graph is not None:
            try:
                hetero_graph = augment_similarity_graph(hetero_graph)
            except Exception as exc:
                print(f"[augment] fallback due to: {type(exc).__name__}: {exc}")
                hetero_graph = _augment_similarity_graph_fallback(hetero_graph)
        else:
            hetero_graph = _augment_similarity_graph_fallback(hetero_graph)

    train(args, hetero_graph, rel_list, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
