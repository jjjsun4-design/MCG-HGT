# -*- coding: utf-8 -*-
"""MCG-HGT training with leakage-controlled supervised graphs.

The default ``nested`` protocol uses an outer test split and an inner validation
split.  Epoch and decision-threshold selection see only the inner validation
set; the outer test set is scored once after a fresh fixed-epoch fit on all
outer-training positives.  Every message graph contains only its stage's
training ``it`` edges and their exact ``ti`` reverse counterparts.

The historical behavior is available only through ``--cv_protocol legacy``.
It is retained for provenance comparisons and must not be used for revised
results. ``fold_isolated`` is retained as the W0 graph-isolation diagnostic; it
still uses one held-out fold for both monitoring and reporting and therefore is
not a submission-grade evaluation protocol.
"""

from __future__ import annotations

import csv
import hashlib
import os
import re
import time
import numpy as np
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
import dgl
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, average_precision_score
from dgl.dataloading import NeighborSampler, as_edge_prediction_sampler, DataLoader
from dgl.dataloading.negative_sampler import GlobalUniform
from torch.optim.swa_utils import AveragedModel   # 用作 EMA

try:
    from checkpoint_contract import (
        CHECKPOINT_SCHEMA_VERSION,
        MODEL_CONFIG_FIELDS,
        architecture_id_for,
    )
except ImportError:  # pragma: no cover
    from .checkpoint_contract import (  # type: ignore
        CHECKPOINT_SCHEMA_VERSION,
        MODEL_CONFIG_FIELDS,
        architecture_id_for,
    )


COMPUTATIONAL_COST_SCHEMA_VERSION = 1


def _cuda_metrics_available(device) -> bool:
    normalized = torch.device(device)
    return normalized.type == "cuda" and torch.cuda.is_available()


def _synchronize_for_timing(device) -> None:
    if _cuda_metrics_available(device):
        torch.cuda.synchronize(torch.device(device))


def _reset_peak_gpu_memory(device) -> None:
    if _cuda_metrics_available(device):
        _synchronize_for_timing(device)
        torch.cuda.reset_peak_memory_stats(torch.device(device))


def _time_operation(timing_device, operation, *args, **kwargs):
    """Return an operation result and synchronized wall-clock duration."""
    _synchronize_for_timing(timing_device)
    start = time.perf_counter()
    result = operation(*args, **kwargs)
    _synchronize_for_timing(timing_device)
    return result, max(0.0, float(time.perf_counter() - start))


def _parameter_counts(model) -> Dict[str, int]:
    parameters = list(model.parameters())
    return {
        "total": int(sum(parameter.numel() for parameter in parameters)),
        "trainable": int(
            sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
        ),
    }


def _architecture_record(model) -> Dict[str, Any]:
    encoder = getattr(model, "encoder", None)
    if encoder is None or not hasattr(encoder, "num_propagation_steps"):
        return {"encoder_type": "feature_only"}
    shared = bool(getattr(encoder, "share_hgt_layers", False))
    steps = int(encoder.num_propagation_steps)
    return {
        "encoder_type": "hgt",
        "hidden_dimension": int(encoder.hidden_dimension),
        "hgt_propagation_steps": steps,
        "attention_heads": int(encoder.heads),
        "hgt_parameter_sets": 1 if shared else steps,
        "weight_shared_hgt": shared,
        "input_projection_hidden_multiplier": int(encoder.proj_hidden_mult),
    }


def _checkpoint_model_metadata(model, args) -> Dict[str, Any]:
    architecture = _architecture_record(model)
    values = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "encoder_type": architecture.get("encoder_type", "feature_only"),
        "in_dim": getattr(args, "in_dim", None),
        "h_dim": architecture.get("hidden_dimension", getattr(args, "h_dim", None)),
        "out_dim": getattr(args, "out_dim", None),
        "hgt_heads": architecture.get("attention_heads", getattr(args, "hgt_heads", None)),
        "num_layers": architecture.get("hgt_propagation_steps", getattr(args, "num_layers", None)),
        "share_hgt_layers": architecture.get("weight_shared_hgt", getattr(args, "share_hgt_layers", False)),
        "hgt_parameter_sets": architecture.get("hgt_parameter_sets", 0),
        "dropout": getattr(args, "dropout", 0.2),
        "input_gate_type": getattr(args, "input_gate_type", "none"),
        "input_gate_reduce": getattr(args, "input_gate_reduce", 4),
        "residual_gate": getattr(args, "residual_gate", False),
        "residual_message_prior": getattr(args, "residual_message_prior", None),
        "gate_bias": getattr(args, "gate_bias", 1.0),
        "feature_fusion": getattr(args, "feature_fusion", "none"),
        "feature_graph_prior": getattr(args, "feature_graph_prior", 0.1),
        "score_gate": getattr(args, "score_gate", "none"),
        "score_fusion": getattr(args, "score_fusion", "none"),
        "score_graph_prior": getattr(args, "score_graph_prior", 0.1),
        "film_condition": getattr(args, "film_condition", "dst"),
        "semantic_gate": getattr(args, "semantic_gate", "none"),
        "sem_hidden": getattr(args, "sem_hidden", 64),
        "sem_gate_bias": getattr(args, "sem_gate_bias", 0.0),
        "head_gate": getattr(args, "head_gate", False),
        "proj_hidden_mult": architecture.get(
            "input_projection_hidden_multiplier",
            getattr(args, "proj_hidden_mult", 2),
        ),
        "proj_dropout": getattr(args, "proj_dropout", 0.2),
    }
    model_config = {field: values[field] for field in MODEL_CONFIG_FIELDS}
    return {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "architecture_id": architecture_id_for(model_config),
        "model_config": model_config,
    }


def _execution_record(args, device) -> Dict[str, Any]:
    return {
        "tf32_requested": bool(getattr(args, "tf32", False)),
        "tf32_active": bool(getattr(args, "tf32_active", False)),
        "fused_adamw_requested": bool(getattr(args, "fused_adamw", False)),
        "fused_adamw_active": bool(getattr(args, "fused_adamw_active", False)),
        "device_type": torch.device(device).type,
    }


def _gpu_memory_record(device) -> Dict[str, Any]:
    if not _cuda_metrics_available(device):
        return {
            "status": "unavailable",
            "peak_allocated_bytes": 0,
            "peak_allocated_mib": 0.0,
        }
    _synchronize_for_timing(device)
    peak_bytes = int(torch.cuda.max_memory_allocated(torch.device(device)))
    return {
        "status": "available",
        "peak_allocated_bytes": peak_bytes,
        "peak_allocated_mib": float(peak_bytes / (1024 ** 2)),
    }


def _computational_cost_record(
    model,
    device,
    training_wall_time_seconds: float,
    inference_wall_time_seconds: float,
    inference_scored_pairs: int,
    args=None,
) -> Dict[str, Any]:
    training_seconds = float(training_wall_time_seconds)
    inference_seconds = float(inference_wall_time_seconds)
    scored_pairs = int(inference_scored_pairs)
    if training_seconds < 0 or inference_seconds < 0 or scored_pairs < 0:
        raise ValueError("Computational-cost measurements must be non-negative")
    throughput = scored_pairs / inference_seconds if inference_seconds > 0 else 0.0
    normalized_device = torch.device(device)
    return {
        "schema_version": COMPUTATIONAL_COST_SCHEMA_VERSION,
        "scope": "outer_fold",
        "device": {
            "label": str(normalized_device),
            "type": normalized_device.type,
            "cuda_metrics_available": _cuda_metrics_available(normalized_device),
        },
        "parameters": _parameter_counts(model),
        "architecture": _architecture_record(model),
        "execution": _execution_record(args, normalized_device),
        "training": {
            "scope": "inner_selection_including_validation_plus_outer_refit",
            "wall_time_seconds": training_seconds,
        },
        "inference": {
            "scope": "outer_test_scoring",
            "wall_time_seconds": inference_seconds,
            "scored_pairs": scored_pairs,
            "throughput_pairs_per_second": float(throughput),
        },
        "gpu_memory": _gpu_memory_record(normalized_device),
    }


def _aggregate_computational_cost(
    fold_records: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    if not fold_records:
        raise ValueError("At least one fold cost record is required")
    parameter_records = {
        (record["parameters"]["total"], record["parameters"]["trainable"])
        for record in fold_records
    }
    if len(parameter_records) != 1:
        raise AssertionError("Model parameter counts differ across outer folds")
    architecture_records = {
        tuple(sorted(record.get("architecture", {}).items()))
        for record in fold_records
    }
    execution_records = {
        tuple(sorted(record.get("execution", {}).items()))
        for record in fold_records
    }
    if len(architecture_records) != 1:
        raise AssertionError("Model architecture differs across outer folds")
    if len(execution_records) != 1:
        raise AssertionError("Execution settings differ across outer folds")
    device_records = {
        (
            record["device"]["label"],
            record["device"]["type"],
            record["device"]["cuda_metrics_available"],
        )
        for record in fold_records
    }
    if len(device_records) != 1:
        raise AssertionError("Compute devices differ across outer folds")

    training_seconds = float(
        sum(record["training"]["wall_time_seconds"] for record in fold_records)
    )
    inference_seconds = float(
        sum(record["inference"]["wall_time_seconds"] for record in fold_records)
    )
    scored_pairs = int(
        sum(record["inference"]["scored_pairs"] for record in fold_records)
    )
    throughput = scored_pairs / inference_seconds if inference_seconds > 0 else 0.0
    peak_bytes = max(
        int(record["gpu_memory"]["peak_allocated_bytes"])
        for record in fold_records
    )
    memory_available = any(
        record["gpu_memory"]["status"] == "available" for record in fold_records
    )
    device_label, device_type, cuda_available = next(iter(device_records))
    total_parameters, trainable_parameters = next(iter(parameter_records))
    return {
        "schema_version": COMPUTATIONAL_COST_SCHEMA_VERSION,
        "scope": "nested_run",
        "fold_count": len(fold_records),
        "device": {
            "label": device_label,
            "type": device_type,
            "cuda_metrics_available": bool(cuda_available),
        },
        "parameters": {
            "total": int(total_parameters),
            "trainable": int(trainable_parameters),
        },
        "architecture": dict(next(iter(architecture_records))),
        "execution": dict(next(iter(execution_records))),
        "training": {
            "scope": "sum_of_outer_fold_inner_selection_and_refit",
            "wall_time_seconds": training_seconds,
        },
        "inference": {
            "scope": "pooled_outer_test_scoring",
            "wall_time_seconds": inference_seconds,
            "scored_pairs": scored_pairs,
            "throughput_pairs_per_second": float(throughput),
        },
        "gpu_memory": {
            "status": "available" if memory_available else "unavailable",
            "peak_allocated_bytes": peak_bytes,
            "peak_allocated_mib": float(peak_bytes / (1024 ** 2)),
        },
    }

try:
    from nested_cv import (
        as_numpy_eids,
        classification_metrics,
        hash_eids,
        make_nested_splits,
        select_threshold,
        write_oof_predictions,
        write_split_manifest,
        write_summary,
    )
except ImportError:  # pragma: no cover - package-style import fallback
    from .nested_cv import (  # type: ignore
        as_numpy_eids,
        classification_metrics,
        hash_eids,
        make_nested_splits,
        select_threshold,
        write_oof_predictions,
        write_split_manifest,
        write_summary,
    )

# =============== Package imports ===============
try:
    from model import HGTModel as Model
    from data import (
        set_seed, process_data, build_graph, compute_loss, remove_unseen_nodes
    )
    try:
        from data import augment_similarity_graph  # type: ignore
    except Exception:
        augment_similarity_graph = None
except ImportError:
    # Fallback for direct execution during local development.
    from model import HGTModel as Model  # type: ignore
    from data import (  # type: ignore
        set_seed, process_data, build_graph, compute_loss, remove_unseen_nodes
    )
    try:
        from data import augment_similarity_graph  # type: ignore
    except Exception:
        augment_similarity_graph = None

# =============== 反向关系映射（用于 exclude='reverse_types'） ===============
def _reverse_map_all(g: dgl.DGLHeteroGraph) -> Dict[str, str]:
    rev = {}
    if ('ingredient', 'it', 'target') in g.canonical_etypes:
        rev['it'] = 'ti'
    if ('target', 'ti', 'ingredient') in g.canonical_etypes:
        rev['ti'] = 'it'
    if ('ingredient', 'is', 'ingredient') in g.canonical_etypes:
        rev['is'] = 'is'
    if ('target', 'ts', 'target') in g.canonical_etypes:
        rev['ts'] = 'ts'
    return rev


def _canonical_etype(g: dgl.DGLHeteroGraph, etype) -> Tuple[str, str, str]:
    return tuple(g.to_canonical_etype(etype))


def _edge_pairs(
    g: dgl.DGLHeteroGraph,
    etype,
    eids: torch.Tensor | Sequence[int] | None = None,
) -> List[Tuple[int, int]]:
    canonical = _canonical_etype(g, etype)
    if eids is None:
        src, dst = g.edges(etype=canonical)
    else:
        eids_t = torch.as_tensor(eids, dtype=torch.long, device=g.device).reshape(-1)
        src, dst = g.find_edges(eids_t, etype=canonical)
    return list(zip(src.detach().cpu().tolist(), dst.detach().cpu().tolist()))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_strict_csv(
    path_value: str | os.PathLike[str] | None,
    expected_columns: Sequence[str],
    label: str,
) -> Tuple[Path | None, List[Tuple[int, Dict[str, str]]], str | None]:
    if path_value is None or not str(path_value).strip():
        return None, [], None
    path = Path(path_value).expanduser()
    try:
        path = path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValueError(f"{label} CSV does not exist: {path}") from exc
    if not path.is_file():
        raise ValueError(f"{label} CSV is not a regular file: {path}")

    expected = tuple(expected_columns)
    rows: List[Tuple[int, Dict[str, str]]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        if len(fieldnames) != len(set(fieldnames)):
            raise ValueError(f"{label} CSV contains duplicate column names")
        if set(fieldnames) != set(expected) or len(fieldnames) != len(expected):
            raise ValueError(
                f"{label} CSV must contain exactly the columns "
                f"{list(expected)}; found {list(fieldnames)}"
            )
        for row in reader:
            line_number = int(reader.line_num)
            if None in row or any(row.get(column) is None for column in expected):
                raise ValueError(f"{label} CSV row {line_number} is malformed")
            rows.append(
                (line_number, {column: str(row[column]) for column in expected})
            )
    return path, rows, _sha256_file(path)


def _parse_nonnegative_id(raw_value: str, label: str, line_number: int) -> int:
    if raw_value != raw_value.strip() or re.fullmatch(r"[0-9]+", raw_value) is None:
        raise ValueError(
            f"{label} CSV row {line_number} must use a non-negative integer ID"
        )
    return int(raw_value)


def _load_node_registry(
    path_value: str | os.PathLike[str] | None,
    key_column: str,
    expected_node_count: int,
    label: str,
) -> Tuple[Tuple[str, ...] | None, Dict[str, Any]]:
    path, rows, sha256 = _read_strict_csv(
        path_value, ("node_id", key_column), label
    )
    if path is None:
        return None, {
            "path": None,
            "sha256": None,
            "count": 0,
            "key_column": key_column,
        }
    if len(rows) != int(expected_node_count):
        raise ValueError(
            f"{label} CSV row count ({len(rows)}) must match graph node count "
            f"({int(expected_node_count)})"
        )

    keys_by_id: List[str | None] = [None] * int(expected_node_count)
    seen_ids: set[int] = set()
    seen_keys: set[str] = set()
    for line_number, row in rows:
        node_id = _parse_nonnegative_id(row["node_id"], label, line_number)
        if node_id in seen_ids:
            raise ValueError(f"{label} CSV contains duplicate node_id {node_id}")
        if node_id >= int(expected_node_count):
            raise ValueError(
                f"{label} CSV node_id {node_id} is outside "
                f"[0, {int(expected_node_count)})"
            )
        key = row[key_column]
        if key != key.strip() or not key or any(ord(char) < 32 for char in key):
            raise ValueError(
                f"{label} CSV row {line_number} has an invalid {key_column}"
            )
        if key in seen_keys:
            raise ValueError(f"{label} CSV contains duplicate {key_column} {key!r}")
        seen_ids.add(node_id)
        seen_keys.add(key)
        keys_by_id[node_id] = key

    expected_ids = set(range(int(expected_node_count)))
    if seen_ids != expected_ids:
        missing = sorted(expected_ids - seen_ids)
        raise ValueError(
            f"{label} CSV node_id values must be continuous from 0 to "
            f"{int(expected_node_count) - 1}; missing {missing[:10]}"
        )
    return tuple(str(key) for key in keys_by_id), {
        "path": str(path),
        "sha256": sha256,
        "count": len(rows),
        "key_column": key_column,
    }


def _load_known_positive_exclusions(
    path_value: str | os.PathLike[str] | None,
    num_compounds: int,
    num_targets: int,
) -> Tuple[set[Tuple[int, int]], Dict[str, Any]]:
    label = "known-positive exclusions"
    path, rows, sha256 = _read_strict_csv(
        path_value, ("compound_id", "target_id"), label
    )
    pairs: set[Tuple[int, int]] = set()
    for line_number, row in rows:
        compound_id = _parse_nonnegative_id(
            row["compound_id"], label, line_number
        )
        target_id = _parse_nonnegative_id(row["target_id"], label, line_number)
        if compound_id >= int(num_compounds):
            raise ValueError(
                f"{label} CSV compound_id {compound_id} is outside "
                f"[0, {int(num_compounds)})"
            )
        if target_id >= int(num_targets):
            raise ValueError(
                f"{label} CSV target_id {target_id} is outside "
                f"[0, {int(num_targets)})"
            )
        pair = (compound_id, target_id)
        if pair in pairs:
            raise ValueError(f"{label} CSV contains duplicate pair {pair}")
        pairs.add(pair)
    return pairs, {
        "path": str(path) if path is not None else None,
        "sha256": sha256,
        "count": len(pairs),
        "row_count": len(rows),
    }


def _prepare_evaluation_inputs(args, graph, sup_rel):
    compound_path = getattr(args, "compound_registry", None)
    target_path = getattr(args, "target_registry", None)
    if bool(compound_path) != bool(target_path):
        raise ValueError(
            "--compound_registry and --target_registry must be provided together"
        )

    num_compounds = int(graph.num_nodes(sup_rel[0]))
    num_targets = int(graph.num_nodes(sup_rel[2]))
    compound_keys, compound_metadata = _load_node_registry(
        compound_path,
        "compound_key",
        num_compounds,
        "compound registry",
    )
    target_keys, target_metadata = _load_node_registry(
        target_path,
        "target_key",
        num_targets,
        "target registry",
    )
    external_pairs, exclusion_metadata = _load_known_positive_exclusions(
        getattr(args, "known_positive_exclusions", None),
        num_compounds,
        num_targets,
    )
    local_pairs = set(_edge_pairs(graph, sup_rel))
    known_pairs = local_pairs | external_pairs
    if not local_pairs.issubset(known_pairs):
        raise AssertionError("Local supervised positives are missing from the oracle")
    exclusion_metadata.update(
        {
            "external_only_count": len(external_pairs - local_pairs),
            "local_positive_count": len(local_pairs),
            "known_positive_union_count": len(known_pairs),
            "local_positives_are_subset": True,
        }
    )
    provenance = {
        "known_positive_exclusions": exclusion_metadata,
        "compound_registry": compound_metadata,
        "target_registry": target_metadata,
    }
    return sorted(known_pairs), compound_keys, target_keys, provenance


def _canonical_pair_fields(
    source_id: int,
    target_id: int,
    compound_keys: Sequence[str] | None,
    target_keys: Sequence[str] | None,
) -> Tuple[str, str, str]:
    if compound_keys is None and target_keys is None:
        return "", "", ""
    if compound_keys is None or target_keys is None:
        raise ValueError("Canonical compound and target registries must be paired")
    compound_key = str(compound_keys[int(source_id)])
    target_key = str(target_keys[int(target_id)])
    return compound_key, target_key, f"{compound_key}::{target_key}"


def _find_reverse_etype(
    g: dgl.DGLHeteroGraph,
    sup_rel,
) -> Tuple[str, str, str]:
    sup_rel = _canonical_etype(g, sup_rel)
    preferred_name = _reverse_map_all(g).get(sup_rel[1])
    candidates = [
        etype
        for etype in g.canonical_etypes
        if etype[0] == sup_rel[2] and etype[2] == sup_rel[0]
    ]
    if preferred_name is not None:
        preferred = [etype for etype in candidates if etype[1] == preferred_name]
        if len(preferred) == 1:
            return preferred[0]
    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one reverse relation for {sup_rel}, found {candidates}"
        )
    return candidates[0]


def _validate_fold_partition(
    g: dgl.DGLHeteroGraph,
    sup_rel,
    train_eids_on_full: torch.Tensor,
    val_eids_on_full: torch.Tensor,
) -> None:
    sup_rel = _canonical_etype(g, sup_rel)
    num_edges = g.num_edges(sup_rel)
    all_pairs = _edge_pairs(g, sup_rel)
    duplicate_pairs = {
        pair: count for pair, count in Counter(all_pairs).items() if count > 1
    }
    if duplicate_pairs:
        raise AssertionError(
            "Supervised interaction pairs must be globally unique before "
            f"cross-validation. duplicates={list(duplicate_pairs.items())[:10]}"
        )
    train_ids = [int(eid) for eid in train_eids_on_full.detach().cpu().tolist()]
    val_ids = [int(eid) for eid in val_eids_on_full.detach().cpu().tolist()]

    if len(train_ids) != len(set(train_ids)) or len(val_ids) != len(set(val_ids)):
        raise AssertionError("Fold EID lists must not contain duplicates")
    if set(train_ids).intersection(val_ids):
        raise AssertionError("Training and held-out EIDs overlap")
    expected = set(range(num_edges))
    observed = set(train_ids).union(val_ids)
    if observed != expected:
        missing = sorted(expected.difference(observed))[:10]
        extra = sorted(observed.difference(expected))[:10]
        raise AssertionError(
            f"Fold EIDs must partition all supervised edges; missing={missing}, extra={extra}"
        )

    train_pairs = set(_edge_pairs(g, sup_rel, train_eids_on_full))
    heldout_pairs = set(_edge_pairs(g, sup_rel, val_eids_on_full))
    overlap = train_pairs.intersection(heldout_pairs)
    if overlap:
        sample = sorted(overlap)[:10]
        raise AssertionError(
            "The same supervised pair occurs in both train and held-out EIDs; "
            f"deduplicate before cross-validation. sample={sample}"
        )


def _matching_reverse_eids(
    g: dgl.DGLHeteroGraph,
    sup_rel,
    reverse_rel,
    train_eids_on_full: torch.Tensor,
) -> torch.Tensor:
    sup_rel = _canonical_etype(g, sup_rel)
    reverse_rel = _canonical_etype(g, reverse_rel)
    forward_pairs = _edge_pairs(g, sup_rel)
    reverse_pairs = [(dst, src) for src, dst in _edge_pairs(g, reverse_rel)]
    if Counter(forward_pairs) != Counter(reverse_pairs):
        missing = Counter(forward_pairs) - Counter(reverse_pairs)
        extra = Counter(reverse_pairs) - Counter(forward_pairs)
        raise AssertionError(
            "Forward and reverse supervised relations are not exact pair-level mirrors; "
            f"missing_reverse={list(missing.items())[:10]}, "
            f"extra_reverse={list(extra.items())[:10]}"
        )

    reverse_eids_by_pair = defaultdict(deque)
    for reverse_eid, pair in enumerate(reverse_pairs):
        reverse_eids_by_pair[pair].append(reverse_eid)

    selected = []
    for pair in _edge_pairs(g, sup_rel, train_eids_on_full):
        if not reverse_eids_by_pair[pair]:
            raise AssertionError(f"Missing reverse edge for training pair {pair}")
        selected.append(reverse_eids_by_pair[pair].popleft())
    return torch.as_tensor(selected, dtype=torch.long, device=g.device)


def _copy_edge_induced_graph(
    g: dgl.DGLHeteroGraph,
    keep_eids: Dict[Tuple[str, str, str], torch.Tensor],
) -> dgl.DGLHeteroGraph:
    graph_data = {}
    normalized_eids = {}
    for etype in g.canonical_etypes:
        eids = torch.as_tensor(
            keep_eids[etype], dtype=torch.long, device=g.device
        ).reshape(-1)
        src, dst = g.find_edges(eids, etype=etype)
        graph_data[etype] = (src, dst)
        normalized_eids[etype] = eids

    num_nodes_dict = {ntype: g.num_nodes(ntype) for ntype in g.ntypes}
    try:
        fold_graph = dgl.heterograph(
            graph_data, num_nodes_dict=num_nodes_dict, device=g.device
        )
    except TypeError:
        fold_graph = dgl.heterograph(
            graph_data, num_nodes_dict=num_nodes_dict
        ).to(g.device)

    for ntype in g.ntypes:
        for key, value in g.nodes[ntype].data.items():
            fold_graph.nodes[ntype].data[key] = value
    for etype in g.canonical_etypes:
        eids = normalized_eids[etype]
        for key, value in g.edges[etype].data.items():
            fold_graph.edges[etype].data[key] = value[eids]
        fold_graph.edges[etype].data[dgl.EID] = eids
    return fold_graph


def _assert_strict_fold_isolation(
    fold_graph: dgl.DGLHeteroGraph,
    sup_rel,
    reverse_rel,
    expected_train_pairs: Iterable[Tuple[int, int]],
    heldout_pairs: Iterable[Tuple[int, int]],
) -> None:
    sup_rel = _canonical_etype(fold_graph, sup_rel)
    reverse_rel = _canonical_etype(fold_graph, reverse_rel)
    expected = Counter(expected_train_pairs)
    forward_observed = Counter(_edge_pairs(fold_graph, sup_rel))
    reverse_observed = Counter(
        (dst, src) for src, dst in _edge_pairs(fold_graph, reverse_rel)
    )
    if forward_observed != expected:
        raise AssertionError(
            f"Fold forward edges differ from train pairs: observed={forward_observed}, "
            f"expected={expected}"
        )
    if reverse_observed != expected:
        raise AssertionError(
            f"Fold reverse edges differ from train pairs: observed={reverse_observed}, "
            f"expected={expected}"
        )

    heldout = set(heldout_pairs)
    leaked_forward = heldout.intersection(forward_observed)
    leaked_reverse = heldout.intersection(reverse_observed)
    if leaked_forward or leaked_reverse:
        raise AssertionError(
            "Held-out supervised pairs leaked into the fold graph; "
            f"forward={sorted(leaked_forward)[:10]}, "
            f"reverse={sorted(leaked_reverse)[:10]}"
        )


@torch.no_grad()
def _build_strict_fold_graph(
    hetero_graph: dgl.DGLHeteroGraph,
    sup_rel,
    train_eids_on_full: torch.Tensor,
    val_eids_on_full: torch.Tensor,
) -> Tuple[dgl.DGLHeteroGraph, torch.Tensor]:
    """Build a fold graph and return its newly numbered training EIDs."""
    sup_rel = _canonical_etype(hetero_graph, sup_rel)
    reverse_rel = _find_reverse_etype(hetero_graph, sup_rel)
    train_eids_on_full = torch.as_tensor(
        train_eids_on_full, dtype=torch.long, device=hetero_graph.device
    ).reshape(-1)
    val_eids_on_full = torch.as_tensor(
        val_eids_on_full, dtype=torch.long, device=hetero_graph.device
    ).reshape(-1)
    _validate_fold_partition(
        hetero_graph, sup_rel, train_eids_on_full, val_eids_on_full
    )

    train_pairs = _edge_pairs(hetero_graph, sup_rel, train_eids_on_full)
    heldout_pairs = _edge_pairs(hetero_graph, sup_rel, val_eids_on_full)
    reverse_train_eids = _matching_reverse_eids(
        hetero_graph, sup_rel, reverse_rel, train_eids_on_full
    )

    keep_eids = {}
    for etype in hetero_graph.canonical_etypes:
        if etype == sup_rel:
            keep_eids[etype] = train_eids_on_full
        elif etype == reverse_rel:
            keep_eids[etype] = reverse_train_eids
        else:
            keep_eids[etype] = hetero_graph.edges(etype=etype, form="eid")

    fold_graph = _copy_edge_induced_graph(hetero_graph, keep_eids)
    fold_train_eids = fold_graph.edges(etype=sup_rel, form="eid")
    expected_renumbered = torch.arange(
        len(train_pairs), dtype=torch.long, device=fold_graph.device
    )
    if not torch.equal(fold_train_eids, expected_renumbered):
        raise AssertionError(
            "Unexpected fold EID numbering after graph reconstruction: "
            f"observed={fold_train_eids.detach().cpu().tolist()}"
        )
    _assert_strict_fold_isolation(
        fold_graph, sup_rel, reverse_rel, train_pairs, heldout_pairs
    )
    return fold_graph, fold_train_eids


def _fold_negative_candidate_ids(
    cv_mode: str,
    train_pairs: Sequence[Tuple[int, int]],
    heldout_pairs: Sequence[Tuple[int, int]],
    num_src: int,
    num_dst: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return train-source, train-target, eval-source, eval-target candidates."""
    cv_mode = cv_mode.upper()
    all_src = np.arange(num_src, dtype=np.int64)
    all_dst = np.arange(num_dst, dtype=np.int64)
    train = np.asarray(train_pairs, dtype=np.int64).reshape(-1, 2)
    heldout = np.asarray(heldout_pairs, dtype=np.int64).reshape(-1, 2)

    if cv_mode == "CVS1":
        return all_src, all_dst, all_src, all_dst
    if cv_mode == "CVS2":
        train_src = np.unique(train[:, 0])
        heldout_src = np.unique(heldout[:, 0])
        if np.intersect1d(train_src, heldout_src).size:
            raise AssertionError("CVS2 train and held-out source nodes overlap")
        return train_src, all_dst, heldout_src, all_dst
    if cv_mode == "CVS3":
        train_dst = np.unique(train[:, 1])
        heldout_dst = np.unique(heldout[:, 1])
        if np.intersect1d(train_dst, heldout_dst).size:
            raise AssertionError("CVS3 train and held-out target nodes overlap")
        return all_src, train_dst, all_src, heldout_dst
    if cv_mode == "CVS4":
        train_src = np.unique(train[:, 0])
        train_dst = np.unique(train[:, 1])
        heldout_src = np.unique(heldout[:, 0])
        heldout_dst = np.unique(heldout[:, 1])
        if np.intersect1d(train_src, heldout_src).size:
            raise AssertionError("CVS4 train and held-out source nodes overlap")
        if np.intersect1d(train_dst, heldout_dst).size:
            raise AssertionError("CVS4 train and held-out target nodes overlap")
        return train_src, train_dst, heldout_src, heldout_dst
    raise ValueError(f"Unknown cv_mode: {cv_mode}")


class _PairBatchLoader:
    """Batch pairs against one isolated message graph.

    The full positive-pair set is used only as a rejection oracle so held-out
    positives cannot be sampled as training negatives. It is never converted
    into message-passing edges.
    """

    def __init__(
        self,
        message_graph: dgl.DGLHeteroGraph,
        sup_rel,
        positive_pairs: Sequence[Tuple[int, int]],
        known_positive_pairs: Sequence[Tuple[int, int]],
        batch_size: int,
        neg_k: int,
        shuffle: bool,
        seed: int,
        candidate_src_ids: Sequence[int] | np.ndarray | None = None,
        candidate_dst_ids: Sequence[int] | np.ndarray | None = None,
    ):
        self.message_graph = message_graph
        self.sup_rel = _canonical_etype(message_graph, sup_rel)
        self.positive_pairs = np.asarray(positive_pairs, dtype=np.int64).reshape(-1, 2)
        if len(self.positive_pairs) == 0:
            raise ValueError("Positive-pair split is empty")
        self.batch_size = max(1, int(batch_size))
        self.neg_k = max(1, int(neg_k))
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.rng = np.random.RandomState(self.seed)
        self.num_nodes_dict = {
            ntype: message_graph.num_nodes(ntype) for ntype in message_graph.ntypes
        }
        self.num_src = self.num_nodes_dict[self.sup_rel[0]]
        self.num_dst = self.num_nodes_dict[self.sup_rel[2]]
        self.candidate_src_ids = self._normalize_candidate_ids(
            candidate_src_ids, self.num_src, "source"
        )
        self.candidate_dst_ids = self._normalize_candidate_ids(
            candidate_dst_ids, self.num_dst, "target"
        )
        known = np.asarray(known_positive_pairs, dtype=np.int64).reshape(-1, 2)
        self.known_positive_keys = np.unique(
            known[:, 0] * self.num_dst + known[:, 1]
        )
        known_in_candidates = np.isin(
            known[:, 0], self.candidate_src_ids
        ) & np.isin(known[:, 1], self.candidate_dst_ids)
        num_known_in_candidates = np.unique(
            known[known_in_candidates, 0] * self.num_dst
            + known[known_in_candidates, 1]
        ).size
        self.available_negative_pairs = (
            len(self.candidate_src_ids) * len(self.candidate_dst_ids)
            - num_known_in_candidates
        )
        if self.available_negative_pairs <= 0:
            raise ValueError("No unobserved pairs are available for negative sampling")
        self._fixed_negative_pairs = None
        if not self.shuffle:
            fixed_count = len(self.positive_pairs) * self.neg_k
            self._fixed_negative_pairs = self._sample_negative_pairs(
                fixed_count, np.random.RandomState(self.seed)
            )

    @staticmethod
    def _normalize_candidate_ids(
        ids: Sequence[int] | np.ndarray | None,
        num_nodes: int,
        label: str,
    ) -> np.ndarray:
        if ids is None:
            values = np.arange(num_nodes, dtype=np.int64)
        else:
            values = np.unique(np.asarray(ids, dtype=np.int64).reshape(-1))
        if values.size == 0:
            raise ValueError(f"Negative-sampling {label} candidate set is empty")
        if values.min() < 0 or values.max() >= num_nodes:
            raise ValueError(
                f"Negative-sampling {label} candidate IDs are outside [0, {num_nodes})"
            )
        return values

    def __len__(self) -> int:
        return (len(self.positive_pairs) + self.batch_size - 1) // self.batch_size

    @property
    def fixed_negative_pairs(self) -> Tuple[Tuple[int, int], ...]:
        """Return the frozen evaluation negatives without exposing mutable state."""

        if self.shuffle or self._fixed_negative_pairs is None:
            raise ValueError("Fixed negatives exist only for non-shuffled loaders")
        return tuple(
            (int(source), int(target))
            for source, target in self._fixed_negative_pairs.tolist()
        )

    def _sample_negative_pairs(
        self, count: int, rng: np.random.RandomState
    ) -> np.ndarray:
        if count > self.available_negative_pairs:
            raise ValueError(
                f"Requested {count} unique negatives, but only "
                f"{self.available_negative_pairs} are available in the candidate space"
            )
        parts = []
        selected_keys = set()
        remaining = int(count)
        while remaining > 0:
            draw = max(1024, remaining * 2)
            src = self.candidate_src_ids[
                rng.randint(0, len(self.candidate_src_ids), size=draw)
            ]
            dst = self.candidate_dst_ids[
                rng.randint(0, len(self.candidate_dst_ids), size=draw)
            ]
            keys = src * self.num_dst + dst
            valid = ~np.isin(keys, self.known_positive_keys, assume_unique=False)
            src = src[valid]
            dst = dst[valid]
            keys = keys[valid]
            if selected_keys:
                not_selected = ~np.isin(
                    keys,
                    np.fromiter(selected_keys, dtype=np.int64),
                    assume_unique=False,
                )
                src = src[not_selected]
                dst = dst[not_selected]
                keys = keys[not_selected]
            if keys.size:
                _, first_indices = np.unique(keys, return_index=True)
                first_indices.sort()
                src = src[first_indices]
                dst = dst[first_indices]
                keys = keys[first_indices]
            candidates = np.column_stack((src, dst))
            if candidates.size == 0:
                continue
            take = min(remaining, len(candidates))
            parts.append(candidates[:take])
            selected_keys.update(int(key) for key in keys[:take])
            remaining -= take
        return np.concatenate(parts, axis=0)

    def _pair_graph(self, pairs: np.ndarray) -> dgl.DGLHeteroGraph:
        src = torch.as_tensor(
            pairs[:, 0], dtype=torch.long, device=self.message_graph.device
        )
        dst = torch.as_tensor(
            pairs[:, 1], dtype=torch.long, device=self.message_graph.device
        )
        try:
            return dgl.heterograph(
                {self.sup_rel: (src, dst)},
                num_nodes_dict=self.num_nodes_dict,
                device=self.message_graph.device,
            )
        except TypeError:
            return dgl.heterograph(
                {self.sup_rel: (src, dst)}, num_nodes_dict=self.num_nodes_dict
            ).to(self.message_graph.device)

    def __iter__(self):
        rng = self.rng
        order = np.arange(len(self.positive_pairs))
        if self.shuffle:
            rng.shuffle(order)
        for start in range(0, len(order), self.batch_size):
            indices = order[start : start + self.batch_size]
            positives = self.positive_pairs[indices]
            if self.shuffle:
                negatives = self._sample_negative_pairs(
                    len(positives) * self.neg_k, rng
                )
            else:
                negative_start = start * self.neg_k
                negative_stop = (start + len(positives)) * self.neg_k
                negatives = self._fixed_negative_pairs[
                    negative_start:negative_stop
                ]
            yield None, self._pair_graph(positives), self._pair_graph(negatives), None

# =============== Edge DataLoader ===============
def _build_edge_loader(
    g: dgl.DGLHeteroGraph,
    etype_name: str,
    eids: torch.Tensor,
    fanout_per_layer,
    batch_size: int,
    device: torch.device,
    neg_k: int = 5,
    shuffle: bool = True,
    exclude_edges: bool = True,
):
    if isinstance(fanout_per_layer, int):
        fanout_per_layer = [fanout_per_layer]

    sampler = as_edge_prediction_sampler(
        NeighborSampler(fanout_per_layer),
        negative_sampler=GlobalUniform(neg_k),
        **({
            "exclude": "reverse_types",
            "reverse_etypes": _reverse_map_all(g)
        } if exclude_edges else {})
    )

    loader = DataLoader(
        g,
        {etype_name: eids},
        sampler,
        device=device,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=0,
    )
    return loader

# =============== 评估用图构造（CVS2/3：仅保留训练监督边） ===============
@torch.no_grad()
def _build_eval_graph_keep_train_sup_only(
    hetero_graph: dgl.DGLHeteroGraph,
    sup_rel_name: str,
    train_eids_on_full: torch.Tensor
) -> dgl.DGLHeteroGraph:
    all_sup_eids = hetero_graph.edges(etype=sup_rel_name, form='eid')
    keep_mask = torch.zeros(all_sup_eids.shape[0], dtype=torch.bool, device=all_sup_eids.device)
    keep_mask[train_eids_on_full] = True
    remove_eids = all_sup_eids[~keep_mask]
    if remove_eids.numel() == 0:
        return hetero_graph
    return dgl.remove_edges(hetero_graph, eids=remove_eids, etype=sup_rel_name)

# =============== CVS1 严格：删正向 + 对应反向 ===============
@torch.no_grad()
def _build_eval_graph_cv1_strict(
    hetero_graph: dgl.DGLHeteroGraph,
    sup_rel_name: str,
    val_eids_on_full: torch.Tensor
) -> dgl.DGLHeteroGraph:
    src_v, dst_v = hetero_graph.find_edges(val_eids_on_full, etype=sup_rel_name)
    g_eval = dgl.remove_edges(hetero_graph, eids=val_eids_on_full, etype=sup_rel_name)
    rev_map = _reverse_map_all(hetero_graph)
    rev_etype = rev_map.get(sup_rel_name, None)
    if rev_etype is not None:
        has_rev = hetero_graph.has_edges_between(dst_v, src_v, etype=rev_etype)
        if has_rev.any():
            eids_rev = hetero_graph.edge_ids(dst_v[has_rev], src_v[has_rev], etype=rev_etype)
            g_eval = dgl.remove_edges(g_eval, eids=eids_rev, etype=rev_etype)
    return g_eval

# =============== CVS1 微泄露：只删正向，保留反向（默认） ===============
@torch.no_grad()
def _build_eval_graph_cv1_keep_reverse(
    hetero_graph: dgl.DGLHeteroGraph,
    sup_rel_name: str,
    val_eids_on_full: torch.Tensor
) -> dgl.DGLHeteroGraph:
    return dgl.remove_edges(hetero_graph, eids=val_eids_on_full, etype=sup_rel_name)

# =============== 图增强兜底实现（与 utlis.augment_similarity_graph 等价） ===============
@torch.no_grad()
def _augment_similarity_graph_fallback(g: dgl.DGLHeteroGraph) -> dgl.DGLHeteroGraph:
    def _write_sim_deg_orig(etype_name: str):
        cand = [ce for ce in g.canonical_etypes if ce[1] == etype_name and ce[0] == ce[2]]
        if not cand:
            return
        ntype = cand[0][0]
        _, dst = g.edges(etype=etype_name)
        num = g.num_nodes(ntype)
        deg = torch.bincount(dst.to(torch.long), minlength=num).to(g.device)
        g.nodes[ntype].data['sim_deg_orig'] = deg

    def _add_missing_self_loops(etype_name: str):
        cand = [ce for ce in g.canonical_etypes if ce[1] == etype_name and ce[0] == ce[2]]
        if not cand:
            return g
        ntype = cand[0][0]
        n = g.num_nodes(ntype)
        nodes = torch.arange(n, device=g.device)
        has = g.has_edges_between(nodes, nodes, etype=etype_name)
        add_nodes = nodes[~has]
        if add_nodes.numel() > 0:
            g2 = dgl.add_edges(g, add_nodes, add_nodes, etype=etype_name)
            return g2
        return g

    _write_sim_deg_orig('is'); _write_sim_deg_orig('ts')
    g = _add_missing_self_loops('is')
    g = _add_missing_self_loops('ts')
    return g

# =============== 评估 ===============
@torch.no_grad()
def evaluate(model, loader, sup_rel, args) -> Tuple[float, float]:
    model.eval()
    pos_all, neg_all = [], []
    with torch.no_grad():
        for _, pos_g, neg_g, blocks in loader:
            pos_score, neg_score = model(args, pos_g, neg_g, blocks, None)
            pos_all.append(pos_score[sup_rel].reshape(-1).cpu())
            neg_all.append(neg_score[sup_rel].reshape(-1).cpu())

    if not pos_all or not neg_all:
        return 0.0, 0.0
    pos = torch.cat(pos_all).numpy()
    neg = torch.cat(neg_all).numpy()
    y_true = np.concatenate([np.ones_like(pos), np.zeros_like(neg)])
    y_pred = np.concatenate([pos, neg])
    auroc = roc_auc_score(y_true, y_pred)
    auprc = average_precision_score(y_true, y_pred)
    return auroc, auprc

# =============== 简单 K 折（CVS1=随机边；CVS2/3=按源/目标分组） ===============
def _make_folds(
    g: dgl.DGLHeteroGraph,
    sup_rel: Tuple[str, str, str],
    k: int,
    cv_mode: str,
    seed: int = 411,
):
    sup_rel_name = sup_rel[1]
    src_all, dst_all = g.edges(etype=sup_rel_name)
    eids_all = g.edges(etype=sup_rel_name, form='eid')
    idx = np.arange(eids_all.shape[0], dtype=np.int64)

    cv_mode = cv_mode.upper()
    if cv_mode == "CVS1":
        kf = KFold(n_splits=k, shuffle=True, random_state=seed)
        folds = []
        for tr, va in kf.split(idx):
            tr_idx = torch.as_tensor(tr, dtype=torch.long, device=eids_all.device)
            va_idx = torch.as_tensor(va, dtype=torch.long, device=eids_all.device)
            folds.append((eids_all[tr_idx], eids_all[va_idx]))
        return folds

    if cv_mode == "CVS2":  # 新配体：按源节点分组
        uniq_src = torch.unique(src_all).cpu().numpy()
        rng = np.random.RandomState(seed)
        rng.shuffle(uniq_src)
        buckets = [[] for _ in range(min(k, len(uniq_src)))]
        for i, u in enumerate(uniq_src):
            buckets[i % len(buckets)].append(u)
        folds = []
        src_np, e_np = src_all.cpu().numpy(), eids_all.cpu().numpy()
        for b in buckets:
            va_mask = np.isin(src_np, b)
            tr_mask = ~va_mask
            folds.append((
                torch.as_tensor(e_np[tr_mask], dtype=torch.long, device=eids_all.device),
                torch.as_tensor(e_np[va_mask], dtype=torch.long, device=eids_all.device),
            ))
        return folds

    if cv_mode == "CVS3":  # 新靶点：按目标节点分组
        uniq_dst = torch.unique(dst_all).cpu().numpy()
        rng = np.random.RandomState(seed)
        rng.shuffle(uniq_dst)
        buckets = [[] for _ in range(min(k, len(uniq_dst)))]
        for i, v in enumerate(uniq_dst):
            buckets[i % len(buckets)].append(v)
        folds = []
        dst_np, e_np = dst_all.cpu().numpy(), eids_all.cpu().numpy()
        for b in buckets:
            va_mask = np.isin(dst_np, b)
            tr_mask = ~va_mask
            folds.append((
                torch.as_tensor(e_np[tr_mask], dtype=torch.long, device=eids_all.device),
                torch.as_tensor(e_np[va_mask], dtype=torch.long, device=eids_all.device),
            ))
        return folds

    raise ValueError(f"Unknown cv_mode: {cv_mode}")

# =============== 训练主流程 ===============
def _build_nested_stage_graph(
    hetero_graph: dgl.DGLHeteroGraph,
    sup_rel,
    stage_train_eids: torch.Tensor,
) -> Tuple[dgl.DGLHeteroGraph, torch.Tensor]:
    """Build an isolated message graph from arbitrary stage-training EIDs."""
    sup_rel = _canonical_etype(hetero_graph, sup_rel)
    all_eids = hetero_graph.edges(etype=sup_rel, form="eid")
    train_np = as_numpy_eids(stage_train_eids)
    all_np = as_numpy_eids(all_eids)
    heldout_np = all_np[~np.isin(all_np, train_np)]
    if train_np.size == 0 or heldout_np.size == 0:
        raise AssertionError(
            "Each nested stage must have non-empty training and non-training EIDs"
        )
    return _build_strict_fold_graph(
        hetero_graph,
        sup_rel,
        torch.as_tensor(train_np, dtype=torch.long, device=hetero_graph.device),
        torch.as_tensor(heldout_np, dtype=torch.long, device=hetero_graph.device),
    )


def _training_negative_candidate_ids(
    cv_mode: str,
    train_pairs: Sequence[Tuple[int, int]],
    num_src: int,
    num_dst: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Training candidates derived without inspecting outer-test endpoints."""
    pairs = np.asarray(train_pairs, dtype=np.int64).reshape(-1, 2)
    if pairs.size == 0:
        raise ValueError("Training positive pairs are empty")
    cv_mode = cv_mode.upper()
    all_src = np.arange(num_src, dtype=np.int64)
    all_dst = np.arange(num_dst, dtype=np.int64)
    if cv_mode == "CVS1":
        return all_src, all_dst
    if cv_mode == "CVS2":
        return np.unique(pairs[:, 0]), all_dst
    if cv_mode == "CVS3":
        return all_src, np.unique(pairs[:, 1])
    if cv_mode == "CVS4":
        return np.unique(pairs[:, 0]), np.unique(pairs[:, 1])
    raise ValueError(f"Unknown cv_mode: {cv_mode}")


@torch.no_grad()
def _score_pair_loader(model, loader, sup_rel, args) -> Dict[str, np.ndarray]:
    model.eval()
    scores = []
    labels = []
    pairs = []
    pair_kinds = []
    for _, pos_graph, neg_graph, blocks in loader:
        pos_score, neg_score = model(args, pos_graph, neg_graph, blocks, None)
        pos_values = pos_score[sup_rel].reshape(-1).detach().cpu().numpy()
        neg_values = neg_score[sup_rel].reshape(-1).detach().cpu().numpy()
        pos_pairs = np.asarray(
            _edge_pairs(pos_graph, sup_rel), dtype=np.int64
        ).reshape(-1, 2)
        neg_pairs = np.asarray(
            _edge_pairs(neg_graph, sup_rel), dtype=np.int64
        ).reshape(-1, 2)
        if len(pos_values) != len(pos_pairs) or len(neg_values) != len(neg_pairs):
            raise AssertionError("Score and pair counts differ during evaluation")
        scores.extend((pos_values, neg_values))
        labels.extend(
            (
                np.ones(len(pos_values), dtype=np.int64),
                np.zeros(len(neg_values), dtype=np.int64),
            )
        )
        pairs.extend((pos_pairs, neg_pairs))
        pair_kinds.extend(
            (
                np.repeat("positive", len(pos_values)),
                np.repeat("negative", len(neg_values)),
            )
        )
    if not scores:
        raise ValueError("Evaluation loader produced no batches")
    result = {
        "score": np.concatenate(scores).astype(np.float64, copy=False),
        "label": np.concatenate(labels).astype(np.int64, copy=False),
        "pairs": np.concatenate(pairs, axis=0).astype(np.int64, copy=False),
        "pair_kind": np.concatenate(pair_kinds),
    }
    if not np.isfinite(result["score"]).all():
        raise FloatingPointError("Non-finite evaluation scores detected")
    return result


def _new_training_state(args, graph, rel_list, device):
    model = Model(args, graph, rel_list).to(device)
    optimizer = _build_adamw(model, args, device)
    ema = AveragedModel(model) if getattr(args, "use_ema", False) else None
    if getattr(args, "use_cosine", False):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=max(1, args.cosine_T0),
            T_mult=max(1, args.cosine_Tmult),
        )
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.lr_period, gamma=args.lr_decay
        )
    return model, optimizer, ema, scheduler


def _build_adamw(model, args, device):
    kwargs = {"lr": args.lr, "weight_decay": args.wd}
    requested = bool(
        getattr(args, "fused_adamw", False)
        and torch.device(device).type == "cuda"
    )
    if requested:
        try:
            optimizer = torch.optim.AdamW(model.parameters(), fused=True, **kwargs)
            args.fused_adamw_active = True
            return optimizer
        except (TypeError, ValueError, RuntimeError):
            pass
    args.fused_adamw_active = False
    return torch.optim.AdamW(model.parameters(), **kwargs)


def _run_training_epoch(
    model,
    optimizer,
    ema,
    scheduler,
    loader,
    args,
    sup_rel,
) -> float:
    model.train()
    losses = []
    ema_decay = float(getattr(args, "ema_decay", 0.999))
    for _, pos_graph, neg_graph, blocks in loader:
        pos_score, neg_score = model(args, pos_graph, neg_graph, blocks, None)
        loss = compute_loss(
            pos_score,
            neg_score,
            sup_rel,
            tau=getattr(args, "tau", 0.07),
            top_m=getattr(args, "top_m", 0),
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        if ema is not None:
            for p_ema, parameter in zip(ema.parameters(), model.parameters()):
                p_ema.data.mul_(ema_decay).add_(
                    parameter.data, alpha=(1.0 - ema_decay)
                )
        losses.append(float(loss.item()))
    scheduler.step()
    if not losses:
        raise ValueError("Training loader produced no batches")
    return float(np.mean(losses))


def _cpu_state_dict(model) -> Dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def _save_torch_checkpoint(payload: Dict[str, Any], path: str | Path) -> None:
    """Save through a Python file handle so PyTorch 1.13 supports Unicode paths."""
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint_path.open("wb") as handle:
        torch.save(payload, handle)


def _fit_with_inner_validation(
    args,
    graph,
    rel_list,
    device,
    train_loader,
    validation_loader,
    sup_rel,
    outer_fold: int,
) -> Dict[str, Any]:
    """Select epoch and raw-score threshold using inner validation only."""
    model, optimizer, ema, scheduler = _new_training_state(
        args, graph, rel_list, device
    )
    best: Dict[str, Any] | None = None
    for epoch in range(1, int(args.num_epochs) + 1):
        loss = _run_training_epoch(
            model, optimizer, ema, scheduler, train_loader, args, sup_rel
        )
        if epoch % max(1, int(args.log_every)) == 0:
            print(f"[Outer {outer_fold}/inner] epoch={epoch} loss={loss:.4f}")
        if epoch % max(1, int(args.val_every)) != 0 and epoch != int(args.num_epochs):
            continue
        evaluation_model = ema.module if ema is not None else model
        validation_scores = _score_pair_loader(
            evaluation_model, validation_loader, sup_rel, args
        )
        threshold, threshold_metrics = select_threshold(
            validation_scores["label"],
            validation_scores["score"],
            getattr(args, "threshold_metric", "youden"),
        )
        monitor_name = getattr(args, "monitor_metric", "auprc")
        monitor_value = float(threshold_metrics[monitor_name])
        if best is None or monitor_value > best["monitor_value"]:
            best = {
                "selected_epoch": int(epoch),
                "threshold_raw": float(threshold),
                "monitor_metric": monitor_name,
                "monitor_value": monitor_value,
                "inner_metrics": threshold_metrics,
                "inner_validation_scores": {
                    key: np.array(value, copy=True)
                    for key, value in validation_scores.items()
                },
                "inner_validation_negative_seed": int(validation_loader.seed),
            }
            print(
                f"[Outer {outer_fold}/inner] selected candidate epoch={epoch} "
                f"{monitor_name}={monitor_value:.4f} threshold={threshold:.6g}"
            )
    if best is None:
        raise AssertionError("No inner-validation checkpoint was selected")
    fixed_epoch = getattr(args, "fixed_selected_epoch", None)
    if fixed_epoch is not None:
        fixed_epoch = int(fixed_epoch)
        if fixed_epoch < 1 or fixed_epoch > int(args.num_epochs):
            raise ValueError(
                "fixed_selected_epoch must be within 1..num_epochs, "
                f"observed {fixed_epoch} and num_epochs={args.num_epochs}"
            )
        # Preserve the inner-validation threshold and metrics, while binding the
        # refit budget to a prespecified value that never consults outer test.
        best["inner_best_epoch"] = int(best["selected_epoch"])
        best["selected_epoch"] = fixed_epoch
        best["epoch_selection_policy"] = "prespecified_fixed"
    else:
        best["inner_best_epoch"] = int(best["selected_epoch"])
        best["epoch_selection_policy"] = "inner_validation"
    return best


def _fit_fixed_epochs(
    args,
    graph,
    rel_list,
    device,
    train_loader,
    sup_rel,
    epochs: int,
    outer_fold: int,
):
    """Fresh fit on all outer-training edges without validation/test access."""
    if int(epochs) < 1:
        raise ValueError("Selected epoch count must be positive")
    model, optimizer, ema, scheduler = _new_training_state(
        args, graph, rel_list, device
    )
    for epoch in range(1, int(epochs) + 1):
        loss = _run_training_epoch(
            model, optimizer, ema, scheduler, train_loader, args, sup_rel
        )
        if epoch % max(1, int(args.log_every)) == 0 or epoch == int(epochs):
            print(f"[Outer {outer_fold}/refit] epoch={epoch} loss={loss:.4f}")
    final_model = ema.module if ema is not None else model
    final_model.eval()
    return final_model


def _prediction_export_rows(
    scores: Dict[str, np.ndarray],
    positive_eids: torch.Tensor,
    args,
    cv_mode: str,
    outer_fold: int,
    split_name: str,
    threshold: float,
    selected_epoch: int,
    monitor_metric: str,
    monitor_value: float,
    split_seed: int,
    model_seed: int,
    evaluation_negative_seed: int,
    compound_keys: Sequence[str] | None = None,
    target_keys: Sequence[str] | None = None,
) -> List[Dict[str, Any]]:
    dataset_name = str(getattr(args, "dataset_name", "HIT")).strip()
    model_name = str(getattr(args, "model_name", "MCG-HGT")).strip()
    if not dataset_name or not model_name:
        raise ValueError("dataset_name and model_name must be non-empty")
    positive_ids = [int(value) for value in as_numpy_eids(positive_eids)]
    positive_index = 0
    rows = []
    for index in range(len(scores["label"])):
        pair_kind = str(scores["pair_kind"][index])
        source_id, target_id = scores["pairs"][index]
        compound_key, target_key, pair_key = _canonical_pair_fields(
            int(source_id), int(target_id), compound_keys, target_keys
        )
        if pair_kind == "positive":
            if positive_index >= len(positive_ids):
                raise AssertionError("More positive scores than positive EIDs")
            edge_id: int | str = positive_ids[positive_index]
            positive_index += 1
            sample_id = f"pos:{pair_key}" if pair_key else f"pos:eid:{edge_id}"
        elif pair_kind == "negative":
            edge_id = ""
            sample_id = (
                f"neg:{pair_key}"
                if pair_key
                else f"neg:{int(source_id)}:{int(target_id)}"
            )
        else:
            raise AssertionError(f"Unknown pair kind: {pair_kind}")
        score = float(scores["score"][index])
        rows.append(
            {
                "dataset": dataset_name,
                "protocol": cv_mode.upper(),
                "mode": "transductive",
                "model": model_name,
                "fold": int(outer_fold),
                "split": split_name,
                "sample_id": sample_id,
                "label": int(scores["label"][index]),
                "score": score,
                "evaluation_protocol": "nested",
                "cv_mode": cv_mode.upper(),
                "outer_fold": int(outer_fold),
                "pair_kind": pair_kind,
                "edge_id": edge_id,
                "source_id": int(source_id),
                "target_id": int(target_id),
                "compound_key": compound_key,
                "target_key": target_key,
                "pair_key": pair_key,
                "score_raw": score,
                "threshold_raw": float(threshold),
                "predicted_label": int(score >= float(threshold)),
                "selected_epoch": int(selected_epoch),
                "inner_monitor_metric": monitor_metric,
                "inner_monitor_value": float(monitor_value),
                "split_seed": int(split_seed),
                "model_seed": int(model_seed),
                "evaluation_negative_seed": int(evaluation_negative_seed),
            }
        )
    if positive_index != len(positive_ids):
        raise AssertionError(
            "Positive EID count differs from exported positive score count"
        )
    return rows


def _train_nested(
    args,
    hetero_graph,
    rel_list,
    device,
    sup_rel,
    known_positive_pairs,
    compound_keys,
    target_keys,
    evaluation_inputs,
):
    cv_mode = args.cv_mode.upper()
    split_seed = int(getattr(args, "split_seed", 411))
    inner_val_fraction = float(getattr(args, "inner_val_fraction", 0.2))
    output_dir = Path(getattr(args, "output_dir", "outputs/nested_eval"))
    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    splits = make_nested_splits(
        hetero_graph,
        sup_rel,
        int(args.k_fold),
        cv_mode,
        inner_val_fraction=inner_val_fraction,
        seed=split_seed,
    )
    fold_start = int(getattr(args, "fold_start", 1))
    fold_end_value = getattr(args, "fold_end", None)
    fold_end = int(args.k_fold if fold_end_value is None else fold_end_value)
    if fold_start < 1 or fold_end < fold_start or fold_end > int(args.k_fold):
        raise ValueError(
            f"Invalid outer-fold range {fold_start}..{fold_end} for k_fold={args.k_fold}"
        )
    manifest_json, manifest_csv = write_split_manifest(
        splits,
        hetero_graph,
        sup_rel,
        output_dir,
        cv_mode,
        split_seed,
        inner_val_fraction,
        compound_keys=compound_keys,
        target_keys=target_keys,
        evaluation_inputs=evaluation_inputs,
    )
    print(
        f"[Protocol] nested outer-test/inner-validation | cv_mode={cv_mode} | "
        f"folds={len(splits)} | manifest={manifest_json}"
    )
    oof_rows: List[Dict[str, Any]] = []
    fold_summaries = []

    selected_splits = [
        split for split in splits if fold_start <= int(split.outer_fold) <= fold_end
    ]
    if len(selected_splits) != fold_end - fold_start + 1:
        raise AssertionError("Requested outer-fold range was not materialized exactly once")

    for split in selected_splits:
        outer_fold = split.outer_fold
        fold_start = time.perf_counter()
        _reset_peak_gpu_memory(device)
        selection_seed = int(getattr(args, "seed", 410)) + outer_fold * 1009
        inner_evaluation_seed = split_seed + outer_fold * 1009 + 100003
        outer_evaluation_seed = split_seed + outer_fold * 1009 + 200003
        set_seed(selection_seed)

        inner_graph, inner_train_sub_eids = _build_nested_stage_graph(
            hetero_graph, sup_rel, split.inner_train_eids
        )
        inner_train_pairs = _edge_pairs(
            inner_graph, sup_rel, inner_train_sub_eids
        )
        inner_val_pairs = _edge_pairs(
            hetero_graph, sup_rel, split.inner_val_eids
        )
        (
            inner_train_src,
            inner_train_dst,
            inner_val_src,
            inner_val_dst,
        ) = _fold_negative_candidate_ids(
            cv_mode,
            inner_train_pairs,
            inner_val_pairs,
            inner_graph.num_nodes(sup_rel[0]),
            inner_graph.num_nodes(sup_rel[2]),
        )
        inner_train_loader = _PairBatchLoader(
            inner_graph,
            sup_rel,
            inner_train_pairs,
            known_positive_pairs,
            args.batch_size,
            args.neg_k,
            shuffle=True,
            seed=selection_seed,
            candidate_src_ids=inner_train_src,
            candidate_dst_ids=inner_train_dst,
        )
        inner_validation_loader = _PairBatchLoader(
            inner_graph,
            sup_rel,
            inner_val_pairs,
            known_positive_pairs,
            args.batch_size,
            1,
            shuffle=False,
            seed=inner_evaluation_seed,
            candidate_src_ids=inner_val_src,
            candidate_dst_ids=inner_val_dst,
        )
        selection, selection_training_seconds = _time_operation(
            device,
            _fit_with_inner_validation,
            args,
            inner_graph,
            rel_list,
            device,
            inner_train_loader,
            inner_validation_loader,
            sup_rel,
            outer_fold,
        )
        threshold = float(selection["threshold_raw"])
        oof_rows.extend(
            _prediction_export_rows(
                selection["inner_validation_scores"],
                split.inner_val_eids,
                args,
                cv_mode,
                outer_fold,
                "inner_val",
                threshold,
                int(selection["selected_epoch"]),
                selection["monitor_metric"],
                float(selection["monitor_value"]),
                split_seed,
                selection_seed,
                int(selection["inner_validation_negative_seed"]),
                compound_keys,
                target_keys,
            )
        )

        # Reinitialize and fit all outer-training positives for the selected
        # number of epochs.  This function receives no validation/test loader.
        refit_seed = int(getattr(args, "seed", 410)) + outer_fold * 1009 + 500003
        set_seed(refit_seed)
        outer_graph, outer_train_sub_eids = _build_nested_stage_graph(
            hetero_graph, sup_rel, split.outer_train_eids
        )
        outer_train_pairs = _edge_pairs(
            outer_graph, sup_rel, outer_train_sub_eids
        )
        outer_train_src, outer_train_dst = _training_negative_candidate_ids(
            cv_mode,
            outer_train_pairs,
            outer_graph.num_nodes(sup_rel[0]),
            outer_graph.num_nodes(sup_rel[2]),
        )
        outer_train_loader = _PairBatchLoader(
            outer_graph,
            sup_rel,
            outer_train_pairs,
            known_positive_pairs,
            args.batch_size,
            args.neg_k,
            shuffle=True,
            seed=refit_seed,
            candidate_src_ids=outer_train_src,
            candidate_dst_ids=outer_train_dst,
        )
        final_model, refit_training_seconds = _time_operation(
            device,
            _fit_fixed_epochs,
            args,
            outer_graph,
            rel_list,
            device,
            outer_train_loader,
            sup_rel,
            int(selection["selected_epoch"]),
            outer_fold,
        )

        # First scoring access to outer test occurs only after selection/refit.
        outer_test_pairs = _edge_pairs(
            hetero_graph, sup_rel, split.outer_test_eids
        )
        _, _, outer_test_src, outer_test_dst = _fold_negative_candidate_ids(
            cv_mode,
            outer_train_pairs,
            outer_test_pairs,
            outer_graph.num_nodes(sup_rel[0]),
            outer_graph.num_nodes(sup_rel[2]),
        )
        outer_test_loader = _PairBatchLoader(
            outer_graph,
            sup_rel,
            outer_test_pairs,
            known_positive_pairs,
            args.batch_size,
            1,
            shuffle=False,
            seed=outer_evaluation_seed,
            candidate_src_ids=outer_test_src,
            candidate_dst_ids=outer_test_dst,
        )
        outer_scores, inference_seconds = _time_operation(
            device,
            _score_pair_loader,
            final_model,
            outer_test_loader,
            sup_rel,
            args,
        )
        outer_metrics = classification_metrics(
            outer_scores["label"], outer_scores["score"], threshold
        )
        fold_cost = _computational_cost_record(
            final_model,
            device,
            selection_training_seconds + refit_training_seconds,
            inference_seconds,
            len(outer_scores["label"]),
            args,
        )
        oof_rows.extend(
            _prediction_export_rows(
                outer_scores,
                split.outer_test_eids,
                args,
                cv_mode,
                outer_fold,
                "outer_test",
                threshold,
                int(selection["selected_epoch"]),
                selection["monitor_metric"],
                float(selection["monitor_value"]),
                split_seed,
                refit_seed,
                outer_evaluation_seed,
                compound_keys,
                target_keys,
            )
        )

        checkpoint_path = checkpoint_dir / (
            f"nested_{cv_mode.lower()}_outer_fold{outer_fold}_"
            f"{args.monitor_metric}.pt"
        )
        _save_torch_checkpoint(
            {
                "state_dict": _cpu_state_dict(final_model),
                **_checkpoint_model_metadata(final_model, args),
                "args": vars(args),
                "protocol": "nested",
                "graph_contract": {"scope": "fold_specific_message_graph"},
                "cv_mode": cv_mode,
                "outer_fold": outer_fold,
                "selected_epoch": int(selection["selected_epoch"]),
                "inner_best_epoch": int(selection["inner_best_epoch"]),
                "epoch_selection_policy": selection["epoch_selection_policy"],
                "threshold_raw": threshold,
                "threshold_metric": getattr(args, "threshold_metric", "youden"),
                "inner_monitor_metric": selection["monitor_metric"],
                "inner_monitor_value": float(selection["monitor_value"]),
                "inner_metrics": selection["inner_metrics"],
                "inner_validation_scores": selection["inner_validation_scores"],
                "inner_validation_negative_seed": int(
                    selection["inner_validation_negative_seed"]
                ),
                "outer_test_negative_seed": int(outer_evaluation_seed),
                "outer_metrics": outer_metrics,
                "computational_cost": fold_cost,
                "evaluation_inputs": evaluation_inputs,
                "split_manifest": str(manifest_json),
                "outer_train_eid_sha256": hash_eids(split.outer_train_eids),
                "outer_test_eid_sha256": hash_eids(split.outer_test_eids),
                "inner_train_eid_sha256": hash_eids(split.inner_train_eids),
                "inner_validation_eid_sha256": hash_eids(split.inner_val_eids),
                "sup_rel": sup_rel,
            },
            checkpoint_path,
        )
        fold_summary = {
            "outer_fold": outer_fold,
            "selected_epoch": int(selection["selected_epoch"]),
            "inner_best_epoch": int(selection["inner_best_epoch"]),
            "epoch_selection_policy": selection["epoch_selection_policy"],
            "threshold_raw": threshold,
            "inner_monitor_metric": selection["monitor_metric"],
            "inner_monitor_value": float(selection["monitor_value"]),
            "inner_metrics": selection["inner_metrics"],
            "outer_metrics": outer_metrics,
            "checkpoint": str(checkpoint_path),
            "elapsed_seconds": float(time.perf_counter() - fold_start),
            "computational_cost": fold_cost,
        }
        fold_summaries.append(fold_summary)
        print(
            f"[Outer {outer_fold}] selected_epoch={selection['selected_epoch']} | "
            f"AUROC={outer_metrics['auroc']:.4f} | "
            f"AUPRC={outer_metrics['auprc']:.4f} | "
            f"ACC={outer_metrics['accuracy']:.4f} | "
            f"SEN={outer_metrics['sensitivity']:.4f} | "
            f"SPE={outer_metrics['specificity']:.4f} | "
            f"MCC={outer_metrics['mcc']:.4f}"
        )
        peak_memory = fold_cost["gpu_memory"]
        print(
            f"[Compute][Outer {outer_fold}] "
            f"parameters={fold_cost['parameters']['total']} | "
            f"training={fold_cost['training']['wall_time_seconds']:.3f}s | "
            f"inference={fold_cost['inference']['wall_time_seconds']:.3f}s | "
            f"throughput="
            f"{fold_cost['inference']['throughput_pairs_per_second']:.3f} pairs/s | "
            f"peak_gpu={peak_memory['status']}/"
            f"{peak_memory['peak_allocated_mib']:.3f} MiB"
        )

    oof_path = write_oof_predictions(oof_rows, output_dir)
    aggregate = {}
    for name in (
        "auroc",
        "auprc",
        "accuracy",
        "sensitivity",
        "specificity",
        "precision",
        "f1",
        "mcc",
    ):
        values = np.asarray(
            [fold["outer_metrics"][name] for fold in fold_summaries],
            dtype=np.float64,
        )
        aggregate[name] = {
            "mean": float(values.mean()),
            "std_population": float(values.std(ddof=0)),
            "std_sample": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "fold_values": values.tolist(),
        }
    run_cost = _aggregate_computational_cost(
        [fold["computational_cost"] for fold in fold_summaries]
    )
    summary = {
        "schema_version": 2,
        "protocol": "nested",
        "cv_mode": cv_mode,
        "mode": "transductive",
        "dataset": str(getattr(args, "dataset_name", "HIT")),
        "model": str(getattr(args, "model_name", "MCG-HGT")),
        "split_seed": split_seed,
        "model_seed": int(getattr(args, "seed", 410)),
        "executed_outer_folds": [int(split.outer_fold) for split in selected_splits],
        "manifest_json": str(manifest_json),
        "manifest_csv": str(manifest_csv),
        "oof_predictions": str(oof_path),
        "evaluation_inputs": evaluation_inputs,
        "folds": fold_summaries,
        "aggregate": aggregate,
        "computational_cost": run_cost,
    }
    summary_path = write_summary(summary, output_dir)
    print(
        f"[Nested CV] AUROC={aggregate['auroc']['mean']:.4f}+/-"
        f"{aggregate['auroc']['std_sample']:.4f} | "
        f"AUPRC={aggregate['auprc']['mean']:.4f}+/-"
        f"{aggregate['auprc']['std_sample']:.4f} | summary={summary_path}"
    )
    peak_memory = run_cost["gpu_memory"]
    print(
        f"[Compute][Nested CV] parameters={run_cost['parameters']['total']} | "
        f"training={run_cost['training']['wall_time_seconds']:.3f}s | "
        f"inference={run_cost['inference']['wall_time_seconds']:.3f}s | "
        f"throughput="
        f"{run_cost['inference']['throughput_pairs_per_second']:.3f} pairs/s | "
        f"peak_gpu={peak_memory['status']}/"
        f"{peak_memory['peak_allocated_mib']:.3f} MiB"
    )
    return summary


def train(args, hetero_graph: dgl.DGLHeteroGraph, rel_list, device):
    set_seed(getattr(args, "seed", 410))

    try:
        sup_rel = _canonical_etype(hetero_graph, "it")
    except Exception:
        sup_rel = _canonical_etype(hetero_graph, rel_list[0])
    sup_rel_name = sup_rel[1]
    cv_protocol = getattr(args, "cv_protocol", "nested").lower()
    if cv_protocol not in {"nested", "fold_isolated", "legacy"}:
        raise ValueError(f"Unknown cv_protocol: {cv_protocol}")

    (
        known_positive_pairs,
        compound_keys,
        target_keys,
        evaluation_inputs,
    ) = _prepare_evaluation_inputs(args, hetero_graph, sup_rel)

    # 兼容旧脚本：把 input_gate_type=etype 映射到 'se'
    if getattr(args, "input_gate_type", "none") == "etype":
        args.input_gate_type = "se"

    # 兼容两个严格冷启动开关
    if getattr(args, "strict_unseen", False):
        args.strict_cold_start = True
    if cv_protocol in {"nested", "fold_isolated"} and getattr(args, "strict_cold_start", False):
        raise ValueError(
            "--strict_cold_start/--strict_unseen removes held-out nodes and is only "
            "available with --cv_protocol legacy. Leakage-controlled protocols keep "
            "all feature/similarity nodes while isolating supervised it/ti edges."
        )

    if cv_protocol == "nested":
        return _train_nested(
            args,
            hetero_graph,
            rel_list,
            device,
            sup_rel,
            known_positive_pairs,
            compound_keys,
            target_keys,
            evaluation_inputs,
        )

    if args.cv_mode.upper() == "CVS4":
        raise ValueError("CVS4 is implemented only by --cv_protocol nested")

    folds = _make_folds(hetero_graph, sup_rel, args.k_fold, args.cv_mode)
    if cv_protocol == "fold_isolated":
        print(
            "[Protocol] fold_isolated is W0 graph isolation only; the held-out "
            "fold is still used for epoch monitoring and fold metrics. This is "
            "not nested outer-test/inner-validation."
        )
    else:
        print(
            "[Protocol] WARNING: legacy reproduces the historical evaluation path "
            "for provenance only. Do not report these values as revised results."
        )

    results = []
    for fold, (train_eids, val_eids) in enumerate(folds, start=1):
        t0 = time.time()
        fanout = [args.fanout] * args.num_layers

        if cv_protocol == "fold_isolated":
            g_train, train_eids_sub = _build_strict_fold_graph(
                hetero_graph, sup_rel, train_eids, val_eids
            )
            train_pairs = _edge_pairs(g_train, sup_rel, train_eids_sub)
            heldout_pairs = _edge_pairs(hetero_graph, sup_rel, val_eids)
            (
                train_negative_src_ids,
                train_negative_dst_ids,
                eval_negative_src_ids,
                eval_negative_dst_ids,
            ) = _fold_negative_candidate_ids(
                args.cv_mode,
                train_pairs,
                heldout_pairs,
                g_train.num_nodes(sup_rel[0]),
                g_train.num_nodes(sup_rel[2]),
            )
            fold_seed = int(getattr(args, "seed", 410)) + fold * 1009
            train_loader = _PairBatchLoader(
                g_train,
                sup_rel,
                train_pairs,
                known_positive_pairs,
                args.batch_size,
                args.neg_k,
                shuffle=True,
                seed=fold_seed,
                candidate_src_ids=train_negative_src_ids,
                candidate_dst_ids=train_negative_dst_ids,
            )
            val_loader = _PairBatchLoader(
                g_train,
                sup_rel,
                heldout_pairs,
                known_positive_pairs,
                args.batch_size,
                1,
                shuffle=False,
                seed=fold_seed + 1,
                candidate_src_ids=eval_negative_src_ids,
                candidate_dst_ids=eval_negative_dst_ids,
            )
            reverse_rel = _find_reverse_etype(g_train, sup_rel)
            print(
                f"[Fold {fold}] W0 fold isolation | "
                f"full_it={hetero_graph.num_edges(sup_rel)} | "
                f"train_it={g_train.num_edges(sup_rel)} | "
                f"train_ti={g_train.num_edges(reverse_rel)} | "
                f"heldout={len(heldout_pairs)}"
            )
            exclude_edges = True
            loader_mode = "isolated_pair_loader"
        else:
            # Historical behavior retained only for provenance comparison.
            g_train = hetero_graph
            if getattr(args, "strict_cold_start", False):
                src_all, dst_all = hetero_graph.edges(etype=sup_rel_name)
                if args.cv_mode.upper() == "CVS2":
                    val_src_nodes = torch.unique(src_all[val_eids]).cpu().numpy()
                    g_train, _, _ = remove_unseen_nodes(
                        "ingredient", hetero_graph, val_src_nodes
                    )
                elif args.cv_mode.upper() == "CVS3":
                    val_dst_nodes = torch.unique(dst_all[val_eids]).cpu().numpy()
                    g_train, _, _ = remove_unseen_nodes(
                        "target", hetero_graph, val_dst_nodes
                    )

            exclude_edges = not (
                args.cv_mode.upper() == "CVS1"
                and getattr(args, "no_exclude_cv1", False)
            )
            train_eids_sub = g_train.edges(etype=sup_rel_name, form="eid")
            train_loader = _build_edge_loader(
                g_train,
                sup_rel_name,
                train_eids_sub,
                fanout,
                args.batch_size,
                device,
                neg_k=args.neg_k,
                shuffle=True,
                exclude_edges=exclude_edges,
            )
            val_loader = _build_edge_loader(
                hetero_graph,
                sup_rel_name,
                val_eids,
                fanout,
                args.batch_size,
                device,
                neg_k=1,
                shuffle=False,
                exclude_edges=exclude_edges,
            )
            loader_mode = "legacy_dgl_edge_loader"

        # ==== 模型（训练在 g_train） ====
        model = Model(args, g_train, rel_list).to(device)
        opt = _build_adamw(model, args, device)

        # ==== EMA ====
        ema = AveragedModel(model) if getattr(args, "use_ema", False) else None
        ema_decay = float(getattr(args, "ema_decay", 0.999))

        # ==== 学习率调度 ====
        if getattr(args, "use_cosine", False):
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                opt, T_0=max(1, args.cosine_T0), T_mult=max(1, args.cosine_Tmult))
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=args.lr_period, gamma=args.lr_decay)

        if getattr(args, "print_model", False):
            print(
                f"[Architecture] hidden={args.h_dim} | "
                f"propagation_steps={args.num_layers} | heads={args.hgt_heads} | "
                f"weight_shared={getattr(args,'share_hgt_layers',False)} | "
                f"hgt_parameter_sets="
                f"{1 if getattr(args,'share_hgt_layers',False) else args.num_layers} | "
                f"proj_hidden_mult={getattr(args,'proj_hidden_mult',2)}\n"
                f"[Model] input_gate={getattr(args,'input_gate_type','none')} "
                f"(reduce={getattr(args,'input_gate_reduce',4)}) | "
                f"residual_gate={getattr(args,'residual_gate',False)} | "
                f"score_gate={getattr(args,'score_gate','none')} | film_cond={getattr(args,'film_condition','dst')} | "
                f"semantic_gate={getattr(args,'semantic_gate','none')} | head_gate={getattr(args,'head_gate',False)} | "
                f"protocol={cv_protocol} | loader={loader_mode} | exclude_edges={exclude_edges} "
                f"(cv_mode={args.cv_mode})"
            )

        best_metric = -float('inf')
        best_ckpt_path = None

        for epoch in tqdm(range(args.num_epochs), desc=f"HGT-Fold{fold}"):
            model.train()
            losses = []
            for _, pos_g, neg_g, blocks in train_loader:
                pos_score, neg_score = model(args, pos_g, neg_g, blocks, None)
                loss = compute_loss(
                    pos_score, neg_score, sup_rel,
                    tau=getattr(args, "tau", 0.07),
                    top_m=getattr(args, "top_m", 0),
                )
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                opt.step()

                # EMA 更新
                if ema is not None:
                    for p_ema, p in zip(ema.parameters(), model.parameters()):
                        p_ema.data.mul_(ema_decay).add_(p.data, alpha=(1.0 - ema_decay))

                losses.append(loss.item())

            scheduler.step()
            if (epoch + 1) % max(1, args.log_every) == 0:
                print(f"[Fold {fold}] epoch={epoch+1} loss={np.mean(losses):.4f}")

            # ==== 周期性验证（用 EMA 或当前参数） ====
            if ((epoch + 1) % max(1, args.val_every) == 0) or ((epoch + 1) == args.num_epochs):
                if cv_protocol == "fold_isolated":
                    g_eval = g_train
                elif args.cv_mode.upper() == "CVS1":
                    mode = getattr(args, "legacy_cv1_eval", "strict")
                    if mode == "strict":
                        g_eval = _build_eval_graph_cv1_strict(
                            hetero_graph, sup_rel_name, val_eids
                        )
                    elif mode == "keep_reverse":
                        g_eval = _build_eval_graph_cv1_keep_reverse(
                            hetero_graph, sup_rel_name, val_eids
                        )
                    elif mode == "full_graph":
                        g_eval = hetero_graph
                    else:
                        raise ValueError(f"Unknown legacy_cv1_eval: {mode}")
                else:
                    g_eval = _build_eval_graph_keep_train_sup_only(
                        hetero_graph, sup_rel_name, train_eids
                    )

                model_eval = Model(args, g_eval, rel_list).to(device)
                state = (ema.module.state_dict() if ema is not None else model.state_dict())
                model_eval.load_state_dict(state, strict=True)

                auroc_val, auprc_val = evaluate(model_eval, val_loader, sup_rel, args)
                monitor = auprc_val if args.monitor_metric == "auprc" else auroc_val

                if best_ckpt_path is None:
                    os.makedirs(args.checkpoint_dir, exist_ok=True)
                    best_ckpt_path = os.path.join(args.checkpoint_dir, f"best_fold{fold}_{args.monitor_metric}.pt")

                if monitor > best_metric:
                    best_metric = float(monitor)
                    _save_torch_checkpoint(
                        {
                            "state_dict": state,  # 保存 EMA 或当前参数
                            **_checkpoint_model_metadata(model_eval, args),
                            "args": vars(args),
                            "protocol": cv_protocol,
                            "graph_contract": {
                                "scope": "fold_specific_message_graph"
                            },
                            "fold": fold,
                            "epoch": int(epoch + 1),
                            "monitor_metric": args.monitor_metric,
                            "best_value": float(best_metric),
                            "sup_rel": sup_rel,
                            "use_ema": bool(ema is not None),
                        },
                        best_ckpt_path,
                    )
                    print(f"[Fold {fold}] ✅ New best {args.monitor_metric}={monitor:.4f} @ epoch {epoch+1} → {best_ckpt_path}")
                else:
                    print(f"[Fold {fold}] val {args.monitor_metric}={monitor:.4f} (best={best_metric:.4f})")

        # ==== 折内最终评估（EMA/当前） ====
        train_time = time.time() - t0
        if cv_protocol == "fold_isolated":
            g_eval = g_train
        elif args.cv_mode.upper() == "CVS1":
            mode = getattr(args, "legacy_cv1_eval", "strict")
            if mode == "strict":
                g_eval = _build_eval_graph_cv1_strict(
                    hetero_graph, sup_rel_name, val_eids
                )
            elif mode == "keep_reverse":
                g_eval = _build_eval_graph_cv1_keep_reverse(
                    hetero_graph, sup_rel_name, val_eids
                )
            elif mode == "full_graph":
                g_eval = hetero_graph
            else:
                raise ValueError(f"Unknown legacy_cv1_eval: {mode}")
        else:
            g_eval = _build_eval_graph_keep_train_sup_only(
                hetero_graph, sup_rel_name, train_eids
            )

        model_eval = Model(args, g_eval, rel_list).to(device)
        final_state = (ema.module.state_dict() if ema is not None else model.state_dict())
        model_eval.load_state_dict(final_state, strict=True)
        model_eval.eval()

        pos_all, neg_all = [], []
        with torch.no_grad():
            for _, pos_g, neg_g, blocks in val_loader:
                pos_score, neg_score = model_eval(args, pos_g, neg_g, blocks, None)
                pos_all.append(pos_score[sup_rel].reshape(-1).cpu())
                neg_all.append(neg_score[sup_rel].reshape(-1).cpu())

        pos = torch.cat(pos_all).numpy() if pos_all else np.array([])
        neg = torch.cat(neg_all).numpy() if neg_all else np.array([])
        y_true = np.concatenate([np.ones_like(pos), np.zeros_like(neg)])
        y_pred = np.concatenate([pos, neg])
        auroc = roc_auc_score(y_true, y_pred)
        auprc = average_precision_score(y_true, y_pred)
        print(f"[Fold {fold}] time={train_time:.1f}s | AUROC={auroc:.4f} | AUPRC={auprc:.4f}")

        results.append((auroc, auprc))

    arr = np.array(results)
    print(f"[CV] AUROC mean={arr[:,0].mean():.4f} std={arr[:,0].std():.4f} | "
          f"AUPRC mean={arr[:,1].mean():.4f} std={arr[:,1].std():.4f}")

# =============== CLI ===============
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    # 设备 / 随机
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=410)

    # K 折 / 采样
    parser.add_argument("--k_fold", type=int, default=10)
    parser.add_argument(
        "--cv_mode",
        choices=["CVS1","CVS2","CVS3","CVS4","cv1","cv2","cv3","cv4"],
        default="CVS1",
    )
    parser.add_argument(
        "--cv_protocol",
        choices=["nested", "fold_isolated", "legacy"],
        default="nested",
        help=(
            "nested separates outer test from inner epoch/threshold selection; "
            "fold_isolated is W0 diagnostic only; legacy is provenance only."
        ),
    )
    parser.add_argument("--split_seed", type=int, default=411)
    parser.add_argument("--inner_val_fraction", type=float, default=0.2)
    parser.add_argument(
        "--threshold_metric", choices=["f1", "youden"], default="youden"
    )
    parser.add_argument("--output_dir", type=str, default="outputs/nested_eval")
    parser.add_argument("--dataset_name", type=str, default="HIT")
    parser.add_argument("--model_name", type=str, default="MCG-HGT")
    parser.add_argument(
        "--encoder_type", choices=["hgt", "feature_only"], default="hgt"
    )
    parser.add_argument("--known_positive_exclusions", type=str, default=None)
    parser.add_argument("--compound_registry", type=str, default=None)
    parser.add_argument("--target_registry", type=str, default=None)
    parser.add_argument(
        "--strict_cold_start", action="store_true", help="Legacy-only option."
    )
    parser.add_argument(
        "--strict_unseen", action="store_true", help="Legacy-only alias."
    )
    parser.add_argument(
        "--no_exclude_cv1", action="store_true", help="Legacy-only option."
    )
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument(
        "--share_hgt_layers",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=("Optionally reuse one HGT parameter set across propagation steps; "
              "the publication configuration uses independent layers."),
    )
    parser.add_argument("--fanout", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--neg_k", type=int, default=5)

    # 数据
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--ligand_embed", type=str, required=True)
    parser.add_argument("--ligand_id_key", type=str, default="node_id")
    parser.add_argument("--target_embed", type=str, required=True)
    parser.add_argument("--target_id_key", type=str, default="node_id")

    # 兼容旧脚本参数
    parser.add_argument("--graph_struct", type=int, default=3)
    parser.add_argument("--method", type=int, default=5)

    # 模型结构
    parser.add_argument("--in_dim", type=int, default=512)
    parser.add_argument("--h_dim", type=int, default=2048)
    parser.add_argument("--out_dim", type=int, default=512)
    parser.add_argument("--hgt_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.2)

    # 门控/语义
    parser.add_argument("--input_gate_type", choices=["none","se","glu","etype"], default="glu",
                        help="'etype' 作为旧版别名，将自动映射到 'se'")
    parser.add_argument("--input_gate_reduce", type=int, default=4)
    parser.add_argument("--residual_gate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--residual_message_prior", type=float, default=None)
    parser.add_argument("--feature_fusion", choices=["none", "type_scalar", "type_dynamic"], default="none")
    parser.add_argument("--feature_graph_prior", type=float, default=0.1)
    parser.add_argument("--score_gate", choices=["none","gmu","film"], default="gmu")
    parser.add_argument("--score_fusion", choices=["none", "feature_residual"], default="none")
    parser.add_argument("--score_graph_prior", type=float, default=0.1)
    parser.add_argument("--film_condition", choices=["src","dst","both"], default="src")
    parser.add_argument("--semantic_gate", choices=["none","etype"], default="etype")
    parser.add_argument("--sem_gate_bias", type=float, default=0.8)
    parser.add_argument("--head_gate", action=argparse.BooleanOptionalAction, default=True)

    # ====== LLM 嵌入消融（新增） ======
    parser.add_argument(
        "--ablate_ligand_llm",
        action="store_true",
        help="是否对配体 embedding 做 LLM 消融（在 process_data 中对整条向量操作）",
    )
    parser.add_argument(
        "--ablate_target_llm",
        action="store_true",
        help="是否对靶标 embedding 做 LLM 消融",
    )
    parser.add_argument(
        "--llm_ablation_mode",
        choices=["zero", "random", "shuffle"],
        default="zero",
        help="LLM 消融方式：zero=置零；random=随机噪声；shuffle=打乱节点-向量对应",
    )

    # 训练
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=None, help="兼容别名，若提供则覆盖 --wd")
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--lr_period", type=int, default=30)
    parser.add_argument("--lr_decay", type=float, default=0.5)
    parser.add_argument("--use_cosine", action="store_true")
    parser.add_argument("--cosine_T0", type=int, default=10)
    parser.add_argument("--cosine_Tmult", type=int, default=2)
    parser.add_argument("--log_every", type=int, default=5)
    parser.add_argument("--val_every", type=int, default=3)

    # InfoNCE
    parser.add_argument("--tau", type=float, default=0.07)
    parser.add_argument("--top_m", type=int, default=0)

    # 输入投影 MLP 配置（已在 model_gated.py 内读取）
    parser.add_argument("--proj_hidden_mult", type=int, default=4)
    parser.add_argument("--proj_dropout", type=float, default=0.2)
    parser.add_argument("--fused_adamw", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)

    # EMA（新增）
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--ema_decay", type=float, default=0.999)

    # 其它
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--monitor_metric", choices=["auprc","auroc"], default="auprc")
    parser.add_argument("--print_model", action="store_true")

    parser.add_argument(
        "--legacy_cv1_eval",
        choices=["strict", "keep_reverse", "full_graph"],
        default="strict",
        help="CVS1 evaluation graph used only with --cv_protocol legacy.",
    )

    # ====== 图增强开关（保持你的方案 B） ======
    parser.add_argument("--augment_sim_loops", action="store_true",
                        help="为 is/ts 同型边补自环并写 sim_deg_orig（默认关闭）")
    parser.add_argument("--augment_cv1", action="store_true",
                        help="CVS1 下也执行同型增强（默认不在 CVS1 执行）")

    args = parser.parse_args()
    if args.weight_decay is not None:
        args.wd = args.weight_decay

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tf32_active = bool(args.tf32 and device.type == "cuda")
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = tf32_active
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = tf32_active
    args.tf32_active = tf32_active

    # 构图 + 特征（此处会根据 ablate_xxx_llm / llm_ablation_mode 在 utlis.process_data 里做消融）
    edges, is_edges, ts_edges, initial_features = process_data(args)
    hetero_graph, rel_list = build_graph(args, edges, is_edges, ts_edges, initial_features, device)

    # === 轻量“图增强”：同型相似边补自环（可控开关，默认 CVS1 不增强） ===
    do_aug = False
    if getattr(args, "augment_sim_loops", False):
        if args.cv_mode.upper() != "CVS1" or getattr(args, "augment_cv1", False):
            do_aug = True

    if do_aug:
        if augment_similarity_graph is not None:
            try:
                hetero_graph = augment_similarity_graph(hetero_graph)
            except Exception as _e:
                print(f"[augment] fallback due to: {type(_e).__name__}: {_e}")
                hetero_graph = _augment_similarity_graph_fallback(hetero_graph)
        else:
            hetero_graph = _augment_similarity_graph_fallback(hetero_graph)

    train(args, hetero_graph, rel_list, device)
