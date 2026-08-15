"""Deterministic nested-CV splits, manifests, and thresholded metrics.

This module is deliberately model-agnostic.  It assigns positive-edge roles
for the outer test and inner validation stages and records every role needed to
audit leakage or join out-of-fold predictions later.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import dgl
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import KFold


@dataclass(frozen=True)
class NestedFoldSplit:
    """Positive-edge roles for one nested outer fold.

    CVS4 uses the excluded roles for cross-quadrant pairs where only one
    endpoint is cold.  Those pairs are absent from both training and the
    double-cold evaluation set.
    """

    outer_fold: int
    outer_train_eids: torch.Tensor
    outer_test_eids: torch.Tensor
    outer_excluded_eids: torch.Tensor
    inner_train_eids: torch.Tensor
    inner_val_eids: torch.Tensor
    inner_excluded_eids: torch.Tensor


def as_numpy_eids(eids: torch.Tensor | Sequence[int]) -> np.ndarray:
    if isinstance(eids, torch.Tensor):
        return eids.detach().cpu().numpy().astype(np.int64, copy=False).reshape(-1)
    return np.asarray(eids, dtype=np.int64).reshape(-1)


def to_graph_eids(
    values: torch.Tensor | Sequence[int] | np.ndarray,
    graph: dgl.DGLHeteroGraph,
) -> torch.Tensor:
    return torch.as_tensor(
        as_numpy_eids(values), dtype=torch.long, device=graph.device
    ).reshape(-1)


def edge_pairs(
    graph: dgl.DGLHeteroGraph,
    sup_rel,
    eids: torch.Tensor | Sequence[int] | None = None,
) -> List[Tuple[int, int]]:
    sup_rel = tuple(graph.to_canonical_etype(sup_rel))
    if eids is None:
        src, dst = graph.edges(etype=sup_rel)
    else:
        src, dst = graph.find_edges(to_graph_eids(eids, graph), etype=sup_rel)
    return list(zip(src.detach().cpu().tolist(), dst.detach().cpu().tolist()))


def _assert_role_partition(
    expected_eids: torch.Tensor | Sequence[int] | np.ndarray,
    roles: Dict[str, torch.Tensor | Sequence[int] | np.ndarray],
    label: str,
) -> None:
    expected = set(int(x) for x in as_numpy_eids(expected_eids))
    observed: set[int] = set()
    for role_name, role_eids in roles.items():
        values = [int(x) for x in as_numpy_eids(role_eids)]
        if len(values) != len(set(values)):
            raise AssertionError(f"{label}/{role_name} contains duplicate EIDs")
        overlap = observed.intersection(values)
        if overlap:
            raise AssertionError(
                f"{label} roles overlap at EIDs {sorted(overlap)[:10]}"
            )
        observed.update(values)
    if observed != expected:
        missing = sorted(expected.difference(observed))[:10]
        extra = sorted(observed.difference(expected))[:10]
        raise AssertionError(
            f"{label} roles do not partition the eligible EIDs; "
            f"missing={missing}, extra={extra}"
        )


def assert_cv_entity_isolation(
    graph: dgl.DGLHeteroGraph,
    sup_rel,
    train_eids: torch.Tensor,
    heldout_eids: torch.Tensor,
    cv_mode: str,
    label: str,
) -> None:
    train_pairs = np.asarray(
        edge_pairs(graph, sup_rel, train_eids), dtype=np.int64
    ).reshape(-1, 2)
    heldout_pairs = np.asarray(
        edge_pairs(graph, sup_rel, heldout_eids), dtype=np.int64
    ).reshape(-1, 2)
    if train_pairs.size == 0 or heldout_pairs.size == 0:
        raise AssertionError(f"{label} train and held-out sets must both be non-empty")
    cv_mode = cv_mode.upper()
    if cv_mode in {"CVS2", "CVS4"}:
        overlap = np.intersect1d(
            np.unique(train_pairs[:, 0]), np.unique(heldout_pairs[:, 0])
        )
        if overlap.size:
            raise AssertionError(
                f"{label} violates {cv_mode} source cold-start isolation: "
                f"{overlap[:10].tolist()}"
            )
    if cv_mode in {"CVS3", "CVS4"}:
        overlap = np.intersect1d(
            np.unique(train_pairs[:, 1]), np.unique(heldout_pairs[:, 1])
        )
        if overlap.size:
            raise AssertionError(
                f"{label} violates {cv_mode} target cold-start isolation: "
                f"{overlap[:10].tolist()}"
            )


def _random_holdout(
    values: np.ndarray,
    fraction: float,
    rng: np.random.RandomState,
) -> np.ndarray:
    values = np.unique(np.asarray(values, dtype=np.int64).reshape(-1))
    if values.size < 2:
        raise ValueError("At least two eligible items are required for inner validation")
    count = int(round(values.size * float(fraction)))
    count = max(1, min(values.size - 1, count))
    order = values.copy()
    rng.shuffle(order)
    return np.sort(order[:count])


def _standard_outer_folds(
    graph: dgl.DGLHeteroGraph,
    sup_rel,
    k: int,
    cv_mode: str,
    seed: int,
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    sup_rel = tuple(graph.to_canonical_etype(sup_rel))
    src_t, dst_t = graph.edges(etype=sup_rel)
    all_eids = graph.edges(etype=sup_rel, form="eid")
    src = src_t.detach().cpu().numpy()
    dst = dst_t.detach().cpu().numpy()
    eids = all_eids.detach().cpu().numpy()
    empty = to_graph_eids([], graph)
    cv_mode = cv_mode.upper()
    if k < 2:
        raise ValueError("k_fold must be at least 2")
    if cv_mode == "CVS1":
        splitter = KFold(n_splits=k, shuffle=True, random_state=seed)
        return [
            (to_graph_eids(eids[train], graph), to_graph_eids(eids[test], graph), empty)
            for train, test in splitter.split(eids)
        ]
    axis = src if cv_mode == "CVS2" else dst if cv_mode == "CVS3" else None
    if axis is None:
        raise ValueError(f"Unknown cv_mode: {cv_mode}")
    entities = np.unique(axis)
    if entities.size < k:
        raise ValueError(
            f"{cv_mode} requires at least k_fold observed cold-axis entities; "
            f"got {entities.size}, k={k}"
        )
    rng = np.random.RandomState(seed)
    rng.shuffle(entities)
    buckets = np.array_split(entities, k)
    folds = []
    for bucket in buckets:
        test_mask = np.isin(axis, bucket)
        folds.append(
            (
                to_graph_eids(eids[~test_mask], graph),
                to_graph_eids(eids[test_mask], graph),
                empty,
            )
        )
    return folds


def _cvs4_outer_folds(
    graph: dgl.DGLHeteroGraph,
    sup_rel,
    k: int,
    seed: int,
    max_attempts: int = 256,
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Create diagonal double-cold folds with cross-quadrant embargoes."""
    sup_rel = tuple(graph.to_canonical_etype(sup_rel))
    src_t, dst_t = graph.edges(etype=sup_rel)
    all_eids = graph.edges(etype=sup_rel, form="eid")
    src = src_t.detach().cpu().numpy().astype(np.int64, copy=False)
    dst = dst_t.detach().cpu().numpy().astype(np.int64, copy=False)
    eids = all_eids.detach().cpu().numpy().astype(np.int64, copy=False)
    unique_src = np.unique(src)
    unique_dst = np.unique(dst)
    if k < 2:
        raise ValueError("CVS4 requires k_fold >= 2")
    if unique_src.size < k or unique_dst.size < k:
        raise ValueError(
            "CVS4 requires at least k_fold observed source and target entities; "
            f"got source={unique_src.size}, target={unique_dst.size}, k={k}"
        )
    for attempt in range(max_attempts):
        rng = np.random.RandomState(int(seed) + attempt * 104729)
        src_order = unique_src.copy()
        dst_order = unique_dst.copy()
        rng.shuffle(src_order)
        rng.shuffle(dst_order)
        src_buckets = [np.sort(x) for x in np.array_split(src_order, k)]
        dst_buckets = [np.sort(x) for x in np.array_split(dst_order, k)]
        candidate = []
        for src_holdout, dst_holdout in zip(src_buckets, dst_buckets):
            src_cold = np.isin(src, src_holdout)
            dst_cold = np.isin(dst, dst_holdout)
            test_mask = src_cold & dst_cold
            train_mask = ~src_cold & ~dst_cold
            excluded_mask = ~(test_mask | train_mask)
            if not test_mask.any() or not train_mask.any():
                candidate = []
                break
            candidate.append(
                (
                    to_graph_eids(eids[train_mask], graph),
                    to_graph_eids(eids[test_mask], graph),
                    to_graph_eids(eids[excluded_mask], graph),
                )
            )
        if candidate:
            return candidate
    raise ValueError(
        "Unable to construct non-empty CVS4 double-cold folds after "
        f"{max_attempts} deterministic attempts. Reduce k_fold or change split_seed."
    )


def _inner_partition(
    graph: dgl.DGLHeteroGraph,
    sup_rel,
    eligible_eids: torch.Tensor,
    cv_mode: str,
    fraction: float,
    seed: int,
    max_attempts: int = 256,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not 0.0 < float(fraction) < 1.0:
        raise ValueError("inner_val_fraction must be strictly between 0 and 1")
    eligible = as_numpy_eids(eligible_eids)
    if eligible.size < 2:
        raise ValueError("Outer training set is too small for inner validation")
    src_t, dst_t = graph.find_edges(to_graph_eids(eligible, graph), etype=sup_rel)
    src = src_t.detach().cpu().numpy().astype(np.int64, copy=False)
    dst = dst_t.detach().cpu().numpy().astype(np.int64, copy=False)
    cv_mode = cv_mode.upper()
    for attempt in range(max_attempts):
        rng = np.random.RandomState(int(seed) + attempt * 130363)
        if cv_mode == "CVS1":
            val_eids = _random_holdout(eligible, fraction, rng)
            val_mask = np.isin(eligible, val_eids)
            train_mask = ~val_mask
            excluded_mask = np.zeros_like(train_mask)
        elif cv_mode == "CVS2":
            heldout_src = _random_holdout(src, fraction, rng)
            val_mask = np.isin(src, heldout_src)
            train_mask = ~val_mask
            excluded_mask = np.zeros_like(train_mask)
        elif cv_mode == "CVS3":
            heldout_dst = _random_holdout(dst, fraction, rng)
            val_mask = np.isin(dst, heldout_dst)
            train_mask = ~val_mask
            excluded_mask = np.zeros_like(train_mask)
        elif cv_mode == "CVS4":
            heldout_src = _random_holdout(src, fraction, rng)
            heldout_dst = _random_holdout(dst, fraction, rng)
            src_cold = np.isin(src, heldout_src)
            dst_cold = np.isin(dst, heldout_dst)
            val_mask = src_cold & dst_cold
            train_mask = ~src_cold & ~dst_cold
            excluded_mask = ~(val_mask | train_mask)
        else:
            raise ValueError(f"Unknown cv_mode: {cv_mode}")
        if train_mask.any() and val_mask.any():
            return (
                to_graph_eids(eligible[train_mask], graph),
                to_graph_eids(eligible[val_mask], graph),
                to_graph_eids(eligible[excluded_mask], graph),
            )
    raise ValueError(
        f"Unable to construct a non-empty {cv_mode} inner split after "
        f"{max_attempts} deterministic attempts"
    )


def make_nested_splits(
    graph: dgl.DGLHeteroGraph,
    sup_rel,
    k: int,
    cv_mode: str,
    inner_val_fraction: float = 0.2,
    seed: int = 411,
) -> List[NestedFoldSplit]:
    """Build deterministic outer-test/inner-validation positive-edge roles."""
    sup_rel = tuple(graph.to_canonical_etype(sup_rel))
    all_eids = graph.edges(etype=sup_rel, form="eid")
    all_pairs = edge_pairs(graph, sup_rel)
    duplicates = {pair: n for pair, n in Counter(all_pairs).items() if n > 1}
    if duplicates:
        raise AssertionError(
            "Supervised interaction pairs must be globally unique before nested "
            f"cross-validation. duplicates={list(duplicates.items())[:10]}"
        )
    cv_mode = cv_mode.upper()
    outer_raw = (
        _cvs4_outer_folds(graph, sup_rel, k, seed)
        if cv_mode == "CVS4"
        else _standard_outer_folds(graph, sup_rel, k, cv_mode, seed)
    )
    splits = []
    for outer_fold, (outer_train, outer_test, outer_excluded) in enumerate(
        outer_raw, start=1
    ):
        _assert_role_partition(
            all_eids,
            {"train": outer_train, "test": outer_test, "excluded": outer_excluded},
            f"outer fold {outer_fold}",
        )
        assert_cv_entity_isolation(
            graph, sup_rel, outer_train, outer_test, cv_mode, f"outer fold {outer_fold}"
        )
        inner_train, inner_val, inner_excluded = _inner_partition(
            graph,
            sup_rel,
            outer_train,
            cv_mode,
            inner_val_fraction,
            int(seed) + outer_fold * 1009,
        )
        _assert_role_partition(
            outer_train,
            {"train": inner_train, "validation": inner_val, "excluded": inner_excluded},
            f"inner fold {outer_fold}",
        )
        assert_cv_entity_isolation(
            graph, sup_rel, inner_train, inner_val, cv_mode, f"inner fold {outer_fold}"
        )
        if set(as_numpy_eids(outer_test)).intersection(as_numpy_eids(inner_val)):
            raise AssertionError("Outer-test EIDs leaked into inner validation")
        splits.append(
            NestedFoldSplit(
                outer_fold,
                outer_train,
                outer_test,
                outer_excluded,
                inner_train,
                inner_val,
                inner_excluded,
            )
        )
    return splits


def hash_eids(eids: torch.Tensor | Sequence[int]) -> str:
    values = np.sort(as_numpy_eids(eids)).astype("<i8", copy=False)
    return hashlib.sha256(values.tobytes()).hexdigest()


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


def _entity_ids_for_eids(
    graph,
    sup_rel,
    eids,
    compound_keys: Sequence[str] | None = None,
    target_keys: Sequence[str] | None = None,
) -> Dict[str, Any]:
    pairs = np.asarray(edge_pairs(graph, sup_rel, eids), dtype=np.int64).reshape(-1, 2)
    source_ids = np.unique(pairs[:, 0]).astype(int).tolist()
    target_ids = np.unique(pairs[:, 1]).astype(int).tolist()
    result: Dict[str, Any] = {
        "source_ids": source_ids,
        "target_ids": target_ids,
    }
    if compound_keys is not None and target_keys is not None:
        result.update(
            {
                "compound_keys": [str(compound_keys[value]) for value in source_ids],
                "target_keys": [str(target_keys[value]) for value in target_ids],
                "pair_keys": [
                    _canonical_pair_fields(
                        int(source_id),
                        int(target_id),
                        compound_keys,
                        target_keys,
                    )[2]
                    for source_id, target_id in pairs
                ],
            }
        )
    return result


def write_split_manifest(
    splits: Sequence[NestedFoldSplit],
    graph: dgl.DGLHeteroGraph,
    sup_rel,
    output_dir: Path,
    cv_mode: str,
    split_seed: int,
    inner_val_fraction: float,
    compound_keys: Sequence[str] | None = None,
    target_keys: Sequence[str] | None = None,
    evaluation_inputs: Dict[str, Any] | None = None,
) -> Tuple[Path, Path]:
    """Write both compact fold metadata and an edge-level role table."""
    sup_rel = tuple(graph.to_canonical_etype(sup_rel))
    if (compound_keys is None) != (target_keys is None):
        raise ValueError("Canonical compound and target registries must be paired")
    if compound_keys is not None:
        if len(compound_keys) != graph.num_nodes(sup_rel[0]):
            raise ValueError("Compound registry size does not match the graph")
        if len(target_keys) != graph.num_nodes(sup_rel[2]):
            raise ValueError("Target registry size does not match the graph")
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "split_manifest.json"
    csv_path = output_dir / "split_manifest_edges.csv"
    all_eids = as_numpy_eids(graph.edges(etype=sup_rel, form="eid"))
    src_t, dst_t = graph.edges(etype=sup_rel)
    src = src_t.detach().cpu().numpy().astype(np.int64, copy=False)
    dst = dst_t.detach().cpu().numpy().astype(np.int64, copy=False)
    manifest_folds = []
    rows = []
    oof_test_union: set[int] = set()

    for split in splits:
        outer_roles = {
            "outer_train": set(int(x) for x in as_numpy_eids(split.outer_train_eids)),
            "outer_test": set(int(x) for x in as_numpy_eids(split.outer_test_eids)),
            "outer_excluded": set(int(x) for x in as_numpy_eids(split.outer_excluded_eids)),
        }
        inner_roles = {
            "inner_train": set(int(x) for x in as_numpy_eids(split.inner_train_eids)),
            "inner_validation": set(int(x) for x in as_numpy_eids(split.inner_val_eids)),
            "inner_excluded": set(int(x) for x in as_numpy_eids(split.inner_excluded_eids)),
        }
        oof_test_union.update(outer_roles["outer_test"])
        for eid in all_eids:
            eid_i = int(eid)
            outer_role = next(name for name, values in outer_roles.items() if eid_i in values)
            inner_role = (
                next(name for name, values in inner_roles.items() if eid_i in values)
                if outer_role == "outer_train"
                else "not_inner_eligible"
            )
            compound_key, target_key, pair_key = _canonical_pair_fields(
                int(src[eid_i]),
                int(dst[eid_i]),
                compound_keys,
                target_keys,
            )
            rows.append(
                {
                    "outer_fold": split.outer_fold,
                    "edge_id": eid_i,
                    "source_id": int(src[eid_i]),
                    "target_id": int(dst[eid_i]),
                    "compound_key": compound_key,
                    "target_key": target_key,
                    "pair_key": pair_key,
                    "outer_role": outer_role,
                    "inner_role": inner_role,
                    "is_oof_evaluation": int(outer_role == "outer_test"),
                }
            )
        fold_record: Dict[str, Any] = {"outer_fold": split.outer_fold}
        for name, values in (
            ("outer_train", split.outer_train_eids),
            ("outer_test", split.outer_test_eids),
            ("outer_excluded", split.outer_excluded_eids),
            ("inner_train", split.inner_train_eids),
            ("inner_validation", split.inner_val_eids),
            ("inner_excluded", split.inner_excluded_eids),
        ):
            fold_record[name] = {
                "count": int(len(as_numpy_eids(values))),
                "eid_sha256": hash_eids(values),
                "eids": [int(x) for x in as_numpy_eids(values)],
                **_entity_ids_for_eids(
                    graph,
                    sup_rel,
                    values,
                    compound_keys,
                    target_keys,
                ),
            }
        manifest_folds.append(fold_record)

    payload = {
        "schema_version": 2,
        "protocol": "nested",
        "cv_mode": cv_mode.upper(),
        "split_seed": int(split_seed),
        "inner_val_fraction": float(inner_val_fraction),
        "num_supervised_edges": int(len(all_eids)),
        "num_outer_folds": int(len(splits)),
        "oof_test_unique_edges": int(len(oof_test_union)),
        "oof_test_coverage_fraction": float(len(oof_test_union) / len(all_eids)),
        "cvs4_note": (
            "CVS4 uses paired source/target buckets. Cross-quadrant edges are "
            "outer_excluded; only edges with both endpoints cold are tested."
            if cv_mode.upper() == "CVS4"
            else None
        ),
        "evaluation_inputs": evaluation_inputs or {},
        "folds": manifest_folds,
    }
    with json_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "outer_fold",
                "edge_id",
                "source_id",
                "target_id",
                "compound_key",
                "target_key",
                "pair_key",
                "outer_role",
                "inner_role",
                "is_oof_evaluation",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def classification_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if labels.size != scores.size or labels.size == 0:
        raise ValueError("labels and scores must be non-empty and aligned")
    if not np.isfinite(scores).all() or not np.isfinite(float(threshold)):
        raise ValueError("scores and threshold must be finite")
    if set(np.unique(labels).tolist()) != {0, 1}:
        raise ValueError("Both positive and negative labels are required")
    pred = (scores >= float(threshold)).astype(np.int64)
    tp = int(np.sum((labels == 1) & (pred == 1)))
    tn = int(np.sum((labels == 0) & (pred == 0)))
    fp = int(np.sum((labels == 0) & (pred == 1)))
    fn = int(np.sum((labels == 1) & (pred == 0)))
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = (
        2.0 * precision * sensitivity / (precision + sensitivity)
        if precision + sensitivity
        else 0.0
    )
    mcc_denominator = float(
        np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    )
    mcc = (tp * tn - fp * fn) / mcc_denominator if mcc_denominator else 0.0
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
        "accuracy": float((tp + tn) / labels.size),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "f1": float(f1),
        "mcc": float(mcc),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def select_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    objective: str = "youden",
) -> Tuple[float, Dict[str, float]]:
    """Select a raw-score threshold deterministically on validation data."""
    objective = objective.lower()
    if objective not in {"f1", "youden"}:
        raise ValueError(f"Unknown threshold objective: {objective}")
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    # Validate inputs and compute rank metrics once.  Re-evaluating every unique
    # threshold against every sample would be quadratic on the full datasets.
    classification_metrics(labels, scores, float(scores[0]) if scores.size else 0.0)
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    group_ends = np.flatnonzero(
        np.r_[sorted_scores[1:] != sorted_scores[:-1], True]
    )
    tp = np.cumsum(sorted_labels)[group_ends].astype(np.float64)
    predicted_positive = (group_ends + 1).astype(np.float64)
    fp = predicted_positive - tp
    total_positive = float(np.sum(labels == 1))
    total_negative = float(np.sum(labels == 0))
    fn = total_positive - tp
    tn = total_negative - fp
    sensitivity = tp / total_positive
    specificity = tn / total_negative
    precision = np.divide(
        tp,
        tp + fp,
        out=np.zeros_like(tp),
        where=(tp + fp) > 0,
    )
    f1 = np.divide(
        2.0 * precision * sensitivity,
        precision + sensitivity,
        out=np.zeros_like(tp),
        where=(precision + sensitivity) > 0,
    )
    values = f1 if objective == "f1" else sensitivity + specificity - 1.0
    best_value = float(np.max(values))
    # Algebraically identical Youden-J values can differ at machine precision
    # after evaluating sensitivity and specificity separately.  Treat those
    # values as ties, matching the independent OOF audit, then take the first
    # threshold because the candidates are sorted from largest to smallest.
    tied = np.isclose(values, best_value, rtol=0.0, atol=1e-12)
    best_index = int(np.flatnonzero(tied)[0])
    best_threshold = float(sorted_scores[group_ends[best_index]])
    best_metrics = classification_metrics(labels, scores, best_threshold)
    best_metrics["threshold_objective_value"] = float(values[best_index])
    return best_threshold, best_metrics


def write_oof_predictions(rows: Sequence[Dict[str, Any]], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "oof_predictions.csv"
    fieldnames = [
        "dataset",
        "protocol",
        "mode",
        "model",
        "fold",
        "split",
        "sample_id",
        "label",
        "score",
        "evaluation_protocol",
        "cv_mode",
        "outer_fold",
        "pair_kind",
        "edge_id",
        "source_id",
        "target_id",
        "compound_key",
        "target_key",
        "pair_key",
        "score_raw",
        "threshold_raw",
        "predicted_label",
        "selected_epoch",
        "inner_monitor_metric",
        "inner_monitor_value",
        "split_seed",
        "model_seed",
        "evaluation_negative_seed",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_summary(summary: Dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "nested_summary.json"
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path
