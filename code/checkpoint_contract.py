"""Pure-Python validation for portable MCG-HGT checkpoints."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple


CHECKPOINT_SCHEMA_VERSION = 1
FULL_RELEASE_GRAPH_SCOPE = "full_release_graph"
FOLD_GRAPH_SCOPES = frozenset({
    "fold_specific_message_graph",
    "strict_inductive_fold_graph",
})

MODEL_CONFIG_FIELDS = (
    "schema_version", "encoder_type", "in_dim", "h_dim", "out_dim",
    "hgt_heads", "num_layers", "share_hgt_layers", "hgt_parameter_sets",
    "dropout", "input_gate_type", "input_gate_reduce", "residual_gate",
    "residual_message_prior", "gate_bias", "feature_fusion",
    "feature_graph_prior", "score_gate", "score_fusion",
    "score_graph_prior", "film_condition", "semantic_gate", "sem_hidden",
    "sem_gate_bias", "head_gate", "proj_hidden_mult", "proj_dropout",
)
MODEL_CONFIG_FIELD_SET = frozenset(MODEL_CONFIG_FIELDS)


def architecture_id_for(config: Mapping[str, Any]) -> str:
    encoder_type = str(config.get("encoder_type", "")).strip().lower()
    if encoder_type == "feature_only":
        return "mcg_hgt_feature_only_v1"
    if encoder_type != "hgt":
        raise ValueError(f"Unsupported encoder_type: {encoder_type!r}")
    return (
        "mcg_hgt_weight_shared_recurrent_v1"
        if bool(config.get("share_hgt_layers"))
        else "mcg_hgt_independent_layers_v1"
    )


def normalize_state_dict_keys(state: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(state, Mapping) or not state:
        raise ValueError("state_dict must be a non-empty mapping")
    keys = [str(key) for key in state]
    prefixed = [key.startswith("module.") for key in keys]
    if any(prefixed) and not all(prefixed):
        raise ValueError("state_dict mixes module.-prefixed and plain keys")
    normalized: Dict[str, Any] = {}
    for key, value in state.items():
        name = str(key)
        if all(prefixed):
            name = name[len("module."):]
        if name in normalized:
            raise ValueError(f"Duplicate normalized state key: {name}")
        normalized[name] = value
    return normalized


def _shape(value: Any) -> Tuple[int, ...]:
    shape = getattr(value, "shape", None)
    return tuple(int(item) for item in shape) if shape is not None else ()


def infer_state_contract(
    state: Mapping[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    normalized = normalize_state_dict_keys(state)
    keys = list(normalized)
    shared = any(key.startswith("encoder.shared_hgt.") for key in keys)
    unshared = any(key.startswith("encoder.layers.") for key in keys)
    if shared and unshared:
        raise ValueError("state_dict contains both shared and independent HGT layouts")
    if not shared and not unshared:
        return normalized, {"encoder_type": "feature_only"}

    facts: Dict[str, Any] = {
        "encoder_type": "hgt",
        "share_hgt_layers": shared,
    }
    if shared:
        step_ids = {
            int(key.split("encoder.norms.", 1)[1].split(".", 1)[0])
            for key in keys if key.startswith("encoder.norms.")
        }
        facts["num_layers"] = len(step_ids)
        adapter_shape = _shape(normalized.get("encoder.shared_input_adapter.weight"))
        output_shape = _shape(normalized.get("encoder.shared_output_norm.weight"))
        if adapter_shape:
            facts["h_dim"] = adapter_shape[0]
        elif step_ids:
            norm_shape = _shape(normalized.get("encoder.norms.0.weight"))
            if norm_shape:
                facts["h_dim"] = norm_shape[0]
        if output_shape:
            facts["out_dim"] = output_shape[0]
    else:
        layer_ids = {
            int(key.split("encoder.layers.", 1)[1].split(".", 1)[0])
            for key in keys if key.startswith("encoder.layers.")
        }
        facts["num_layers"] = len(layer_ids)
        first_shape = _shape(normalized.get("encoder.norms.0.weight"))
        last_shape = _shape(
            normalized.get(f"encoder.norms.{len(layer_ids) - 1}.weight")
        )
        if first_shape:
            facts["h_dim"] = first_shape[0]
        if last_shape:
            facts["out_dim"] = last_shape[0]

    facts["residual_gate"] = any(
        key.startswith("encoder.res_gate.") for key in keys
    )
    facts["head_gate"] = any(
        key.startswith("encoder.head_alphas.") for key in keys
    )
    if any("encoder.input_gates." in key and ".Wa." in key for key in keys):
        facts["input_gate_type"] = "glu"
    elif any("encoder.input_gates." in key and ".gate." in key for key in keys):
        facts["input_gate_type"] = "se"
    else:
        facts["input_gate_type"] = "none"
    if any(key.startswith("encoder.feature_graph_logits.") for key in keys):
        facts["feature_fusion"] = "type_scalar"
    elif any(key.startswith("encoder.feature_graph_gates.") for key in keys):
        facts["feature_fusion"] = "type_dynamic"
    else:
        facts["feature_fusion"] = "none"
    facts["score_fusion"] = (
        "feature_residual"
        if any(key.startswith("feature_pred.") for key in keys)
        else "none"
    )
    if any(key.startswith("pred.scorer.cond_mlp.") for key in keys):
        facts["score_gate"] = "film"
    elif any(key.startswith("pred.scorer.gate.") for key in keys):
        facts["score_gate"] = "gmu"
    else:
        facts["score_gate"] = "none"
    facts["semantic_gate"] = (
        "etype" if any(key.startswith("pred.rel_gate.") for key in keys) else "none"
    )
    for key, value in normalized.items():
        if key.startswith("encoder.input_proj.") and key.endswith(".3.weight"):
            shape = _shape(value)
            if len(shape) == 2 and shape[0] > 0:
                facts["in_dim"] = shape[0]
                facts["proj_hidden_mult"] = max(1, shape[1] // shape[0])
                break
    return normalized, facts


def validate_model_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(config, Mapping):
        raise ValueError("checkpoint model_config must be a mapping")
    unknown = sorted(set(config) - MODEL_CONFIG_FIELD_SET)
    missing = sorted(MODEL_CONFIG_FIELD_SET - set(config))
    if unknown:
        raise ValueError(f"Unknown model_config fields: {', '.join(unknown)}")
    if missing:
        raise ValueError(f"Missing model_config fields: {', '.join(missing)}")
    result = {field: config[field] for field in MODEL_CONFIG_FIELDS}
    if int(result["schema_version"]) != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported model_config schema_version={result['schema_version']!r}"
        )
    if str(result["encoder_type"]).lower() == "hgt":
        expected_sets = (
            1 if bool(result["share_hgt_layers"]) else int(result["num_layers"])
        )
        if int(result["hgt_parameter_sets"]) != expected_sets:
            raise ValueError("hgt_parameter_sets conflicts with sharing configuration")
    return result


def resolve_checkpoint_contract(
    payload: Mapping[str, Any],
    state: Mapping[str, Any],
    cli_values: Mapping[str, Any] | None = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if not isinstance(payload, Mapping):
        raise ValueError("Portable inference requires a checkpoint payload mapping")
    if int(payload.get("checkpoint_schema_version", -1)) != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            "Missing or unsupported checkpoint_schema_version"
        )
    model_config = validate_model_config(payload.get("model_config", {}))
    architecture_id = str(payload.get("architecture_id", "")).strip()
    expected_id = architecture_id_for(model_config)
    if architecture_id != expected_id:
        raise ValueError(
            f"architecture_id={architecture_id!r} conflicts with model_config; "
            f"expected {expected_id!r}"
        )
    checkpoint_args = payload.get("args", {})
    if hasattr(checkpoint_args, "__dict__"):
        checkpoint_args = vars(checkpoint_args)
    if checkpoint_args is None:
        checkpoint_args = {}
    if not isinstance(checkpoint_args, Mapping):
        raise ValueError("checkpoint args must be a mapping when present")
    for field in MODEL_CONFIG_FIELDS:
        if field == "schema_version" or field not in checkpoint_args:
            continue
        if checkpoint_args[field] != model_config[field]:
            raise ValueError(
                f"checkpoint args.{field} conflicts with model_config.{field}"
            )

    normalized, state_facts = infer_state_contract(state)
    for field, observed in state_facts.items():
        if model_config.get(field) != observed:
            raise ValueError(
                f"state_dict requires {field}={observed!r}, but model_config "
                f"records {model_config.get(field)!r}"
            )
    for field, value in (cli_values or {}).items():
        if field not in MODEL_CONFIG_FIELD_SET or field == "schema_version":
            continue
        if value is not None and value != model_config[field]:
            raise ValueError(
                f"CLI {field}={value!r} conflicts with immutable checkpoint "
                f"value {model_config[field]!r}"
            )
    return normalized, model_config


def validate_graph_scope(
    payload: Mapping[str, Any], *, allow_legacy_unknown: bool = False
) -> str:
    contract = payload.get("graph_contract", {}) if isinstance(payload, Mapping) else {}
    scope = (
        str(contract.get("scope", "")).strip().lower()
        if isinstance(contract, Mapping)
        else str(contract).strip().lower()
    )
    if scope == FULL_RELEASE_GRAPH_SCOPE:
        return scope
    if scope in FOLD_GRAPH_SCOPES:
        raise ValueError(
            f"Checkpoint graph scope {scope!r} is fold-specific and cannot be "
            "used by generic full-graph inference"
        )
    if allow_legacy_unknown:
        return scope or "legacy_unknown"
    raise ValueError(
        "Generic inference accepts only graph_contract.scope='full_release_graph'; "
        "unknown or missing graph scope is rejected by default"
    )


def validate_portable_checkpoint_payload(payload: Mapping[str, Any]) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError("Checkpoint must contain a payload mapping")
    state = None
    for key in ("model_state_dict", "state_dict", "model"):
        candidate = payload.get(key)
        if isinstance(candidate, Mapping):
            state = candidate
            break
    if state is None:
        raise ValueError("Checkpoint payload does not contain a state_dict mapping")
    validate_graph_scope(payload)
    resolve_checkpoint_contract(payload, state)
