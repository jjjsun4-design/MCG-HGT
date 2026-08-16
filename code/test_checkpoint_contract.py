from __future__ import annotations

import copy
import unittest

from checkpoint_contract import (
    CHECKPOINT_SCHEMA_VERSION,
    MODEL_CONFIG_FIELDS,
    resolve_checkpoint_contract,
    validate_graph_scope,
    validate_portable_checkpoint_payload,
)


class FakeTensor:
    def __init__(self, *shape):
        self.shape = shape


def manuscript_state(prefix=""):
    return {
        prefix + "encoder.layers.0.k_linear.weight": FakeTensor(2048, 512),
        prefix + "encoder.layers.1.k_linear.weight": FakeTensor(2048, 2048),
        prefix + "encoder.layers.2.k_linear.weight": FakeTensor(512, 2048),
        prefix + "encoder.norms.0.weight": FakeTensor(2048),
        prefix + "encoder.norms.1.weight": FakeTensor(2048),
        prefix + "encoder.norms.2.weight": FakeTensor(512),
        prefix + "encoder.res_gate.0.weight": FakeTensor(2048, 4096),
        prefix + "encoder.res_gate.0.bias": FakeTensor(2048),
        prefix + "encoder.res_gate.1.weight": FakeTensor(2048, 4096),
        prefix + "encoder.res_gate.1.bias": FakeTensor(2048),
        prefix + "encoder.res_gate.2.weight": FakeTensor(512, 1024),
        prefix + "encoder.res_gate.2.bias": FakeTensor(512),
        prefix + "encoder.head_alphas.0": FakeTensor(8),
        prefix + "encoder.input_gates.ingredient.Wa.weight": FakeTensor(512, 512),
        prefix + "encoder.input_proj.ingredient.weight": FakeTensor(512, 2304),
        prefix + "pred.scorer.W_b": FakeTensor(512, 512),
        prefix + "pred.scorer.gate.weight": FakeTensor(1, 1536),
        prefix + "pred.rel_gate.ingredient__it__target.0.weight": FakeTensor(64, 1536),
    }


def model_config():
    values = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "encoder_type": "hgt", "in_dim": 512, "h_dim": 2048,
        "out_dim": 512, "hgt_heads": 8, "num_layers": 3,
        "share_hgt_layers": False, "hgt_parameter_sets": 3,
        "dropout": 0.2, "input_gate_type": "glu",
        "input_gate_reduce": 4, "residual_gate": True,
        "residual_message_prior": None, "gate_bias": 1.0,
        "feature_fusion": "none", "feature_graph_prior": 0.1,
        "score_gate": "gmu", "score_fusion": "none",
        "score_graph_prior": 0.1, "film_condition": "src",
        "semantic_gate": "etype", "sem_hidden": 64,
        "sem_gate_bias": 0.8, "head_gate": True,
        "input_projection": "type_specific_linear",
        "residual_gate_input": "message_residual_concat",
        "gmu_gate_input": "projected_source_target_hadamard",
        "bilinear_form": "full_matrix",
        "semantic_gate_input": "source_target_hadamard",
    }
    return {field: values[field] for field in MODEL_CONFIG_FIELDS}


def payload(state=None, scope="full_release_graph"):
    config = model_config()
    return {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "architecture_id": "mcg_hgt_independent_layers_v2",
        "model_config": config,
        "args": dict(config),
        "graph_contract": {"scope": scope},
        "state_dict": state or manuscript_state(),
    }


class CheckpointContractTests(unittest.TestCase):
    def test_valid_manuscript_formula_and_module_prefix(self):
        validate_portable_checkpoint_payload(payload())
        normalized, config = resolve_checkpoint_contract(
            payload(manuscript_state("module.")), manuscript_state("module.")
        )
        self.assertIn("encoder.layers.0.k_linear.weight", normalized)
        self.assertFalse(config["share_hgt_layers"])

    def test_legacy_formula_shapes_are_rejected(self):
        cases = {
            "residual_gate_input": (
                "encoder.res_gate.0.weight", FakeTensor(2048, 2048)
            ),
            "gmu_gate_input": (
                "pred.scorer.gate.weight", FakeTensor(1, 1024)
            ),
            "semantic_gate_input": (
                "pred.rel_gate.ingredient__it__target.0.weight",
                FakeTensor(64, 1024),
            ),
        }
        for expected_field, (key, value) in cases.items():
            with self.subTest(expected_field=expected_field):
                state = manuscript_state()
                state[key] = value
                item = payload(state)
                with self.assertRaisesRegex(ValueError, expected_field):
                    resolve_checkpoint_contract(item, state)

        state = manuscript_state()
        del state["pred.scorer.W_b"]
        item = payload(state)
        with self.assertRaisesRegex(ValueError, "bilinear_form"):
            resolve_checkpoint_contract(item, state)

    def test_cli_metadata_and_state_conflicts_fail(self):
        item = payload()
        with self.assertRaisesRegex(ValueError, "CLI film_condition"):
            resolve_checkpoint_contract(
                item, item["state_dict"], {"film_condition": "dst"}
            )
        conflicting = copy.deepcopy(item)
        conflicting["args"]["hgt_heads"] = 4
        with self.assertRaisesRegex(ValueError, "args.hgt_heads"):
            resolve_checkpoint_contract(
                conflicting, conflicting["state_dict"]
            )
        conflicting = copy.deepcopy(item)
        conflicting["model_config"]["num_layers"] = 2
        conflicting["model_config"]["hgt_parameter_sets"] = 2
        conflicting["args"]["num_layers"] = 2
        conflicting["args"]["hgt_parameter_sets"] = 2
        with self.assertRaisesRegex(ValueError, "state_dict requires num_layers"):
            resolve_checkpoint_contract(
                conflicting, conflicting["state_dict"]
            )

    def test_graph_scope_default_deny_and_override(self):
        with self.assertRaisesRegex(ValueError, "accepts only"):
            validate_graph_scope({})
        self.assertEqual(
            validate_graph_scope({}, allow_legacy_unknown=True),
            "legacy_unknown",
        )
        for scope in ("fold_specific_message_graph", "strict_inductive_fold_graph"):
            with self.assertRaisesRegex(ValueError, "fold-specific"):
                validate_graph_scope(
                    {"graph_contract": {"scope": scope}},
                    allow_legacy_unknown=True,
                )

    def test_missing_or_unknown_config_fields_fail(self):
        item = payload()
        del item["model_config"]["film_condition"]
        with self.assertRaisesRegex(ValueError, "Missing model_config"):
            validate_portable_checkpoint_payload(item)
        item = payload()
        item["model_config"]["unexpected"] = 1
        with self.assertRaisesRegex(ValueError, "Unknown model_config"):
            validate_portable_checkpoint_payload(item)


if __name__ == "__main__":
    unittest.main()
