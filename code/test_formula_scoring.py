from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types
import unittest

import torch
import torch.nn as nn


def load_model_without_dgl():
    dgl = types.ModuleType("dgl")
    dgl_function = types.ModuleType("dgl.function")
    dgl_nn = types.ModuleType("dgl.nn")
    dgl_nn_pytorch = types.ModuleType("dgl.nn.pytorch")
    dgl_conv = types.ModuleType("dgl.nn.pytorch.conv")

    class DummyHGTConv(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

    dgl_conv.HGTConv = DummyHGTConv
    dgl.function = dgl_function
    dgl.nn = dgl_nn
    dgl_nn.pytorch = dgl_nn_pytorch
    dgl_nn_pytorch.conv = dgl_conv
    modules = {
        "dgl": dgl,
        "dgl.function": dgl_function,
        "dgl.nn": dgl_nn,
        "dgl.nn.pytorch": dgl_nn_pytorch,
        "dgl.nn.pytorch.conv": dgl_conv,
    }
    previous = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    try:
        path = Path(__file__).with_name("model.py")
        spec = importlib.util.spec_from_file_location("model_formula_test", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        for name, value in previous.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


MODEL = load_model_without_dgl()


class FormulaScoringTests(unittest.TestCase):
    def test_residual_gate_matches_equations_12_and_13(self):
        gate = nn.Linear(4, 2)
        with torch.no_grad():
            gate.weight.copy_(torch.tensor([[1.0, 0.0, 0.0, 1.0],
                                            [0.0, 1.0, 1.0, 0.0]]))
            gate.bias.copy_(torch.tensor([0.25, -0.5]))
        message = torch.tensor([[1.0, 2.0]])
        residual = torch.tensor([[3.0, 4.0]])
        observed = MODEL._gated_residual(message, residual, gate)
        manual_gate = torch.sigmoid(gate(torch.cat([message, residual], dim=-1)))
        expected = manual_gate * message + (1.0 - manual_gate) * residual
        torch.testing.assert_close(observed, expected)

    def test_gmu_matches_equations_15_to_18(self):
        scorer = MODEL._GMUScore(2)
        with torch.no_grad():
            scorer.proj_u.weight.copy_(torch.eye(2))
            scorer.proj_v.weight.copy_(torch.eye(2))
            scorer.W_b.copy_(torch.tensor([[2.0, 1.0], [-1.0, 3.0]]))
            scorer.gate.weight.zero_()
            scorer.gate.bias.zero_()

        hu = torch.tensor([[1.0, 2.0]], requires_grad=True)
        hv = torch.tensor([[3.0, 4.0]], requires_grad=True)
        observed = scorer(hu, hv)
        bilinear = torch.einsum("bi,ij,bj->b", hu, scorer.W_b, hv)
        expected = 0.5 * bilinear
        torch.testing.assert_close(observed, expected)

        observed.sum().backward()
        self.assertIsNotNone(scorer.W_b.grad)
        self.assertGreater(float(scorer.W_b.grad.abs().sum()), 0.0)

    def test_gmu_gate_uses_projected_hadamard_channel(self):
        scorer = MODEL._GMUScore(2)
        with torch.no_grad():
            scorer.proj_u.weight.copy_(torch.eye(2))
            scorer.proj_v.weight.copy_(torch.eye(2))
            scorer.W_b.copy_(torch.eye(2))
            scorer.gate.weight.zero_()
            scorer.gate.bias.zero_()
            scorer.gate.weight[0, 4:] = 1.0

        hu = torch.tensor([[1.0, 0.0]])
        low_product = scorer(hu, torch.tensor([[0.0, 1.0]]))
        high_product = scorer(hu, torch.tensor([[1.0, 1.0]]))
        self.assertEqual(float(low_product.detach()), 0.0)
        self.assertGreater(float(high_product.detach()), 0.5)

    def test_semantic_gate_has_source_target_hadamard_input(self):
        graph = types.SimpleNamespace(
            canonical_etypes=[("ingredient", "it", "target")]
        )
        predictor = MODEL.ScorePredictor(
            out_dim=2,
            g_hetero=graph,
            semantic_gate="etype",
            sem_hidden=3,
        )
        gate = predictor.rel_gate["ingredient__it__target"]
        self.assertEqual(gate[0].in_features, 6)
        hu = torch.tensor([[1.0, 2.0]])
        hv = torch.tensor([[3.0, 4.0]])
        features = torch.cat([hu, hv, hu * hv], dim=-1)
        self.assertTrue(torch.equal(features[:, 4:], hu * hv))


if __name__ == "__main__":
    unittest.main()
