from __future__ import annotations

import ast
import unittest
from pathlib import Path

import inference
import main


ROOT = Path(__file__).resolve().parent


def source_defaults(path: Path) -> dict[str, object]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    defaults: dict[str, object] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        function = node.func
        if not isinstance(function, ast.Attribute) or function.attr != "add_argument":
            continue
        flag_node = node.args[0]
        if not isinstance(flag_node, ast.Constant) or not isinstance(flag_node.value, str):
            continue
        flag = flag_node.value
        for keyword in node.keywords:
            if keyword.arg == "default" and isinstance(keyword.value, ast.Constant):
                defaults[flag] = keyword.value.value
    return defaults


class PublicationDefaultsTests(unittest.TestCase):
    def test_training_entrypoint_matches_reported_configuration(self):
        parser = main.build_parser()
        args = parser.parse_args([])
        self.assertEqual(args.cv_protocol, "nested")
        self.assertEqual(args.cv_mode, "CVS1")
        self.assertEqual(args.num_epochs, 200)
        self.assertEqual(args.batch_size, 1024)
        self.assertEqual(args.wd, 1e-4)
        self.assertEqual(args.in_dim, 512)
        self.assertEqual(args.h_dim, 2048)
        self.assertEqual(args.out_dim, 512)
        self.assertEqual(args.num_layers, 3)
        self.assertEqual(args.hgt_heads, 8)
        self.assertFalse(args.share_hgt_layers)
        self.assertEqual(args.proj_hidden_mult, 4)
        self.assertEqual(args.legacy_cv1_eval, "strict")

        cv_action = next(action for action in parser._actions if action.dest == "cv_mode")
        self.assertIn("CVS4", cv_action.choices)

    def test_secondary_training_parser_has_the_same_defaults(self):
        defaults = source_defaults(ROOT / "training.py")
        expected = {
            "--num_epochs": 200,
            "--batch_size": 1024,
            "--wd": 1e-4,
            "--h_dim": 2048,
            "--num_layers": 3,
            "--hgt_heads": 8,
            "--share_hgt_layers": False,
            "--proj_hidden_mult": 4,
            "--legacy_cv1_eval": "strict",
        }
        self.assertEqual({key: defaults[key] for key in expected}, expected)

    def test_inference_fallback_matches_the_publication_architecture(self):
        defaults = inference._ARCH_DEFAULTS
        self.assertEqual(defaults["h_dim"], 2048)
        self.assertEqual(defaults["num_layers"], 3)
        self.assertEqual(defaults["hgt_heads"], 8)
        self.assertFalse(defaults["share_hgt_layers"])
        self.assertEqual(defaults["proj_hidden_mult"], 4)


if __name__ == "__main__":
    unittest.main()
