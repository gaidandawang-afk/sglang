import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Optional
from unittest.mock import Mock

import torch


REPO_ROOT = Path(__file__).resolve().parents[4]
WEIGHT_UPDATER_PATH = (
    REPO_ROOT
    / "python/sglang/srt/model_executor/model_runner_components/weight_updater.py"
)
QWEN3_MOE_PATH = REPO_ROOT / "python/sglang/srt/models/qwen3_moe.py"


def load_functions(names):
    tree = ast.parse(WEIGHT_UPDATER_PATH.read_text(encoding="utf-8"))
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    namespace = {
        "Any": Any,
        "Callable": Callable,
        "DefaultModelLoader": object,
        "Optional": Optional,
        "torch": SimpleNamespace(device=object),
    }
    module = ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[]))
    exec(compile(module, str(WEIGHT_UPDATER_PATH), "exec"), namespace)
    return {name: namespace[name] for name in names}


def load_class_method(path, class_name, method_name):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )
    method.decorator_list = []
    namespace = {"Dict": dict, "List": list, "torch": torch}
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[method_name]


class TestNpuPartialWeightReload(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        functions = load_functions(
            {
                "_should_skip_full_model_postprocess_for_filtered_npu_reload",
                "_load_weights_for_disk_update",
            }
        )
        cls.should_skip = staticmethod(
            functions[
                "_should_skip_full_model_postprocess_for_filtered_npu_reload"
            ]
        )
        cls.load_weights = staticmethod(functions["_load_weights_for_disk_update"])
        cls.validate_probe = staticmethod(
            load_class_method(
                QWEN3_MOE_PATH,
                "Qwen3MoeForCausalLM",
                "validate_ft_reloaded_expert_probe",
            )
        )

    def test_only_filtered_unquantized_npu_reload_skips_postprocess(self):
        weight_filter = lambda _name: True

        self.assertTrue(
            self.should_skip("npu", SimpleNamespace(quantization=None), weight_filter)
        )
        self.assertFalse(
            self.should_skip("npu", SimpleNamespace(quantization=None), None)
        )
        self.assertFalse(
            self.should_skip("cuda", SimpleNamespace(quantization=None), weight_filter)
        )
        self.assertFalse(
            self.should_skip("npu", SimpleNamespace(quantization="fp8"), weight_filter)
        )

    def test_filtered_npu_reload_calls_model_loader_without_full_postprocess(self):
        loader = SimpleNamespace(load_weights_and_postprocess=Mock())
        model = SimpleNamespace(load_weights=Mock())
        weights = object()

        result = self.load_weights(
            loader,
            model,
            weights,
            object(),
            skip_full_model_postprocess=True,
        )

        self.assertIs(result, model)
        model.load_weights.assert_called_once_with(weights)
        loader.load_weights_and_postprocess.assert_not_called()

    def test_normal_reload_keeps_full_postprocess(self):
        loader = SimpleNamespace(load_weights_and_postprocess=Mock())
        model = SimpleNamespace(load_weights=Mock())
        weights = object()
        target_device = object()

        result = self.load_weights(
            loader,
            model,
            weights,
            target_device,
            skip_full_model_postprocess=False,
        )

        self.assertIs(result, model)
        loader.load_weights_and_postprocess.assert_called_once_with(
            model, weights, target_device
        )
        model.load_weights.assert_not_called()

    def test_qwen_filter_reports_checkpoint_pair_coverage(self):
        generate_filter = load_class_method(
            QWEN3_MOE_PATH,
            "Qwen3MoeForCausalLM",
            "generate_weight_name_filter",
        )
        weight_filter = generate_filter(None, {2: [64, 65]})

        self.assertTrue(
            weight_filter("model.layers.2.mlp.experts.64.gate_proj.weight")
        )
        self.assertTrue(
            weight_filter("model.layers.2.mlp.experts.65.down_proj.weight")
        )
        self.assertFalse(
            weight_filter("model.layers.3.mlp.experts.64.gate_proj.weight")
        )

        stats = weight_filter._sglang_ft_reload_stats
        self.assertEqual(stats["expected_pairs"], {(2, 64), (2, 65)})
        self.assertEqual(stats["selected_pairs"], {(2, 64), (2, 65)})
        self.assertEqual(stats["selected_weight_names"], 2)
        self.assertEqual(stats["probe_pair"], (2, 64))

        observe_weight = weight_filter._sglang_ft_observe_weight
        source_weight = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)
        for component in ("gate_proj", "up_proj", "down_proj"):
            observe_weight(
                f"model.layers.2.mlp.experts.64.{component}.weight",
                source_weight,
            )
        self.assertEqual(
            set(stats["source_probes"]),
            {"gate_proj", "up_proj", "down_proj"},
        )
        self.assertEqual(
            stats["source_probes"]["gate_proj"],
            [0.0, 1.0, 2.0, 3.0],
        )

    def test_qwen_probe_checks_original_rank_local_slot_contents(self):
        gate = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)
        up = gate + 20
        down = gate + 40
        w13 = torch.zeros((64, 6, 4), dtype=torch.bfloat16)
        w2 = torch.zeros((64, 3, 4), dtype=torch.bfloat16)
        w13[1, :3] = gate
        w13[1, 3:] = up
        w2[1] = down

        experts = SimpleNamespace(
            moe_tp_size=1,
            _expert_storage_rank=2,
            _num_local_routed=64,
            quant_method=SimpleNamespace(load_up_proj_weight_first=False),
            w13_weight=SimpleNamespace(data=w13),
            w2_weight=SimpleNamespace(data=w2),
        )
        layers = [None, None, SimpleNamespace(mlp=SimpleNamespace(experts=experts))]
        model = SimpleNamespace(model=SimpleNamespace(layers=layers))
        metadata = SimpleNamespace(logical_to_all_physical=lambda *_args: [129])
        self.validate_probe.__globals__["get_global_expert_location_metadata"] = (
            lambda: metadata
        )

        def make_probe(weight):
            return weight.reshape(-1)[:4].to(torch.float32).tolist()

        reload_stats = {
            "probe_pair": (2, 65),
            "source_probes": {
                "gate_proj": make_probe(gate),
                "up_proj": make_probe(up),
                "down_proj": make_probe(down),
            },
        }

        result = self.validate_probe(model, reload_stats, tp_rank=2)

        self.assertEqual(result["physical_expert_id"], 129)
        self.assertEqual(result["local_slot"], 1)
        self.assertEqual(result["max_abs_diff"], 0.0)

        reload_stats["source_probes"]["gate_proj"][0] += 1
        with self.assertRaisesRegex(RuntimeError, "incorrect physical slot"):
            self.validate_probe(model, reload_stats, tp_rank=2)


if __name__ == "__main__":
    unittest.main()
