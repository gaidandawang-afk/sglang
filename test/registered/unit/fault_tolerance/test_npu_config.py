import ast
import logging
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, Dict, List, Tuple
import unittest
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[4]


def load_method(path, class_name, method_name, namespace):
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
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[method_name]


class TestNPUFaultToleranceConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.handle = staticmethod(
            load_method(
                REPO_ROOT / "python/sglang/srt/server_args.py",
                "ServerArgs",
                "_handle_fault_tolerance",
                {
                    "is_npu": lambda: True,
                    "logger": logging.getLogger(__name__),
                    "os": os,
                },
            )
        )
        cls.parse_apply_args = staticmethod(
            load_method(
                REPO_ROOT / "python/sglang/srt/fault_tolerance/manager.py",
                "FaultToleranceManager",
                "_parse_apply_args",
                {"Any": Any, "Dict": Dict, "List": List, "Tuple": Tuple},
            )
        )

    def make_args(self):
        return SimpleNamespace(
            enable_fault_tolerance=True,
            fault_tolerance_on_error_strategy="pause",
            elastic_ep_backend="mc2",
            fault_tolerance_gloo_timeout=10,
            fault_tolerance_timeout=20,
            fault_tolerance_communication_abort_timeout=7,
        )

    def test_npu_timeout_environment_is_configured(self):
        controller = ModuleType("sglang.srt.fault_tolerance.controller")
        controller.is_ft_supported_config = lambda _: (True, "")
        modules = {
            "sglang": ModuleType("sglang"),
            "sglang.srt": ModuleType("sglang.srt"),
            "sglang.srt.fault_tolerance": ModuleType("sglang.srt.fault_tolerance"),
            "sglang.srt.fault_tolerance.controller": controller,
        }

        with patch.dict(os.environ, {}, clear=True), patch.dict(sys.modules, modules):
            self.handle(self.make_args())
            self.assertEqual(os.environ["TASK_QUEUE_ENABLE"], "0")
            self.assertEqual(os.environ["HCCL_EVENT_TIMEOUT"], "7")
            self.assertEqual(os.environ["HCCL_EXEC_TIMEOUT"], "6")
            self.assertEqual(os.environ["ACL_STREAM_TIMEOUT"], "7000")

    def test_gloo_timeout_must_be_less_than_command_timeout(self):
        args = self.make_args()
        args.fault_tolerance_gloo_timeout = 20

        with self.assertRaisesRegex(ValueError, "gloo_timeout"):
            self.handle(args)

    def test_external_task_queue_setting_must_be_zero(self):
        with patch.dict(os.environ, {"TASK_QUEUE_ENABLE": "1"}, clear=True):
            with self.assertRaisesRegex(ValueError, "TASK_QUEUE_ENABLE"):
                self.handle(self.make_args())

    def test_per_request_timeout_must_exceed_gloo_timeout(self):
        manager = SimpleNamespace(
            server_args=SimpleNamespace(
                fault_tolerance_timeout=20,
                fault_tolerance_gloo_timeout=10,
                device="npu",
                elastic_ep_backend="mc2",
            ),
            state=SimpleNamespace(dp_size=4),
        )
        request = {
            "instruction": "scale_down",
            "params": {"timeout": 10, "ranks": [0]},
        }

        with self.assertRaisesRegex(ValueError, "greater than"):
            self.parse_apply_args(manager, request)


if __name__ == "__main__":
    unittest.main()
