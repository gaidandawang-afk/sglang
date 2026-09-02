import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

MODULE_PATH = (
    Path(__file__).resolve().parents[4]
    / "python/sglang/srt/fault_tolerance/npu_adapter.py"
)


def load_adapter_module():
    spec = importlib.util.spec_from_file_location("npu_adapter_under_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestNPUFaultToleranceAdapter(unittest.TestCase):
    def test_recover_device_runtime_order(self):
        events = []
        npu = SimpleNamespace(
            set_device=lambda device: events.append(("set_device", device)),
            stop_device=lambda device_id: events.append(("stop", device_id)),
            restart_device=lambda device_id: events.append(("restart", device_id)),
        )
        torch_npu = SimpleNamespace(
            npu=npu,
            distributed=SimpleNamespace(
                reinit_process_group=lambda group, rebuild_link: events.append(
                    ("reinit", group, rebuild_link)
                )
            ),
        )
        device = object()
        torch_module = SimpleNamespace(device=Mock(return_value=device))

        with patch.dict(sys.modules, {"torch": torch_module}):
            adapter_module = load_adapter_module()
        adapter = adapter_module.NPUFaultToleranceAdapter(device_id=2, dp_rank=5)

        with patch.dict(sys.modules, {"torch_npu": torch_npu}):
            adapter.recover_device_runtime()

        self.assertEqual(
            events,
            [
                ("set_device", device),
                ("stop", 2),
                ("restart", 2),
                ("reinit", None, False),
            ],
        )


if __name__ == "__main__":
    unittest.main()
