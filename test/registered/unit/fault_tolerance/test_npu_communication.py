import importlib.util
from datetime import timedelta
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import Mock, patch


MODULE_PATH = (
    Path(__file__).resolve().parents[4]
    / "python/sglang/srt/fault_tolerance/npu_communication.py"
)


def load_communication_module():
    dist = ModuleType("torch.distributed")
    dist.PrefixStore = Mock(side_effect=lambda prefix, store: (prefix, store))
    dist.TCPStore = Mock()
    dist.ProcessGroup = object
    dist.barrier = Mock()
    dist.all_gather_into_tensor = Mock()
    torch = ModuleType("torch")
    torch.distributed = dist

    spec = importlib.util.spec_from_file_location(
        "npu_communication_under_test", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    with patch.dict(sys.modules, {"torch": torch, "torch.distributed": dist}):
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
        finally:
            sys.modules.pop(spec.name, None)
    return module


class TestNPUFaultToleranceCommunication(unittest.TestCase):
    def test_rebuild_survivor_group_uses_compact_rank(self):
        module = load_communication_module()
        communication = module.NPUFaultToleranceCommunication(
            store=object(),
            original_rank=2,
            original_world_size=4,
            gloo_timeout_sec=5,
            control_group=object(),
            active_original_ranks=(0, 1, 2, 3),
        )
        new_group = object()
        device_group = object()
        create_group = Mock(return_value=new_group)
        install_context = Mock()
        context = Mock()
        utils = ModuleType("sglang.srt.utils")
        utils.init_custom_process_group = create_group
        parallel_state = ModuleType("sglang.srt.distributed.parallel_state")
        parallel_state.get_moe_ep_group = Mock(
            return_value=ModuleType("moe_ep_group")
        )
        parallel_state.get_moe_ep_group.return_value.device_group = device_group
        process_group_context = ModuleType("sglang.srt.eplb.process_group_context")
        process_group_context.EPLBProcessGroupContext = context
        process_group_context.set_eplb_process_group_context = install_context

        with patch.dict(
            sys.modules,
            {
                utils.__name__: utils,
                parallel_state.__name__: parallel_state,
                process_group_context.__name__: process_group_context,
            },
        ):
            communication.rebuild_survivor_control_group([False, True, True, True])

        create_group.assert_called_once()
        kwargs = create_group.call_args.kwargs
        self.assertEqual(kwargs["world_size"], 3)
        self.assertEqual(kwargs["rank"], 1)
        self.assertEqual(kwargs["timeout"], timedelta(seconds=5))
        module.dist.barrier.assert_called_once_with(group=new_group)
        self.assertIs(communication.control_group, new_group)
        self.assertEqual(communication.active_original_ranks, (1, 2, 3))
        context.assert_called_once_with(
            control_group=new_group,
            device_group=device_group,
            active_original_ranks=(1, 2, 3),
            control_group_uses_cpu=True,
        )
        install_context.assert_called_once_with(context.return_value)

    def test_rebuild_rejects_inactive_local_rank(self):
        module = load_communication_module()
        communication = module.NPUFaultToleranceCommunication(
            store=object(),
            original_rank=0,
            original_world_size=4,
            gloo_timeout_sec=5,
            control_group=object(),
            active_original_ranks=(0, 1, 2, 3),
        )

        with self.assertRaisesRegex(ValueError, "not active"):
            communication.rebuild_survivor_control_group([False, True, True, True])

    def test_collective_wait_is_bounded(self):
        module = load_communication_module()
        work = Mock()
        work.wait.return_value = True
        module.dist.all_gather_into_tensor.return_value = work

        module.all_gather_into_tensor_with_timeout(
            object(), object(), group="survivor", timeout_sec=5
        )

        work.wait.assert_called_once_with(timeout=timedelta(seconds=5))

    def test_collective_timeout_raises(self):
        module = load_communication_module()
        work = Mock()
        work.wait.return_value = False
        module.dist.all_gather_into_tensor.return_value = work

        with self.assertRaisesRegex(RuntimeError, "timed out"):
            module.all_gather_into_tensor_with_timeout(
                object(), object(), group="survivor", timeout_sec=5
            )


if __name__ == "__main__":
    unittest.main()
