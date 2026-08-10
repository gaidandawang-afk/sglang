"""Unit coverage for NPU-only DeepEP elastic_info propagation."""

import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.elastic_ep.elastic_ep import ElasticEPStateManager
from sglang.srt.layers.moe.token_dispatcher import deepep
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _PatchedBuffer:
    def low_latency_dispatch(self, *, elastic_info=None):
        pass

    def low_latency_combine(self, *, elastic_info=None):
        pass


class _UnpatchedBuffer:
    def low_latency_dispatch(self):
        pass

    def low_latency_combine(self):
        pass


class TestNpuMC2DeepEPArguments(CustomTestCase):
    def _runtime_patches(self, buffer_cls, tensor):
        args = SimpleNamespace(
            enable_fault_tolerance=True,
            elastic_ep_backend="mc2",
        )
        state = SimpleNamespace(
            npu_mc2_elastic_info=SimpleNamespace(tensor=tensor)
        )
        return (
            patch.object(deepep, "_is_npu", True),
            patch.object(deepep, "Buffer", buffer_cls, create=True),
            patch(
                "sglang.srt.runtime_context.get_server_args", return_value=args
            ),
            patch.object(ElasticEPStateManager, "instance", return_value=state),
        )

    def test_fixed_tensor_is_passed_only_to_patched_npu_ft_deepep(self):
        tensor = torch.arange(12, dtype=torch.int32)
        with ExitStack() as stack:
            for patcher in self._runtime_patches(_PatchedBuffer, tensor):
                stack.enter_context(patcher)
            kwargs = deepep._npu_mc2_elastic_info_kwargs()

        self.assertIs(kwargs["elastic_info"], tensor)

        with patch.object(deepep, "_is_npu", False):
            self.assertEqual(deepep._npu_mc2_elastic_info_kwargs(), {})

    def test_unpatched_deepep_fails_before_dispatch(self):
        tensor = torch.arange(12, dtype=torch.int32)
        with ExitStack() as stack:
            for patcher in self._runtime_patches(_UnpatchedBuffer, tensor):
                stack.enter_context(patcher)
            with self.assertRaisesRegex(RuntimeError, "apply patches/ascend"):
                deepep._npu_mc2_elastic_info_kwargs()


if __name__ == "__main__":
    unittest.main()
