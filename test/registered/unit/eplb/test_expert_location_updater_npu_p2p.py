"""Unit tests for offset-zero NPU buffers in EPLB expert migration."""

import unittest
from unittest.mock import patch

import torch
from sglang.srt.eplb import expert_location_updater
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _FakeDevice:
    def __init__(self, device_type: str):
        self.type = device_type

    def __hash__(self):
        return hash(self.type)


class _FakeTensor:
    def __init__(self, device_type: str, storage_offset: int = 0):
        self.device = _FakeDevice(device_type)
        self._storage_offset = storage_offset
        self.shape = (4, 4)
        self.dtype = torch.float16

    def storage_offset(self):
        return self._storage_offset

    def data_ptr(self):
        return id(self)

    def stride(self):
        return (4, 1)


class _FakeP2POp:
    def __init__(self, op, tensor, peer, group=None, tag=0):
        self.op = op
        self.tensor = tensor
        self.peer = peer
        self.group = group
        self.tag = tag


class TestExpertLocationUpdaterNPUP2P(CustomTestCase):
    def test_internal_format_copy_uses_npu_raw_copy_helper(self):
        destination = _FakeTensor("npu", storage_offset=7)
        source = _FakeTensor("npu", storage_offset=0)

        with (
            patch(
                "sglang.srt.hardware_backend.npu.utils.is_npu_internal_format_tensor",
                return_value=True,
            ),
            patch(
                "sglang.srt.hardware_backend.npu.utils.copy_npu_formatted_tensor_"
            ) as formatted_copy,
        ):
            expert_location_updater._copy_expert_tensor_(destination, source)

        formatted_copy.assert_called_once_with(destination, source)

    def test_plain_tensor_copy_keeps_pytorch_path(self):
        destination = torch.zeros(2)
        source = torch.tensor([1.0, 2.0])

        expert_location_updater._copy_expert_tensor_(destination, source)

        self.assertTrue(torch.equal(destination, source))

    def test_only_nonzero_offset_npu_views_need_staging(self):
        self.assertFalse(
            expert_location_updater._needs_npu_p2p_staging(
                _FakeTensor("npu", storage_offset=0)
            )
        )
        self.assertTrue(
            expert_location_updater._needs_npu_p2p_staging(
                _FakeTensor("npu", storage_offset=7)
            )
        )
        self.assertFalse(
            expert_location_updater._needs_npu_p2p_staging(
                _FakeTensor("cuda", storage_offset=7)
            )
        )

    def test_recv_preserves_p2p_metadata_and_copies_back(self):
        original = _FakeTensor("npu", storage_offset=7)
        staged = _FakeTensor("npu", storage_offset=0)
        group = object()
        op = _FakeP2POp(
            torch.distributed.irecv,
            original,
            peer=2,
            group=group,
            tag=11,
        )

        with (
            patch.object(expert_location_updater, "P2POp", _FakeP2POp),
            patch.object(
                expert_location_updater,
                "_new_npu_offset_zero_staging_like",
                return_value=staged,
            ),
        ):
            staged_ops, send_copy_infos, recv_copy_infos = (
                expert_location_updater._stage_npu_p2p_ops([op])
            )

        self.assertIs(staged_ops[0].tensor, staged)
        self.assertEqual(staged_ops[0].peer, 2)
        self.assertIs(staged_ops[0].group, group)
        self.assertEqual(staged_ops[0].tag, 11)
        self.assertEqual(send_copy_infos, [])
        self.assertEqual(recv_copy_infos, [(original, staged)])

        with patch.object(expert_location_updater, "_copy_expert_tensor_") as copy:
            expert_location_updater._copy_expert_tensors_(recv_copy_infos)

        copy.assert_called_once_with(original, staged)

    def test_multicast_reuses_one_staged_send_tensor(self):
        original = _FakeTensor("npu", storage_offset=7)
        staged = _FakeTensor("npu", storage_offset=0)
        ops = [
            _FakeP2POp(torch.distributed.isend, original, peer=1),
            _FakeP2POp(torch.distributed.isend, original, peer=2),
        ]

        with (
            patch.object(expert_location_updater, "P2POp", _FakeP2POp),
            patch.object(
                expert_location_updater,
                "_new_npu_offset_zero_staging_like",
                return_value=staged,
            ) as new_staging,
        ):
            staged_ops, send_copy_infos, recv_copy_infos = (
                expert_location_updater._stage_npu_p2p_ops(ops)
            )

        new_staging.assert_called_once_with(original)
        self.assertIs(staged_ops[0].tensor, staged_ops[1].tensor)
        self.assertEqual(send_copy_infos, [(staged, original)])
        self.assertEqual(recv_copy_infos, [])

        with patch.object(expert_location_updater, "_copy_expert_tensor_") as copy:
            expert_location_updater._copy_expert_tensors_(send_copy_infos)

        copy.assert_called_once_with(staged, original)

    def test_weight_update_uses_staged_buffers(self):
        routed_expert_weights = [
            torch.tensor(
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                ]
            )
        ]
        temp_buffers = [torch.empty_like(routed_expert_weights[0])]
        observed_ops = []
        observed_send_payloads = []

        class FakeRequest:
            def __init__(self, op):
                self.op = op

            def wait(self):
                if self.op.op == torch.distributed.irecv:
                    self.op.tensor.copy_(torch.tensor([20.0, 21.0]))

        def fake_batch_isend_irecv(ops):
            observed_ops.extend(ops)
            observed_send_payloads.extend(
                op.tensor.clone() for op in ops if op.op == torch.distributed.isend
            )
            return [FakeRequest(op) for op in ops]

        with (
            patch.object(expert_location_updater, "P2POp", _FakeP2POp),
            patch.object(
                expert_location_updater,
                "_needs_npu_p2p_staging",
                return_value=True,
            ),
            patch.object(
                expert_location_updater,
                "_new_npu_offset_zero_staging_like",
                side_effect=torch.empty_like,
            ),
            patch.object(
                expert_location_updater.torch.distributed,
                "batch_isend_irecv",
                side_effect=fake_batch_isend_irecv,
            ),
            patch.object(
                expert_location_updater.ElasticEPStateManager,
                "instance",
                return_value=None,
            ),
            patch.object(
                expert_location_updater.envs.SGLANG_EPLB_P2P_BATCH_CHUNK_SIZE,
                "get",
                return_value=4,
            ),
        ):
            expert_location_updater.update_expert_weights_single_layer(
                routed_experts_weights=routed_expert_weights,
                temp_buffers=temp_buffers,
                old_physical_to_logical_map=[0, 1, 2, 3],
                new_physical_to_logical_map=[0, 2, 1, 3],
                num_local_physical_experts=2,
                num_gpu_per_node=2,
                rank=0,
                world_size=2,
            )

        self.assertEqual(len(observed_ops), 2)
        self.assertTrue(all(op.tensor.storage_offset() == 0 for op in observed_ops))
        self.assertEqual(len(observed_send_payloads), 1)
        self.assertTrue(
            torch.equal(observed_send_payloads[0], torch.tensor([3.0, 4.0]))
        )
        self.assertTrue(
            torch.equal(
                routed_expert_weights[0],
                torch.tensor(
                    [
                        [1.0, 2.0],
                        [20.0, 21.0],
                    ]
                ),
            )
        )


if __name__ == "__main__":
    unittest.main()
