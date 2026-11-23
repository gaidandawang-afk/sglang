import torch

from sglang.srt.elastic_ep.npu_mc2 import NpuMC2ElasticInfo
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_repeated_shrink_keeps_physical_slots_and_compacts_mc2_ids():
    info = NpuMC2ElasticInfo(ep_size=4, local_experts=2, device="cpu")
    data_ptr = info.tensor.data_ptr()
    ids = torch.arange(-1, 8, dtype=torch.int32)

    info.install((0, 2, 3))
    assert info.tensor.tolist() == [1, 3, 0, 6, 0, -1, 1, 2, 0, 2, 3, -1]
    assert info.compact_expert_ids(ids).tolist() == [-1, 0, 1, -1, -1, 2, 3, 4, 5]

    info.install((2, 3))
    assert info.tensor.data_ptr() == data_ptr
    assert info.tensor.tolist() == [1, 2, 0, 4, -1, -1, 0, 1, 2, 3, -1, -1]
    assert info.compact_expert_ids(ids).tolist() == [-1, -1, -1, -1, -1, 0, 1, 2, 3]
