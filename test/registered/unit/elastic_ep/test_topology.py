import pytest

from sglang.srt.elastic_ep.topology import collapse_physical_rank_status


def test_physical_health_uses_all_members_rule():
    assert collapse_physical_rank_status(
        [True, True, True, True, True, False], attn_replica_size=2
    ) == [True, True, False]
    assert collapse_physical_rank_status([True, False, True], attn_replica_size=1) == [
        True,
        False,
        True,
    ]


def test_physical_health_rejects_incomplete_replica():
    with pytest.raises(ValueError, match="must be divisible"):
        collapse_physical_rank_status([True, True, False], attn_replica_size=2)
