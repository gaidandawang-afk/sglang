from sglang.srt.eplb.process_group_context import EPLBProcessGroupContext


def test_default_context_preserves_original_rank_namespace():
    context = EPLBProcessGroupContext()

    assert context.is_active(3)
    assert context.to_group_rank(3) == 3
    assert context.is_group_root(0)


def test_survivor_context_maps_original_to_compact_ranks_after_rank_zero_failure():
    context = EPLBProcessGroupContext(
        group=object(),
        active_original_ranks=(1, 2, 3),
    )

    assert not context.is_active(0)
    assert context.is_active(1)
    assert context.to_group_rank(1) == 0
    assert context.to_group_rank(3) == 2
    assert context.is_group_root(1)
    assert not context.is_group_root(0)
