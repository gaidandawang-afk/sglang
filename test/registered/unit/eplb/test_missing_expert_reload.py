import ast
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List
from unittest.mock import Mock


REPO_ROOT = Path(__file__).resolve().parents[4]
MANAGER_PATH = REPO_ROOT / "python/sglang/srt/eplb/eplb_manager.py"
QWEN_PATH = REPO_ROOT / "python/sglang/srt/models/qwen3_moe.py"
QWEN_MTP_PATH = REPO_ROOT / "python/sglang/srt/models/qwen3_moe_mtp.py"


def load_manager_recovery_function():
    tree = ast.parse(MANAGER_PATH.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "update_expert_location_with_recovery"
    )
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = {
        "ExpertLocationMetadata": object,
        "ExpertLocationUpdater": object,
        "List": List,
        "get_server_args": lambda: SimpleNamespace(
            model_path="model", load_format="safetensors"
        ),
        "logger": logging.getLogger(__name__),
        "nn": SimpleNamespace(Module=object),
    }
    exec(compile(module, str(MANAGER_PATH), "exec"), namespace)
    return namespace["update_expert_location_with_recovery"]


def load_qwen_filter_factory():
    tree = ast.parse(QWEN_PATH.read_text(encoding="utf-8"))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Qwen3MoeForCausalLM"
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "generate_weight_name_filter"
    )
    stub = ast.ClassDef(
        name="Qwen3MoeForCausalLM",
        bases=[],
        keywords=[],
        body=[method],
        decorator_list=[],
    )
    module = ast.fix_missing_locations(ast.Module(body=[stub], type_ignores=[]))
    namespace = {"Dict": Dict, "List": List}
    exec(compile(module, str(QWEN_PATH), "exec"), namespace)
    return namespace["Qwen3MoeForCausalLM"].generate_weight_name_filter


def make_recovery_inputs(model):
    updater = SimpleNamespace(
        update=Mock(return_value={3: [7]}),
    )
    return dict(
        expert_location_updater=updater,
        model=model,
        new_expert_location_metadata=object(),
        update_layer_ids=[3],
        nnodes=1,
        tp_rank=1,
        expert_backup_client=None,
        ep_dispatch_algorithm="none",
        init_lplb_solvers_callable=Mock(),
    )


def test_qwen_filter_selects_only_requested_routed_experts():
    weight_name_filter = load_qwen_filter_factory()({3: [7], 5: [2]})

    assert weight_name_filter(
        "model.layers.3.mlp.experts.7.gate_proj.weight"
    )
    assert weight_name_filter("model.layers.3.mlp.experts.7.up_proj.weight")
    assert weight_name_filter("mtp.layers.5.mlp.experts.2.down_proj.weight")
    assert not weight_name_filter(
        "model.layers.3.mlp.experts.8.gate_proj.weight"
    )
    assert not weight_name_filter(
        "model.layers.4.mlp.experts.7.gate_proj.weight"
    )
    assert not weight_name_filter(
        "model.layers.3.mlp.shared_experts.gate_proj.weight"
    )


def test_qwen_mtp_inherits_the_model_owned_filter():
    tree = ast.parse(QWEN_MTP_PATH.read_text(encoding="utf-8"))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "Qwen3MoeForCausalLMMTP"
    )

    assert any(
        isinstance(base, ast.Name) and base.id == "Qwen3MoeForCausalLM"
        for base in cls.bases
    )
    assert not any(
        isinstance(node, ast.FunctionDef)
        and node.name == "generate_weight_name_filter"
        for node in cls.body
    )


def test_missing_reload_requires_model_owned_filter():
    recover = load_manager_recovery_function()
    update_from_disk = Mock()
    inputs = make_recovery_inputs(
        SimpleNamespace(routed_experts_weights_of_layer={})
    )

    try:
        recover(
            **inputs,
            update_weights_from_disk_callable=update_from_disk,
        )
    except RuntimeError as exc:
        assert "model-owned" in str(exc)
    else:
        raise AssertionError("missing reload must reject a model without a filter")

    update_from_disk.assert_not_called()


def test_missing_reload_passes_only_the_model_selected_checkpoint_weights():
    recover = load_manager_recovery_function()
    model = SimpleNamespace(
        routed_experts_weights_of_layer={},
        generate_weight_name_filter=load_qwen_filter_factory(),
    )
    inputs = make_recovery_inputs(model)
    selected = []

    def successful_update(model_path, load_format, *, weight_name_filter):
        names = [
            "model.layers.3.mlp.experts.7.gate_proj.weight",
            "model.layers.3.mlp.experts.8.gate_proj.weight",
            "model.layers.3.mlp.shared_experts.gate_proj.weight",
        ]
        selected.extend(name for name in names if weight_name_filter(name))
        return True, "ok"

    recover(
        **inputs,
        update_weights_from_disk_callable=successful_update,
    )

    assert selected == ["model.layers.3.mlp.experts.7.gate_proj.weight"]


def test_missing_reload_propagates_failure_and_rejects_empty_selection():
    recover = load_manager_recovery_function()
    qwen_filter = load_qwen_filter_factory()
    model = SimpleNamespace(
        routed_experts_weights_of_layer={},
        generate_weight_name_filter=qwen_filter,
    )
    inputs = make_recovery_inputs(model)

    def failed_update(model_path, load_format, *, weight_name_filter):
        assert weight_name_filter(
            "model.layers.3.mlp.experts.7.gate_proj.weight"
        )
        return False, "reload failed"

    try:
        recover(
            **inputs,
            update_weights_from_disk_callable=failed_update,
        )
    except RuntimeError as exc:
        assert "reload failed" in str(exc)
    else:
        raise AssertionError("reload failure must fail the scale-down")

    def empty_update(model_path, load_format, *, weight_name_filter):
        assert not weight_name_filter(
            "model.layers.3.mlp.shared_experts.gate_proj.weight"
        )
        return True, "ok"

    try:
        recover(
            **inputs,
            update_weights_from_disk_callable=empty_update,
        )
    except RuntimeError as exc:
        assert "matched zero" in str(exc)
    else:
        raise AssertionError("empty selective reload must fail the scale-down")
