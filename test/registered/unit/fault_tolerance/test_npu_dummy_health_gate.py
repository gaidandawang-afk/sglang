import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
from unittest.mock import Mock


REPO_ROOT = Path(__file__).resolve().parents[4]
BASE_RUNNER_PATH = REPO_ROOT / "python/sglang/srt/model_executor/runner/base_runner.py"
MODEL_RUNNER_PATH = REPO_ROOT / "python/sglang/srt/model_executor/model_runner.py"


class FillTensor:
    def __init__(self):
        self.filled = None

    def __getitem__(self, index):
        return self

    def fill_(self, value):
        self.filled = value


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


def test_recovery_dummy_is_one_valid_token_through_model_runner_forward():
    set_dp_buffer_len = Mock()
    set_is_extend_in_batch = Mock()
    torch = SimpleNamespace(inference_mode=nullcontext)
    run_dummy = load_method(
        BASE_RUNNER_PATH,
        "BaseRunner",
        "run_dummy_via_model_runner",
        {
            "Optional": Optional,
            "torch": torch,
            "set_dp_buffer_len": set_dp_buffer_len,
            "set_is_extend_in_batch": set_is_extend_in_batch,
        },
    )

    buffers = SimpleNamespace(seq_lens=FillTensor(), seq_lens_cpu=FillTensor())
    forward_batch = SimpleNamespace(
        dp_local_start_pos=7,
        dp_local_num_tokens=9,
        dp_padding_mode=SimpleNamespace(is_max_len=lambda: False),
    )
    active_mask = [True, False, True, True]
    prepare = Mock(return_value=(forward_batch, "pp-proxy", 3, 1, [1, 0, 1, 1]))
    model_runner = SimpleNamespace(forward=Mock(return_value="healthy"))
    runner = SimpleNamespace(
        model_runner=model_runner,
        _alloc_dummy_decode_buffers=Mock(return_value=buffers),
        _prepare_dummy_forward_batch=prepare,
    )

    output = run_dummy(runner, batch_size=1, active_mask=active_mask)

    assert output == "healthy"
    assert buffers.seq_lens.filled == 1
    assert buffers.seq_lens_cpu.filled == 1
    prepare.assert_called_once_with(1, buffers=buffers, active_mask=active_mask)
    model_runner.forward.assert_called_once_with(
        forward_batch, pp_proxy_tensors="pp-proxy"
    )
    assert forward_batch.dp_local_start_pos is None
    assert forward_batch.dp_local_num_tokens is None
    set_dp_buffer_len.assert_called_once_with(3, 1, False, [1, 0, 1, 1])
    set_is_extend_in_batch.assert_called_once_with(False)


def test_model_runner_health_gate_delegates_dummy_and_synchronizes_device():
    dummy = load_method(
        MODEL_RUNNER_PATH,
        "ModelRunner",
        "run_npu_fault_tolerance_dummy_batch",
        {},
    )
    synchronize = Mock()
    health_sync = load_method(
        MODEL_RUNNER_PATH,
        "ModelRunner",
        "synchronize_npu_fault_tolerance_health_gate",
        {
            "torch": SimpleNamespace(
                get_device_module=lambda device: SimpleNamespace(
                    synchronize=synchronize
                )
            )
        },
    )
    eager_runner = SimpleNamespace(run_dummy_via_model_runner=Mock())
    model_runner = SimpleNamespace(eager_runner=eager_runner, device="npu:1")

    dummy(model_runner, [True, False])
    health_sync(model_runner)

    eager_runner.run_dummy_via_model_runner.assert_called_once_with(
        batch_size=1, active_mask=[True, False]
    )
    synchronize.assert_called_once_with()
