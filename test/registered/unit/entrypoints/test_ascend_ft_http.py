"""HTTP boundary of the v0.1.0 fault-tolerance contract."""

import json
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from sglang.srt.entrypoints import http_server
from sglang.srt.fault_tolerance.protocol import ScaleDownRequest
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


@pytest.fixture
def http(monkeypatch):
    manager = Mock()
    tokenizer = SimpleNamespace(ascend_ft=manager)
    monkeypatch.setattr(
        http_server, "_global_state", SimpleNamespace(tokenizer_manager=tokenizer)
    )
    app = FastAPI()
    app.get("/v1/fault_tolerance/status")(http_server.ascend_ft_status)
    app.post("/v1/fault_tolerance/apply")(http_server.ascend_ft_apply)
    app.post("/generate")(http_server.generate_request)
    return TestClient(app), tokenizer, manager


def test_apply_preserves_request_id_and_status_fields(http):
    client, _, manager = http
    request_id = "client-0007"
    accepted = {
        "message": "Request accepted; poll /v1/fault_tolerance/status for updates.",
        "request_id": request_id,
    }
    manager.submit.return_value = HTTPStatus.ACCEPTED, accepted
    response = client.post(
        "/v1/fault_tolerance/apply",
        json={
            "instruction": "scale_down",
            "params": {"removed_dp_ranks": [0]},
            "request_id": request_id,
        },
    )
    assert response.status_code == 202
    assert response.json() == accepted
    (request,) = manager.submit.call_args.args
    assert isinstance(request, ScaleDownRequest)
    assert request.request_id == request_id
    assert request.params.removed_dp_ranks == [0]

    status = {
        "schema_version": 1,
        "total_engines": 2,
        "engines": [{"id": 0, "status": "dead"}, {"id": 1, "status": "healthy"}],
        "last_ft_request_id": request_id,
    }
    manager.status.return_value = status
    response = client.get("/v1/fault_tolerance/status")
    assert response.status_code == 200
    assert response.json() == status


@pytest.mark.parametrize(
    "body",
    [
        {"instruction": "retry", "request_id": 7},
        {"instruction": "scale_down", "params": {"removed_dp_ranks": [True]}},
        {"instruction": "scale_down", "params": {"removed_dp_ranks": [-1]}},
        {"instruction": "scale_down", "params": {}},
        {"instruction": "pause", "params": {}},
    ],
)
def test_bad_envelope_rejected_before_submission(http, body):
    client, _, manager = http
    response = client.post("/v1/fault_tolerance/apply", json=body)
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == 400
    assert error["type"] == "Bad Request"
    assert error["param"] is None
    assert error["message"]
    manager.submit.assert_not_called()


def test_busy_and_disabled_have_http_error_envelope(http):
    client, tokenizer, manager = http
    manager.submit.return_value = HTTPStatus.CONFLICT, {"message": "operation_running"}
    req = {"instruction": "retry", "request_id": "client-0008"}
    busy = client.post("/v1/fault_tolerance/apply", json=req)
    assert busy.status_code == 409
    assert busy.json()["error"]["message"] == "operation_running"

    tokenizer.ascend_ft = None
    for response in (
        client.get("/v1/fault_tolerance/status"),
        client.post("/v1/fault_tolerance/apply", json=req),
    ):
        assert response.status_code == 503
        assert response.json() == {
            "error": {
                "message": "fault_tolerance_disabled",
                "type": "Service Unavailable",
                "param": None,
                "code": 503,
            }
        }


@pytest.mark.parametrize("stream", [False, True])
def test_inactive_dp_503_before_headers(http, stream):
    client, tokenizer, _ = http

    async def refuse(obj, _request):
        # Pre-admission can also reject after tokenization, before the DPC send.
        if obj.routed_dp_rank == 0:
            raise HTTPException(
                status_code=503, detail="routed_dp_rank=0 is not active"
            )
        yield {}

    tokenizer.generate_request = refuse
    response = client.post(
        "/generate", json={"text": "hi", "routed_dp_rank": 0, "stream": stream}
    )
    assert response.status_code == 503
    assert response.json()["error"]["message"] == "routed_dp_rank=0 is not active"


def test_admitted_stream_keeps_first_event(http):
    client, tokenizer, _ = http

    async def generate(obj, _request):
        obj.normalize_batch_and_arguments()
        yield {"text": "a", "meta_info": {"finish_reason": None}}
        yield {"text": "ab", "meta_info": {"finish_reason": {"type": "stop"}}}

    tokenizer.generate_request = generate
    tokenizer.create_abort_task = lambda _obj: None
    response = client.post("/generate", json={"text": "hi", "stream": True})
    assert response.status_code == 200
    events = [
        line.removeprefix("data: ")
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    assert events[-1] == "[DONE]"
    assert [json.loads(event)["text"] for event in events[:-1]] == ["a", "ab"]
