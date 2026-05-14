import os
import types
import unittest

if os.name == "nt":
    raise unittest.SkipTest("SGLang runtime imports require POSIX modules.")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from sglang.srt.fault_tolerance.middleware import FaultToleranceAdmissionMiddleware
from sglang.srt.fault_tolerance.state import FaultToleranceState


class TestFaultToleranceMiddleware(unittest.TestCase):
    def test_blocks_regular_request_when_not_running(self):
        manager = types.SimpleNamespace(
            enabled=True,
            state=FaultToleranceState.COMM_ABORTED,
        )
        app = FastAPI()
        app.add_middleware(
            FaultToleranceAdmissionMiddleware, manager_getter=lambda: manager
        )

        @app.get("/generate")
        def generate():
            return {"ok": True}

        @app.get("/fault_tolerance/status")
        def status():
            return {"ok": True}

        client = TestClient(app)

        blocked = client.get("/generate")
        allowed = client.get("/fault_tolerance/status")

        self.assertEqual(blocked.status_code, 503)
        self.assertEqual(blocked.json()["error"]["state"], "COMM_ABORTED")
        self.assertEqual(allowed.status_code, 200)


if __name__ == "__main__":
    unittest.main()
