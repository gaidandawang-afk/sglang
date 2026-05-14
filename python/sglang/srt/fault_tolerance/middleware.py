from __future__ import annotations

from typing import Callable, Iterable, Optional

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from sglang.srt.fault_tolerance.state import FaultToleranceState


DEFAULT_ALLOWED_PREFIXES = (
    "/fault_tolerance/status",
    "/fault_tolerance/apply",
    "/health",
    "/health_generate",
    "/metrics",
    "/ping",
    "/openapi.json",
)


class FaultToleranceAdmissionMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app,
        manager_getter: Callable[[], Optional[object]],
        allowed_prefixes: Iterable[str] = DEFAULT_ALLOWED_PREFIXES,
    ):
        super().__init__(app)
        self.manager_getter = manager_getter
        self.allowed_prefixes = tuple(allowed_prefixes)

    async def dispatch(self, request: Request, call_next):
        manager = self.manager_getter()
        if manager is None or not getattr(manager, "enabled", False):
            return await call_next(request)

        path = request.url.path
        if path.startswith(self.allowed_prefixes):
            return await call_next(request)

        state = manager.state
        if state == FaultToleranceState.RUNNING:
            return await call_next(request)

        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "type": "fault_tolerance_unavailable",
                    "message": (
                        "SGLang engine is controlled by fault tolerance state "
                        f"{state.value}"
                    ),
                    "state": state.value,
                }
            },
        )
