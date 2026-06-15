from __future__ import annotations

from typing import Callable

from fastapi.responses import ORJSONResponse
from starlette.middleware.base import BaseHTTPMiddleware


FT_ADMISSION_BYPASS_PATHS = {
    "/fault_tolerance/status",
    "/fault_tolerance/apply",
    "/health",
    "/health_generate",
    "/metrics",
    "/ping",
}


class FaultToleranceAdmissionMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, global_state_getter: Callable):
        super().__init__(app)
        self.global_state_getter = global_state_getter

    async def dispatch(self, request, call_next):
        if request.url.path in FT_ADMISSION_BYPASS_PATHS:
            return await call_next(request)

        global_state = self.global_state_getter()
        tokenizer_manager = getattr(global_state, "tokenizer_manager", None)
        fault_tolerance = getattr(tokenizer_manager, "fault_tolerance", None)
        if fault_tolerance is None or not fault_tolerance.enabled:
            return await call_next(request)
        if fault_tolerance.admission_open():
            return await call_next(request)

        return ORJSONResponse(
            content=fault_tolerance.unavailable_error(),
            status_code=503,
        )
