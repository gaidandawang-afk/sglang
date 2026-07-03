from __future__ import annotations

from typing import Any

from fastapi.responses import ORJSONResponse


FT_ADMISSION_BYPASS_PATHS = {
    "/fault_tolerance/status",
    "/fault_tolerance/apply",
    "/health",
    "/metrics",
    "/ping",
}


def should_reject_fault_tolerance_request(tokenizer_manager: Any, path: str) -> bool:
    if path in FT_ADMISSION_BYPASS_PATHS:
        return False
    ft = getattr(tokenizer_manager, "fault_tolerance", None)
    return bool(ft and ft.should_reject_admission())


def fault_tolerance_unavailable_response(message: str = "fault_tolerance_paused"):
    return ORJSONResponse(
        content={"success": False, "message": message},
        status_code=503,
    )
