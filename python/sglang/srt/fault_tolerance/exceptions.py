"""Fault tolerance internal exceptions."""


class FaultToleranceError(RuntimeError):
    """Base class for fault tolerance control-plane failures."""


class FaultToleranceDisabledError(FaultToleranceError):
    """Raised when an FT-only operation is requested while FT is disabled."""


class SentinelCommandTimeout(FaultToleranceError):
    """Raised when a scheduler sentinel command does not finish in time."""


class SchedulerTerminateRequested(BaseException):
    """Raised on the scheduler main thread when FT requests termination."""
