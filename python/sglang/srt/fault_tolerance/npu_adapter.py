import logging

import torch

logger = logging.getLogger(__name__)


class NPUFaultToleranceAdapter:
    """NPU-only recovery operations used by the generic FT control flow."""

    def __init__(self, *, device_id: int, dp_rank: int):
        self.device_id = device_id
        self.dp_rank = dp_rank

    def configure_operation_timeout(self, timeout_sec: float) -> None:
        if timeout_sec <= 0:
            return
        import torch_npu

        torch_npu.npu.set_op_timeout_ms(int(timeout_sec * 1000))

    def recover_device_runtime(self) -> None:
        import torch_npu

        device = torch.device("npu", self.device_id)
        logger.info(
            "NPU FT device recovery step=bind_device phase=begin dp_rank=%s "
            "device_id=%s",
            self.dp_rank,
            self.device_id,
        )
        torch_npu.npu.set_device(device)
        logger.info(
            "NPU FT device recovery step=bind_device phase=complete dp_rank=%s "
            "device_id=%s",
            self.dp_rank,
            self.device_id,
        )

        logger.info(
            "NPU FT device recovery step=stop_device phase=begin dp_rank=%s "
            "device_id=%s",
            self.dp_rank,
            self.device_id,
        )
        torch_npu.npu.stop_device(self.device_id)
        logger.info(
            "NPU FT device recovery step=stop_device phase=complete dp_rank=%s "
            "device_id=%s",
            self.dp_rank,
            self.device_id,
        )

        logger.info(
            "NPU FT device recovery step=restart_device phase=begin dp_rank=%s "
            "device_id=%s",
            self.dp_rank,
            self.device_id,
        )
        torch_npu.npu.restart_device(self.device_id)
        logger.info(
            "NPU FT device recovery step=restart_device phase=complete dp_rank=%s "
            "device_id=%s",
            self.dp_rank,
            self.device_id,
        )

        logger.info(
            "NPU FT device recovery step=reinit_process_group phase=begin "
            "dp_rank=%s device_id=%s",
            self.dp_rank,
            self.device_id,
        )
        torch_npu.distributed.reinit_process_group(None, False)
        logger.info(
            "NPU FT device recovery step=reinit_process_group phase=complete "
            "dp_rank=%s device_id=%s",
            self.dp_rank,
            self.device_id,
        )
