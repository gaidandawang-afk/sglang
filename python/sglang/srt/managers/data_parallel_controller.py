# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A controller that dispatches requests to multiple data parallel workers."""

import faulthandler
import logging
import multiprocessing as mp
import os
import signal
import tempfile
import threading
import time
from enum import Enum, auto
from typing import Callable, List, Optional

import psutil
import setproctitle
import zmq

from sglang.srt.environ import envs
from sglang.srt.fault_tolerance.process_registry import SchedulerProcessRegistry
from sglang.srt.fault_tolerance.rank_space import (
    FT_RANK_SPACE_DP_ROUTE,
    active_ranks_broadcast_rank_space,
    is_dp_route_rank_space,
)
from sglang.srt.layers.dp_attention import compute_dp_attention_world_info
from sglang.srt.managers.io_struct import (
    AbortReq,
    ActiveRanksOutput,
    ActiveRanksUpdateReqOutput,
    BatchTokenizedEmbeddingReqInput,
    BatchTokenizedGenerateReqInput,
    BlockReqInput,
    FaultToleranceCommandReqInput,
    FaultToleranceCommandReqOutput,
    FaultToleranceRankFaultOutput,
    FaultToleranceRankRejoinOutput,
    ProfileReq,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
    WatchLoadUpdateReq,
)
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.scheduler import run_scheduler_process
from sglang.srt.observability.cpu_monitor import start_cpu_monitor_thread
from sglang.srt.observability.req_time_stats import DPControllerReqTimeStats
from sglang.srt.observability.trace import process_tracing_init, trace_set_thread_info
from sglang.srt.server_args import (
    DP_ATTENTION_HANDSHAKE_PORT_DELTA,
    PortArgs,
    ServerArgs,
)
from sglang.srt.utils import numa_utils
from sglang.srt.utils.common import (
    configure_logger,
    kill_itself_when_parent_died,
    maybe_reindex_device_id,
)
from sglang.srt.utils.network import (
    NetworkAddress,
    bind_port,
    get_zmq_socket,
    get_zmq_socket_on_host,
)
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter
from sglang.srt.utils.watchdog import Watchdog
from sglang.srt.utils.watchdog import SubprocessWatchdog
from sglang.utils import TypeBasedDispatcher, get_exception_traceback

logger = logging.getLogger(__name__)

SCHEDULER_PIDS_ARG = "scheduler_pids"


class LoadBalanceMethod(Enum):
    """Load balance method."""

    ROUND_ROBIN = auto()
    FOLLOW_BOOTSTRAP_ROOM = auto()
    TOTAL_REQUESTS = auto()
    TOTAL_TOKENS = auto()

    @classmethod
    def from_str(cls, method: str):
        method = method.upper()
        try:
            return cls[method]
        except KeyError as exc:
            raise ValueError(f"Invalid load balance method: {method}") from exc


class DPBudget:
    def __init__(self, dp_size: int):
        self.dp_size = dp_size
        self.total_requests = [0] * dp_size
        self.total_tokens = [0] * dp_size

    def update_budget(self, load_update: WatchLoadUpdateReq):
        """Update the budget."""
        for load in load_update.loads:
            self.total_requests[load.dp_rank] = (
                load.num_running_reqs + load.num_waiting_reqs
            )
            self.total_tokens[load.dp_rank] = load.num_total_tokens

    def dispatch(self, method: LoadBalanceMethod, estimated_tokens: int = 0):
        if method == LoadBalanceMethod.TOTAL_REQUESTS:
            target_rank = self.total_requests.index(min(self.total_requests))
        elif method == LoadBalanceMethod.TOTAL_TOKENS:
            # Use total_requests as a tie-breaker when total_tokens are equal
            target_rank = min(
                range(self.dp_size),
                key=lambda i: (self.total_tokens[i], self.total_requests[i]),
            )
        else:
            return None

        # Increment the load of that worker by one as a heuristic
        self.total_requests[target_rank] += 1
        self.total_tokens[target_rank] += estimated_tokens
        return target_rank


class DataParallelController:
    """A controller that dispatches requests to multiple data parallel workers."""

    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        run_scheduler_process_func: Callable,
    ) -> None:
        # Parse args
        self.server_args = server_args
        self.port_args = port_args
        self.load_balance_method = LoadBalanceMethod.from_str(
            server_args.load_balance_method
        )
        self.run_scheduler_process_func = run_scheduler_process_func

        # For DP balance
        self.global_balance_id = 0

        # Init inter-process communication
        self.context = zmq.Context(1 + server_args.dp_size)
        if server_args.node_rank == 0:
            self.recv_from_tokenizer = get_zmq_socket(
                self.context, zmq.PULL, port_args.scheduler_input_ipc_name, False
            )
            self.send_to_tokenizer = get_zmq_socket(
                self.context, zmq.PUSH, port_args.tokenizer_ipc_name, False
            )
        else:
            self.send_to_tokenizer = None

        # Dispatch method
        self.round_robin_counter = 0
        dispatch_lookup = {
            LoadBalanceMethod.ROUND_ROBIN: self.round_robin_scheduler,
            LoadBalanceMethod.FOLLOW_BOOTSTRAP_ROOM: self.follow_bootstrap_room_scheduler,
            LoadBalanceMethod.TOTAL_REQUESTS: self.total_requests_scheduler,
            LoadBalanceMethod.TOTAL_TOKENS: self.total_tokens_scheduler,
        }
        self.dispatching = dispatch_lookup[self.load_balance_method]

        # Load balance budget
        self.dp_budget = DPBudget(server_args.dp_size)

        # To protect changing env vars to set CUDA_VISIBLE_DEVICES.
        self.env_lock = threading.Lock()

        # Launch data parallel workers
        self.scheduler_procs = []
        self.scheduler_process_registry = SchedulerProcessRegistry(
            dp_size=server_args.dp_size,
            tp_size=server_args.tp_size,
            attn_cp_size=server_args.attn_cp_size,
            enable_dp_attention=server_args.enable_dp_attention,
        )
        self.workers: List[zmq.Socket] = [None] * server_args.dp_size
        self.ft_control_workers: List[Optional[zmq.Socket]] = []
        self.status: List[bool] = [True] * server_args.dp_size
        self._init_ft_control_channels(server_args, port_args)

        if server_args.enable_dp_attention:
            self.launch_dp_attention_schedulers(server_args, port_args)
            # When local control broadcast is enabled, send control messages to
            # every DP group leader (attn_tp_rank=0) so each leader broadcasts
            # within its own attn_tp_group instead of the full tp_group.
            # Otherwise fall back to the original behaviour: send to only the
            # first leader, which then broadcasts over the full tp_group.
            local_ctrl = server_args.enable_dp_attention_local_control_broadcast
            self.control_message_step = 1 if local_ctrl else server_args.tp_size
        else:
            self.launch_dp_schedulers(server_args, port_args)
            self.control_message_step = 1

        self._scheduler_watchdog = None
        if self.scheduler_procs and server_args.node_rank == 0:
            self._scheduler_watchdog = SubprocessWatchdog(
                processes=self.scheduler_procs,
                process_names=[
                    f"scheduler_dp_{i}" for i in range(len(self.scheduler_procs))
                ],
                on_exit=self._handle_scheduler_process_exit,
            )
            self._scheduler_watchdog.start()

        self.init_dispatcher()

        self.soft_watchdog = Watchdog.create(
            debug_name="DataParallelController",
            watchdog_timeout=server_args.soft_watchdog_timeout,
            soft=True,
            test_stuck_time=envs.SGLANG_TEST_STUCK_DP_CONTROLLER.get(),
        )

        if server_args.enable_metrics:
            start_cpu_monitor_thread("data_parallel_controller")

    def send_to_all_workers(self, obj):
        for i, worker in enumerate(self.workers):
            if self.status[i]:
                worker.send_pyobj(obj)

    def _init_ft_control_channels(
        self, server_args: ServerArgs, port_args: PortArgs
    ) -> None:
        if not (server_args.enable_fault_tolerance and server_args.enable_dp_attention):
            return
        if port_args.ft_control_ipc_names is None:
            port_args.ft_control_ipc_names = [
                f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}"
                for _ in range(server_args.tp_size)
            ]
        self.ft_control_workers = [
            get_zmq_socket(self.context, zmq.PUSH, endpoint, True)
            for endpoint in port_args.ft_control_ipc_names
        ]

    def _dp_rank_for_scheduler_index(self, scheduler_index: int) -> Optional[int]:
        return self.scheduler_process_registry.dp_rank_for_scheduler_index(
            scheduler_index
        )

    def send_control_message(self, obj):
        # Send control messages to first worker of tp group
        self._refresh_worker_liveness()
        for rank in range(0, len(self.workers), self.control_message_step):
            if self.status[rank]:
                self.workers[rank].send_pyobj(obj)

    def _send_ft_command_failure(
        self, request_id: str, rank: int, message: str
    ) -> None:
        if self.send_to_tokenizer is None:
            return
        self.send_to_tokenizer.send_pyobj(
            FaultToleranceCommandReqOutput(
                request_id=request_id,
                rank=rank,
                success=False,
                message=message,
            )
        )

    def _has_dead_scheduler_rank(self) -> bool:
        return self.scheduler_process_registry.has_dead_scheduler_rank(
            self.scheduler_procs
        )

    def _ft_control_rank_for_target(self, target_rank: int) -> Optional[int]:
        return self.scheduler_process_registry.ft_control_rank_for_target(
            target_rank,
            control_message_step=self.control_message_step,
            worker_count=len(self.workers),
        )

    def _is_ft_control_rank_reachable(self, control_rank: int) -> bool:
        return self.scheduler_process_registry.is_ft_control_rank_reachable(
            control_rank,
            control_message_step=self.control_message_step,
            worker_count=len(self.workers),
            status=self.status,
            processes=self.scheduler_procs,
        )

    def _send_ft_command_direct(self, obj: FaultToleranceCommandReqInput) -> bool:
        if not self.ft_control_workers:
            return False
        for rank in obj.target_ranks:
            if not (
                0 <= rank < len(self.ft_control_workers)
                and rank < len(self.scheduler_procs)
            ):
                self._send_ft_command_failure(
                    obj.request_id,
                    rank,
                    "unknown rank",
                )
                continue
            proc = self.scheduler_procs[rank]
            if not self.scheduler_process_registry.is_rank_alive(rank, proc):
                self._send_ft_command_failure(
                    obj.request_id,
                    rank,
                    "ft_target_rank_unreachable",
                )
                continue
            logger.info(
                "DPC forwarding direct FT command: id=%s command=%s rank=%s",
                obj.request_id,
                obj.command,
                rank,
            )
            self.ft_control_workers[rank].send_pyobj(obj)
        return True

    def send_fault_tolerance_command(self, obj: FaultToleranceCommandReqInput):
        self._refresh_worker_liveness()
        if obj.command == "shutdown":
            for rank in obj.target_ranks:
                success = True
                message = "shutdown requested"
                if not (0 <= rank < len(self.scheduler_procs)):
                    success = False
                    message = "unknown rank"
                else:
                    proc = self.scheduler_procs[rank]
                    replacement_pid = self.scheduler_process_registry.replacement_pid(
                        rank, proc
                    )
                    dp_rank = self._dp_rank_for_scheduler_index(rank)
                    if dp_rank is not None:
                        self.status[dp_rank] = False
                    if replacement_pid is not None:
                        logger.info(
                            "DPC shutting down rejoined scheduler for fault tolerance: "
                            "id=%s rank=%s pid=%s",
                            obj.request_id,
                            rank,
                            replacement_pid,
                        )
                        try:
                            psutil.Process(replacement_pid).terminate()
                        except psutil.NoSuchProcess:
                            message = "already stopped"
                    elif proc is not None and proc.is_alive():
                        logger.info(
                            "DPC shutting down scheduler for fault tolerance: "
                            "id=%s rank=%s pid=%s",
                            obj.request_id,
                            rank,
                            proc.pid,
                        )
                        proc.terminate()
                    else:
                        message = "already stopped"
                if self.send_to_tokenizer is not None:
                    self.send_to_tokenizer.send_pyobj(
                        FaultToleranceCommandReqOutput(
                            request_id=obj.request_id,
                            rank=rank,
                            success=success,
                            message=message,
                        )
                    )
            return

        if self._send_ft_command_direct(obj):
            return

        if (
            self.server_args.enable_dp_attention
            and self.control_message_step != 1
            and self._has_dead_scheduler_rank()
        ):
            for rank in obj.target_ranks:
                self._send_ft_command_failure(
                    obj.request_id,
                    rank,
                    "ft_control_broadcast_contains_dead_rank",
                )
            return

        control_ranks = set()
        for target_rank in obj.target_ranks:
            control_rank = self._ft_control_rank_for_target(target_rank)
            if control_rank is None or not self._is_ft_control_rank_reachable(
                control_rank
            ):
                self._send_ft_command_failure(
                    obj.request_id,
                    target_rank,
                    "ft_control_rank_unreachable",
                )
                continue
            control_ranks.add(control_rank)

        # Fallback path for configurations without direct FT control sockets.
        # In DP-attention without local control broadcast, only rank 0 receives
        # from the DPC and then broadcasts control_reqs over the full tp_group,
        # so fail above instead of entering this path when a dead member could
        # make that bootstrap broadcast hang.
        for rank in sorted(control_ranks):
            if 0 <= rank < len(self.workers):
                logger.info(
                    "DPC forwarding FT command: id=%s command=%s control_rank=%s",
                    obj.request_id,
                    obj.command,
                    rank,
                )
                self.workers[rank].send_pyobj(obj)

    def _handle_scheduler_process_exit(self, index, proc, name):
        if self.send_to_tokenizer is None:
            return False
        if self.scheduler_process_registry.should_ignore_process_exit(index, proc):
            logger.info(
                "Ignoring old scheduler process exit after FT rejoin: "
                "rank=%s old_pid=%s replacement_pid=%s",
                index,
                getattr(proc, "pid", None),
                self.scheduler_process_registry.pids[index],
            )
            return True
        dp_rank = self._dp_rank_for_scheduler_index(index)
        if dp_rank is not None:
            route_was_active = self.status[dp_rank]
            self.status[dp_rank] = False
            if not self.server_args.enable_fault_tolerance:
                if route_was_active:
                    self._send_active_ranks_to_live_workers()
                return True
        if not self.scheduler_process_registry.mark_process_exit_reported(index):
            return True
        self.send_to_tokenizer.send_pyobj(
            FaultToleranceRankFaultOutput(
                rank=index,
                fault_type="kill",
                message=f"{name} pid={proc.pid} exitcode={proc.exitcode}",
            )
        )
        return True

    def _send_active_ranks_to_live_workers(self):
        if self.server_args.elastic_ep_backend is None:
            return
        active_ranks = ActiveRanksOutput(
            status=list(self.status),
            rank_space=active_ranks_broadcast_rank_space(
                mask_size=len(self.status),
                scheduler_rank_count=len(self.scheduler_procs),
            ),
        )
        for rank, worker in enumerate(self.workers):
            if rank < len(self.status) and self.status[rank]:
                worker.send_pyobj(active_ranks)

    def handle_load_update_req(self, obj):
        self.dp_budget.update_budget(obj)

    def update_active_ranks(self, ranks: ActiveRanksOutput):
        success = True
        message = "active ranks updated"
        try:
            if not is_dp_route_rank_space(ranks.rank_space):
                raise ValueError(
                    f"DPC active-ranks update expects dp_route rank_space, "
                    f"got {ranks.rank_space}"
                )
            self.status = list(ranks.status)
            self._refresh_worker_liveness()
        except Exception as exc:
            success = False
            message = str(exc)
            logger.exception("Failed to update active DP ranks")
        finally:
            if ranks.request_id is not None and self.send_to_tokenizer is not None:
                self.send_to_tokenizer.send_pyobj(
                    ActiveRanksUpdateReqOutput(
                        request_id=ranks.request_id,
                        success=success,
                        message=message,
                    )
                )

    def handle_ft_rank_rejoin(self, event: FaultToleranceRankRejoinOutput):
        if not (0 <= event.rank < len(self.scheduler_procs)):
            logger.warning(
                "Ignoring FT rejoin registration for unknown rank=%s pid=%s",
                event.rank,
                event.pid,
            )
            return
        old_pid = self.scheduler_process_registry.register_rejoin(
            event.rank, event.pid
        )
        logger.info(
            "Registered FT rejoined scheduler process: rank=%s old_pid=%s "
            "new_pid=%s message=%s",
            event.rank,
            old_pid,
            event.pid,
            event.message,
        )

    def _refresh_worker_liveness(self):
        """Keep DPC routing state aligned with scheduler process liveness."""
        changed = False
        for rank, proc in enumerate(self.scheduler_procs):
            if proc is None:
                continue
            dp_rank = self._dp_rank_for_scheduler_index(rank)
            if dp_rank is None:
                continue
            if self.scheduler_process_registry.is_rank_alive(rank, proc):
                continue
            route_changed = False
            if self.status[dp_rank]:
                self.status[dp_rank] = False
                changed = True
                route_changed = True
            new_fault = (
                self.server_args.enable_fault_tolerance
                and self.send_to_tokenizer is not None
                and self.scheduler_process_registry.mark_process_exit_reported(rank)
            )
            if route_changed or new_fault:
                logger.warning(
                    "Mark DP rank %s inactive because scheduler process rank=%s "
                    "pid=%s exited",
                    dp_rank,
                    rank,
                    proc.pid,
                )
            if new_fault:
                self.send_to_tokenizer.send_pyobj(
                    FaultToleranceRankFaultOutput(
                        rank=rank,
                        fault_type="kill",
                        message=(
                            f"scheduler rank={rank} pid={proc.pid} is not alive"
                        ),
                    )
                )

        if changed and not self.server_args.enable_fault_tolerance:
            self._send_active_ranks_to_live_workers()

    def dispatching_with_trace(self, req: Req):
        self._refresh_worker_liveness()
        req.time_stats = DPControllerReqTimeStats.new_from_obj(req.time_stats)

        req.time_stats.set_dp_dispatch_time()
        self.dispatching(req)
        req.time_stats.set_dp_dispatch_finish_time()

    def dispatch_batch_generate(self, batch_req: BatchTokenizedGenerateReqInput):
        for req in batch_req:
            self.dispatching_with_trace(req)

    def dispatch_batch_embedding(self, batch_req: BatchTokenizedEmbeddingReqInput):
        for req in batch_req:
            self.dispatching_with_trace(req)

    def init_dispatcher(self):
        self._request_dispatcher = TypeBasedDispatcher(
            [
                (TokenizedGenerateReqInput, self.dispatching_with_trace),
                (TokenizedEmbeddingReqInput, self.dispatching_with_trace),
                (BatchTokenizedGenerateReqInput, self.dispatch_batch_generate),
                (BatchTokenizedEmbeddingReqInput, self.dispatch_batch_embedding),
                (BlockReqInput, self.send_to_all_workers),
                (ProfileReq, self.send_to_all_workers),
                (WatchLoadUpdateReq, self.handle_load_update_req),
                (ActiveRanksOutput, self.update_active_ranks),
                (FaultToleranceCommandReqInput, self.send_fault_tolerance_command),
                (FaultToleranceRankRejoinOutput, self.handle_ft_rank_rejoin),
            ]
        )
        self._request_dispatcher.add_fallback_fn(self.send_control_message)

    def launch_dp_schedulers(self, server_args, port_args):
        base_gpu_id = 0

        threads = []
        sockets = []
        ready_events = []
        for dp_rank in range(server_args.dp_size):
            tmp_port_args = PortArgs.init_new(server_args)
            tmp_port_args.tokenizer_ipc_name = port_args.tokenizer_ipc_name
            tmp_port_args.detokenizer_ipc_name = port_args.detokenizer_ipc_name

            # This port is checked free in PortArgs.init_new.
            # We hold it first so that the next dp worker gets a different port
            sockets.append(bind_port(tmp_port_args.nccl_port))

            ready_event = threading.Event()
            ready_events.append(ready_event)

            # Create a thread for each worker
            thread = threading.Thread(
                target=self.launch_tensor_parallel_group_thread,
                args=(server_args, tmp_port_args, base_gpu_id, dp_rank, ready_event),
            )
            threads.append(thread)
            base_gpu_id += (
                server_args.tp_size * server_args.pp_size * server_args.gpu_id_step
            )

            if server_args.node_rank == 0:
                self.workers[dp_rank] = get_zmq_socket(
                    self.context,
                    zmq.PUSH,
                    tmp_port_args.scheduler_input_ipc_name,
                    True,
                )

        # Free all sockets before starting the threads to launch TP workers
        for sock in sockets:
            sock.close()

        # Start all threads
        for thread in threads:
            thread.start()
        for event in ready_events:
            event.wait()

    def launch_tensor_parallel_group_thread(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        base_gpu_id: int,
        dp_rank: int,
        ready_event: threading.Event,
    ):
        self.launch_tensor_parallel_group(server_args, port_args, base_gpu_id, dp_rank)
        ready_event.set()

        # This thread cannot be closed because otherwise the `kill_itself_when_parent_died`
        # function in scheduler.py will kill the scheduler.
        while True:
            time.sleep(30 * 24 * 3600)

    def _broadcast_worker_ports(
        self, server_args: ServerArgs, worker_ports: Optional[List[int]] = None
    ) -> List[int]:
        """Broadcast worker ports from node 0 to all other nodes.

        Node 0 acts as the server, waiting for all other nodes to connect and
        sending them the pre-allocated worker ports. Other nodes act as clients,
        connecting to node 0 to receive their copy of the worker ports.

        Args:
            server_args: Server arguments containing node configuration.
            worker_ports: Pre-allocated worker ports to broadcast.

        Returns:
            List of worker ports (same on all nodes after broadcast).
        """
        # Determine the endpoint for inter-node communication
        if server_args.dist_init_addr is None:
            na = NetworkAddress(
                server_args.host or "127.0.0.1",
                server_args.port + DP_ATTENTION_HANDSHAKE_PORT_DELTA,
            )
        else:
            na = NetworkAddress.parse(server_args.dist_init_addr)
            na = NetworkAddress(na.host, na.port + DP_ATTENTION_HANDSHAKE_PORT_DELTA)
        endpoint = na.to_tcp()

        if server_args.node_rank == 0:
            # Node 0: Broadcast worker ports to all other nodes
            return self._broadcast_ports_as_server(
                endpoint, server_args.nnodes - 1, worker_ports
            )
        else:
            # Other nodes: Receive worker ports from node 0
            return self._receive_ports_as_client(endpoint, server_args.node_rank)

    def _broadcast_ports_as_server(
        self, endpoint: str, expected_clients: int, worker_ports: List[int]
    ) -> List[int]:
        """Broadcast worker ports to all client nodes."""
        logger.debug(f"Broadcasting worker ports to {expected_clients} client nodes")
        logger.debug(f"Worker ports: {worker_ports}")

        rep_socket = get_zmq_socket(self.context, zmq.REP, endpoint, True)

        try:
            connected_clients = 0
            while connected_clients < expected_clients:
                # Wait for client handshake
                client_rank = rep_socket.recv().decode()
                logger.debug(f"Received handshake from node {client_rank}")

                # Send worker ports to client
                rep_socket.send_pyobj(worker_ports)
                connected_clients += 1
                logger.debug(
                    f"Sent worker ports to {connected_clients}/{expected_clients} nodes"
                )

            logger.debug("Worker port broadcast completed")
            return worker_ports
        finally:
            if self.server_args.elastic_ep_backend is None:
                rep_socket.close()
            else:
                threading.Thread(
                    target=self._reply_ports_as_server,
                    args=(rep_socket, worker_ports),
                    daemon=True,
                ).start()

    def _reply_ports_as_server(self, rep_socket: zmq.Socket, worker_ports: List[int]):
        """
        Runs as a background thread to broadcast worker ports for recovered EP ranks
        """
        while True:
            # Wait for client handshake
            try:
                client_rank = rep_socket.recv().decode()
            except Exception:
                logger.exception(
                    "Failed to recv/decode handshake in reply thread; continue"
                )
                continue
            logger.debug(f"Received handshake from node {client_rank}")

            # Send worker ports to client
            rep_socket.send_pyobj(worker_ports)
            logger.debug(f"Sent worker ports to node {client_rank}")

    def _receive_ports_as_client(self, endpoint: str, node_rank: int) -> List[int]:
        """Receive worker ports from the server node."""
        logger.debug(f"Connecting to node 0 to receive worker ports")

        req_socket = get_zmq_socket(self.context, zmq.REQ, endpoint, False)
        req_socket.setsockopt(zmq.RCVTIMEO, 600 * 1000)  # 10 minute timeout
        req_socket.setsockopt(zmq.SNDTIMEO, 600 * 1000)

        try:
            # Send handshake with our node rank
            req_socket.send(str(node_rank).encode())

            # Receive worker ports
            worker_ports = req_socket.recv_pyobj()
            logger.debug(f"Received {len(worker_ports)} worker ports from node 0")
            return worker_ports
        except zmq.Again:
            logger.error("Timeout waiting for worker ports from node 0")
            raise RuntimeError(
                "Failed to receive worker ports from node 0 within timeout"
            )
        finally:
            req_socket.close()

    def launch_dp_attention_schedulers(
        self, server_args: ServerArgs, port_args: PortArgs
    ):
        if server_args.dist_init_addr is None:
            bind_host = "127.0.0.1"
        else:
            bind_host = NetworkAddress.parse(server_args.dist_init_addr).host

        # Pre-allocate worker ports on node 0 to avoid conflicts
        worker_ports = []
        if server_args.node_rank == 0:
            for dp_rank in range(server_args.dp_size):
                worker_port, worker_socket = get_zmq_socket_on_host(
                    self.context, zmq.PUSH, host=bind_host
                )
                worker_ports.append(worker_port)
                self.workers[dp_rank] = worker_socket
                logger.debug(
                    "Assigned port %s to worker %s on host %s",
                    worker_port,
                    dp_rank,
                    bind_host,
                )

        broadcasted_ports = self._broadcast_worker_ports(
            server_args, worker_ports if worker_ports else None
        )
        self.launch_tensor_parallel_group(
            server_args, port_args, 0, None, broadcasted_ports
        )

    def launch_tensor_parallel_group(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        base_gpu_id: int,
        dp_rank: Optional[int],
        worker_ports: Optional[List[int]] = None,
    ):
        if not server_args.enable_dp_attention:
            logger.info(f"Launch DP{dp_rank} starting at GPU #{base_gpu_id}.")

        memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=server_args.enable_memory_saver
        )

        scheduler_pipe_readers = []

        pp_size_per_node = max(server_args.pp_size // server_args.nnodes, 1)
        nnodes_per_pp_rank = max(server_args.nnodes // server_args.pp_size, 1)
        pp_rank_range = range(
            pp_size_per_node * (server_args.node_rank // nnodes_per_pp_rank),
            pp_size_per_node * (server_args.node_rank // nnodes_per_pp_rank + 1),
        )

        nnodes_per_tp_group = nnodes_per_pp_rank
        tp_size_per_node = server_args.tp_size // nnodes_per_tp_group
        tp_rank_range = range(
            tp_size_per_node * (server_args.node_rank % nnodes_per_tp_group),
            tp_size_per_node * (server_args.node_rank % nnodes_per_tp_group + 1),
        )

        attn_cp_rank = 0
        moe_dp_rank = 0
        for pp_rank in pp_rank_range:
            for tp_rank in tp_rank_range:
                rank_port_args = port_args

                if server_args.enable_dp_attention:
                    # dp attention has different sharding logic
                    _, _, dp_rank = compute_dp_attention_world_info(
                        server_args.enable_dp_attention,
                        tp_rank,
                        server_args.tp_size,
                        server_args.dp_size,
                        server_args.attn_cp_size,
                    )
                    # compute zmq ports for this dp rank
                    rank_port_args = PortArgs.init_new(
                        server_args, dp_rank, worker_ports
                    )
                    # Data parallelism reuses the tensor parallelism group,
                    # so all dp ranks should use the same nccl port.
                    rank_port_args.nccl_port = port_args.nccl_port
                    rank_port_args.ft_control_ipc_names = (
                        port_args.ft_control_ipc_names
                    )

                reader, writer = mp.Pipe(duplex=False)
                gpu_id = (
                    server_args.base_gpu_id
                    + base_gpu_id
                    + ((pp_rank % pp_size_per_node) * tp_size_per_node)
                    + (tp_rank % tp_size_per_node) * server_args.gpu_id_step
                )
                attn_dp_size = (
                    server_args.dp_size if server_args.enable_dp_attention else 1
                )

                # Parallelism hierarchy (outermost to innermost):
                # - Attention: Global(TP) -> DP -> ATTN_CP -> ATTN_TP (innermost)
                # - MoE: Global(TP) -> MOE_DP -> EP -> MOE_TP (innermost)
                attn_tp_size = (
                    server_args.tp_size // attn_dp_size // server_args.attn_cp_size
                )
                attn_cp_rank = (tp_rank // attn_tp_size) % server_args.attn_cp_size
                moe_dp_rank = tp_rank // (
                    server_args.tp_size // server_args.moe_dp_size
                )
                moe_ep_rank = (
                    tp_rank
                    % (server_args.tp_size // server_args.moe_dp_size)
                    // (
                        server_args.tp_size
                        // server_args.moe_dp_size
                        // server_args.ep_size
                    )
                )

                with self.env_lock, maybe_reindex_device_id(gpu_id) as gpu_id:
                    proc = mp.Process(
                        target=self.run_scheduler_process_func,
                        args=(
                            server_args,
                            rank_port_args,
                            gpu_id,
                            tp_rank,
                            attn_cp_rank,
                            moe_dp_rank,
                            moe_ep_rank,
                            pp_rank,
                            dp_rank,
                            writer,
                        ),
                    )
                    with (
                        memory_saver_adapter.configure_subprocess(),
                        numa_utils.configure_subprocess(server_args, gpu_id),
                    ):
                        proc.start()
                self.scheduler_procs.append(proc)
                self.scheduler_process_registry.append_process(proc)
                scheduler_pipe_readers.append(reader)

        # Wait for model to finish loading
        scheduler_info = []
        for i in range(len(scheduler_pipe_readers)):
            scheduler_info.append(scheduler_pipe_readers[i].recv())

        self.max_total_num_tokens = scheduler_info[0]["max_total_num_tokens"]
        self.max_req_input_len = scheduler_info[0]["max_req_input_len"]

    def maybe_external_dp_rank_routing(self, req: Req):
        if req.routed_dp_rank is not None:
            if not self.status[req.routed_dp_rank]:
                self._reject_req(
                    req, f"routed_dp_rank={req.routed_dp_rank} is inactive"
                )
                return True
            logger.debug(f"Direct routing to DP rank {req.routed_dp_rank}")
            self.workers[req.routed_dp_rank].send_pyobj(req)
            return True
        return False

    def _ensure_active_rank_available(self):
        return any(self.status)

    def _reject_req(self, req: Req, message: str):
        logger.warning("Rejecting DP request %s: %s", getattr(req, "rid", ""), message)
        if self.send_to_tokenizer is not None:
            self.send_to_tokenizer.send_pyobj(
                AbortReq(rid=req.rid, abort_message=message)
            )

    def _next_active_rank(self, preferred_rank: Optional[int] = None) -> int:
        if not self._ensure_active_rank_available():
            raise RuntimeError("No active DP rank available")
        if preferred_rank is not None and self.status[preferred_rank]:
            return preferred_rank
        for offset in range(len(self.workers)):
            rank = (self.round_robin_counter + offset) % len(self.workers)
            if self.status[rank]:
                self.round_robin_counter = (rank + 1) % len(self.workers)
                return rank
        raise RuntimeError("No active DP rank available")

    def round_robin_scheduler(self, req: Req):
        if self.maybe_external_dp_rank_routing(req):
            return
        if not self._ensure_active_rank_available():
            self._reject_req(req, "no active DP rank")
            return

        target_rank = self._next_active_rank()
        logger.debug(f"Choose worker {target_rank}")
        self.workers[target_rank].send_pyobj(req)

    def follow_bootstrap_room_scheduler(self, req: Req):
        if self.maybe_external_dp_rank_routing(req):
            return
        if not self._ensure_active_rank_available():
            self._reject_req(req, "no active DP rank")
            return

        assert req.bootstrap_room is not None, (
            "req.bootstrap_room should not be None. Do not send requests directly to "
            "prefill or decode instances; send to the router instead."
        )
        target_rank = self._next_active_rank(req.bootstrap_room % len(self.workers))
        self.workers[target_rank].send_pyobj(req)

    def total_requests_scheduler(self, req: Req):
        if self.maybe_external_dp_rank_routing(req):
            return
        if not self._ensure_active_rank_available():
            self._reject_req(req, "no active DP rank")
            return
        target_worker = self.dp_budget.dispatch(LoadBalanceMethod.TOTAL_REQUESTS)
        if not self.status[target_worker]:
            target_worker = self._next_active_rank()
        self.workers[target_worker].send_pyobj(req)

    def total_tokens_scheduler(self, req: Req):
        if self.maybe_external_dp_rank_routing(req):
            return
        if not self._ensure_active_rank_available():
            self._reject_req(req, "no active DP rank")
            return
        estimated_tokens = len(req.input_ids)
        target_worker = self.dp_budget.dispatch(
            LoadBalanceMethod.TOTAL_TOKENS, estimated_tokens=estimated_tokens
        )
        if not self.status[target_worker]:
            target_worker = self._next_active_rank()
        self.workers[target_worker].send_pyobj(req)

    def event_loop(self):
        while True:
            while True:
                self.soft_watchdog.feed()
                try:
                    recv_req = self.recv_from_tokenizer.recv_pyobj(zmq.NOBLOCK)
                except zmq.ZMQError:
                    break
                self._request_dispatcher(recv_req)


def run_data_parallel_controller_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    pipe_writer,
    run_scheduler_process_func: Callable = run_scheduler_process,
):
    setproctitle.setproctitle("sglang::data_parallel_controller")
    faulthandler.enable()
    kill_itself_when_parent_died()
    parent_process = psutil.Process().parent()

    configure_logger(server_args)
    if server_args.enable_trace:
        process_tracing_init(server_args.otlp_traces_endpoint, "sglang")
        thread_label = "DP Controller"
        if server_args.disaggregation_mode == "prefill":
            thread_label = "Prefill DP Controller"
        elif server_args.disaggregation_mode == "decode":
            thread_label = "Decode DP Controller"
        trace_set_thread_info(thread_label)

    try:
        controller = DataParallelController(
            server_args, port_args, run_scheduler_process_func
        )
        scheduler_pids = [
            proc.pid for proc in controller.scheduler_procs if proc is not None
        ]
        pipe_writer.send(
            {
                "status": "ready",
                "max_total_num_tokens": controller.max_total_num_tokens,
                "max_req_input_len": controller.max_req_input_len,
                SCHEDULER_PIDS_ARG: scheduler_pids,
            }
        )
        if server_args.node_rank == 0:
            controller.event_loop()
        for proc in controller.scheduler_procs:
            proc.join()
            logger.error(
                f"Scheduler or DataParallelController {proc.pid} terminated with {proc.exitcode}"
            )
    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"DataParallelController hit an exception: {traceback}")
        parent_process.send_signal(signal.SIGQUIT)
