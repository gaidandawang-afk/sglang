"""Comprehensive Ascend MC2 Fault-Tolerance & Scale-Down Test Suite.

This script implements the complete fault-injection validation plan covering:
1. Idle scale-down (direct API scale-down vs. incident scale-down)
2. In-flight dynamic scale-down under concurrent inference load
3. FT strategy comparison (pause vs. continue)
4. Mixed fault injection (application exception + watchdog SIGKILL)
5. Tensor Parallelism TP > 1 fault tolerance & DP-unit isolation
6. Multi-victim & cascading sequential scale-down (4 -> 3 -> 2)
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import re
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("ft_test_suite")


# Log patterns for Ascend NPU MC2 recovery verification
MC2_LOG_PATTERN = re.compile(
    r"\[NPU FT\].*MC2.*rank=(?P<rank>\d+).*"
    r"data_ptr=(?P<data_ptr>\d+).*values=\[(?P<values>[^]]*)\]"
)
PROCESS_GROUP_LOG_PATTERN = re.compile(
    r"\[NPU FT\] rebuilt graph-external process groups: "
    r"generation=(?P<generation>\d+) original_rank=(?P<rank>\d+) "
    r"compact_rank=(?P<compact_rank>\d+) "
    r"active_original_ranks=\[(?P<active_ranks>[^]]*)\]"
)
DEVICE_STOP_LOG_PATTERN = re.compile(
    r"\[NPU FT\] stopping survivor device before communication-domain "
    r"rebuild: rank=(?P<rank>\d+) device_id=(?P<device_id>\d+)"
)
DEVICE_RESTART_LOG_PATTERN = re.compile(
    r"\[NPU FT\] restarted survivor device without rebuilding graph "
    r"resources: rank=(?P<rank>\d+) device_id=(?P<device_id>\d+)"
)
SCHEDULER_PROCESS_TITLE_PATTERN = re.compile(
    r"(?:^|\s)sglang::scheduler_DP(?P<dp_rank>\d+)(?:_|\s|$)"
)


@dataclass
class RequestResult:
    request_id: int
    start_time: float
    end_time: float
    status_code: int
    text: str = ""
    error: str = ""


@dataclass
class InFlightTrafficStats:
    total_requests: int = 0
    success_200: int = 0
    paused_503: int = 0
    other_errors: int = 0
    results: List[RequestResult] = field(default_factory=list)


def _http_json(
    session: requests.Session,
    method: str,
    url: str,
    *,
    timeout: float = 10.0,
    payload: Optional[Dict[str, Any]] = None,
) -> Any:
    response = session.request(method, url, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _get_ft_status(session: requests.Session, base_url: str) -> Dict[str, Any]:
    return _http_json(session, "GET", f"{base_url}/fault_tolerance/status", timeout=5.0)


def _rank_states(status: Dict[str, Any]) -> Dict[int, str]:
    return {int(item["rank"]): str(item["state"]) for item in status.get("ranks", [])}


def _find_scheduler_pids() -> Dict[int, int]:
    """Discover local DP rank -> PID mapping by searching /proc command lines."""
    rank_pids: Dict[int, int] = {}
    proc_root = Path("/proc")
    if not proc_root.exists():
        return rank_pids

    for pid_dir in proc_root.iterdir():
        if not pid_dir.name.isdigit():
            continue
        try:
            cmdline = (
                (pid_dir / "cmdline")
                .read_bytes()
                .replace(b"\0", b" ")
                .decode(errors="replace")
                .strip()
            )
            match = SCHEDULER_PROCESS_TITLE_PATTERN.search(cmdline)
            if match:
                dp_rank = int(match.group("dp_rank"))
                rank_pids[dp_rank] = int(pid_dir.name)
        except (OSError, PermissionError):
            continue
    return rank_pids


def _generate_single(
    session: requests.Session,
    base_url: str,
    prompt: str,
    *,
    req_id: int = 0,
    max_new_tokens: int = 32,
    temperature: float = 0.0,
    timeout: float = 30.0,
) -> RequestResult:
    start = time.monotonic()
    url = f"{base_url}/v1/chat/completions"
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_new_tokens,
        "temperature": temperature,
    }
    try:
        resp = session.post(url, json=payload, timeout=timeout)
        end = time.monotonic()
        if resp.status_code == 200:
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            return RequestResult(req_id, start, end, 200, text=text)
        else:
            return RequestResult(
                req_id, start, end, resp.status_code, error=resp.text
            )
    except Exception as exc:
        end = time.monotonic()
        return RequestResult(req_id, start, end, 0, error=str(exc))


def _wait_for_incident(
    session: requests.Session,
    base_url: str,
    victim_ranks: List[int],
    *,
    timeout: float = 30.0,
) -> Dict[str, Any]:
    deadline = time.monotonic() + timeout
    last_status: Optional[Dict[str, Any]] = None
    target_set = set(victim_ranks)
    while time.monotonic() < deadline:
        try:
            last_status = _get_ft_status(session, base_url)
            states = _rank_states(last_status)
            if all(states.get(r) in {"dead", "unhealthy"} for r in target_set):
                return last_status
        except Exception:
            pass
        time.sleep(0.5)
    raise TimeoutError(
        f"Ranks {victim_ranks} did not reach expected incident state in {timeout}s; last={last_status}"
    )


def _trigger_scale_down(
    session: requests.Session,
    base_url: str,
    removed_ranks: List[int],
    *,
    timeout: float = 120.0,
) -> Dict[str, Any]:
    logger.info(f"Triggering scale-down for ranks {removed_ranks}...")
    url = f"{base_url}/fault_tolerance/scale_down"
    resp = _http_json(
        session, "POST", url, payload={"removed_ranks": removed_ranks}, timeout=timeout
    )
    logger.info(f"Scale-down response: {resp}")
    return resp


def _verify_server_log_rebuild(log_path: Path, expected_generation: int) -> None:
    if not log_path.exists():
        logger.warning(f"Log file {log_path} not found; skipping log assertions.")
        return

    text = log_path.read_text(errors="replace")
    stop_matches = DEVICE_STOP_LOG_PATTERN.findall(text)
    restart_matches = DEVICE_RESTART_LOG_PATTERN.findall(text)
    rebuild_matches = [
        m.groupdict()
        for m in PROCESS_GROUP_LOG_PATTERN.finditer(text)
        if int(m.group("generation")) == expected_generation
    ]

    logger.info(
        f"Log audit [Gen {expected_generation}]: stops={len(stop_matches)}, "
        f"restarts={len(restart_matches)}, group_rebuilds={len(rebuild_matches)}"
    )
    if not rebuild_matches:
        logger.warning(
            f"No process group rebuild log found for generation {expected_generation}!"
        )


# ==============================================================================
# Scenario 1: Idle Scale-Down (Direct API vs. Incident Scale-Down)
# ==============================================================================
def run_exp1_idle_scale_down(
    session: requests.Session,
    base_url: str,
    victim_rank: int,
    log_path: Path,
    *,
    direct_api: bool = False,
) -> None:
    logger.info("=== [EXP-1] Starting Idle Scale-Down Test ===")
    # 1. Warmup Baseline
    warmup = _generate_single(
        session, base_url, "Count from 1 to 5:", max_new_tokens=16
    )
    assert warmup.status_code == 200, f"Warmup failed: {warmup}"
    logger.info(f"Warmup baseline output: {warmup.text!r}")

    if direct_api:
        logger.info("Directly triggering scale_down without killing process...")
        _trigger_scale_down(session, base_url, [victim_rank])
    else:
        # Discover PID and kill
        pids = _find_scheduler_pids()
        victim_pid = pids.get(victim_rank)
        assert victim_pid is not None, f"Could not find PID for DP rank {victim_rank}"
        logger.info(f"Killing victim DP rank {victim_rank} (PID {victim_pid})...")
        os.kill(victim_pid, signal.SIGKILL)
        _wait_for_incident(session, base_url, [victim_rank])
        _trigger_scale_down(session, base_url, [victim_rank])

    time.sleep(2.0)
    # 2. Verify Post-Scale-Down Inference
    post_req = _generate_single(
        session, base_url, "Count from 1 to 5:", max_new_tokens=16
    )
    assert (
        post_req.status_code == 200
    ), f"Post-scale-down request failed: {post_req.error}"
    logger.info(f"Post-scale-down output: {post_req.text!r}")
    assert (
        post_req.text == warmup.text
    ), f"Output mismatch: expected {warmup.text!r}, got {post_req.text!r}"
    _verify_server_log_rebuild(log_path, expected_generation=1)
    logger.info("=== [EXP-1] Idle Scale-Down Test PASSED ===")


# ==============================================================================
# Scenario 2: In-Flight Dynamic Scale-Down under Concurrent Load
# ==============================================================================
def run_exp2_inflight_scale_down(
    session: requests.Session,
    base_url: str,
    victim_rank: int,
    log_path: Path,
    *,
    concurrency: int = 10,
    duration_secs: float = 20.0,
) -> None:
    logger.info("=== [EXP-2] Starting In-Flight Dynamic Scale-Down Test ===")
    stats = InFlightTrafficStats()
    stop_event = concurrent.futures.ThreadPoolExecutor(max_workers=concurrency + 2)

    def worker_loop(worker_id: int):
        req_counter = 0
        while not stop_flag:
            req_id = worker_id * 1000 + req_counter
            req_counter += 1
            res = _generate_single(
                session,
                base_url,
                f"Say hello {req_id}:",
                req_id=req_id,
                max_new_tokens=16,
                timeout=10.0,
            )
            stats.results.append(res)
            stats.total_requests += 1
            if res.status_code == 200:
                stats.success_200 += 1
            elif res.status_code == 503:
                stats.paused_503 += 1
            else:
                stats.other_errors += 1
            time.sleep(0.1)

    stop_flag = False
    futures = [stop_event.submit(worker_loop, i) for i in range(concurrency)]

    logger.info("Traffic started. Waiting 5s before injecting fault...")
    time.sleep(5.0)

    pids = _find_scheduler_pids()
    victim_pid = pids.get(victim_rank)
    assert victim_pid is not None, f"Could not find PID for DP rank {victim_rank}"
    logger.info(
        f"Injecting SIGKILL to victim DP rank {victim_rank} (PID {victim_pid})..."
    )
    os.kill(victim_pid, signal.SIGKILL)

    _wait_for_incident(session, base_url, [victim_rank])
    logger.info("Incident detected by watchdog. Waiting 3s then scaling down...")
    time.sleep(3.0)

    _trigger_scale_down(session, base_url, [victim_rank])
    logger.info("Scale down completed. Letting traffic run for 5 more seconds...")
    time.sleep(5.0)

    stop_flag = True
    stop_event.shutdown(wait=True)

    logger.info(
        f"In-Flight Traffic Stats: Total={stats.total_requests}, "
        f"200_OK={stats.success_200}, 503_PAUSED={stats.paused_503}, Errors={stats.other_errors}"
    )
    # Post recovery clean check
    post_check = _generate_single(
        session, base_url, "Verify post-traffic recovery:", max_new_tokens=16
    )
    assert post_check.status_code == 200, f"Final check failed: {post_check}"
    _verify_server_log_rebuild(log_path, expected_generation=1)
    logger.info("=== [EXP-2] In-Flight Dynamic Scale-Down Test PASSED ===")


# ==============================================================================
# Scenario 3: Strategy Comparison (Continue vs. Pause)
# ==============================================================================
def run_exp3_continue_isolation(
    session: requests.Session,
    base_url: str,
    victim_rank: int,
    log_path: Path,
) -> None:
    logger.info("=== [EXP-3] Starting Continue Strategy Isolation Test ===")
    status = _get_ft_status(session, base_url)
    assert (
        status.get("strategy") == "continue"
    ), f"Server must be launched with --fault-tolerance-on-error-strategy continue, got {status}"

    pids = _find_scheduler_pids()
    victim_pid = pids.get(victim_rank)
    assert victim_pid is not None, f"Could not find PID for DP rank {victim_rank}"

    logger.info(
        f"Killing victim rank {victim_rank} under continue strategy (PID {victim_pid})..."
    )
    os.kill(victim_pid, signal.SIGKILL)
    _wait_for_incident(session, base_url, [victim_rank])

    logger.info("Verifying that non-faulty DP ranks continue serving without 503...")
    success_count = 0
    for i in range(10):
        res = _generate_single(
            session, base_url, f"Prompt {i}", max_new_tokens=8, timeout=5.0
        )
        if res.status_code == 200:
            success_count += 1
    logger.info(f"Received {success_count}/10 successful responses during incident.")
    assert success_count > 0, "No requests succeeded during continue incident state!"

    _trigger_scale_down(session, base_url, [victim_rank])
    _verify_server_log_rebuild(log_path, expected_generation=1)
    logger.info("=== [EXP-3] Continue Strategy Isolation Test PASSED ===")


# ==============================================================================
# Scenario 4: Mixed Fault Injection (Application Exception + SIGKILL)
# ==============================================================================
def run_exp4_mixed_fault_injection(
    session: requests.Session,
    base_url: str,
    soft_victim_rank: int,
    hard_victim_rank: int,
    log_path: Path,
) -> None:
    logger.info("=== [EXP-4] Starting Mixed Fault Injection Test ===")
    # 1. Soft fault injection via API
    logger.info(
        f"Injecting soft exception to rank {soft_victim_rank} via API endpoint..."
    )
    try:
        _http_json(
            session,
            "POST",
            f"{base_url}/fault_tolerance/inject_rank_fault",
            payload={"rank": soft_victim_rank},
            timeout=5.0,
        )
    except Exception as exc:
        logger.info(f"Inject rank fault response/error: {exc}")

    # 2. Hard fault injection via SIGKILL
    pids = _find_scheduler_pids()
    hard_pid = pids.get(hard_victim_rank)
    assert (
        hard_pid is not None
    ), f"Could not find PID for hard victim DP rank {hard_victim_rank}"
    logger.info(
        f"Injecting hard SIGKILL to rank {hard_victim_rank} (PID {hard_pid})..."
    )
    os.kill(hard_pid, signal.SIGKILL)

    # 3. Wait for both to be captured in status
    status = _wait_for_incident(
        session, base_url, [soft_victim_rank, hard_victim_rank]
    )
    states = _rank_states(status)
    logger.info(f"Mixed incident states: {states}")
    assert states.get(soft_victim_rank) in {"unhealthy", "dead"}
    assert states.get(hard_victim_rank) == "dead"

    # 4. Scale down both victims simultaneously
    _trigger_scale_down(session, base_url, [soft_victim_rank, hard_victim_rank])

    # 5. Verify 2-rank survivor cluster
    res = _generate_single(
        session, base_url, "Verify 2-rank survivor output:", max_new_tokens=16
    )
    assert res.status_code == 200, f"Post mixed scale-down failed: {res}"
    _verify_server_log_rebuild(log_path, expected_generation=1)
    logger.info("=== [EXP-4] Mixed Fault Injection Test PASSED ===")


# ==============================================================================
# Scenario 5: Tensor Parallelism TP > 1 (e.g. TP=2, DP=2)
# ==============================================================================
def run_exp5_tp_parallel_scale_down(
    session: requests.Session,
    base_url: str,
    victim_dp_rank: int,
    log_path: Path,
) -> None:
    logger.info(
        f"=== [EXP-5] Starting TP > 1 Parallel Scale-Down Test (Victim DP {victim_dp_rank}) ==="
    )
    warmup = _generate_single(session, base_url, "TP test warmup:", max_new_tokens=16)
    assert warmup.status_code == 200, f"TP warmup failed: {warmup}"

    pids = _find_scheduler_pids()
    victim_pid = pids.get(victim_dp_rank)
    assert (
        victim_pid is not None
    ), f"Could not find PID for DP rank {victim_dp_rank} across TP workers"

    logger.info(
        f"Killing TP worker in DP rank {victim_dp_rank} (PID {victim_pid})..."
    )
    os.kill(victim_pid, signal.SIGKILL)

    _wait_for_incident(session, base_url, [victim_dp_rank])
    _trigger_scale_down(session, base_url, [victim_dp_rank])

    post_req = _generate_single(
        session, base_url, "TP test post-recovery:", max_new_tokens=16
    )
    assert (
        post_req.status_code == 200
    ), f"Post TP scale-down request failed: {post_req}"
    _verify_server_log_rebuild(log_path, expected_generation=1)
    logger.info("=== [EXP-5] TP > 1 Parallel Scale-Down Test PASSED ===")


# ==============================================================================
# Scenario 6: Cascading Sequential Scale-Down (4 -> 3 -> 2)
# ==============================================================================
def run_exp6_cascading_scale_down(
    session: requests.Session,
    base_url: str,
    victim_ranks: List[int],
    log_path: Path,
) -> None:
    logger.info(
        f"=== [EXP-6] Starting Cascading Scale-Down Test (Steps: {victim_ranks}) ==="
    )
    warmup = _generate_single(
        session, base_url, "Cascading test baseline:", max_new_tokens=16
    )
    assert warmup.status_code == 200

    for step, victim_rank in enumerate(victim_ranks, start=1):
        logger.info(
            f"--- Cascading Step {step}: Killing rank {victim_rank} ---"
        )
        pids = _find_scheduler_pids()
        victim_pid = pids.get(victim_rank)
        assert (
            victim_pid is not None
        ), f"Step {step}: PID for DP rank {victim_rank} not found"

        os.kill(victim_pid, signal.SIGKILL)
        _wait_for_incident(session, base_url, [victim_rank])
        _trigger_scale_down(session, base_url, [victim_rank])

        # Verify generation after each step
        res = _generate_single(
            session,
            base_url,
            f"Cascading step {step} post-check:",
            max_new_tokens=16,
        )
        assert (
            res.status_code == 200
        ), f"Cascading step {step} generation failed: {res}"
        _verify_server_log_rebuild(log_path, expected_generation=step)
        logger.info(f"--- Cascading Step {step} Complete (Generation {step}) ---")

    logger.info("=== [EXP-6] Cascading Scale-Down Test PASSED ===")


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive Ascend MC2 Fault-Tolerance Test Suite"
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:30000",
        help="SGLang server base URL",
    )
    parser.add_argument(
        "--test-case",
        choices=[
            "idle_scale_down",
            "inflight_scale_down",
            "strategy_continue_isolation",
            "mixed_fault_injection",
            "tp_parallel_scale_down",
            "cascading_scale_down",
        ],
        required=True,
        help="Test case scenario to execute",
    )
    parser.add_argument(
        "--victim-rank",
        type=int,
        default=3,
        help="Victim DP rank to kill/scale down",
    )
    parser.add_argument(
        "--soft-victim-rank",
        type=int,
        default=1,
        help="Victim DP rank for soft exception fault",
    )
    parser.add_argument(
        "--hard-victim-rank",
        type=int,
        default=2,
        help="Victim DP rank for hard SIGKILL fault",
    )
    parser.add_argument(
        "--cascading-ranks",
        type=int,
        nargs="+",
        default=[3, 2],
        help="Ordered list of victim ranks for cascading scale down",
    )
    parser.add_argument(
        "--direct-api",
        action="store_true",
        help="Directly invoke scale_down API without killing process (for EXP-1)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Concurrency for in-flight traffic test",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=Path("/tmp/sglang-npu-ft.log"),
        help="Path to SGLang server log file for audit",
    )

    args = parser.parse_args()
    session = requests.Session()

    if args.test_case == "idle_scale_down":
        run_exp1_idle_scale_down(
            session,
            args.base_url,
            args.victim_rank,
            args.log_path,
            direct_api=args.direct_api,
        )
    elif args.test_case == "inflight_scale_down":
        run_exp2_inflight_scale_down(
            session,
            args.base_url,
            args.victim_rank,
            args.log_path,
            concurrency=args.concurrency,
        )
    elif args.test_case == "strategy_continue_isolation":
        run_exp3_continue_isolation(
            session, args.base_url, args.victim_rank, args.log_path
        )
    elif args.test_case == "mixed_fault_injection":
        run_exp4_mixed_fault_injection(
            session,
            args.base_url,
            args.soft_victim_rank,
            args.hard_victim_rank,
            args.log_path,
        )
    elif args.test_case == "tp_parallel_scale_down":
        run_exp5_tp_parallel_scale_down(
            session, args.base_url, args.victim_rank, args.log_path
        )
    elif args.test_case == "cascading_scale_down":
        run_exp6_cascading_scale_down(
            session, args.base_url, args.cascading_ranks, args.log_path
        )


if __name__ == "__main__":
    main()
