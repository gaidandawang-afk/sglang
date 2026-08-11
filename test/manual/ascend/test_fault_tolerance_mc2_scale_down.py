"""Manual Ascend MC2 scale-down validation.

Run this only against a disposable four-rank server. The script deliberately
sends SIGKILL to the Scheduler PID supplied with --victim-pid unless an
existing incident is selected instead.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import time
from pathlib import Path
from typing import Any

import requests


MC2_LOG_PATTERN = re.compile(
    r"\[NPU FT\].*MC2.*rank=(?P<rank>\d+).*"
    r"data_ptr=(?P<data_ptr>\d+).*values=\[(?P<values>[^]]*)\]"
)
ORIGINAL_GLOO_PREWARM_LOG_PATTERN = re.compile(
    r"\[NPU FT\] prewarmed original graph-external MLP-sync Gloo group: "
    r"original_rank=(?P<rank>\d+)"
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


def _http_json(
    session: requests.Session,
    method: str,
    url: str,
    *,
    timeout: float,
    payload: dict[str, Any] | None = None,
) -> Any:
    response = session.request(method, url, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _rank_states(status: dict[str, Any]) -> dict[int, str]:
    return {int(item["rank"]): str(item["state"]) for item in status["ranks"]}


def _validate_victim_pid_rank(victim_pid: int, victim_rank: int) -> None:
    """Reject a Linux Scheduler PID that belongs to a different DP rank."""

    cmdline_path = Path(f"/proc/{victim_pid}/cmdline")
    if not cmdline_path.exists():
        return
    try:
        process_title = (
            cmdline_path.read_bytes()
            .replace(b"\0", b" ")
            .decode(errors="replace")
            .strip()
        )
    except OSError:
        # Some hardened /proc mounts hide command lines. The server-side DPC
        # target logs remain the fallback evidence in that environment.
        return

    match = SCHEDULER_PROCESS_TITLE_PATTERN.search(process_title)
    if match is None:
        return
    pid_dp_rank = int(match.group("dp_rank"))
    if pid_dp_rank != victim_rank:
        raise ValueError(
            f"--victim-pid {victim_pid} belongs to DP rank {pid_dp_rank}, "
            f"but --victim-rank is {victim_rank}; refusing to inject two "
            "different rank failures"
        )


def _wait_for_incident(
    session: requests.Session,
    base_url: str,
    *,
    victim_rank: int,
    deadline: float,
) -> dict[str, Any]:
    last_status: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        last_status = _http_json(
            session,
            "GET",
            f"{base_url}/fault_tolerance/status",
            timeout=5,
        )
        if _rank_states(last_status).get(victim_rank) in {"dead", "unhealthy"}:
            return last_status
        time.sleep(0.5)
    raise TimeoutError(
        f"rank {victim_rank} did not become dead/unhealthy; last={last_status}"
    )


def _generate(
    session: requests.Session,
    base_url: str,
    *,
    prompt: str,
    max_new_tokens: int,
    timeout: float,
) -> Any:
    return _http_json(
        session,
        "POST",
        f"{base_url}/generate",
        timeout=timeout,
        payload={
            "text": prompt,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": max_new_tokens,
            },
        },
    )


def _response_text(response: Any) -> Any:
    if isinstance(response, dict):
        return response.get("text")
    if isinstance(response, list):
        return [_response_text(item) for item in response]
    return None


def _parse_mc2_log(path: Path, original_world_size: int, victim_rank: int) -> dict:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    prewarmed_original_ranks = sorted(
        {
            int(match.group("rank"))
            for line in lines
            if (match := ORIGINAL_GLOO_PREWARM_LOG_PATTERN.search(line))
            is not None
        }
    )
    expected_original_ranks = list(range(original_world_size))
    if prewarmed_original_ranks != expected_original_ranks:
        raise AssertionError(
            "missing original MLP-sync Gloo prewarm logs: "
            f"got={prewarmed_original_ranks} expected={expected_original_ranks}"
        )

    matches = []
    for line_index, line in enumerate(lines):
        match = MC2_LOG_PATTERN.search(line)
        if match is None:
            continue
        values = [
            int(item.strip())
            for item in match.group("values").split(",")
            if item.strip()
        ]
        matches.append(
            {
                "rank": int(match.group("rank")),
                "data_ptr": int(match.group("data_ptr")),
                "values": values,
                "line_index": line_index,
            }
        )

    if len(matches) < 2:
        raise AssertionError(
            "server log must contain MC2 elastic_info initialization and update"
        )
    initial_by_rank = {
        item["rank"]: item for item in matches if item["values"][0] == 0
    }
    updates = [item for item in matches if item["values"][0] == 1]
    if not updates:
        raise AssertionError("server log has no committed scale-down elastic_info")
    for update in updates:
        initial = initial_by_rank.get(update["rank"])
        if initial is None:
            raise AssertionError(
                f"rank {update['rank']} has no elastic_info initialization log"
            )
        if initial["data_ptr"] != update["data_ptr"]:
            raise AssertionError(
                "MC2 elastic_info address changed across graph replay on "
                f"rank {update['rank']}: "
                f"{initial['data_ptr']} != {update['data_ptr']}"
            )

    updated = updates[-1]
    values = updated["values"]
    expected_size = 4 + 2 * original_world_size
    if len(values) != expected_size:
        raise AssertionError(
            f"elastic_info length {len(values)} != expected {expected_size}"
        )
    if values[1] != original_world_size - 1:
        raise AssertionError(f"unexpected effective EP size: {values[1]}")
    original_to_effective = values[4 : 4 + original_world_size]
    effective_to_original = values[4 + original_world_size :]
    if original_to_effective[victim_rank] != -1:
        raise AssertionError("victim rank was not masked in elastic_info")
    expected_survivors = [
        rank for rank in range(original_world_size) if rank != victim_rank
    ]
    if effective_to_original[: original_world_size - 1] != expected_survivors:
        raise AssertionError(
            "effective-to-original mapping mismatch: "
            f"{effective_to_original} vs {expected_survivors}"
        )
    if effective_to_original[-1] != -1:
        raise AssertionError("fixed-width reverse mapping is not padded with -1")

    process_group_updates = []
    for line_index, line in enumerate(lines):
        match = PROCESS_GROUP_LOG_PATTERN.search(line)
        if match is None:
            continue
        process_group_updates.append(
            {
                "generation": int(match.group("generation")),
                "rank": int(match.group("rank")),
                "compact_rank": int(match.group("compact_rank")),
                "active_ranks": [
                    int(item.strip())
                    for item in match.group("active_ranks").split(",")
                    if item.strip()
                ],
                "line_index": line_index,
            }
        )
    expected_survivors = [
        rank for rank in range(original_world_size) if rank != victim_rank
    ]
    latest_process_group_by_rank = {
        item["rank"]: item for item in process_group_updates
    }
    if sorted(latest_process_group_by_rank) != expected_survivors:
        raise AssertionError(
            "missing graph-external process-group rebuild logs: "
            f"got={sorted(latest_process_group_by_rank)} "
            f"expected={expected_survivors}"
        )
    for compact_rank, original_rank in enumerate(expected_survivors):
        item = latest_process_group_by_rank[original_rank]
        if item["active_ranks"] != expected_survivors:
            raise AssertionError(
                f"rank {original_rank} rebuilt with wrong membership: {item}"
            )
        if item["compact_rank"] != compact_rank:
            raise AssertionError(
                f"rank {original_rank} has wrong compact rank: {item}"
            )

    device_stops = {}
    device_restarts = {}
    for line_index, line in enumerate(lines):
        stop_match = DEVICE_STOP_LOG_PATTERN.search(line)
        if stop_match is not None:
            device_stops[int(stop_match.group("rank"))] = {
                "device_id": int(stop_match.group("device_id")),
                "line_index": line_index,
            }
        restart_match = DEVICE_RESTART_LOG_PATTERN.search(line)
        if restart_match is not None:
            device_restarts[int(restart_match.group("rank"))] = {
                "device_id": int(restart_match.group("device_id")),
                "line_index": line_index,
            }

    if sorted(device_stops) != expected_survivors:
        raise AssertionError(
            "missing survivor stop_device logs: "
            f"got={sorted(device_stops)} expected={expected_survivors}"
        )
    if sorted(device_restarts) != expected_survivors:
        raise AssertionError(
            "missing survivor restart_device logs: "
            f"got={sorted(device_restarts)} expected={expected_survivors}"
        )
    latest_update_by_rank = {item["rank"]: item for item in updates}
    for original_rank in expected_survivors:
        stop = device_stops[original_rank]
        restart = device_restarts[original_rank]
        process_group = latest_process_group_by_rank[original_rank]
        elastic_update = latest_update_by_rank[original_rank]
        if stop["device_id"] != restart["device_id"]:
            raise AssertionError(
                f"rank {original_rank} stopped and restarted different devices"
            )
        if not (
            stop["line_index"]
            < restart["line_index"]
            < process_group["line_index"]
            < elastic_update["line_index"]
        ):
            raise AssertionError(
                "wrong NPU FT recovery ordering for rank "
                f"{original_rank}: stop={stop}, restart={restart}, "
                f"process_group={process_group}, elastic={elastic_update}"
            )

    return {
        "prewarmed_original_ranks": prewarmed_original_ranks,
        "initial_by_rank": initial_by_rank,
        "updates": updates,
        "process_group_updates": process_group_updates,
        "device_stops": device_stops,
        "device_restarts": device_restarts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--victim-rank", type=int, default=1)
    fault = parser.add_mutually_exclusive_group(required=True)
    fault.add_argument(
        "--victim-pid",
        type=int,
        help="Scheduler PID for --victim-rank to terminate with SIGKILL",
    )
    fault.add_argument(
        "--wait-for-existing-incident",
        action="store_true",
        help="do not kill a process; wait for an externally injected incident",
    )
    parser.add_argument("--original-world-size", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--post-requests", type=int, default=3)
    parser.add_argument(
        "--prompt",
        default="Write one short sentence explaining fault tolerance.",
    )
    parser.add_argument(
        "--server-log",
        type=Path,
        help="server log used to verify fixed elastic_info address and mapping",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("/tmp/sglang_npu_mc2_scale_down_report.json"),
    )
    args = parser.parse_args()

    if not 0 <= args.victim_rank < args.original_world_size:
        parser.error("--victim-rank must be in the original rank namespace")
    if args.post_requests <= 0:
        parser.error("--post-requests must be positive")

    session = requests.Session()
    base_url = args.base_url.rstrip("/")
    initial_status = _http_json(
        session,
        "GET",
        f"{base_url}/fault_tolerance/status",
        timeout=5,
    )
    if _rank_states(initial_status) != {
        rank: "healthy" for rank in range(args.original_world_size)
    }:
        raise AssertionError(f"server is not initially healthy: {initial_status}")

    baseline = _generate(
        session,
        base_url,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        timeout=args.timeout,
    )

    if args.victim_pid is not None:
        _validate_victim_pid_rank(args.victim_pid, args.victim_rank)
        os.kill(args.victim_pid, signal.SIGKILL)

    incident = _wait_for_incident(
        session,
        base_url,
        victim_rank=args.victim_rank,
        deadline=time.monotonic() + args.timeout,
    )
    scale_down = _http_json(
        session,
        "POST",
        f"{base_url}/fault_tolerance/apply",
        timeout=args.timeout,
        payload={
            "instruction": "scale_down",
            "params": {
                "ranks": [args.victim_rank],
                "timeout": int(args.timeout),
            },
        },
    )
    if not scale_down.get("success"):
        raise AssertionError(f"scale-down was not committed: {scale_down}")

    final_status = _http_json(
        session,
        "GET",
        f"{base_url}/fault_tolerance/status",
        timeout=5,
    )
    final_states = _rank_states(final_status)
    expected_final = {
        rank: ("dead" if rank == args.victim_rank else "healthy")
        for rank in range(args.original_world_size)
    }
    if final_states != expected_final:
        raise AssertionError(
            f"unexpected final FT state: {final_states} != {expected_final}"
        )

    post_responses = [
        _generate(
            session,
            base_url,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            timeout=args.timeout,
        )
        for _ in range(args.post_requests)
    ]
    baseline_text = _response_text(baseline)
    post_texts = [_response_text(item) for item in post_responses]
    if baseline_text is None or any(text is None for text in post_texts):
        raise AssertionError("generation response did not contain text")
    if any(text != baseline_text for text in post_texts):
        raise AssertionError(
            "deterministic generation changed after scale-down: "
            f"baseline={baseline_text!r} post={post_texts!r}"
        )

    mc2_log_check = None
    if args.server_log is not None:
        mc2_log_check = _parse_mc2_log(
            args.server_log,
            args.original_world_size,
            args.victim_rank,
        )

    report = {
        "initial_status": initial_status,
        "baseline": baseline,
        "incident": incident,
        "scale_down": scale_down,
        "final_status": final_status,
        "post_responses": post_responses,
        "mc2_log_check": mc2_log_check,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"PASS: NPU MC2 scale-down report written to {args.report}")


if __name__ == "__main__":
    main()
