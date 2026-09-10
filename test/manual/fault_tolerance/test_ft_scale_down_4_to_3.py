"""Single-node manual FT test, matching server_tool's GSM8K scale-chain-r3.

Run in the prepared Linux GPU container with the paired Mooncake adaptation.
Set FT_MODEL, FT_GSM8K_DATA and CUDA_VISIBLE_DEVICES (exactly four idle GPUs).
python -m pytest -v -s test/manual/fault_tolerance/test_ft_scale_down_4_to_3.py
"""

import hashlib
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import unittest
import uuid
from pathlib import Path

import psutil
import requests


class TestFaultToleranceScaleDown4To3(unittest.TestCase):
    def setUp(self):
        self.assertEqual(sys.platform, "linux", "Run inside the Linux GPU container")
        self.model = os.environ.get("FT_MODEL") or os.environ.get("MODEL_PATH")
        self.assertTrue(self.model, "Set FT_MODEL to DeepSeek-V2-Lite-Chat")
        self.data = Path(os.environ["FT_GSM8K_DATA"]).resolve()
        self.assertTrue(self.data.is_file(), str(self.data))
        self.assertGreaterEqual(len(self.data.read_text().splitlines()), 261)
        devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        self.assertTrue(len(devices) == len(set(devices)) == 4 and all(devices))
        port = int(os.environ.get("FT_PORT", "23000"))
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", port))
        self.url = f"http://127.0.0.1:{port}"
        self.artifacts = Path(os.environ.get("FT_RESULTS_DIR", "/tmp/ft-scale-down")) / uuid.uuid4().hex
        self.artifacts.mkdir(parents=True)
        self.log_path = self.artifacts / "server.log"
        print(f"Artifacts: {self.artifacts}", flush=True)
        self._save("data", {"path": str(self.data), "sha256": hashlib.sha256(self.data.read_bytes()).hexdigest()})
        env = os.environ.copy()
        # Same single-node transport and precision settings as scale-chain-r3.
        env.update({
            "MC_FORCE_TCP": "1", "MC_INTRANODE_NVLINK": "1",
            "NCCL_IB_DISABLE": "1", "SGLANG_HOST_IP": "127.0.0.1", "HOST_IP": "127.0.0.1",
            "SGLANG_JIT_DEEPGEMM_PRECOMPILE": "0", "SGLANG_OPT_USE_JIT_EP_ACTIVATION": "0",
            "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK": "false",
            "SGLANG_DEEPEP_BF16_DISPATCH": "1", "SGLANG_TRITON_PREFILL_TRUNCATION_ALIGN_SIZE": "128",
        })
        for key in ("MOONCAKE_PROTOCOL", "MOONCAKE_EP_FORCE_FALLBACK"):
            env.pop(key, None)
        args = [
            "--tp-size", "4", "--dp-size", "4", "--ep-size", "4",
            "--enable-dp-attention", "--enable-dp-lm-head", "--moe-dense-tp-size", "1",
            "--dtype", "auto", "--load-format", "auto", "--trust-remote-code",
            "--elastic-ep-backend", "mooncake", "--moe-a2a-backend", "mooncake",
            "--deepep-mode", "low_latency", "--enable-eplb",
            "--eplb-algorithm", "elasticity_aware", "--ep-dispatch-algorithm", "static",
            "--ep-num-redundant-experts", "64", "--moe-runner-backend", "deep_gemm",
            "--attention-backend", "triton", "--sampling-backend", "pytorch",
            "--enable-deterministic-inference", "--disable-overlap-schedule",
            "--disable-custom-all-reduce", "--skip-server-warmup",
            "--enable-fault-tolerance", "--fault-tolerance-on-error-strategy", "pause",
            "--fault-tolerance-pause-timeout", "300", "--fault-tolerance-timeout", "60",
            "--elastic-ep-scale-timeout", "150", "--watchdog-timeout", "120",
            "--cuda-graph-backend-decode", "disabled", "--cuda-graph-backend-prefill", "disabled",
            "--mem-fraction-static", "0.75", "--max-running-requests", "8",
            "--max-total-tokens", "16384", "--chunked-prefill-size", "512", "--context-length", "8192",
        ]
        command = [sys.executable, "-u", "-m", "sglang.launch_server", "--model-path", self.model,
                   "--host", "127.0.0.1", "--port", str(port), *args]
        self._save("launch-command", command)
        log = self.log_path.open("w")
        self.addCleanup(log.close)
        self.server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        self.addCleanup(self._stop_server)
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            self.assertIsNone(self.server.poll(), f"Server exited: {self.log_path}")
            try:
                if requests.get(f"{self.url}/health", timeout=5).status_code == 200:
                    return
            except requests.ConnectionError:
                pass
            time.sleep(1)
        self.fail(f"Server did not start within 600s: {self.log_path}")

    def _stop_server(self):
        # Dedicated session created above; includes descendants after parent exit.
        try:
            os.killpg(self.server.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        self.server.wait(timeout=30)

    def _wait_status(self, expected, request_id=None):
        deadline = time.monotonic() + 90
        last = None
        while time.monotonic() < deadline:
            self.assertIsNone(self.server.poll(), f"Server exited: {self.log_path}")
            response = requests.get(f"{self.url}/v1/fault_tolerance/status", timeout=10)
            response.raise_for_status()
            last = response.json()
            if request_id and last.get("last_ft_request_id") == request_id:
                self.assertNotIn("ft_error", last, last)
            states = {e["id"]: e["status"] for e in last["engines"]}
            if states == expected and (request_id is None or last.get("last_ft_request_id") == request_id):
                return last
            time.sleep(0.5)
        self.fail(f"Status did not converge within 90s: {last}")

    def _save(self, name, value):
        (self.artifacts / f"{name}.json").write_text(json.dumps(value, indent=2))

    def _generate(self, rank=None, logprobs=False):
        body = {"text": "The capital of France is", "sampling_params": {"temperature": 0, "max_new_tokens": 16}}
        if rank is not None:
            body["routed_dp_rank"] = rank
        if logprobs:
            body.update(return_logprob=True, logprob_start_len=0)
        return requests.post(f"{self.url}/generate", json=body, timeout=90)

    def _schedulers(self):
        ranks = {}
        for process in psutil.Process(self.server.pid).children(recursive=True):
            try:
                title = " ".join(process.cmdline())
            except psutil.NoSuchProcess:
                continue
            # TP=DP=EP=4: physical TP rank identifies the corresponding DP rank.
            match = re.search(r"sglang::scheduler.*_TP(\d+)_EP(\d+)(?:_|\b)", title)
            if match:
                rank, ep = map(int, match.groups())
                self.assertEqual(rank, ep, title)
                self.assertNotIn(rank, ranks, title)
                ranks[rank] = process
        return ranks

    def _gsm8k(self, stage):
        offset = self.log_path.stat().st_size
        command = [sys.executable, "-u", "-m", "sglang.test.run_eval",
                   "--base-url", self.url, "--model", self.model, "--eval-name", "gsm8k",
                   "--api", "completion", "--gsm8k-data-path", str(self.data),
                   "--num-examples", "256", "--num-shots", "5", "--num-threads", "50",
                   "--max-tokens", "512", "--temperature", "0"]
        path = self.artifacts / f"{stage}.log"
        with path.open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1800)
        client_log = path.read_text()
        for source in re.findall(r"Writing (?:report|results) to (.+)", client_log):
            source = Path(source.strip())
            shutil.copyfile(source, self.artifacts / f"{stage}{source.suffix}")
        # Evaluator retries/errors must not be mistaken for ordinary wrong answers.
        with self.log_path.open("rb") as log:
            log.seek(offset)
            codes = re.findall(r'POST /v1/completions HTTP/1\.1" (\d+)', log.read().decode(errors="replace"))
        self.assertEqual(codes, ["200"] * 256, f"Incomplete evaluation; inspect {path}")
        for marker in ("Bad Request Error", "Rate limit exception", "All retry attempts exhausted"):
            self.assertNotIn(marker, client_log)
        return json.loads((self.artifacts / f"{stage}.json").read_text())["score"]

    def test_pause_kill_scale_down(self):
        healthy = {rank: "healthy" for rank in range(4)}
        self._save("initial-status", self._wait_status(healthy))
        schedulers = self._schedulers()
        self.assertEqual(set(schedulers), set(range(4)))
        self._save("scheduler-pids", {rank: proc.pid for rank, proc in schedulers.items()})
        for rank in range(4):
            response = self._generate(rank)
            self.assertEqual(response.status_code, 200, response.text)
        baseline = self._gsm8k("baseline")
        # Confirm DP0 has begun decode before killing rank 3, as in server_tool.
        stream = requests.post(f"{self.url}/generate", json={
            "text": "Count upward slowly, writing one integer per line.",
            "routed_dp_rank": 0, "stream": True,
            "sampling_params": {"temperature": 0, "max_new_tokens": 256},
        }, stream=True, timeout=(10, 60))
        self.addCleanup(stream.close)
        self.assertEqual(stream.status_code, 200)
        for line in stream.iter_lines(chunk_size=1):
            if line.startswith(b"data: ") and line != b"data: [DONE]":
                event = json.loads(line[6:])
                info = event.get("meta_info", {})
                if info.get("dp_rank") == 0 and info.get("completion_tokens", 0) > 0:
                    self._save("trigger-stream-first-decode", event)
                    break
        else:
            self.fail("Stream ended before DP0 decode was observed")
        schedulers[3].kill()
        schedulers[3].wait(timeout=15)
        fault = self._wait_status({0: "unhealthy", 1: "unhealthy", 2: "unhealthy", 3: "dead"})
        self._save("fault-status", fault)
        stream.close()
        self.assertEqual(self._generate(0).status_code, 503)
        request_id = f"scale-down-{uuid.uuid4().hex}"
        response = requests.post(f"{self.url}/v1/fault_tolerance/apply", json={
            "instruction": "scale_down", "params": {"removed_dp_ranks": [3]}, "request_id": request_id,
        }, timeout=10)
        self.assertEqual(response.status_code, 202, response.text)
        self._save("apply-response", response.json())
        self.assertEqual(response.json()["request_id"], request_id)
        healthy[3] = "dead"
        self._save("completed-status", self._wait_status(healthy, request_id))
        survivors = self._schedulers()
        self.assertEqual({rank: proc.pid for rank, proc in survivors.items()},
                         {rank: schedulers[rank].pid for rank in range(3)})
        for rank in range(3):
            response = self._generate(rank, logprobs=True)
            self.assertEqual(response.status_code, 200, response.text)
            self.assertTrue(response.json()["meta_info"]["input_token_logprobs"])
        removed = self._generate(3)
        self.assertEqual(removed.status_code, 503)
        self.assertIn("routed_dp_rank=3 is not active", removed.text)
        self.assertEqual(self._generate().status_code, 200)
        after = self._gsm8k("after-scale-down")
        self._save("accuracy-comparison", {"baseline": baseline, "after_scale_down": after, "delta": after - baseline})
        print(f"GSM8K: {baseline:.2%} -> {after:.2%}; artifacts: {self.artifacts}")
        self.assertGreaterEqual(after, baseline, "Accuracy decreased; inspect saved reports")
