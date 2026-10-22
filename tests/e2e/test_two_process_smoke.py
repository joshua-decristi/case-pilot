"""Deterministic two-process smoke: real uvicorn backend + simulator.

No paid model calls (CASEPILOT_TEST_MODE + HTTP scripted-model hook).
Explicit env only — overrides any repo .env tokens/ports/paths.

Sole purpose: one real-stack path proving CLI→backend wiring, HTTP discover,
auth gate, authorized task persistence, and restart/idempotent reapproval.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest

from casepilot.backend.agent import AgentFinding
from casepilot.backend.app import CASE_PROPOSALS
from casepilot.backend.hooks import TEST_MODE_ENV
from casepilot.shared.contracts import (
    BackendRoutes,
    ProposalStatus,
    RunStatus,
    SimulatorRoutes,
)
from casepilot.simulator.seed import seed
from tests.helpers import _finding, _tool_then_finding

CASE_ID = "CASE-001"
TOKEN_SIM = "e2e-sim-token"
TOKEN_OP = "e2e-op-token"
REPO_ROOT = Path(__file__).resolve().parents[2]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _isolated_os_env() -> dict[str, str]:
    """Carry only process plumbing; drop parent tokens/API keys/.env-fed secrets."""
    keep_prefixes = ("UV_", "PYTHON", "VIRTUAL", "CONDA", "SSL_", "REQUESTS_")
    keep_exact = {
        "PATH",
        "HOME",
        "USER",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "TMPDIR",
        "TMP",
        "TEMP",
        "XDG_CACHE_HOME",
        "XDG_CONFIG_HOME",
        "SYSTEMROOT",
        "WINDIR",
        "COMSPEC",
    }
    out: dict[str, str] = {}
    for key, value in os.environ.items():
        if key in keep_exact or key.startswith(keep_prefixes):
            out[key] = value
    return out


def _wait_ready(url: str, *, timeout_s: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_s
    last: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as resp:
                if 200 <= resp.status < 300:
                    return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last = exc
            time.sleep(0.1)
    raise RuntimeError(f"not ready at {url}: {last}")


def _stop_proc(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def _port_free(port: int, *, timeout_s: float = 10.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
                return
            except OSError:
                time.sleep(0.05)
    raise RuntimeError(f"port {port} still busy after stop")


@dataclass
class LiveStack:
    backend_port: int
    simulator_port: int
    backend_url: str
    simulator_url: str
    backend_db: Path
    simulator_db: Path
    env: dict[str, str]
    simulator_proc: subprocess.Popen[bytes] | None = None
    backend_proc: subprocess.Popen[bytes] | None = None

    def start_backend(self) -> subprocess.Popen[bytes]:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "casepilot.backend.app:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(self.backend_port),
            ],
            cwd=str(REPO_ROOT),
            env=self.env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        self.backend_proc = proc
        try:
            _wait_ready(f"{self.backend_url}{BackendRoutes.HEALTH}")
        except Exception:
            err = b""
            if proc.stderr is not None:
                err = proc.stderr.read()[:800]
            _stop_proc(proc)
            raise RuntimeError(f"backend failed to start: {err.decode(errors='replace')}") from None
        return proc

    def stop_backend(self) -> None:
        if self.backend_proc is not None:
            _stop_proc(self.backend_proc)
            self.backend_proc = None
        _port_free(self.backend_port)

    def shutdown(self) -> None:
        self.stop_backend()
        if self.simulator_proc is not None:
            _stop_proc(self.simulator_proc)
            self.simulator_proc = None


@pytest.fixture()
def stack(tmp_path: Path) -> Iterator[LiveStack]:
    backend_port = _free_port()
    simulator_port = _free_port()
    backend_db = tmp_path / "backend.sqlite3"
    simulator_db = tmp_path / "simulator.sqlite3"
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()

    assert seed(db_path=simulator_db, reset=True) == 0

    backend_url = f"http://127.0.0.1:{backend_port}"
    simulator_url = f"http://127.0.0.1:{simulator_port}"

    # Explicit settings only. OS env overrides repo .env; parent secrets are not inherited.
    env = {
        **_isolated_os_env(),
        TEST_MODE_ENV: "1",
        "SIMULATOR_SERVICE_TOKEN": TOKEN_SIM,
        "OPERATOR_TOKEN": TOKEN_OP,
        "BACKEND_HOST": "127.0.0.1",
        "BACKEND_PORT": str(backend_port),
        "SIMULATOR_HOST": "127.0.0.1",
        "SIMULATOR_PORT": str(simulator_port),
        "BACKEND_BASE_URL": backend_url,
        "SIMULATOR_BASE_URL": simulator_url,
        "BACKEND_DB_PATH": str(backend_db),
        "SIMULATOR_DB_PATH": str(simulator_db),
        "TRACE_DIR": str(trace_dir),
        "OPENROUTER_API_KEY": "",
        "INVESTIGATION_TIMEOUT_SECONDS": "30",
        "TOOL_HTTP_TIMEOUT_SECONDS": "10",
        "CASEPILOT_COST_LEDGER_PATH": str(tmp_path / "cost-ledger-unused.json"),
    }

    live = LiveStack(
        backend_port=backend_port,
        simulator_port=simulator_port,
        backend_url=backend_url,
        simulator_url=simulator_url,
        backend_db=backend_db,
        simulator_db=simulator_db,
        env=env,
    )

    sim = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "casepilot.simulator.app:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(simulator_port),
        ],
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    live.simulator_proc = sim
    try:
        _wait_ready(f"{simulator_url}{SimulatorRoutes.HEALTH}")
        live.start_backend()
        if sim.poll() is not None:
            err = (sim.stderr.read() if sim.stderr else b"").decode()[:500]
            raise RuntimeError(f"simulator exited early: {sim.returncode} {err}")
        yield live
    finally:
        live.shutdown()


def _install_script(client: httpx.Client, finding: AgentFinding) -> None:
    mode = client.get("/v1/_test/mode")
    assert mode.status_code == 200, mode.text
    assert mode.json()["test_mode"] == "1"
    responses = _tool_then_finding(finding)
    resp = client.put(
        "/v1/_test/scripted-model",
        json={"responses": [m.model_dump(mode="json") for m in responses]},
    )
    assert resp.status_code == 200, resp.text


def _cli(stack: LiveStack, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "casepilot.cli", *args],
        cwd=str(REPO_ROOT),
        env=stack.env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def _sim_task_row(db_path: Path, proposal_id: str) -> sqlite3.Row | None:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute(
            "SELECT * FROM follow_up_tasks WHERE proposal_id = ?",
            (proposal_id,),
        ).fetchone()
    finally:
        conn.close()


def test_two_process_cli_http_smoke(stack: LiveStack) -> None:
    """CLI investigate → HTTP discover/auth/approve → restart + idempotent reapproval."""
    finding = _finding(CASE_ID)
    assert finding.proposed_task is not None

    with httpx.Client(base_url=stack.backend_url, timeout=45.0) as client:
        _install_script(client, finding)

        # CLI initiates investigation (real CLI→backend sockets)
        inv_cli = _cli(stack, "investigate", CASE_ID, "--json")
        assert inv_cli.returncode == 0, inv_cli.stderr + inv_cli.stdout
        inv_payload = json.loads(inv_cli.stdout)
        run_id = inv_payload["run_id"]
        result = inv_payload["result"]
        assert result["status"] == RunStatus.COMPLETED.value
        assert result["approval_required"] is True
        proposal_id = result["proposal_id"]
        assert proposal_id

        # HTTP discover run + proposal
        run_got = client.get(BackendRoutes.RUN.format(run_id=run_id))
        assert run_got.status_code == 200, run_got.text
        assert run_got.json()["status"] == RunStatus.COMPLETED.value

        listed = client.get(CASE_PROPOSALS.format(case_id=CASE_ID))
        assert listed.status_code == 200, listed.text
        assert proposal_id in {p["proposal_id"] for p in listed.json()}

        got = client.get(BackendRoutes.PROPOSAL.format(proposal_id=proposal_id))
        assert got.status_code == 200
        assert got.json()["status"] == ProposalStatus.PENDING.value

        # One unauthorized auth-propagation check
        no_auth = client.post(
            BackendRoutes.APPROVE.format(proposal_id=proposal_id),
            json={"operator_identity": "e2e"},
        )
        assert no_auth.status_code == 401

        # Authorized approve → simulator task persisted
        ok = client.post(
            BackendRoutes.APPROVE.format(proposal_id=proposal_id),
            json={"operator_identity": "e2e-operator"},
            headers={"Authorization": f"Bearer {TOKEN_OP}"},
        )
        assert ok.status_code == 200, ok.text
        assert ok.json()["proposal"]["status"] == ProposalStatus.EXECUTED.value

        row = _sim_task_row(stack.simulator_db, proposal_id)
        assert row is not None
        assert row["case_id"] == CASE_ID
        assert row["title"] == finding.proposed_task.title

    # Backend restart: durable recovery + idempotent reapproval
    stack.stop_backend()
    stack.start_backend()

    with httpx.Client(base_url=stack.backend_url, timeout=45.0) as client:
        run_after = client.get(BackendRoutes.RUN.format(run_id=run_id))
        assert run_after.status_code == 200, run_after.text
        assert run_after.json()["status"] == RunStatus.COMPLETED.value
        assert run_after.json()["result"]["proposal_id"] == proposal_id

        prop_after = client.get(BackendRoutes.PROPOSAL.format(proposal_id=proposal_id))
        assert prop_after.status_code == 200
        assert prop_after.json()["status"] == ProposalStatus.EXECUTED.value

        post_restart = client.post(
            BackendRoutes.APPROVE.format(proposal_id=proposal_id),
            json={"operator_identity": "e2e-operator"},
            headers={"Authorization": f"Bearer {TOKEN_OP}"},
        )
        assert post_restart.status_code == 200
        assert post_restart.json()["proposal"]["status"] == ProposalStatus.EXECUTED.value
