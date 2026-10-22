"""Launch backend and simulator with auth/readiness checks and clean shutdown."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def _url(host: str, port: int, path: str = "/health") -> str:
    return f"http://{host}:{port}{path}"


def _check_auth_settings() -> None:
    """Fail fast if required local auth tokens are missing (no silent bypass)."""
    try:
        from pydantic import ValidationError

        from casepilot.shared.config import get_settings
    except ImportError as exc:
        raise RuntimeError(f"Cannot import settings for auth check: {exc}") from exc

    try:
        get_settings()
    except ValidationError as exc:
        missing = sorted(
            {
                str(err.get("loc", ("?",))[0]).upper()
                for err in exc.errors()
                if err.get("type") == "missing"
            }
        )
        hint = (
            "Required local auth tokens are missing "
            f"({', '.join(missing) or 'see ValidationError'}). "
            "Run: make provision-local-auth\n"
            "Refusing to start without SIMULATOR_SERVICE_TOKEN and OPERATOR_TOKEN "
            "(no silent bypass)."
        )
        raise RuntimeError(hint) from None
    print("auth settings: OK (tokens present; values not printed)")


def _wait_ready(name: str, url: str, timeout_s: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as resp:
                if 200 <= resp.status < 300:
                    print(f"{name} ready: {url}")
                    return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            time.sleep(0.25)
    raise RuntimeError(f"{name} not ready at {url}: {last_error}")


def main() -> int:
    backend_host = os.environ.get("BACKEND_HOST", "127.0.0.1")
    simulator_host = os.environ.get("SIMULATOR_HOST", "127.0.0.1")
    backend_port = _env_int("BACKEND_PORT", 8000)
    simulator_port = _env_int("SIMULATOR_PORT", 8001)

    procs: list[subprocess.Popen[bytes]] = []

    def shutdown() -> None:
        for proc in reversed(procs):
            if proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
        for proc in reversed(procs):
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

    try:
        _check_auth_settings()

        procs.append(
            subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "uvicorn",
                    "casepilot.simulator.app:app",
                    "--host",
                    simulator_host,
                    "--port",
                    str(simulator_port),
                ],
                cwd=ROOT,
            )
        )
        procs.append(
            subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "uvicorn",
                    "casepilot.backend.app:app",
                    "--host",
                    backend_host,
                    "--port",
                    str(backend_port),
                ],
                cwd=ROOT,
            )
        )

        _wait_ready("simulator", _url(simulator_host, simulator_port))
        _wait_ready("backend", _url(backend_host, backend_port))
        print("Both services ready. Ctrl+C to stop.")

        while True:
            for name, proc in (("simulator", procs[0]), ("backend", procs[1])):
                code = proc.poll()
                if code is not None:
                    raise RuntimeError(f"{name} exited early with code {code}")
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nShutting down...")
        return 0
    except Exception as exc:
        print(f"dev launcher failed: {exc}", file=sys.stderr)
        return 1
    finally:
        shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
