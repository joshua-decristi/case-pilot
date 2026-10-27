"""Launch a pinned local Phoenix viewer as a separate process (not a backend dep)."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PHOENIX_DIR = (ROOT / ".data" / "phoenix").resolve()
PHOENIX_DB = PHOENIX_DIR / "phoenix.db"

# Server stays outside the app lockfile. App runtime remains project Python (3.14).
PHOENIX_VERSION = "20.26.1"


def main() -> int:
    if shutil.which("uvx") is None and shutil.which("uv") is None:
        print("error: uv/uvx not found on PATH", file=sys.stderr)
        return 2

    PHOENIX_DIR.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PHOENIX_WORKING_DIR"] = str(PHOENIX_DIR)
    env["PHOENIX_SQL_DATABASE_URL"] = f"sqlite:///{PHOENIX_DB}"
    env["PHOENIX_HOST"] = "127.0.0.1"
    env["PHOENIX_PORT"] = env.get("PHOENIX_PORT") or "6006"
    # Official product-telemetry switch (read by this pinned server).
    env["PHOENIX_TELEMETRY_ENABLED"] = "false"
    # Do not enable Phoenix's own remote OTLP server instrumentation.
    env.pop("PHOENIX_SERVER_INSTRUMENTATION_OTLP_TRACE_COLLECTOR_HTTP_ENDPOINT", None)
    env.pop("PHOENIX_SERVER_INSTRUMENTATION_OTLP_TRACE_COLLECTOR_GRPC_ENDPOINT", None)

    cmd = [
        "uvx",
        "--python",
        "3.14",
        "--from",
        f"arize-phoenix=={PHOENIX_VERSION}",
        "phoenix",
        "serve",
    ]

    grpc_port = env.get("PHOENIX_GRPC_PORT") or "4317"
    print(f"Phoenix working dir: {PHOENIX_DIR}")
    print(f"Phoenix SQLite:      {PHOENIX_DB}")
    print(f"UI / OTLP HTTP:     http://127.0.0.1:{env['PHOENIX_PORT']} (loopback)")
    print(
        f"note: unused OTLP gRPC intake still binds all interfaces on port {grpc_port} "
        f"(Phoenix default; HTTP/UI stay localhost-only)"
    )
    sys.stdout.flush()

    os.execvpe(cmd[0], cmd, env)


if __name__ == "__main__":
    raise SystemExit(main())
