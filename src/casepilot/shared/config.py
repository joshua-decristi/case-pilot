"""Application settings loaded from environment / .env (extra keys ignored)."""

from __future__ import annotations

from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Local runtime configuration.

    Tokens and API keys use SecretStr so repr/str do not leak secrets.
    Service and operator tokens have no defaults — supply via env or constructor
    (tests pass explicit values). OpenRouter key is optional until live runs.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # OpenRouter (live model runs only)
    openrouter_api_key: SecretStr | None = None
    openrouter_model: str = "z-ai/glm-5.3-flash"

    # Auth — required for authenticated runtime services; tests supply values
    simulator_service_token: SecretStr
    operator_token: SecretStr

    # Loopback service bind / base URLs
    backend_host: str = "127.0.0.1"
    backend_port: int = 8000
    simulator_host: str = "127.0.0.1"
    simulator_port: int = 8001
    backend_base_url: str = "http://127.0.0.1:8000"
    simulator_base_url: str = "http://127.0.0.1:8001"

    # Local data paths
    backend_db_path: Path = Path(".data/backend.sqlite3")
    simulator_db_path: Path = Path(".data/simulator.sqlite3")
    trace_dir: Path = Path("traces")

    # Local OTLP tracing (Phoenix viewer is a separate process; loopback only)
    tracing_enabled: bool = True
    tracing_otlp_endpoint: str = "http://127.0.0.1:6006/v1/traces"

    # Execution budgets (draft defaults from spec)
    investigation_timeout_seconds: float = 60.0
    tool_http_timeout_seconds: float = 10.0
    max_tool_calls: int = 12
    max_model_turns: int = 8


def get_settings(**overrides: object) -> Settings:
    """Build settings from env; optional overrides for tests.

    BaseSettings.model_validate still resolves missing fields from the environment,
    so tokens remain required at runtime without defeating the type checker.
    """
    return Settings.model_validate(overrides)
