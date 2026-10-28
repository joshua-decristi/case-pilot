"""Focused settings security checks (schema/enum/route units deleted)."""

from __future__ import annotations

from casepilot.shared import get_settings


def test_settings_secrets_not_leaked_in_repr() -> None:
    settings = get_settings(
        _env_file=None,
        simulator_service_token="super-secret-sim",
        operator_token="super-secret-op",
        openrouter_api_key="sk-secret-key",
    )
    text = repr(settings)
    assert "super-secret-sim" not in text
    assert "super-secret-op" not in text
    assert "sk-secret-key" not in text
    assert settings.simulator_service_token.get_secret_value() == "super-secret-sim"
    assert settings.operator_token.get_secret_value() == "super-secret-op"
