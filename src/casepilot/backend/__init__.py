"""Investigation backend: API, agent, proposals, traces."""

from casepilot.backend.app import create_app
from casepilot.backend.hooks import (
    TEST_MODE_ENV,
    clear_test_hooks,
    install_scripted_model,
    is_test_mode,
)

__all__ = [
    "TEST_MODE_ENV",
    "clear_test_hooks",
    "create_app",
    "install_scripted_model",
    "is_test_mode",
]
