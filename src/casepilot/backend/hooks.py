"""Explicit test hooks for scripted models. Never active unless test mode is set."""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage

TEST_MODE_ENV = "CASEPILOT_TEST_MODE"

# Module-level only consulted when CASEPILOT_TEST_MODE is truthy.
_scripted_messages: list[AIMessage] | None = None
_model_factory: Callable[[], BaseChatModel] | None = None


def is_test_mode() -> bool:
    return os.environ.get(TEST_MODE_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def install_scripted_model(
    responses: Sequence[AIMessage | dict[str, Any]],
) -> None:
    """Install scripted AIMessage turns for the next agent runs.

    Requires CASEPILOT_TEST_MODE. Raises RuntimeError otherwise so live
    processes never silently pick up scripted behavior.
    """
    if not is_test_mode():
        raise RuntimeError(
            f"install_scripted_model requires {TEST_MODE_ENV}=1 "
            "(refusing to install scripted model outside explicit test mode)"
        )
    global _scripted_messages, _model_factory
    msgs: list[AIMessage] = []
    for item in responses:
        if isinstance(item, AIMessage):
            msgs.append(item)
        else:
            msgs.append(AIMessage.model_validate(item))
    _scripted_messages = msgs
    _model_factory = None


def install_model_factory(factory: Callable[[], BaseChatModel]) -> None:
    """Install a custom chat-model factory. Requires CASEPILOT_TEST_MODE."""
    if not is_test_mode():
        raise RuntimeError(
            f"install_model_factory requires {TEST_MODE_ENV}=1 "
            "(refusing custom model factory outside explicit test mode)"
        )
    global _scripted_messages, _model_factory
    _model_factory = factory
    _scripted_messages = None


def clear_test_hooks() -> None:
    global _scripted_messages, _model_factory
    _scripted_messages = None
    _model_factory = None


def has_test_hook() -> bool:
    """True when a scripted queue or custom factory is installed."""
    return _model_factory is not None or _scripted_messages is not None


def resolve_test_model() -> BaseChatModel | None:
    """Return a test model if test mode + hook installed; else None.

    Callers in test mode must not treat None as permission to use a live paid
    provider — see default_model_factory fail-closed behavior.
    """
    if not is_test_mode():
        return None
    if _model_factory is not None:
        return _model_factory()
    if _scripted_messages is not None:
        from casepilot.backend.scripted import ScriptedChatModel

        return ScriptedChatModel(responses=list(_scripted_messages))
    return None
