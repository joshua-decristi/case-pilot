"""Scripted chat model for deterministic agent tests (bind_tools-compatible)."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field, PrivateAttr


class ScriptedChatModel(BaseChatModel):
    """Replay fixed AIMessage turns. Shared index survives bind_tools copies."""

    responses: list[AIMessage] = Field(default_factory=list)
    _idx: list[int] = PrivateAttr(default_factory=lambda: [0])

    @property
    def _llm_type(self) -> str:
        return "casepilot-scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> ScriptedChatModel:
        other = self.model_copy()
        other._idx = self._idx
        return other

    def _generate(
        self,
        messages: list[Any],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        i = self._idx[0]
        if i >= len(self.responses):
            raise RuntimeError(f"ScriptedChatModel exhausted after {i} turns")
        msg = self.responses[i]
        self._idx[0] = i + 1
        return ChatResult(generations=[ChatGeneration(message=msg)])

    async def _agenerate(
        self,
        messages: list[Any],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        return self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
