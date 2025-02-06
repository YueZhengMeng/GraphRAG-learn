# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""The Chat-based language model."""

from typing_extensions import Unpack

from graphrag.llm.types import (
    LLM,
    CompletionInput,
    CompletionLLM,
    CompletionOutput,
    LLMInput,
    LLMOutput,
)


class OpenAIHistoryTrackingLLM(LLM[CompletionInput, CompletionOutput]):
    """An OpenAI History-Tracking LLM."""

    _delegate: CompletionLLM

    def __init__(self, delegate: CompletionLLM):
        self._delegate = delegate

    async def __call__(
        self,
        input: CompletionInput,
        **kwargs: Unpack[LLMInput],
    ) -> LLMOutput[CompletionOutput]:
        """Call the LLM."""
        history = kwargs.get("history") or []
        output = await self._delegate(input, **kwargs)
        # 核心逻辑
        # 将LLM的输出以system role添加到history中，并返回
        # 我对此有些疑惑，不应该是assistant role吗？
        # 而且，input不需要以user role添加到history中吗？
        return LLMOutput(
            output=output.output,
            json=output.json,
            history=[*history, {"role": "system", "content": output.output}],
        )
