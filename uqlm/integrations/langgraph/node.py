# Copyright 2025 CVS Health and/or one of its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio

from uqlm.integrations.base import resolve_adapter
import uqlm.integrations.langgraph.adapters  # noqa: F401 — triggers adapter registration


class UQLMNode:
    def __init__(
        self,
        scorer,
        *,
        input_key: str = "messages",
        output_key: str = "uq",
        mode: str = "score_response",
        prompt_field: str = "prompt",
        response_field: str = "response",
        num_responses: int = 5,
        adapter_kwargs: dict | None = None,
    ):
        self.scorer = scorer
        self.input_key = input_key
        self.output_key = output_key
        self.mode = mode
        self.prompt_field = prompt_field
        self.response_field = response_field
        self.num_responses = num_responses
        self.adapter_kwargs = adapter_kwargs

    async def __acall__(self, state: dict) -> dict:
        adapter = resolve_adapter(self.scorer)
        prompt = state[self.prompt_field]
        response = state.get(self.response_field)

        extra_kwargs = dict(self.adapter_kwargs or {})
        for key in ("sampled_responses", "reference_solution"):
            if key in state and key not in extra_kwargs:
                extra_kwargs[key] = state[key]

        payload = await adapter.run(self.scorer, prompt=prompt, response=response, mode=self.mode, num_responses=self.num_responses, **extra_kwargs)
        payload["scorer"] = type(self.scorer).__name__
        return {self.output_key: payload}

    def __call__(self, state: dict) -> dict:
        return asyncio.run(self.__acall__(state))


def make_uqlm_node(scorer, **kwargs):
    node = UQLMNode(scorer, **kwargs)

    async def _node(state: dict) -> dict:
        return await node.__acall__(state)

    return _node
