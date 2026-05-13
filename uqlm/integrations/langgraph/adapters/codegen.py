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

from uqlm.scorers.shortform.codegen import CodeGenUQ
from uqlm.integrations.base import register_adapter

_SKIP_KEYS = frozenset({"prompts", "responses", "sampled_responses", "raw_responses", "raw_sampled_responses", "logprob", "sampled_logprob"})


class CodeGenUQAdapter:
    scorer_type = CodeGenUQ

    async def run(self, scorer, *, prompt, response, mode, num_responses, reference_solution=None, **kwargs):
        result = await scorer.generate_and_score(prompts=[prompt], num_responses=num_responses, show_progress_bars=False)
        data = result.data
        responses_list = data.get("responses", [])
        scores = {k: v[0] for k, v in data.items() if k not in _SKIP_KEYS and isinstance(v, list) and len(v) > 0}
        primary = [responses_list[0]] if responses_list else []
        extra = {}
        if reference_solution is not None:
            extra["reference_solution"] = reference_solution
        return {"scores": scores, "responses": primary, "extra": extra}


register_adapter(CodeGenUQAdapter())
