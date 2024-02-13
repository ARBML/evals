import random
import textwrap
from typing import Any
from collections import Counter
import evals
import evals.metrics
from evals.prompt.base import is_chat_prompt

class Rater(evals.Eval):
    def __init__(
        self,
        samples_jsonl: str,
        *args,
        max_tokens: int = 500,
        num_few_shot: int = 0,
        few_shot_jsonl: str = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.samples_jsonl = samples_jsonl
        self.max_tokens = max_tokens
        self.num_few_shot = int(num_few_shot)
        self.few_shot_jsonl = few_shot_jsonl
        if self.num_few_shot > 0:
            assert (
                few_shot_jsonl is not None
            ), "few shot requires few shot sample dataset"
            self.few_shot_jsonl = few_shot_jsonl
            self.few_shot = evals.get_jsonl(self.few_shot_jsonl)

    def eval_sample(self, sample: Any, *_):
        prompt = sample["input"]

        if self.num_few_shot > 0:
            assert is_chat_prompt(sample["input"]), "few shot requires chat prompt"
            prompt = sample["input"][:-1]
            for s in self.few_shot[: self.num_few_shot]:
                prompt += s["sample"]
            prompt += sample["input"][-1:]

        sampled = evals.sample_freeform(
            self.model_spec,
            prompt,
            max_tokens=self.max_tokens,
        )
        if 'B' in sampled:
            chosen_answer = 'B'
        elif 'C' in sampled:
            chosen_answer = 'C'
        else:
            chosen_answer = 'A'
        evals.record.default_recorder().record_event(
            type="rater",
            data=dict(
                choice=chosen_answer,
                sampled = sampled
            ),
        )
            

    def run(self, recorder):
        """
        Called by the `oaieval` CLI to run the eval. The `eval_all_samples` method calls `eval_sample`.
        """
        samples = evals.get_jsonl(self.samples_jsonl)
        self.eval_all_samples(recorder, samples)
        events = recorder.get_events("rater")

        output = Counter(map(lambda e: e.data["choice"], events))

        return {d:c for d,c in output.most_common()}
