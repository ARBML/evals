import random
from typing import Any

import evals
import evals.metrics
from evals.prompt.base import is_chat_prompt

from .utils import pos_tagging_accuracy

class Pos_Tagging(evals.Eval):
    def __init__(
        self,
        samples_jsonl: str,
        *args,
        max_tokens: int = 1400,
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
        expected = sample["ideal"]

        if self.num_few_shot > 0:
            assert is_chat_prompt(sample["input"]), "few shot requires chat prompt"
            prompt = sample["input"][:-1]

            random_fewshots = random.sample(
                self.few_shot,
                self.num_few_shot
                if self.num_few_shot < len(self.few_shot)
                else len(self.few_shot),
            )

            for s in random_fewshots:
                prompt += s["sample"]

            prompt += sample["input"][-1:]
            # print(prompt)

        assert isinstance(
            expected, str
        ), "ideal entry in jsonl (i.e. expected) should be string"

        sampled = evals.sample_freeform(
            self.model_spec,
            prompt,
            max_tokens=self.max_tokens,
        )

        if expected is not None:
            accuracy = pos_tagging_accuracy(predicted_tags=sampled, true_tags=expected)

            evals.record.default_recorder().record_event(
                type="tagging",
                data=dict(
                    accuracy=accuracy,
                    sampled=sampled,
                    expected=expected,
                ),
            )

    def run(self, recorder):
        """
        Called by the `oaieval` CLI to run the eval. The `eval_all_samples` method calls `eval_sample`.
        """
        samples = evals.get_jsonl(self.samples_jsonl)
        self.eval_all_samples(recorder, samples)
        events = recorder.get_events("tagging")

        accuracy = list(map(lambda e: e.data["accuracy"], events))

        get_average = lambda items: sum(items) / len(items)

        average_acc = get_average(accuracy)

        return {
            "accuracy": average_acc,
        }
