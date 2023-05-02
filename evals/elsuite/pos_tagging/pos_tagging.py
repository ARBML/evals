import re
import random
from typing import Any
import difflib

import evals
import evals.metrics
from evals.prompt.base import is_chat_prompt


def match_tag_sequences(predicted_tags, true_tags):
    def clean_tag_lines(lines):
        cleaned_lines = []
        for line in lines:
            # Remove quotes at the beginning or end of the lines
            line = re.sub(r'^"|"$', "", line)

            # Remove any character that is not Arabic or punctuation
            line = re.sub(r"[^\u0600-\u06FF\s\.\?!،؛]", "", line)

            # Print the cleaned line
            cleaned_lines.append(line)
        return cleaned_lines

    matcher = difflib.Differ()
    match_sequence = matcher.compare(
        clean_tag_lines(true_tags),
        clean_tag_lines(predicted_tags),
    )
    return list(match_sequence)


def pos_tagging_accuracy(predicted_tags, true_tags):
    # Convert the predicted and true tags to lists
    predicted_tags = predicted_tags.strip().split("\n")
    true_tags = true_tags.strip().split("\n")

    match_sequence = match_tag_sequences(
        predicted_tags=predicted_tags,
        true_tags=true_tags,
    )

    # print(len(match_sequence))
    # print(len(true_tags))
    # print(len(predicted_tags))

    # Initialize the count of correct tags
    correct_count = 0
    matched_index = 0
    for line in match_sequence:
        # if the current line only presents in the predicted sequence, just continue without any penalization.
        if line.startswith("+"):
            # remove that element from the predicted tags to have equal length lists
            predicted_tags.pop(matched_index)
            continue
        if line.startswith("?"):
            continue
        # if the current line is present in the true sequence but not in the predicted sequence, penalize (matched_index incremented and correct_count was not) and continue
        if line.startswith("-"):
            predicted_tags.insert(matched_index, "X")
            matched_index += 1
            continue
        # print(line)
        # print(predicted_tags[matched_index])
        # print(true_tags[matched_index])
        # print(f"{matched_index:=}")
        # print("*" * 100)
        predicted_tag = predicted_tags[matched_index].split(":")[1].strip()
        true_tag = true_tags[matched_index].split(":")[1].strip()
        if predicted_tag == true_tag:
            correct_count += 1
        matched_index += 1

    # Calculate the accuracy and return it
    accuracy = correct_count / len(true_tags)
    return accuracy


class POSTagger(evals.Eval):
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
