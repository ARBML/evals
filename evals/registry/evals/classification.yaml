# Define a base eval
classification:
  id: classification.dev.match-v1
  metrics: [accuracy]
  description: Evaluate classification ability
# Define the eval
classification.dev.match-v1:
  class: evals.elsuite.classification:Classification
  args:
    train_jsonl: /tmp/train.jsonl
    test_jsonl: /tmp/test.jsonl
