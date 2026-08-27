# A Deterministic Threshold Classifier

## Abstract

This toy experiment evaluates a threshold classifier on a fixed binary dataset.
It is intentionally small so that the complete reproduction harness can be
demonstrated without external datasets, model downloads, GPUs, or network access.

## Method

Given a scalar input `x`, the classifier predicts class 1 when `x >= 0.5` and
class 0 otherwise. The experiment uses seed 7 and contains 20 fixed examples.
No training dependency is required.

## Evaluation

The primary metric is classification accuracy. The expected full-experiment
accuracy is 0.90 with an absolute tolerance of 0.01. The experiment must write
`metrics.json`, `predictions.json`, and `labels.json` so an independent verifier
can recompute the metric from raw outputs.

## Reproduction protocol

The repository exposes five gates: static check, unit test, smoke test, reduced
experiment, and full experiment. Every gate must pass before the final report is
assembled.
