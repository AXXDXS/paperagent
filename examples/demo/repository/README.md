# ReproAgent demo repository

This repository is the deterministic workload used by `repro-agent demo`.
It has no third-party dependencies and does not require a network connection.

The five-tier commands are:

```bash
python -m compileall -q .
python -m unittest -q
python train.py --tier smoke_test
python train.py --tier reduced_experiment
python train.py --tier full_experiment
```

`train.py` writes metrics, predictions, and labels to
`REPRO_AGENT_OUTPUT_DIR` when provided by the harness.
