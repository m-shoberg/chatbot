#!/usr/bin/env python3
from fairseq.checkpoint_utils import load_model_ensemble_and_task

# model = load_model_ensemble_and_task(["./pytorch_model.pt"], arg_overrides={"config_yaml": "./config.yaml", "data": "./"})
model = load_model_ensemble_and_task(["./pytorch_model.pt"], arg_overrides={"data": "./"})
