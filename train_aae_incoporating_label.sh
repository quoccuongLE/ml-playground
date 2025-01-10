#!/bin/bash
SEED=${1:-"-1"}

python aae_conditional_train.py --num_epochs 200 &