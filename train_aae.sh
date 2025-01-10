#!/bin/bash
SEED=${1:-"-1"}

python aae_train.py --num_epochs 200 &