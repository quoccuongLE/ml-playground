#!/bin/bash
SEED=${1:-"-1"}

current_date=$(date +"%Y-%m-%d_%H:%M:%S")

python aae_train.py > log/log_aae_train_$current_date.txt &

# python aae_train.py --seed 906
