#!/bin/bash
current_date=$(date +"%Y-%m-%d_%H:%M:%S")

python aae_train.py > log_aae_train_$current_date.txt &
