#!/usr/bin/env bash
python ./src/main.py --epochs 500 --train_loss mae --test_loss mae --data_log ./log/data_log_0180.txt --reload_model ./model/cuda_True_epoch_150 --reload_epoch 150 2>&1 | tee ./log/training_log_0180c.txt
python ./src/main.py --limited 2>&1 | tee ./src/training_log.txt