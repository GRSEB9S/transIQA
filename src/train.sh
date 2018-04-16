#!/usr/bin/env bash

python ./src/main.py --limited 2>&1 | tee ./src/training_log.txt