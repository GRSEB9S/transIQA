#!/usr/bin/env bash


# LIVE ft with mse
python ./src/fine_tune.py --mode ft --load_model ./model/scratch/cuda_True_epoch_550\
 --model_save ./model/ft/live_mse --data_log ./log/ft/live_180_mse.txt\
 --train_loss mse --test_loss mse\
 --lr 1e-6\
 2>&1 | tee ./log/ft/running/run_live_180_mse.txt

# LIVE ft with mae
python ./src/fine_tune.py --mode ft --load_model ./model/scratch/cuda_True_epoch_550\
 --model_save ./model/ft/live_mae --data_log ./log/ft/live_180_mae.txt\
 --lr 1e-6\
 2>&1 | tee ./log/ft/running/run_live_180_mae.txt

# ----------

# LIVE ft2 with mae
python ./src/fine_tune.py --mode ft2 --load_model ./model/scratch/cuda_True_epoch_550\
 --model_save ./model/ft2/live_mae --data_log ./log/ft2/live_180_mae.txt\
 --lr 1e-6\
 2>&1 | tee ./log/ft2/running/run_live_180_mae.txt

# LIVE ft2 with mae
# python ./src/fine_tune.py --mode ft2 --load_model ./model/scratch/cuda_True_epoch_550\
# --model_save ./model/ft2/live --model_epoch 200 --data_log ./log/ft2/live_180.txt

# ----------

# LIVE ft12 with mae
# python ./src/fine_tune.py --mode ft12 --load_model ./model/scratch/cuda_True_epoch_550\
# --model_save ./model/ft12/live --model_epoch 200 --data_log ./log/ft12/live_180.txt

# LIVE ft12 with mae
python ./src/fine_tune.py --mode ft12 --load_model ./model/scratch/cuda_True_epoch_550\
 --model_save ./model/ft12/live_mae --data_log ./log/ft12/live_180_mae.txt\
 --lr 1e-6\
 2>&1 | tee ./log/ft12/running/run_live_180_mae.txt

# ----------

# LIVE ft2 with mse
# python ./src/fine_tune.py --mode ft2 --load_model ./model/scratch/cuda_True_epoch_550\
# --model_save ./model/ft2/live_mse  --data_log ./log/ft2/live_180_mse.txt
# --train_loss mse --test_loss mse

# LIVE ft12 with mse
# python ./src/fine_tune.py --mode ft12 --load_model ./model/scratch/cuda_True_epoch_550\
# --model_save ./model/ft12/live_mse  --data_log ./log/ft12/live_180_mse.txt\
# --train_loss mse --test_loss mse

# ----------
# ----------

# TID2013 ft with mse
# python ./src/fine_tune.py --mode ft --dataset tid2013 --load_model ./model/scratch/cuda_True_epoch_550\
# --model_save ./model/ft/tid2013_mse --data_log ./log/ft/tid2013_180_mse.txt
# --train_loss mse --test_loss mse

# TID2013 ft2 with mse
# python ./src/fine_tune.py --mode ft2 --dataset tid2013 --load_model ./model/scratch/cuda_True_epoch_550\
# --model_save ./model/ft2/tid2013_mse --data_log ./log/ft2/tid2013_180_mse.txt
# --train_loss mse --test_loss mse

# TID2013 ft12 with mse
# python ./src/fine_tune.py --mode ft12 --dataset tid2013 --load_model ./model/scratch/cuda_True_epoch_550\
# --model_save ./model/ft12/tid2013_mse --data_log ./log/ft12/tid2013_180_mse.txt
# --train_loss mse --test_loss mse

# TID2013 ft with mae
# python ./src/fine_tune.py --mode ft --dataset tid2013 --load_model ./model/scratch/cuda_True_epoch_550\
# --model_save ./model/ft/tid2013_mae --data_log ./log/ft/tid2013_180_mae.txt
# --train_loss mae --test_loss mae

# TID2013 ft2 with mae
# python ./src/fine_tune.py --mode ft2 --dataset tid2013 --load_model ./model/scratch/cuda_True_epoch_550\
# --model_save ./model/ft2/tid2013_mae --data_log ./log/ft2/tid2013_180_mae.txt
# --train_loss mae --test_loss mae

# TID2013 ft12 with mae
# python ./src/fine_tune.py --mode ft12 --dataset tid2013 --load_model ./model/scratch/cuda_True_epoch_550\
# --model_save ./model/ft12/tid2013_mae --data_log ./log/ft12/tid2013_180_mae.txt
# --train_loss mae --test_loss mae


# ----------
# ----------

# train from scratch with mse
python ./src/main.py --epochs 500 --data_log ./log/scratch/face_180_mse.txt\
 --train_loss mse --test_loss mse\
 2>&1 | tee ./log/scratch/running/run_face_180_mse.txt