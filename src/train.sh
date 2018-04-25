#!/usr/bin/env bash

# train live with ft2 with mae
# python ./src/fine_tune.py --mode ft2  --dataset live --load_model ./model/scratch/cuda_True_epoch_550--model_save ./model/ft2/live --model_epoch 200 --data_log ./log/ft2/live_180.txt

# train live with ft12 with mae
# python ./src/fine_tune.py --mode ft12  --dataset live --load_model ./model/scratch/cuda_True_epoch_550--model_save ./model/ft12/live --model_epoch 200 --data_log ./log/ft12/live_180.txt

# train live with ft2 with mse
python ./src/fine_tune.py --mode ft2 --train_loss mse --test_loss mse --load_model ./model/scratch/cuda_True_epoch_550 --model_save ./model/ft2/live_mse  --data_log ./log/ft2/live_180_mse.txt

# train live with ft12 with mse
python ./src/fine_tune.py --mode ft12 --train_loss mse --test_loss mse --load_model ./model/scratch/cuda_True_epoch_550 --model_save ./model/ft12/live_mse  --data_log ./log/ft12/live_180_mse.txt

# train tid2013 with ft with mse
python ./src/fine_tune.py --mode ft --dataset tid2013 --train_loss mse --test_loss mse --load_model ./model/scratch/cuda_True_epoch_550 --model_save ./model/ft/tid2013_mse --data_log ./log/ft/tid2013_180_mse.txt

# train tid2013 with ft2 with mse
python ./src/fine_tune.py --mode ft2 --dataset tid2013 --train_loss mse --test_loss mse --load_model ./model/scratch/cuda_True_epoch_550 --model_save ./model/ft2/tid2013_mse --data_log ./log/ft2/tid2013_180_mse.txt

# train tid2013 with ft12 with mse
python ./src/fine_tune.py --mode ft12 --dataset tid2013 --train_loss mse --test_loss mse --load_model ./model/scratch/cuda_True_epoch_550 --model_save ./model/ft12/tid2013_mse --data_log ./log/ft12/tid2013_180_mse.txt

# train tid2013 with ft with mae
python ./src/fine_tune.py --mode ft --dataset tid2013 --train_loss mae --test_loss mae --load_model ./model/scratch/cuda_True_epoch_550 --model_save ./model/ft/tid2013_mae --data_log ./log/ft/tid2013_180_mae.txt

# train tid2013 with ft2 with mae
python ./src/fine_tune.py --mode ft2 --dataset tid2013 --train_loss mae --test_loss mae --load_model ./model/scratch/cuda_True_epoch_550 --model_save ./model/ft2/tid2013_mae --data_log ./log/ft2/tid2013_180_mae.txt

# train tid2013 with ft12 with mae
python ./src/fine_tune.py --mode ft12 --dataset tid2013 --train_loss mae --test_loss mae --load_model ./model/scratch/cuda_True_epoch_550 --model_save ./model/ft12/tid2013_mae --data_log ./log/ft12/tid2013_180_mae.txt
