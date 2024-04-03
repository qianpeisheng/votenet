# !/bin/bash

# change to the parent directory to run the script
cd ..

CUDA_VISIBLE_DEVICES=1 python3 train_CIL.py --dataset scannet --log_dir log_scannet --num_point 40000 --seed 43 --total_budget 2000