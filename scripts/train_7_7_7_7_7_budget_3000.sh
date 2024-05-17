# !/bin/bash

CUDA_VISIBLE_DEVICES=0 python train_CIL_object.py --dataset scannet --log_dir log_scannet_b_3000 --num_point 40000 --seed 42  --total_budget 3000