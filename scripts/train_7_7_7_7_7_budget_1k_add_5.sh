# !/bin/bash

CUDA_VISIBLE_DEVICES=1 python train_CIL_object.py --dataset scannet --log_dir log_scannet_b_1k_a_5 --num_point 40000 --seed 42  --total_budget 1000 --fixed_insertion_number 5