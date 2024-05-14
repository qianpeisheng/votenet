# !/bin/bash


CUDA_VISIBLE_DEVICES=1 python3 train_CIL.py --dataset scannet --log_dir log_scannet --num_point 40000 --seed 43 --total_budget 2000 --duplicate_memory_bank 2

CUDA_VISIBLE_DEVICES=1 python3 train_CIL.py --dataset scannet --log_dir log_scannet --num_point 40000 --seed 43 --total_budget 2000 --duplicate_memory_bank 3

CUDA_VISIBLE_DEVICES=1 python3 train_CIL.py --dataset scannet --log_dir log_scannet --num_point 40000 --seed 43 --total_budget 1000 --duplicate_memory_bank 2

CUDA_VISIBLE_DEVICES=1 python3 train_CIL.py --dataset scannet --log_dir log_scannet --num_point 40000 --seed 43 --total_budget 1000 --duplicate_memory_bank 2