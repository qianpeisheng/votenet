# !/bin/bash

CUDA_VISIBLE_DEVICES=0 python train_CIL.py --dataset scannet --log_dir log_scannet --num_point 40000 --seed 42