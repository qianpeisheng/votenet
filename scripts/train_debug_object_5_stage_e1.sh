# !/bin/bash

CUDA_VISIBLE_DEVICES=1 python train_CIL_object.py --dataset scannet --log_dir debug_log_scannet --num_point 40000 --seed 42  --debug --eval_freq 1 --batch_size 1