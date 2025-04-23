#!/bin/bash

source_path=$1
model_path=$2

for repeat in 2
do
    for NAME in fortress fern flower horns orchids room leaves trex
    do
        CUDA_VISIBLE_DEVICES=0 python train_llff.py -s $source_path/$NAME --model_path $model_path/run$repeat/$NAME -r 8 --eval --n_sparse 3  --rand_pcd --iterations 6000 --lambda_dssim 0.2 --densify_grad_threshold 0.0013 --prune_threshold 0.01 --densify_until_iter 6000 --percent_dense 0.01 --position_lr_init 0.016 --position_lr_final 0.00016 --position_lr_max_steps 5500 --position_lr_start 500 --split_opacity_thresh 0.1 --error_tolerance 0.00025 --scaling_lr 0.003 --shape_pena 0.002 --opa_pena 0.001 --near 10 \
--dropout --dropout_rate 0.4 --consistency_lambda 0.2 --edge_threshold 0.001 --size_threshold_rate 50 --test_iterations 1000 2000 3000 4000 5000 6000&
        wait;
        CUDA_VISIBLE_DEVICES=0 python render.py -s $source_path/$NAME --model_path $model_path/run$repeat/$NAME -r 8 --near 10&
        wait;
        CUDA_VISIBLE_DEVICES=0 python metrics.py --model_path $model_path/run$repeat/$NAME&
        wait;
    done
done