#!/bin/bash

# 检查用户是否传入了自定义的 source_path 和 model_path
if [ $# -lt 2 ]; then
    echo "Usage: $0 <source_path> <model_path>"
    echo "Example: $0 /path/to/source /path/to/model"
    exit 1
fi

# 获取用户传入的路径
source_path=$1
model_path=$2

for repeat in 2
do
    for NAME in leaves horns
    do
        CUDA_VISIBLE_DEVICES=0 python train_llff_mvs.py -s $source_path/$NAME --model_path $model_path/run$repeat/$NAME -r 8 --eval --n_sparse 3  --mvs_pcd --iterations 6000 --lambda_dssim 0.25 --densify_grad_threshold 0.0008 --prune_threshold 0.01 --densify_until_iter 6000 --percent_dense 0.01 --position_lr_init 0.0005 --position_lr_final 0.000005 --position_lr_max_steps 5500 --position_lr_start 500 --split_opacity_thresh 0.1 --error_tolerance 0.01 --scaling_lr 0.005 --shape_pena 0.002 --opa_pena 0.001 --scale_pena 0 --test_iterations 1000 2000 3000 4500 6000 --near 10 \
        --dropout --dropout_rate 0.4 --consistency_lambda 0.4 --edge_threshold 0.0005 --size_threshold_rate 50&
        wait;
        CUDA_VISIBLE_DEVICES=0 python render.py -s $source_path/$NAME --model_path $model_path/run$repeat/$NAME -r 8 --near 10&
        wait;
        CUDA_VISIBLE_DEVICES=0 python metrics.py --model_path $model_path/run$repeat/$NAME&
        wait;
    done
done

for repeat in 2
do
    for NAME in fortress flower fern room
    do
        CUDA_VISIBLE_DEVICES=0 python train_llff_mvs.py -s $source_path/$NAME --model_path $model_path/run$repeat/$NAME -r 8 --eval --n_sparse 3  --mvs_pcd --iterations 6000 --lambda_dssim 0.25 --densify_grad_threshold 0.0008 --prune_threshold 0.01 --densify_until_iter 6000 --percent_dense 0.01 --position_lr_init 0.0005 --position_lr_final 0.000005 --position_lr_max_steps 5500 --position_lr_start 500 --split_opacity_thresh 0.1 --error_tolerance 0.01 --scaling_lr 0.005 --shape_pena 0.002 --opa_pena 0.001 --scale_pena 0 --test_iterations 1000 2000 3000 4500 6000 --near 10 \
        --dropout --dropout_rate 0.4 --consistency_lambda 0.4 --edge_threshold 0.001 --size_threshold_rate 80&
        wait;
        CUDA_VISIBLE_DEVICES=0 python render.py -s $source_path/$NAME --model_path $model_path/run$repeat/$NAME -r 8 --near 10&
        wait;
        CUDA_VISIBLE_DEVICES=0 python metrics.py --model_path $model_path/run$repeat/$NAME&
        wait;
    done
done

for repeat in 2
do
    for NAME in trex orchids
    do
        CUDA_VISIBLE_DEVICES=0 python train_llff_mvs.py -s $source_path/$NAME --model_path $model_path/run$repeat/$NAME -r 8 --eval --n_sparse 3  --mvs_pcd --iterations 6000 --lambda_dssim 0.25 --densify_grad_threshold 0.0008 --prune_threshold 0.01 --densify_until_iter 6000 --percent_dense 0.01 --position_lr_init 0.0005 --position_lr_final 0.000005 --position_lr_max_steps 5500 --position_lr_start 500 --split_opacity_thresh 0.1 --error_tolerance 0.01 --scaling_lr 0.005 --shape_pena 0.002 --opa_pena 0.001 --scale_pena 0 --test_iterations 1000 2000 3000 4500 6000 --near 10 \
        --dropout --dropout_rate 0.4 --consistency_lambda 0.4 --edge_threshold 0.001 --size_threshold_rate 50&
        wait;
        CUDA_VISIBLE_DEVICES=0 python render.py -s $source_path/$NAME --model_path $model_path/run$repeat/$NAME -r 8 --near 10&
        wait;
        CUDA_VISIBLE_DEVICES=0 python metrics.py --model_path $model_path/run$repeat/$NAME&
        wait;
    done
done
