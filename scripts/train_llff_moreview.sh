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

# # view xxx
# for repeat in 2
# do
#     for NAME in leaves trex
#     do
#         CUDA_VISIBLE_DEVICES=0 python train_llff_moreviews.py -s $source_path/$NAME --model_path $model_path/run$repeat/$NAME -r 8 --eval --n_sparse 6  --rand_pcd --iterations 12000 --lambda_dssim 0.2 --densify_grad_threshold 0.0013 --prune_threshold 0.01 --densify_until_iter 9000 --percent_dense 0.01 --position_lr_init 0.016 --position_lr_final 0.00016 --position_lr_max_steps 8500 --position_lr_start 500 --split_opacity_thresh 0.1 --error_tolerance 0.01 --scaling_lr 0.005 --test_iterations 1000 2000 3000 4500 6000 9000 12000  --shape_pena 0.002 --opa_pena 0.001 --near 10\
#         --dropout --dropout_rate 0.4 --consistency_lambda 0.2 --edge_threshold 0.001 --size_threshold_rate 50&
#         wait;
#         CUDA_VISIBLE_DEVICES=0 python render.py -s $source_path/$NAME --model_path $model_path/run$repeat/$NAME -r 8 --near 10&
#         wait;
#         CUDA_VISIBLE_DEVICES=0 python metrics.py --model_path $model_path/run$repeat/$NAME&
#         wait;
#     done
# done


# view xxx
for repeat in 1
do
    for NAME in leaves fern flower orchids room trex fortress horns
    do
        CUDA_VISIBLE_DEVICES=0 python train_llff_moreviews.py -s $source_path/$NAME --model_path $model_path/run$repeat/$NAME -r 8 --eval --n_sparse 9  --rand_pcd --iterations 12000 --lambda_dssim 0.2 --densify_grad_threshold 0.0013 --prune_threshold 0.01 --densify_until_iter 9000 --percent_dense 0.01 --position_lr_init 0.016 --position_lr_final 0.00016 --position_lr_max_steps 8500 --position_lr_start 500 --split_opacity_thresh 0.1 --error_tolerance 0.01 --scaling_lr 0.005 --test_iterations 1000 2000 3000 4500 6000 9000 12000  --shape_pena 0.002 --opa_pena 0.001 \
        --dropout --dropout_rate 0.4 --consistency_lambda 0.2 --edge_threshold 0.001 --size_threshold_rate 50&
        wait;
        CUDA_VISIBLE_DEVICES=0 python render.py -s $source_path/$NAME --model_path $model_path/run$repeat/$NAME -r 8 --near 10&
        wait;
        CUDA_VISIBLE_DEVICES=0 python metrics.py --model_path $model_path/run$repeat/$NAME&
        wait;
    done
done