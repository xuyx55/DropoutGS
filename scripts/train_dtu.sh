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

# 开始训练和渲染
for repeat in 1
do
    for NAME in scan30 scan34 scan41 scan45 scan82 scan103 scan38 scan21 scan40 scan55 scan63 scan31 scan8 scan110 scan114
    do
        CUDA_VISIBLE_DEVICES=0 python train_dtu.py --dataset DTU -s $source_path/Rectified/$NAME --model_path $model_path/run$repeat/$NAME -r 4 --eval --n_sparse 3 --rand_pcd --iterations 6000 --lambda_dssim 0.6 --densify_grad_threshold 0.001 --prune_threshold 0.01 --densify_until_iter 6000 --percent_dense 0.1 --position_lr_init 0.0016 --position_lr_final 0.000016 --position_lr_max_steps 5500 --position_lr_start 500 --test_iterations 100 1000 2000 3000 4500 6000 --save_iterations 100 500 1000 3000 6000 --error_tolerance 0.01 --opacity_lr 0.05 --scaling_lr 0.003 --shape_pena 0.005 --opa_pena 0.001 --scale_pena 0.005 \
        --dropout --dropout_rate 0.3 --consistency_lambda 0.5 --edge_threshold 0.001 --size_threshold_rate 1&
        wait;

        CUDA_VISIBLE_DEVICES=0 python render.py -s $source_path/Rectified/$NAME --model_path $model_path/run$repeat/$NAME -r 4&
        wait;
        
        bash ./scripts/copy_mask_dtu.sh ${model_path}/run${repeat} ${source_path}/submission_data/idrmasks $NAME

        CUDA_VISIBLE_DEVICES=0 python metrics_dtu.py --model_path $model_path/run$repeat/$NAME&
        wait;
    done
done

# beifen
# for repeat in 1 2 3
# do
#     for NAME in scan30 scan34 scan41 scan45 scan82 scan103 scan38 scan21 scan40 scan55 scan63 scan31 scan8 scan110 scan114
#     do
#         CUDA_VISIBLE_DEVICES=0 python train_dtu_aug_edge_arg.py --dataset DTU -s /home/chenzy85/ws/dataset/dtu/Rectified/$NAME --model_path /home/chenzy85/ws/code/DNGaussian_edge/output/dtu/dtu_0.5_0.1_0.005_1/run$repeat/$NAME -r 4 --eval --n_sparse 3 --rand_pcd --iterations 6000 --lambda_dssim 0.6 --densify_grad_threshold 0.001 --prune_threshold 0.01 --densify_until_iter 6000 --percent_dense 0.1 --position_lr_init 0.0016 --position_lr_final 0.000016 --position_lr_max_steps 5500 --position_lr_start 500 --test_iterations 100 1000 2000 3000 4500 6000 --save_iterations 100 500 1000 3000 6000 --error_tolerance 0.01 --opacity_lr 0.05 --scaling_lr 0.003 --shape_pena 0.005 --opa_pena 0.001 --scale_pena 0.005 \
#         --dropout --dropout_rate 0.5 --consistency_lambda 0.1 --edge_threshold 0.005 --size_threshold_rate 1&
#         wait;
#         CUDA_VISIBLE_DEVICES=0 python render.py -s /home/chenzy85/ws/dataset/dtu/Rectified/$NAME --model_path /home/chenzy85/ws/code/DNGaussian_edge/output/dtu/dtu_0.5_0.1_0.005_1/run$repeat/$NAME -r 4&
#         wait;
#         cp -r /home/chenzy85/ws/code/DNGaussian/output/DTU/baseline/run1/$NAME/mask /home/chenzy85/ws/code/DNGaussian_edge/output/dtu/dtu_0.5_0.1_0.005_1/run$repeat/$NAME/mask&
#         wait;
#         CUDA_VISIBLE_DEVICES=0 python metrics_dtu.py --model_path /home/chenzy85/ws/code/DNGaussian_edge/output/dtu/dtu_0.5_0.1_0.005_1/run$repeat/$NAME&
#         wait;
#     done
# done