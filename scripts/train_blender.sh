# # drums & materials  0.5mask+0.3lambda drums, 0.3mask+0.1lambda materials
# # ship lego ficus hotdog   0.3mask_0.2lambda
# # chair 0.3mask_0.2lambda
# # mic 0.2mask_0.2lambda

# v2
# scenes with no sh (materials drums) 0.5lambda 0.1mask no edge
# lego fiucs 0.1lambda 0.1mask 0.001edge+1size
# hotdog 0.5lambda 0.1 mask 0.01edge+1size
# ship 0.2lambda 0.2 mask 0.001edge+1size
# mic 0.2lambda 0.2mask 0.01edge+1size
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

for repeat in 1
do
    for NAME in materials drums
    do
        CUDA_VISIBLE_DEVICES=0 python train_blender.py -s $source_path/$NAME --model_path $model_path/run$repeat/$NAME -r 2 --eval --n_sparse 8 --rand_pcd --iterations 6000 --lambda_dssim 0.6 --white_background \
            --densify_grad_threshold 0.001 --prune_threshold 0.01 --densify_until_iter 6000 --percent_dense 0.01 \
            --densify_from_iter 500 \
            --position_lr_init 0.00016 --position_lr_final 0.0000016 --position_lr_max_steps 1000 --position_lr_start 5000 \
            --test_iterations 1000 2000 3000 4500 6000 --save_iterations 1000 2000 3000 6000 \
            --hard_depth_start 0 --soft_depth_start 9999999 \
            --split_opacity_thresh 0.1 --error_tolerance 0.001 \
            --scaling_lr 0.005 \
            --shape_pena 0.000 --opa_pena 0.000 --scale_pena 0.000 \
            --dropout --dropout_rate 0.1 --consistency_lambda 0.5 --edge_threshold 1000 --size_threshold_rate 1.0&
        wait;
        python render.py -s $source_path/$NAME --model_path $model_path/run$repeat/$NAME -r 2 &
        wait;
        python metrics.py --model_path $model_path/run$repeat/$NAME &
        wait;
    done
done &
wait;

for repeat in 1
do
    for NAME in lego ficus
    do
        CUDA_VISIBLE_DEVICES=0 python train_blender.py -s $source_path/$NAME --model_path $model_path/run$repeat/$NAME -r 2 --eval --n_sparse 8 --rand_pcd --iterations 6000 --lambda_dssim 0.2 --white_background \
            --densify_grad_threshold 0.0002 --prune_threshold 0.005 --densify_until_iter 6000 --percent_dense 0.01 \
            --densify_from_iter 500 \
            --position_lr_init 0.00016 --position_lr_final 0.0000016 --position_lr_max_steps 1000 --position_lr_start 5000 \
            --test_iterations 1000 2000 3000 4500 6000 --save_iterations 1000 2000 3000 6000 \
            --hard_depth_start 0 \
            --error_tolerance 0.01 \
            --scaling_lr 0.005 \
            --shape_pena 0.000 --opa_pena 0.000 --scale_pena 0.000 \
            --use_SH \
            --dropout --dropout_rate 0.1 --consistency_lambda 0.1 --edge_threshold 0.001 --size_threshold_rate 1.0&
        wait;
        python render_sh.py -s $source_path/$NAME --model_path $model_path/run$repeat/$NAME -r 2 &
        wait;
        python metrics.py --model_path $model_path/run$repeat/$NAME &
        wait;
    done
done &
wait;

for repeat in 1
do
    for NAME in hotdog
    do
        CUDA_VISIBLE_DEVICES=0 python train_blender.py -s $source_path/$NAME --model_path $model_path/run$repeat/$NAME -r 2 --eval --n_sparse 8 --rand_pcd --iterations 6000 --lambda_dssim 0.2 --white_background \
            --densify_grad_threshold 0.0002 --prune_threshold 0.005 --densify_until_iter 6000 --percent_dense 0.01 \
            --densify_from_iter 500 \
            --position_lr_init 0.00016 --position_lr_final 0.0000016 --position_lr_max_steps 1000 --position_lr_start 5000 \
            --test_iterations 1000 2000 3000 4500 6000 --save_iterations 1000 2000 3000 6000 \
            --hard_depth_start 0 \
            --error_tolerance 0.01 \
            --scaling_lr 0.005 \
            --shape_pena 0.000 --opa_pena 0.000 --scale_pena 0.000 \
            --use_SH \
            --dropout --dropout_rate 0.1 --consistency_lambda 0.5 --edge_threshold 0.01 --size_threshold_rate 1.0&
        wait;
        python render_sh.py -s $source_path/$NAME --model_path $model_path/run$repeat/$NAME -r 2 &
        wait;
        python metrics.py --model_path $model_path/run$repeat/$NAME &
        wait;
    done
done &
wait;


for repeat in 1
do
    for NAME in ship
    do
        CUDA_VISIBLE_DEVICES=0 python train_blender.py -s $source_path/$NAME --model_path $model_path/run$repeat/$NAME -r 2 --eval --n_sparse 8 --rand_pcd --iterations 6000 --lambda_dssim 0.2 --white_background \
            --densify_grad_threshold 0.0002 --prune_threshold 0.005 --densify_until_iter 6000 --percent_dense 0.01 \
            --densify_from_iter 500 \
            --position_lr_init 0.00016 --position_lr_final 0.0000016 --position_lr_max_steps 1000 --position_lr_start 5000 \
            --test_iterations 1000 2000 3000 4500 6000 --save_iterations 1000 2000 3000 6000 \
            --hard_depth_start 0 \
            --error_tolerance 0.01 \
            --scaling_lr 0.005 \
            --shape_pena 0.000 --opa_pena 0.000 --scale_pena 0.000 \
            --use_SH \
            --dropout --dropout_rate 0.2 --consistency_lambda 0.2 --edge_threshold 0.001 --size_threshold_rate 1.0&
        wait;
        python render_sh.py -s $source_path/$NAME --model_path $model_path/run$repeat/$NAME -r 2 &
        wait;
        python metrics.py --model_path $model_path/run$repeat/$NAME &
        wait;
    done
done &
wait;

# for repeat in 1
# do
#     for NAME in mic chair
#     do
#         CUDA_VISIBLE_DEVICES=0 python train_blender.py -s $source_path/$NAME --model_path $model_path/run$repeat/$NAME -r 2 --eval --n_sparse 8 --rand_pcd --iterations 30000 --lambda_dssim 0.2 --white_background \
#             --densify_grad_threshold 0.0002 --prune_threshold 0.005 --densify_until_iter 15000 --percent_dense 0.01 \
#             --densify_from_iter 500 \
#             --position_lr_init 0.00016 --position_lr_final 0.0000016 --position_lr_max_steps 30000 --position_lr_start 0 \
#             --test_iterations 1000 2000 3000 4500 6000 --save_iterations 1000 2000 3000 6000 \
#             --hard_depth_start 99999 \
#             --error_tolerance 0.2 \
#             --scaling_lr 0.005 \
#             --shape_pena 0.000 --opa_pena 0.000 --scale_pena 0.000 \
#             --use_SH \
#             --dropout --dropout_rate 0.2 --consistency_lambda 0.2 --edge_threshold 0.01 --size_threshold_rate 1.0&
#         wait;
#         python render_sh.py -s $source_path/$NAME --model_path $model_path/run$repeat/$NAME -r 2 &
#         wait;
#         python metrics.py --model_path $model_path/run$repeat/$NAME &
#         wait;
#     done
# done &
# wait;