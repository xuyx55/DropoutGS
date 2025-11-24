# DropoutGS: Dropping Out Gaussians for Better Sparse-view Rendering

This is the official repository for our CVPR 2025 paper **DropoutGS: Dropping Out Gaussians for Better Sparse-view Rendering.**

[Arxiv](https://arxiv.org/abs/2504.09491) |[ Paper](assets/paper.pdf) | [Project](https://xuyx55.github.io/DropoutGS/)

![image](assets/main.svg)

## Installation

Tested on Ubuntu 18.04, CUDA 11.3, PyTorch 1.12.1

``````
conda env create --file environment.yml
conda activate dropoutgs

cd submodules
git clone https://gitlab.inria.fr/bkerbl/simple-knn.git
pip install ./diff-gaussian-rasterization ./simple-knn
``````

If encountering installation problem of the `diff-gaussian-rasterization` or `gridencoder`, you may get some help from [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) and [torch-ngp](https://github.com/ashawkey/torch-ngp).

## Evaluation

### LLFF

1. Download LLFF from [the official download link](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1).
2. Generate monocular depths by DPT:

   ```bash
   cd dpt
   python get_depth_map_for_llff_dtu.py --root_path $<dataset_path_for_llff> --benchmark LLFF
   ```
3. Start training and testing:

   ```bash
   # for example
   bash scripts/train_llff.sh $<dataset_path_for_llff_scene> output/llff/$<llff_scene>
   ```

### DTU

1. Download DTU dataset

   - Download the DTU dataset "Rectified (123 GB)" from the [official website](https://roboimagedata.compute.dtu.dk/?page_id=36/), and extract it.
   - Download masks (used for evaluation only) from [this link](https://drive.google.com/file/d/1d4BglbLvsgskyetUDRUb3miOfpZQq_kV/view?usp=sharing).
2. Organize DTU for few-shot setting

   ```bash
   bash scripts/organize_dtu_dataset.sh $rectified_path
   ```
3. Format

   - Poses: following [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting), run `convert.py` to get the poses and the undistorted images by COLMAP.
   - Render Path: following [LLFF](https://github.com/Fyusion/LLFF) to get the `poses_bounds.npy` from the COLMAP data. (Optional)
4. Generate monocular depths by DPT:

   ```bash
   cd dpt
   python get_depth_map_for_llff_dtu.py --root_path $<dataset_path_for_dtu> --benchmark DTU
   ```
5. Set the mask path and the expected output model path in `copy_mask_dtu.sh` for evaluation. (default: "data/dtu/submission_data/idrmasks" and "output/dtu")
6. Start training and testing:

   ```bash
   # for example
   bash scripts/train_dtu.sh $<dataset_path_for_dtu_scene> output/dtu/$<dtu_scan>
   ```

### Blender

1. Download the NeRF Synthetic dataset from [here](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1?usp=sharing).
2. Generate monocular depths by DPT:

   ```bash
   cd dpt
   python get_depth_map_for_blender.py --root_path $<dataset_path_for_blender>
   ```
3. Start training and testing:

   ```bash
   # for example
   # there are some special settings for different scenes in the Blender dataset, please refer to "run_blender.sh".
   bash scripts/train_blender.sh $<dataset_path_for_blender_scene> output/blender/$<blender_scene>
   ```

## Reproducing Results

Due to the randomness of the densification process and random initialization, the metrics may be unstable in some scenes, especially PSNR.

### MVS Point Cloud Initialization

If more stable performance is needed, we recommend trying the dense initialization from [FSGS](https://github.com/VITA-Group/FSGS).

Here we provide an example script for LLFF:

```bash
# Following FSGS to get the "data/llff/$<scene>/3_views/dense/fused.ply" first
bash scripts/train_llff_mvs.sh $<dataset_path_for_llff_scene> output/llff_dense/$<llff_scene>
```

## Customized Dataset

Similar to Gaussian Splatting, our method can read standard COLMAP format datasets. Please customize your sampling rule in `scenes/dataset_readers.py`, and see how to organize a COLMAP-format dataset from raw RGB images referring to our preprocessing of DTU.

## Citation

Consider citing as below if you find this repository helpful to your project:

```
@article{xu2025dropoutgs,
      title={DropoutGS: Dropping Out Gaussians for Better Sparse-view Rendering},
      author={Xu, Yexing and Wang, Longguang and Chen, Minglin and Ao, Sheng and Li, Li and Guo, Yulan},
      journal={arXiv preprint arXiv:2504.09491},
      year={2025}
    }
```

## Acknowledgement

This code is developed on [DNGaussian](https://github.com/Fictionarry/DNGaussian). Thanks for the great project and the friendly authors!
