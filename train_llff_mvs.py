#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import time
import torch
import torchvision
from os import makedirs
from random import randint
from utils.graphics_utils import fov2focal
from utils.loss_utils import l1_loss, loss_depth_smoothness, patch_norm_mse_loss, patch_norm_mse_loss_global, ssim
# from utils.loss_utils import mssim as ssim
from gaussian_renderer import render, render_for_depth, render_for_opa  # , network_gui
import sys
from scene import RenderScene, Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import random
import torch.nn.functional as F
try:
    from torch.utils.tensorboard import SummaryWriter
    print('Launch TensorBoard')
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def edge_aware_normal_loss(I):
    '''
	borrow from AtomGS
    '''
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).float().unsqueeze(0).unsqueeze(0).to(I.device)/4
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).float().unsqueeze(0).unsqueeze(0).to(I.device)/4

    dI_dx = torch.cat([F.conv2d(I[i].unsqueeze(0), sobel_x) for i in range(I.shape[0])])
    dI_dx = torch.mean(torch.abs(dI_dx), 0, keepdim=True)
    dI_dy = torch.cat([F.conv2d(I[i].unsqueeze(0), sobel_y) for i in range(I.shape[0])])
    dI_dy = torch.mean(torch.abs(dI_dy), 0, keepdim=True)

    weights_x = (dI_dx)
    weights_y = (dI_dy)

    loss_x = weights_x
    loss_y = weights_y
    loss = (loss_x + loss_y).norm(dim=0, keepdim=True)

    loss = torch.nn.functional.interpolate(loss.unsqueeze(0), (I.shape[1], I.shape[2]))
    return loss[0]
def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, near_range):
    start_time = time.time()
    init_memory = torch.cuda.memory_allocated()
    print(init_memory)
    
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset, opt)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    scene_sprical = RenderScene(dataset, gaussians, spiral=True)
    gaussians.training_setup(opt)
    if checkpoint:
        # (model_params, first_iter) = torch.load(checkpoint)
        # gaussians.restore(model_params, opt)
        (model_params, _) = torch.load(checkpoint)
        gaussians.load_shape(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    viewpoint_sprical_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress", ascii=True, dynamic_ncols=True)
    first_iter += 1

    patch_range = (5, 17) # LLFF

    time_accum = 0

    for iteration in range(first_iter, opt.iterations + 1):        

        iter_start.record()

        gaussians.update_learning_rate(max(iteration - opt.position_lr_start, 0))

        # Every 1000 its we increase the levels of SH up to a maximum degree
        # if iteration % 1000 == 0:
        #     gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        if not viewpoint_sprical_stack:
            viewpoint_sprical_stack = scene_sprical.getRenderCameras().copy()


        gt_image = viewpoint_cam.original_image.cuda()

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        # -------------------------------------------------- DEPTH --------------------------------------------
        if iteration > opt.hard_depth_start:
            render_pkg = render_for_depth(viewpoint_cam, gaussians, pipe, background)
            depth = render_pkg["depth"]

            # Depth loss
            loss_hard = 0
            depth_mono = 255.0 - viewpoint_cam.depth_mono

            loss_l2_dpt = patch_norm_mse_loss(depth[None,...], depth_mono[None,...], randint(patch_range[0], patch_range[1]), opt.error_tolerance)
            loss_hard += 0.1 * loss_l2_dpt # 0.1

            
            if iteration > 3000:
                loss_hard += 0.1 * loss_depth_smoothness(depth[None, ...], depth_mono[None, ...])   # 0.1

            loss_global = patch_norm_mse_loss_global(depth[None,...], depth_mono[None,...], randint(patch_range[0], patch_range[1]), opt.error_tolerance)
            loss_hard += 0.1 * loss_global    # 1

            loss_hard.backward()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

        # -------------------------------------------------- pnt --------------------------------------------
        if iteration > opt.soft_depth_start :
            render_pkg = render_for_opa(viewpoint_cam, gaussians, pipe, background)
            viewspace_point_tensor, visibility_filter = render_pkg["viewspace_points"], render_pkg["visibility_filter"]
            depth, alpha = render_pkg["depth"], render_pkg["alpha"]

            # Depth loss
            loss_pnt = 0
            depth_mono = 255.0 - viewpoint_cam.depth_mono

            loss_l2_dpt = patch_norm_mse_loss(depth[None,...], depth_mono[None,...], randint(patch_range[0], patch_range[1]), opt.error_tolerance)
            loss_pnt += 0.1 * loss_l2_dpt  # 0.1

            if iteration > 3000:
                loss_pnt += 0.1 * loss_depth_smoothness(depth[None, ...], depth_mono[None, ...])   # 0.1

            loss_global = patch_norm_mse_loss_global(depth[None,...], depth_mono[None,...], randint(patch_range[0], patch_range[1]), opt.error_tolerance)
            loss_pnt += 0.1 * loss_global   # 1

            loss_pnt.backward()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
        

        # ---------------------------------------------- Photometric --------------------------------------------
        edge = edge_aware_normal_loss(viewpoint_cam.original_image)
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, edge = edge[0], dropout=args.dropout, dropout_rate = args.dropout_rate)       
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        # depth
        depth, opacity, alpha = render_pkg["depth"], render_pkg["opacity"], render_pkg['alpha']  # [visibility_filter]

        # Loss
        Ll1 = l1_loss(image, gt_image)
        loss = Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))


        # Reg
        loss_reg = torch.tensor(0., device=loss.device)
        shape_pena = (gaussians.get_scaling.max(dim=1).values / gaussians.get_scaling.min(dim=1).values).mean()
        scale_pena = ((gaussians.get_scaling.max(dim=1, keepdim=True).values)**2).mean()
        opa_pena = 1 - (opacity[opacity > 0.2]**2).mean() + ((1 - opacity[opacity < 0.2])**2).mean()

        loss_reg += opt.shape_pena*shape_pena + opt.scale_pena*scale_pena + opt.opa_pena*opa_pena
        loss += loss_reg * 0.1

        # consistency 
        if args.dropout:
            consistency_L1_loss = l1_loss(render_pkg['render'], render_pkg['dropout_render']) 
            consistency_loss = (1.0 - opt.lambda_dssim) * consistency_L1_loss + opt.lambda_dssim * (1.0 - ssim(render_pkg['render'], render_pkg['dropout_render']))
            loss += args.consistency_lambda * consistency_loss
        
        loss.backward()

        # ================================================================================

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            if not loss.isnan():
                ema_loss_for_log = 0.4 * (loss.item()) + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "MEM":f"{total_memory/(iteration - first_iter + 1)}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            clean_iterations = testing_iterations + [first_iter]
            clean_views(iteration, clean_iterations, scene, gaussians, pipe, background)
            time_accum += iter_start.elapsed_time(iter_end)
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration, render(viewpoint_cam, gaussians, pipe, background)["color"])

            # Densification
            if iteration < opt.densify_until_iter and iteration not in clean_iterations:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, render_pkg["pixels"], render_pkg["imgrad_pixels"])

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:  
                    size_threshold = max_dist = None
                            
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.prune_threshold, scene.cameras_extent, size_threshold, opt.split_opacity_thresh, max_dist, edge_threshold=args.edge_threshold, size_threshold_rate=args.size_threshold_rate)
                
                # if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                #     gaussians.reset_opacity()

            if (iteration - 1) % 25 == 0:
                viewpoint_sprical_cam = viewpoint_sprical_stack.pop(0)
                mask_near = None
                if iteration > 2000:
                    for idx, view in enumerate(scene_sprical.getRenderCameras().copy()):
                        mask_temp = (gaussians.get_xyz - view.camera_center.repeat(gaussians.get_xyz.shape[0], 1)).norm(dim=1, keepdim=True) < near_range
                        mask_near = mask_near + mask_temp if mask_near is not None else mask_temp
                    gaussians.prune_points(mask_near.squeeze())


                ## render process
                # if (iteration + 25) > (opt.iterations):
                #     while viewpoint_sprical_stack:
                #         render_one_step(iteration, time_accum / 1000, dataset, viewpoint_sprical_cam, gaussians, render, (pipe, background), save=False)
                #         iteration += 1
                #         viewpoint_sprical_cam = viewpoint_sprical_stack.pop(0)
                # render_one_step(iteration, time_accum / 1000, dataset, viewpoint_sprical_cam, gaussians, render, (pipe, background), save=((iteration + 25) > (opt.iterations)))


            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
            if iteration == opt.iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt_latest.pth")
            

def prepare_output_and_logger(args, opt):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
    with open(os.path.join(args.model_path, "opt_args"), 'w') as opt_log_f:
        opt_log_f.write(str(Namespace(**vars(opt))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


@torch.no_grad()
def clean_views(iteration, test_iterations, scene, gaussians, pipe, background):
    if iteration in test_iterations:
        visible_pnts = None
        for viewpoint_cam in scene.getTrainCameras().copy():
            render_pkg = render(viewpoint_cam, gaussians, pipe, background)
            visibility_filter = render_pkg["visibility_filter"]
            if visible_pnts is None:
                visible_pnts = visibility_filter
            visible_pnts += visibility_filter
        unvisible_pnts = ~visible_pnts
        gaussians.prune_points(unvisible_pnts)


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, depth_loss=torch.tensor(0), reg_loss=torch.tensor(0)):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('train_loss_patches/depth_kl_loss', depth_loss.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/reg_loss', reg_loss.item(), iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'eval', 'cameras' : scene.getEvalCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_results = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_results["render"], 0.0, 1.0)
                    depth = render_results["depth"]
                    depth = 1 - (depth - depth.min()) / (depth.max() - depth.min())
                    alpha = render_results["alpha"]
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    bg_mask = (gt_image.max(0, keepdim=True).values < 30/255)
                    bg_mask_clone = bg_mask.clone()
                    for i in range(1, 50):
                        bg_mask[:, i:] *= bg_mask_clone[:, :-i]
                    white_mask = (gt_image.min(0, keepdim=True).values > 240/255)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}_alpha/alpha".format(viewpoint.image_name), alpha[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}_alpha/mask".format(viewpoint.image_name), bg_mask[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}_alpha/white_mask".format(viewpoint.image_name), white_mask[None], global_step=iteration)

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()



def render_one_step(iteration, time, dataset, viewpoint, gaussians, renderFunc, renderArgs, save=False):
    torch.cuda.empty_cache()
    time_path = os.path.join(dataset.model_path, 'time')
    makedirs(time_path, exist_ok=True)
    render_results = renderFunc(viewpoint, gaussians, *renderArgs)
    image = torch.clamp(render_results["render"], 0.0, 1.0)
    torchvision.utils.save_image(image, os.path.join(time_path, '{0:05d}'.format(iteration) + ".png"))
    
    import matplotlib.font_manager as fm # to create font
    from PIL import Image, ImageDraw, ImageFont  
    
    img = Image.open(os.path.join(time_path, '{0:05d}'.format(iteration) + ".png"))  
    draw = ImageDraw.Draw(img)  
    font = ImageFont.truetype(fm.findfont(fm.FontProperties(family='DejaVu Sans')), 30)  
    text = format(time, '.2f') + ' s'
    x = 10
    y = 0 
    draw.text((x-1, y), text, font=font, fill='black')  
    draw.text((x+1, y), text, font=font, fill='black')  
    draw.text((x, y-1), text, font=font, fill='black')  
    draw.text((x, y+1), text, font=font, fill='black')  
    draw.text((x-1, y-1), text, font=font, fill='black')  
    draw.text((x+1, y-1), text, font=font, fill='black')  
    draw.text((x-1, y+1), text, font=font, fill='black')  
    draw.text((x+1, y+1), text, font=font, fill='black')  
    draw.text((x, y), text, font=font, fill='white')

    img.save(os.path.join(time_path, '{0:05d}'.format(iteration) + ".png"))


    torch.cuda.empty_cache()
    if save:
        # os.system(f"ffmpeg -i " + time_path + f"/%5d.png -q 2 " + dataset.model_path + "/out_time_{}.mp4 -y".format(dataset.model_path.split('/')[-1]))
        os.system(f'ffmpeg -f image2 -pattern_type glob -i "' + time_path + f'/*.png" -q 2 ' + dataset.model_path + "/out_time_{}.mp4 -y".format(dataset.model_path.split('/')[-1]))





if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    # parser.add_argument('--ip', type=str, default="127.0.0.1")
    # parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])   # 2000, 4000, 5000, 6000
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[6000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--near", type=int, default=0)
    parser.add_argument("--dropout", action="store_true")
    parser.add_argument("--dropout_rate", type=float, default=0.0)
    parser.add_argument("--edge_threshold", type=float, default=1000.0)
    parser.add_argument("--size_threshold_rate", type=float, default=1000.0)
    parser.add_argument("--consistency_lambda", type=float, default=0.0)

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    # args.checkpoint_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.near)

    # All done
    print("\nTraining complete.")
