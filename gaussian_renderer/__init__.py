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

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from scene.gaussian_model_sh import GaussianModelSH
from utils.sh_utils import eval_sh
from utils.graphics_utils import fov2focal


def render_neural(viewpoint_camera, pc : GaussianModel):
    dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_xyz.shape[0], 1))
    dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
    sigma, color = pc.neural_renderer(pc.get_xyz.cuda(), dir_pp_normalized.cuda())
    # opacity = 1 - torch.exp(-sigma.view(-1, 1))
    opacity = sigma.view(-1, 1)
    return pc.combine_opacity(opacity), color


def mip_scales(viewpoint_camera, pc : GaussianModel):
    dist_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_xyz.shape[0], 1)).norm(dim=1, keepdim=True)
    focal = fov2focal(viewpoint_camera.FoVx, viewpoint_camera.image_width)
    return dist_pp / focal 



def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, inference=False, dropout=False, dropout_rate=0.3, edge = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    # default
    if edge is None:
        # edge = torch.zeros(viewpoint_camera.original_image.shape[1:]).cuda()
        edge = torch.zeros(int(viewpoint_camera.image_height), int(viewpoint_camera.image_width)).cuda()
        
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    dropout_screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
        dropout_screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    # opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None

    # pre
    opacity, colors_precomp = render_neural(viewpoint_camera, pc)
    if inference:
        opacity = pc.get_opacity_

    # sh
    # opacity = pc.get_opacity
    # shs = pc.get_features

    # Ashawkey version
    rendered_image, radii, rendered_depth, rendered_alpha, pixels, imgrad_pixels  = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
        edge = edge
    )
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return_dict = {"render": rendered_image,
            "depth": rendered_depth, 
            "alpha": rendered_alpha,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "opacity": opacity,
            "color": colors_precomp,
            "pixels": pixels,
            "imgrad_pixels": imgrad_pixels}
    
    dropout_means2D = dropout_screenspace_points
    if dropout == True:
        mask = (torch.rand((means3D.shape[0],)) >= dropout_rate).cuda()
        dropout_rendered_image, dropout_radii, _, _, _, _ = rasterizer(
            means3D = means3D*mask[...,None],
            means2D = dropout_means2D,
            shs = shs,
            colors_precomp = colors_precomp*mask[...,None],
            opacities = opacity*mask[...,None],
            scales = scales*mask[...,None],
            rotations = rotations*mask[...,None],
            cov3D_precomp = cov3D_precomp,
            edge = torch.zeros(int(viewpoint_camera.image_height), int(viewpoint_camera.image_width)).cuda())
        return_dict.update({'dropout_render':dropout_rendered_image, 'dropout_viewspace_points':dropout_screenspace_points, 'dropout_radii':dropout_radii, 'mask':mask})

    return return_dict




def render_for_depth(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, value=0.95, edge=None):
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # default
    if edge is None:
        # edge = torch.zeros(viewpoint_camera.original_image.shape[1:]).cuda()
        edge = torch.zeros(int(viewpoint_camera.image_height), int(viewpoint_camera.image_width)).cuda()
        
    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    # opacity = pc.get_opacity
    opacity = torch.ones(pc.get_xyz.shape[0], 1, device=pc.get_xyz.device) * value

    with torch.no_grad():
        scales = None
        rotations = None
        cov3D_precomp = None
        if pipe.compute_cov3D_python:
            cov3D_precomp = pc.get_covariance(scaling_modifier)
        else:
            scales = pc.get_scaling.detach()
            rotations = pc.get_rotation.detach()

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = torch.ones_like(pc.get_xyz)
        

    # Ashawkey version
    rendered_image, radii, rendered_depth, rendered_alpha, _, _ = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp, 
        edge=edge
    )


    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "depth": rendered_depth, 
            "alpha": rendered_alpha,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}



def render_for_opa(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, edge=None):
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # default
    if edge is None:
        # edge = torch.zeros(viewpoint_camera.original_image.shape[1:]).cuda()
        edge = torch.zeros(int(viewpoint_camera.image_height), int(viewpoint_camera.image_width)).cuda()

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz.detach()
    means2D = screenspace_points
    opacity = pc.get_opacity

    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling.detach()
        rotations = pc.get_rotation.detach()

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = torch.ones_like(pc.get_xyz)
        

    # Ashawkey version
    rendered_image, radii, rendered_depth, rendered_alpha, _, _ = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
        edge=edge
    )


    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "depth": rendered_depth, 
            "alpha": rendered_alpha,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "opacity": opacity}




#----------- for SH




def render_sh(viewpoint_camera, pc : GaussianModelSH, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, inference=False, dropout=False, dropout_rate=0.3, edge=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    dropout_screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
        dropout_screenspace_points.retain_grad()
    except:
        pass

    # default
    if edge is None:
        # edge = torch.zeros(viewpoint_camera.original_image.shape[1:]).cuda()
        edge = torch.zeros(int(viewpoint_camera.image_height), int(viewpoint_camera.image_width)).cuda()
        
    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    # opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None

    # sh
    opacity = pc.get_opacity
    shs = pc.get_features

    # Ashawkey version
    rendered_image, radii, rendered_depth, rendered_alpha, pixels, imgrad_pixels = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
        edge=edge
    )


    shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
    dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
    dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
    sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
    color = torch.clamp_min(sh2rgb + 0.5, 0.0)


    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return_dict = {"render": rendered_image,
            "depth": rendered_depth, 
            "alpha": rendered_alpha,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "opacity": opacity,
            "color": color,
            "pixels": pixels,
            "imgrad_pixels": imgrad_pixels}

    dropout_means2D = dropout_screenspace_points
    if dropout == True:
        mask = (torch.rand((means3D.shape[0],)) >= dropout_rate).cuda()
        dropout_rendered_image, dropout_radii, _, _, _, _ = rasterizer(
            means3D = means3D*mask[...,None],
            means2D = dropout_means2D,
            shs = shs*mask[..., None, None],
            colors_precomp = colors_precomp,
            opacities = opacity*mask[...,None]*1/(1-dropout_rate),
            scales = scales*mask[...,None],
            rotations = rotations*mask[...,None],
            cov3D_precomp = cov3D_precomp,
            # edge=torch.zeros(viewpoint_camera.original_image.shape[1:]).cuda())
            edge=torch.zeros(int(viewpoint_camera.image_height), int(viewpoint_camera.image_width)).cuda())
        
        return_dict.update({'disturb_render':dropout_rendered_image, 'disturb_viewspace_points':dropout_screenspace_points, 'disturb_radii':dropout_radii, 'mask':mask})

    return return_dict



def render_for_depth_sh(viewpoint_camera, pc : GaussianModelSH, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, value=0.95, edge=None):
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # default
    if edge is None:
        # edge = torch.zeros(viewpoint_camera.original_image.shape[1:]).cuda()
        edge = torch.zeros(int(viewpoint_camera.image_height), int(viewpoint_camera.image_width)).cuda()
        
    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    # opacity = pc.get_opacity
    opacity = torch.ones(pc.get_xyz.shape[0], 1, device=pc.get_xyz.device) * value

    with torch.no_grad():
        scales = None
        rotations = None
        cov3D_precomp = None
        if pipe.compute_cov3D_python:
            cov3D_precomp = pc.get_covariance(scaling_modifier)
        else:
            scales = pc.get_scaling.detach()
            rotations = pc.get_rotation.detach()

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = torch.ones_like(pc.get_xyz)
        

    # Ashawkey version
    rendered_image, radii, rendered_depth, rendered_alpha, _, _ = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
        edge=edge
    )


    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "depth": rendered_depth, 
            "alpha": rendered_alpha,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}



def render_for_opa_sh(viewpoint_camera, pc : GaussianModelSH, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, edge=None):
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # default
    if edge is None:
        # edge = torch.zeros(viewpoint_camera.original_image.shape[1:]).cuda()
        edge = torch.zeros(int(viewpoint_camera.image_height), int(viewpoint_camera.image_width)).cuda()
        
    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz.detach()
    means2D = screenspace_points
    opacity = pc.get_opacity

    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling.detach()
        rotations = pc.get_rotation.detach()

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = torch.ones_like(pc.get_xyz)
        

    # Ashawkey version
    rendered_image, radii, rendered_depth, rendered_alpha, _, _ = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
        edge=edge
    )


    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "depth": rendered_depth, 
            "alpha": rendered_alpha,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "opacity": opacity}


