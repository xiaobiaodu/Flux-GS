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
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from diff_gaussian_rasterization_fluxgs import GaussianRasterizationSettings, GaussianRasterizer

def render_fluxgs(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, mult, scaling_modifier = 1.0, override_color = None, get_flag=None, metric_map = None, render_size=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    # screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    screenspace_points = torch.zeros((pc.get_xyz.shape[0], 4), dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    if metric_map==None:
        metric_map=torch.zeros(int(viewpoint_camera.image_height)*int(viewpoint_camera.image_width), dtype=torch.int, device='cuda')

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height) if render_size is None else render_size[0],
        image_width=int(viewpoint_camera.image_width) if render_size is None else render_size[1],
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        mult = mult,
        prefiltered=False,
        debug=pipe.debug,
        get_flag=get_flag,
        metric_map = metric_map
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None

    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    elif pc.vq_enabled:
        scales = pc.get_svq_scale
        rotations = pc.get_svq_rotation
        # opacity = pc.get_svq_opacity
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation
        opacity = pc.get_opacity

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            
            if pc.net_enabled:
                cont_feature = pc.mlp_cont(pc.contract_to_unisphere(means3D.clone().detach(), torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], device='cuda')))
                if pc.vq_enabled:
                    app_feature = pc.get_svq_appearance
                    space_feature = torch.cat([cont_feature, app_feature[:,0:3]],dim=-1)
                    view_feature = torch.cat([cont_feature, app_feature[:,3:6]],dim=-1)
                else:
                    space_feature = torch.cat([cont_feature, pc._features_static],dim=-1)
                    view_feature = torch.cat([cont_feature, pc._features_view],dim=-1)
                shs = pc.mlp_view(view_feature).reshape(-1,pc.max_sh_rest,3).float()
                dc = pc.mlp_dc(space_feature).reshape(-1,1,3).float()
                opacity = pc.opacity_activation(pc.mlp_opacity(space_feature).float())
                shs = torch.cat([dc, shs], dim=1)
                shs = shs +  pc.get_features_offset(shs, opacity)

            elif pc.shoffset_enabled:
                shs = pc.get_features 
                shs = shs + pc.get_features_offset(shs, opacity)
            else:
                shs = pc.get_features 

   
    else:
        colors_precomp = override_color


  
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, accum_metric_counts = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : (radii > 0).nonzero(),
            "radii": radii,
            "accum_metric_counts" : accum_metric_counts}