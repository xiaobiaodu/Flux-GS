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
import numpy as np
import os, random, time
from random import randint
from lpipsPyTorch import lpips
from utils.loss_utils import l1_loss
from fused_ssim import fused_ssim as fast_ssim
from gaussian_renderer import render_fluxgs, network_gui_ws
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

from utils.fast_utils import compute_gaussian_score_mobilegs2, sampling_cameras, sample_cameras_stratified, compute_gaussian_pruning_mobilegs2
from utils.compress_utils import save_comp, write_storage, save_comp_web
from utils.sh_utils import mc_project_sh_rgb, project_sh_mc
from torch import nn


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, websockets):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))

    # record time
    optim_start = torch.cuda.Event(enable_timing=True)
    optim_end = torch.cuda.Event(enable_timing=True)
    total_time = 0.0

    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    bg = torch.rand((3), device="cuda") if opt.random_background else background

    for iteration in range(first_iter, opt.iterations + 1):

        if websockets:
            if network_gui_ws.curr_id >= 0 and network_gui_ws.curr_id < len(scene.getTrainCameras()):
                cam = scene.getTrainCameras()[network_gui_ws.curr_id]
                net_image = render_fluxgs(cam, gaussians, pipe, background, opt.mult, 1.0)["render"]
                network_gui_ws.latest_width = cam.image_width
                network_gui_ws.latest_height = cam.image_height
                network_gui_ws.latest_result = net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())

        iter_start.record()
        
        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        # if iteration % 1000 == 0:
        #     gaussians.oneupSHdegree()

        if iteration == args.svq_itr:
            gaussians.apply_svq(args)


        loss_mv = 0

        for _ in range(pipe.mv):

            # Pick a random Camera
            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras().copy()
                viewpoint_indices = list(range(len(viewpoint_stack)))
            rand_idx = randint(0, len(viewpoint_indices) - 1)
            viewpoint_cam = viewpoint_stack.pop(rand_idx)
            _ = viewpoint_indices.pop(rand_idx)


        


            # Render
            if (iteration - 1) == debug_from:
                pipe.debug = True

            render_scale = None
            # if iteration < 2000:
            #     render_scale = 8
            # elif iteration < 5000:
            #     render_scale = 4
            # elif iteration < 10000:
            #     render_scale = 2
            

            gt_image = viewpoint_cam.original_image.cuda()
            if render_scale is not None:
                gt_image = torch.nn.functional.interpolate(gt_image[None], scale_factor=1/render_scale, mode="bilinear", 
                                                        recompute_scale_factor=True, antialias=True)[0]
                

            render_pkg = render_fluxgs(viewpoint_cam, gaussians, pipe, bg, opt.mult, render_size=gt_image.shape[-2:])
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

            # Loss
            Ll1 = l1_loss(image, gt_image)
            ssim_value = fast_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

            loss_mv += loss
        loss_mv.backward()

        iter_end.record()


        # if iteration > 1000:
        #     print("dc grad is None:", gaussians._features_dc.grad is None)
        #     print("dc grad mean:", 
        #         None if gaussians._features_dc.grad is None 
        #         else gaussians._features_dc.grad.abs().mean().item())


        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            iter_time = iter_start.elapsed_time(iter_end)
            # Log and save
            # if iteration % 5000 == 0 and iteration >= 30000:
            #     print(len(gaussians.get_xyz))
            #     training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_time, testing_iterations, scene, render_fluxgs, (pipe, background, opt.mult))
            
            optim_start.record()
            




            if iteration == opt.iterations:
                print("before quantization")
                training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_time, testing_iterations, scene, render_fluxgs, (pipe, background, opt.mult))

                save_dict = gaussians.encode()     
                save_comp_web(scene.model_path + "/comp.json", save_dict)
                
                actual_storage = os.path.getsize(scene.model_path + "/comp.json")
                with open(scene.model_path + "/storage.txt", 'w') as f:  
                    byte = {'xyz': 0, 'scale':0, 'rotation':0, 'app':0, 'MLPs':0, 'opacity':0}
                    f.write(write_storage(save_dict, byte, gaussians.get_xyz.shape[0]))
                    f.write("Actual storage: " + str(round(actual_storage/2**20, 2)) + " MB")
                gaussians.decode(save_dict, decompress=True)
                
                print("after quantization")


            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    my_viewpoint_stack = scene.getTrainCameras().copy()
                    camlist = sample_cameras_stratified(my_viewpoint_stack)

                    importance_score, pruning_score = compute_gaussian_score_mobilegs2(camlist, gaussians, pipe, bg, opt, DENSIFY=True)                    
                    gaussians.densify_and_prune_mobilegs2(max_screen_size = size_threshold, 
                                                min_opacity = 0.005, 
                                                extent = scene.cameras_extent, 
                                                radii=radii,
                                                args = opt,
                                                importance_score = importance_score,
                                                pruning_score = pruning_score,
                                                importance_quantile= opt.importance_quantile)

                if (iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter)) and iteration < args.net_itr:
                    gaussians.reset_opacity()

          
            if iteration % 3000 == 0 and iteration > 15_000 and iteration < 30_000:
                my_viewpoint_stack = scene.getTrainCameras().copy()
                camlist = sample_cameras_stratified(my_viewpoint_stack)

                pruning_score = compute_gaussian_pruning_mobilegs2(camlist, gaussians, pipe, bg, opt)                    
                gaussians.final_prune_mobilegs2(min_opacity = 0.1, pruning_score = pruning_score, pruning_quantile=opt.pruning_quantile)



            
            # Optimization step
            if iteration < opt.iterations:
                
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

    
        
                if iteration >= args.svq_itr:
                    gaussians.optimizer_code.step()

                if iteration > args.nn_iter:
                    gaussians.shs_nn_optimizer.step()
                    gaussians.shs_nn_optimizer.zero_grad(set_to_none = True)

                if iteration > args.net_itr:
                    gaussians.optimizer_net.step()
                    gaussians.optimizer_net.zero_grad(set_to_none=True)
                    gaussians.scheduler_net.step()
            
            if iteration == args.net_itr:
                current_opt_dict = gaussians.optimizer.state_dict()

                gaussians.construct_net()
                gaussians.training_setup(opt)
                

            if iteration == args.nn_iter:
         
                
                # cam_T = []
                # for cam in  scene.getTrainCameras().copy():
                #     cam_T.append(cam.camera_center[None])
                # cam_T = torch.cat(cam_T, dim=0).to("cuda")

                static, view = project_sh_mc(gaussians.get_features, num_samples=opt.num_mc_points)  # same with dc

                static = static[:,0]
                view = view[:,0]
                # gaussians._features_dc = nn.Parameter(new_dc.requires_grad_(True))
                # gaussians._features_rest = nn.Parameter(new_rest.requires_grad_(True))
                gaussians._features_static = nn.Parameter(static.requires_grad_(True))
                gaussians._features_view = nn.Parameter(view.requires_grad_(True))
        

                gaussians.active_sh_degree = 1
                gaussians.shoffset_enabled = True
                gaussians.training_setup(opt)
                gaussians.init_shsnn(opt)


            #     training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_time, testing_iterations, scene, render_fluxgs, (pipe, background, opt.mult))


            # record time
            optim_end.record()
            torch.cuda.synchronize()
            optim_time = optim_start.elapsed_time(optim_end)
            total_time += (iter_time + optim_time) / 1e3

    scene.save(iteration)
    print(f"Gaussian number: {gaussians._xyz.shape[0]}")
    print(f"Training time: {total_time}")
    with torch.no_grad():
        training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_time, testing_iterations, scene, render_fluxgs, (pipe, background, opt.mult))

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str)
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    # if iteration in testing_iterations:
    torch.cuda.empty_cache()
    validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                            {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

    for config in validation_configs:
        if config['cameras'] and len(config['cameras']) > 0:
            l1_test = 0.0
            psnr_test, ssim_test, lpips_test = 0.0, 0.0, 0.0
            for idx, viewpoint in enumerate(config['cameras']):
                image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                if tb_writer and (idx < 5):
                    tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                    if iteration == testing_iterations[0]:
                        tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                l1_test += l1_loss(image, gt_image).mean().double()
                psnr_test += psnr(image, gt_image).mean().double()
                ssim_test += fast_ssim(image.unsqueeze(0), gt_image.unsqueeze(0)).mean().double()
                lpips_test += lpips(image, gt_image, net_type='vgg').mean().double()
            psnr_test /= len(config['cameras'])
            ssim_test /= len(config['cameras'])
            lpips_test /= len(config['cameras'])
            l1_test /= len(config['cameras'])          
            print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
            if tb_writer:
                tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                tb_writer.add_scalar(config['name'] + '/loss_viewpoint - lpips', lpips_test, iteration)

    if tb_writer:
        tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
    torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--websockets", action='store_true', default=False)
    parser.add_argument("--benchmark_dir", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    if(args.websockets):
        network_gui_ws.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    training(
        lp.extract(args), 
        op.extract(args), 
        pp.extract(args), 
        args.test_iterations, 
        args.save_iterations, 
        args.checkpoint_iterations, 
        args.start_checkpoint, 
        args.debug_from, 
        args.websockets
    )

    # All done
    print("\nTraining complete.")
