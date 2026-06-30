import torch
from PIL import ImageFilter
from gaussian_renderer import render_fluxgs
from .loss_utils import l1_loss
from fused_ssim import fused_ssim as fast_ssim
import torchvision.transforms as transforms
import random


import torch
import numpy as np # Used only for random.choice if preferred, or use torch






def estimate_scene_center(centers, directions):
    """
    Finds the 'center of attention' by calculating the closest point 
    to all camera look-at rays (Least Squares Intersection).
    """
    N = centers.shape[0]
    device = centers.device
    
    # Create Identity matrices stack: (N, 3, 3)
    I = torch.eye(3, device=device).unsqueeze(0).expand(N, 3, 3)
    
    # Reshape directions for outer product: (N, 3, 1)
    d = directions.unsqueeze(-1)
    
    # Projection matrices: I - d*d^T
    # This projects a vector onto the plane perpendicular to the ray
    P_matrices = I - torch.matmul(d, d.transpose(1, 2))
    
    # We want to solve Sum(P_i) * Center = Sum(P_i * c_i)
    # A * x = b
    
    # A = Sum(P_i) -> Shape (3, 3)
    A = torch.sum(P_matrices, dim=0)
    
    # b = Sum(P_i * c_i) -> Shape (3, 1)
    # Perform P * c per camera first
    Pc = torch.matmul(P_matrices, centers.unsqueeze(-1))
    b = torch.sum(Pc, dim=0)
    
    # Solve linear system. lstsq is more robust than solve for singular matrices.
    # We use .solution to get the result tensor
    scene_center = torch.linalg.lstsq(A, b).solution.squeeze()
    
    return scene_center


def get_camera_directions(R_w2c):
    """
    Extracts camera look vectors (Z-axis) from World-to-Camera rotation matrices.
    
    Args:
        R_w2c: (N, 3, 3) World-to-Camera rotation matrices
    Returns:
        directions: (N, 3) Look vectors in world space
    """
    # The Camera Z-axis in World Space is the 3rd row of R_w2c.
    # Why? R_c2w = R_w2c.T. The Z-axis is the 3rd column of R_c2w.
    # 3rd column of Transpose(R) == 3rd row of R.
    return R_w2c[:, 2, :] 

def sample_cameras_stratified(my_viewpoint_stack, azimuth_bins=12, elevation_bins=6, seed=None, num_cams=6):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- 1. Load Data (Simplified) ---
    cam_centers_list = []
    cam_R_w2c_list = []
    
    for cam in my_viewpoint_stack:
        # T is already c2w (Center), so we just use it directly.
        cam_centers_list.append(torch.from_numpy(cam.T).float().view(1, 3))
        
        # R is w2c. We keep it as w2c.
        cam_R_w2c_list.append(torch.from_numpy(cam.R).float().unsqueeze(0)) # (1, 3, 3)

    # Stack them
    centers = torch.cat(cam_centers_list, dim=0).to(device) # Shape (N, 3)
    R_w2c = torch.cat(cam_R_w2c_list, dim=0).to(device)     # Shape (N, 3, 3)

    # --- 2. Geometry & Bins ---
    # We only need to calculate directions now
    directions = get_camera_directions(R_w2c)
    
    # Calculate scene center using your existing helper (which is correct)
    target_center = estimate_scene_center(centers, directions)
    
    rel_pos = centers - target_center
    azimuths = torch.atan2(rel_pos[:, 1], rel_pos[:, 0])
    radii = torch.norm(rel_pos, dim=1) + 1e-8
    elevations = torch.asin(rel_pos[:, 2] / radii)
    
    az_edges = torch.linspace(-torch.pi, torch.pi, steps=azimuth_bins + 1, device=device)
    el_edges = torch.linspace(-torch.pi / 2, torch.pi / 2, steps=elevation_bins + 1, device=device)
    
    az_indices = torch.bucketize(azimuths, az_edges)
    el_indices = torch.bucketize(elevations, el_edges)
    
    grid_width = elevation_bins + 2 
    bin_ids = az_indices * grid_width + el_indices
    
    # --- 3. Binning Logic ---
    bin_ids_cpu = bin_ids.cpu().tolist()
    bins = {} 
    
    for cam_idx, b_id in enumerate(bin_ids_cpu):
        if b_id not in bins:
            bins[b_id] = []
        bins[b_id].append(cam_idx)
        
    # --- 4. Selection ---
    selected_indices = []
    unique_bins = sorted(bins.keys())
    
    for b_id in unique_bins:
        candidates = bins[b_id]
        chosen = np.random.choice(candidates)
        selected_indices.append(chosen)

    # Shuffle to ensure random coverage if we truncate
    np.random.shuffle(selected_indices)
    
    # Slice to budget
    final_indices = selected_indices[:num_cams]
    final_indices.sort()
    
    # Return actual objects
    selected_cams = [my_viewpoint_stack[i] for i in final_indices]
    
    return selected_cams


def sampling_cameras(my_viewpoint_stack):
    ''' Randomly sample a given number of cameras from the viewpoint stack'''

    num_cams = 20
    camlist = []
    for _ in range(num_cams):
        loc = random.randint(0, len(my_viewpoint_stack) - 1)
        camlist.append(my_viewpoint_stack.pop(loc))
    
    return camlist

def get_loss(reconstructed_image, original_image):
    l1_loss = torch.mean(torch.abs(reconstructed_image - original_image), 0).detach()
    l1_loss_norm = (l1_loss - torch.min(l1_loss)) / (torch.max(l1_loss) - torch.min(l1_loss))

    return l1_loss_norm

def compute_photometric_loss(viewpoint_cam, image):
    gt_image = viewpoint_cam.original_image.cuda()
    Ll1 = l1_loss(image, gt_image)
    loss = (1.0 - 0.2) * Ll1 + 0.2 * (1.0 - fast_ssim(image.unsqueeze(0), gt_image.unsqueeze(0)))
    return loss

def normalize(config_value, value_tensor):
    multiplier = config_value
    value_tensor[value_tensor.isnan()] = 0

    valid_indices = (value_tensor > 0)
    valid_value = value_tensor[valid_indices].to(torch.float32)

    ret_value = torch.zeros_like(value_tensor, dtype=torch.float32)
    ret_value[valid_indices] = multiplier * (valid_value / torch.median(valid_value))

    return ret_value

def compute_gaussian_score_mobilegs2(camlist, gaussians, pipe, bg, args, DENSIFY = False):
    """Compute multi-view consistency scores for Gaussians to guide densification.

    For each camera in `camlist` the function renders the scene and computes a
    photometric loss and a binary metric map of high-error pixels. It accumulates
    per-Gaussian counts of views that flagged the Gaussian and a weighted
    photometric score across views.

    Args:
        camlist (list): list of viewpoint camera objects to render from.
        gaussians: current Gaussian representation (model/state) used for rendering.
        pipe: rendering pipeline/context required by `render`.
        bg: background used for rendering.
        args: runtime config containing thresholds (e.g. `loss_thresh`).
        DENSIFY (bool): whether to compute and return the importance score
            used for densification. If False, only the pruning score is computed.

    Returns:
        importance_score (Tensor): per-Gaussian integer counts of how many views
            marked the Gaussian as high-error (floor-averaged across views).
            This output is only returned if `DENSIFY` is True.
        pruning_score (Tensor): normalized (0..1) per-Gaussian score used to
            prioritize densification (higher means worse reconstruction consistency).
    """

    full_metric_counts = None
    full_metric_score = None

    for view in range(len(camlist)):
        my_viewpoint_cam = camlist[view]
        render_image = render_fluxgs(my_viewpoint_cam, gaussians, pipe, bg, args.mult)["render"]
        photometric_loss = compute_photometric_loss(my_viewpoint_cam, render_image)

        gt_image = my_viewpoint_cam.original_image.cuda()
        get_flag = True
        l1_loss_norm = get_loss(render_image, gt_image)
        
        metric_map = (l1_loss_norm > args.loss_thresh).int()

        render_pkg = render_fluxgs(my_viewpoint_cam, gaussians, pipe, bg, args.mult, get_flag = get_flag, metric_map = metric_map)

        accum_loss_counts = render_pkg["accum_metric_counts"]

        if DENSIFY:
            if full_metric_counts is None:
                full_metric_counts = accum_loss_counts.clone()
            else:
                full_metric_counts += accum_loss_counts

        if full_metric_score is None:
            full_metric_score = photometric_loss * accum_loss_counts.clone()
        else:
            full_metric_score += photometric_loss * accum_loss_counts

    pruning_score = (full_metric_score - torch.min(full_metric_score)) / (torch.max(full_metric_score) - torch.min(full_metric_score))
    
    if DENSIFY:
        importance_score = torch.div(full_metric_counts, len(camlist), rounding_mode='floor')
    else:
        importance_score = None
    return importance_score, pruning_score




def compute_gaussian_pruning_mobilegs2(camlist, gaussians, pipe, bg, args):
    """Compute multi-view consistency scores for Gaussians to guide densification.

    For each camera in `camlist` the function renders the scene and computes a
    photometric loss and a binary metric map of high-error pixels. It accumulates
    per-Gaussian counts of views that flagged the Gaussian and a weighted
    photometric score across views.

    Args:
        camlist (list): list of viewpoint camera objects to render from.
        gaussians: current Gaussian representation (model/state) used for rendering.
        pipe: rendering pipeline/context required by `render`.
        bg: background used for rendering.
        args: runtime config containing thresholds (e.g. `loss_thresh`).
        DENSIFY (bool): whether to compute and return the importance score
            used for densification. If False, only the pruning score is computed.

    Returns:
        importance_score (Tensor): per-Gaussian integer counts of how many views
            marked the Gaussian as high-error (floor-averaged across views).
            This output is only returned if `DENSIFY` is True.
        pruning_score (Tensor): normalized (0..1) per-Gaussian score used to
            prioritize densification (higher means worse reconstruction consistency).
    """

    full_metric_counts = None
    full_metric_score = None

    for view in range(len(camlist)):
        my_viewpoint_cam = camlist[view]
        render_image = render_fluxgs(my_viewpoint_cam, gaussians, pipe, bg, args.mult)["render"]
        photometric_loss = compute_photometric_loss(my_viewpoint_cam, render_image)

        gt_image = my_viewpoint_cam.original_image.cuda()
        get_flag = True
        l1_loss_norm = get_loss(render_image, gt_image)
        
        metric_map = (l1_loss_norm < 0.01).int()

        render_pkg = render_fluxgs(my_viewpoint_cam, gaussians, pipe, bg, args.mult, get_flag = get_flag, metric_map = metric_map)

        accum_loss_counts = render_pkg["accum_metric_counts"]

        if full_metric_score is None:
            full_metric_score = photometric_loss * accum_loss_counts.clone()
        else:
            full_metric_score += photometric_loss * accum_loss_counts

    pruning_score = (full_metric_score - torch.min(full_metric_score)) / (torch.max(full_metric_score) - torch.min(full_metric_score))
    

    return  pruning_score