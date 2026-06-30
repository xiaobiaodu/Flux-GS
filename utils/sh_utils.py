#  Copyright 2021 The PlenOctree Authors.
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#  this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#  this list of conditions and the following disclaimer in the documentation
#  and/or other materials provided with the distribution.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.

import torch,  math
import numpy as np
C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]   


def eval_sh(deg, sh, dirs):
    """
    Evaluate spherical harmonics at unit directions
    using hardcoded SH polynomials.
    Works with torch/np/jnp.
    ... Can be 0 or more batch dimensions.
    Args:
        deg: int SH deg. Currently, 0-3 supported
        sh: jnp.ndarray SH coeffs [..., C, (deg + 1) ** 2]
        dirs: jnp.ndarray unit directions [..., 3]
    Returns:
        [..., C]
    """
    assert deg <= 4 and deg >= 0
    coeff = (deg + 1) ** 2
    assert sh.shape[-1] >= coeff

    result = C0 * sh[..., 0]
    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = (result -
                C1 * y * sh[..., 1] +
                C1 * z * sh[..., 2] -
                C1 * x * sh[..., 3])

        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (result +
                    C2[0] * xy * sh[..., 4] +
                    C2[1] * yz * sh[..., 5] +
                    C2[2] * (2.0 * zz - xx - yy) * sh[..., 6] +
                    C2[3] * xz * sh[..., 7] +
                    C2[4] * (xx - yy) * sh[..., 8])

            if deg > 2:
                result = (result +
                C3[0] * y * (3 * xx - yy) * sh[..., 9] +
                C3[1] * xy * z * sh[..., 10] +
                C3[2] * y * (4 * zz - xx - yy)* sh[..., 11] +
                C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12] +
                C3[4] * x * (4 * zz - xx - yy) * sh[..., 13] +
                C3[5] * z * (xx - yy) * sh[..., 14] +
                C3[6] * x * (xx - 3 * yy) * sh[..., 15])

                if deg > 3:
                    result = (result + C4[0] * xy * (xx - yy) * sh[..., 16] +
                            C4[1] * yz * (3 * xx - yy) * sh[..., 17] +
                            C4[2] * xy * (7 * zz - 1) * sh[..., 18] +
                            C4[3] * yz * (7 * zz - 3) * sh[..., 19] +
                            C4[4] * (zz * (35 * zz - 30) + 3) * sh[..., 20] +
                            C4[5] * xz * (7 * zz - 3) * sh[..., 21] +
                            C4[6] * (xx - yy) * (7 * zz - 1) * sh[..., 22] +
                            C4[7] * xz * (xx - 3 * yy) * sh[..., 23] +
                            C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)) * sh[..., 24])
    return result



def RGB2SH(rgb):
    return (rgb - 0.5) / C0

def SH2RGB(sh):
    return sh * C0 + 0.5



def sample_sphere(num_samples, device):
    """
    Uniformly sample directions on unit sphere.

    Returns:
        dirs: [S, 3]
    """
    u = torch.rand(num_samples, device=device)
    v = torch.rand(num_samples, device=device)

    theta = 2.0 * math.pi * u
    phi = torch.acos(2.0 * v - 1.0)

    x = torch.sin(phi) * torch.cos(theta)
    y = torch.sin(phi) * torch.sin(theta)
    z = torch.cos(phi)

    return torch.stack([x, y, z], dim=-1)





def eval_real_sh_3(dirs, origin_degree=3):
    """
    Real spherical harmonics basis up to l=3
    Using exact constants provided (Sloan / 3DGS convention)

    Args:
        dirs: [S, 3] normalized directions

    Returns:
        Y: [S, 16]
    """
    x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]

    Y = []

    # l = 0
    Y.append(torch.full_like(x, C0))

    if origin_degree>=1:
        Y += [
            -C1 * y,
            C1 * z,
            -C1 * x,
        ]

    if origin_degree>=2:
        Y += [
            C2[0] * x * y,
            C2[1] * y * z,
            C2[2] * (3.0 * z * z - 1.0),
            C2[3] * x * z,
            C2[4] * (x * x - y * y),
        ]

    if origin_degree==3:
        Y += [
            C3[0] * y * (3.0 * x * x - y * y),
            C3[1] * x * y * z,
            C3[2] * y * (5.0 * z * z - 1.0),
            C3[3] * z * (5.0 * z * z - 3.0),
            C3[4] * x * (5.0 * z * z - 1.0),
            C3[5] * z * (x * x - y * y),
            C3[6] * x * (x * x - 3.0 * y * y),
        ]

    return torch.stack(Y, dim=1)  

def fibonacci_sphere(n, device):
    i = torch.arange(n, device=device)
    phi = math.pi * (3.0 - math.sqrt(5.0))
    y = 1.0 - 2.0 * (i + 0.5) / n
    r = torch.sqrt(1.0 - y * y)
    theta = phi * i
    x = torch.cos(theta) * r
    z = torch.sin(theta) * r
    return torch.stack([x, y, z], dim=-1)


@torch.no_grad()
def mc_project_sh_rgb(
    sh,                # [N, 16, 3]
    origin_degree,
    target_degree,      # 0, 1, or 2
    num_samples=2048,
):
    """
    Monte-Carlo projection: SH3 -> SH_L

    Returns:
        sh_low: [N, (L+1)^2, 3]
    """
    assert sh.ndim == 3 and sh.shape[2] == 3
    assert target_degree in (0, 1, 2)

    device = sh.device
    N = sh.shape[0]
    K = (target_degree + 1) ** 2

    dirs = fibonacci_sphere(num_samples, device)     # [S, 3]

    #  Evaluate SH basis
    Y3 = eval_real_sh_3(dirs, origin_degree=origin_degree)                      # [S, 16]
    YL = Y3[:, :K]                                 # [S, K]

    # Evaluate function f(ω) = Σ c Y
    #    f: [N, S, 3]
    f = torch.einsum("sm,nmc->nsc", Y3, sh)

    #  Monte-Carlo projection
    #    c_L = (4π / S) * Σ f(ω) Y_L(ω)
    sh_low = (4.0 * math.pi / num_samples) * torch.einsum(
        "sk,nsc->nkc", YL, f
    )

    return sh_low



 

@torch.no_grad()
def residual_aware_sh3_to_sh1(
    sh3, gaussians_xyz, cams_xyz,
    num_samples=512, alpha=2.0,
    cam_chunk=16, dir_chunk=64,
):
    device = sh3.device
    N = sh3.shape[0]
    M = cams_xyz.shape[0]

    dirs = fibonacci_sphere(num_samples, device)          # [S, 3]
    Y3 = eval_real_sh_3(dirs, origin_degree=3)            # [S, 16]
    Y1 = Y3[:, :4]                                       # [S, 4]

    # 累计最小二乘项
    AtA = torch.zeros(N, 4, 4, device=device)             # Σ w Y Y^T
    Atb = torch.zeros(N, 4, 3, device=device)             # Σ w Y f

    for j in range(0, num_samples, dir_chunk):
        dirs_chunk = dirs[j:j+dir_chunk]                  # [s, 3]
        Y3_chunk = Y3[j:j+dir_chunk]                      # [s, 16]
        Y1_chunk = Y1[j:j+dir_chunk]                      # [s, 4]

        f_chunk = torch.einsum("sm,nmc->nsc", Y3_chunk, sh3)  # [N, s, 3]

        # vis_w_chunk: [N, s]
        vis_w_chunk = torch.zeros(N, f_chunk.shape[1], device=device)

        for i in range(0, M, cam_chunk):
            cams_chunk = cams_xyz[i:i+cam_chunk]          # [c, 3]
            view_dirs = cams_chunk[None] - gaussians_xyz[:, None]  
            view_dirs = view_dirs / (view_dirs.norm(dim=-1, keepdim=True) + 1e-6)

            dots = torch.einsum("ncd,sd->ncs", view_dirs, dirs_chunk)  # [N, c, s]
            vis_w_chunk += dots.clamp(min=0).sum(dim=1)

        vis_w_chunk = vis_w_chunk / M

        # residual-aware 权重
        var_w = f_chunk.var(dim=2)                 # [N, s]
        w = vis_w_chunk * (1.0 + alpha * var_w)    # [N, s]

        # 累计 AtA, Atb
        # Y1_chunk: [s,4]
        Y = Y1_chunk[None]                         # [1, s, 4]
        wY = w[..., None] * Y                      # [N, s, 4]

        AtA += torch.einsum("nsd,nse->nde", wY, Y) # [N, 4, 4]
        Atb += torch.einsum("nsd,nsc->ndc", wY, f_chunk)  # [N, 4, 3]

    # 解 4x4 线性系统（每个 Gaussian 一个）
    eps = 1e-6 * torch.eye(4, device=device)[None]
    AtA = AtA + eps

    sh1 = torch.linalg.solve(AtA, Atb)   # [N, 4, 3]

    return sh1




def eval_sh_dirs(deg, sh, dirs):
    """
    Evaluate Spherical Harmonics for N points across K directions.
    
    Args:
        deg (int): Degree (0 to 4).
        sh (torch.Tensor): SH coefficients. Shape (N, (deg+1)**2, 3).
        dirs (torch.Tensor): Unit sampling directions. Shape (K, 3).
    
    Returns:
        torch.Tensor: RGB colors. Shape (N, K, 3).
    """
    
    # 1. Constants
    C0 = 0.28209479177387814
    C1 = 0.4886025119029199
    C2 = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396]
    C3 = [-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154, -0.4570457994644658, 1.445305721320277, -0.5900435899266435]
    C4 = [2.5033429417967046, -1.7701307697799304, 0.9461746957575601, -0.6690465435572892, 0.10578554691520431, -0.6690465435572892, 0.47308734787878004, -1.7701307697799304, 0.6258357354491761]

    assert 0 <= deg <= 4
    
    # -----------------------------------------------------------
    # Construct Basis Matrix (K, Coeffs)
    # -----------------------------------------------------------
    
    # Normalize dirs: (K, 3)
    dirs = dirs / (dirs.norm(dim=1, keepdim=True) + 1e-6)
    
    x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2] # Shapes: (K,)
    
    # Precomputations
    x2, y2, z2 = x*x, y*y, z*z
    xy, yz, xz = x*y, y*z, x*z
    
    basis = []
    
    # Band 0 (l=0) -> 1 coeff
    basis.append(torch.full_like(x, C0))
    
    if deg >= 1: # Band 1 -> 3 coeffs
        basis.append(C1 * y)
        basis.append(C1 * z)
        basis.append(C1 * x)
        
    if deg >= 2: # Band 2 -> 5 coeffs
        basis.append(C2[0] * xy)
        basis.append(C2[1] * yz)
        basis.append(C2[2] * (3 * z2 - 1))
        basis.append(C2[3] * xz)
        basis.append(C2[4] * (x2 - y2))
        
    if deg >= 3: # Band 3 -> 7 coeffs
        basis.append(C3[0] * y * (3 * x2 - y2))
        basis.append(C3[1] * xy * z)
        basis.append(C3[2] * y * (5 * z2 - 1))
        basis.append(C3[3] * z * (5 * z2 - 3))
        basis.append(C3[4] * x * (5 * z2 - 1))
        basis.append(C3[5] * z * (x2 - y2))
        basis.append(C3[6] * x * (x2 - 3 * y2))
        
    if deg >= 4: # Band 4 -> 9 coeffs
        basis.append(C4[0] * xy * (x2 - y2))
        basis.append(C4[1] * y * z * (3 * x2 - y2))
        basis.append(C4[2] * xy * (7 * z2 - 1))
        basis.append(C4[3] * y * z * (7 * z2 - 3))
        basis.append(C4[4] * (35 * z2 * z2 - 30 * z2 + 3))
        basis.append(C4[5] * x * z * (7 * z2 - 3))
        basis.append(C4[6] * (x2 - y2) * (7 * z2 - 1))
        basis.append(C4[7] * x * z * (x2 - 3 * y2))
        basis.append(C4[8] * (x2 * (x2 - 3 * y2) - y2 * (3 * x2 - y2)))

    # Stack to form (K, Total_Coeffs)
    # Note: simple stack assumes individual tensors are (K,)
    basis_matrix = torch.stack(basis, dim=1) 
    
    # -----------------------------------------------------------
    # Evaluation (Broadcasting)
    # -----------------------------------------------------------
    
    # sh:           (N, Coeffs, 3)  <- The 'n' and 'c' and 'color' dimensions
    # basis_matrix: (K, Coeffs)     <- The 'k' and 'c' dimensions
    # Output:       (N, K, 3)       <- The 'n', 'k', and 'color' dimensions
    
    # Einstein Summation is the cleanest way to map this:
    # n=N points, c=Coeffs, d=rgb dimension, k=K samples
    rgb = torch.einsum('ncd, kc -> nkd', sh, basis_matrix)
    
    return rgb


def project_sh_mc(sh_coeffs, num_samples=2048):
    """
    Monte Carlo Specular Energy Aggregator (Eq. 4)
    
    sh_coeffs: (N, 16, 3) - Input 3rd order SH coefficients
    num_samples: K random directions on the unit sphere for MC sampling
    
    Returns:
        e_mag: (N, 1, 3) - Mean positive high-frequency residual energy
        e_dir: (N, 1, 3) - Compact directional first moment of the energy
    """
    assert sh_coeffs.ndim == 3 and sh_coeffs.shape[-1] == 3
    assert sh_coeffs.shape[1] >= 16

    N = sh_coeffs.shape[0]
    device = sh_coeffs.device
    
    # Uniform spherical sampling, matching Eq. 3 in the paper.
    xi = torch.rand(num_samples, device=device)
    theta = torch.acos(1.0 - 2.0 * xi)
    phi = torch.rand(num_samples, device=device) * 2 * math.pi
    
    x = torch.sin(theta) * torch.cos(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(theta)
    dirs = torch.stack([x, y, z], dim=1)  # (K, 3)
    
    # c_res(d_k) = c_i(d_k) - c_i^2(d_k)
    samples_full = eval_sh_dirs(deg=3, sh=sh_coeffs[:, :16, :], dirs=dirs)  # (N, K, 3)
    samples_base = eval_sh_dirs(deg=2, sh=sh_coeffs[:, :9, :], dirs=dirs)   # (N, K, 3)
    positive_residual = torch.clamp(samples_full - samples_base, min=0.0)
    
    # E_mag = 1/K * sum_k max(0, c_res(d_k))
    e_mag = positive_residual.mean(dim=1, keepdim=True)
    
    # Eq. 4 uses d_k outer max(0, c_res(d_k)). This codebase stores a compact
    # 3-vector latent, so collapse RGB energy to a scalar before accumulating
    # the directional first moment.
    energy = positive_residual.mean(dim=-1)  # (N, K)
    e_dir = torch.einsum("kd,nk->nd", dirs, energy) / float(num_samples)
    e_dir = e_dir.unsqueeze(1)
    
    return e_mag, e_dir
