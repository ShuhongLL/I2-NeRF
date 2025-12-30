import os.path

from internal import stepfun
from internal import math
from internal import utils
import torch
import torch.nn.functional as F


def lift_gaussian(d, t_mean, t_var, r_var, diag):
    """Lift a Gaussian defined along a ray to 3D coordinates."""
    mean = d[..., None, :] * t_mean[..., None]
    eps = torch.finfo(d.dtype).eps
    # eps = 1e-3
    d_mag_sq = torch.sum(d ** 2, dim=-1, keepdim=True).clamp_min(eps)

    if diag:
        d_outer_diag = d ** 2
        null_outer_diag = 1 - d_outer_diag / d_mag_sq
        t_cov_diag = t_var[..., None] * d_outer_diag[..., None, :]
        xy_cov_diag = r_var[..., None] * null_outer_diag[..., None, :]
        cov_diag = t_cov_diag + xy_cov_diag
        return mean, cov_diag
    else:
        d_outer = d[..., :, None] * d[..., None, :]
        eye = torch.eye(d.shape[-1], device=d.device)
        null_outer = eye - d[..., :, None] * (d / d_mag_sq)[..., None, :]
        t_cov = t_var[..., None, None] * d_outer[..., None, :, :]
        xy_cov = r_var[..., None, None] * null_outer[..., None, :, :]
        cov = t_cov + xy_cov
        return mean, cov


def conical_frustum_to_gaussian(d, t0, t1, base_radius, diag, stable=True):
    """Approximate a conical frustum as a Gaussian distribution (mean+cov).

  Assumes the ray is originating from the origin, and base_radius is the
  radius at dist=1. Doesn't assume `d` is normalized.

  Args:
    d: the axis of the cone
    t0: the starting distance of the frustum.
    t1: the ending distance of the frustum.
    base_radius: the scale of the radius as a function of distance.
    diag: whether or the Gaussian will be diagonal or full-covariance.
    stable: whether or not to use the stable computation described in
      the paper (setting this to False will cause catastrophic failure).

  Returns:
    a Gaussian (mean and covariance).
  """
    if stable:
        # Equation 7 in the paper (https://arxiv.org/abs/2103.13415).
        mu = (t0 + t1) / 2  # The average of the two `t` values.
        hw = (t1 - t0) / 2  # The half-width of the two `t` values.
        eps = torch.finfo(d.dtype).eps
        # eps = 1e-3
        t_mean = mu + (2 * mu * hw ** 2) / (3 * mu ** 2 + hw ** 2).clamp_min(eps)
        denom = (3 * mu ** 2 + hw ** 2).clamp_min(eps)
        t_var = (hw ** 2) / 3 - (4 / 15) * hw ** 4 * (12 * mu ** 2 - hw ** 2) / denom ** 2
        r_var = (mu ** 2) / 4 + (5 / 12) * hw ** 2 - (4 / 15) * (hw ** 4) / denom
    else:
        # Equations 37-39 in the paper.
        t_mean = (3 * (t1 ** 4 - t0 ** 4)) / (4 * (t1 ** 3 - t0 ** 3))
        r_var = 3 / 20 * (t1 ** 5 - t0 ** 5) / (t1 ** 3 - t0 ** 3)
        t_mosq = 3 / 5 * (t1 ** 5 - t0 ** 5) / (t1 ** 3 - t0 ** 3)
        t_var = t_mosq - t_mean ** 2
    r_var *= base_radius ** 2
    return lift_gaussian(d, t_mean, t_var, r_var, diag)


def cylinder_to_gaussian(d, t0, t1, radius, diag):
    """Approximate a cylinder as a Gaussian distribution (mean+cov).

  Assumes the ray is originating from the origin, and radius is the
  radius. Does not renormalize `d`.

  Args:
    d: the axis of the cylinder
    t0: the starting distance of the cylinder.
    t1: the ending distance of the cylinder.
    radius: the radius of the cylinder
    diag: whether or the Gaussian will be diagonal or full-covariance.

  Returns:
    a Gaussian (mean and covariance).
  """
    t_mean = (t0 + t1) / 2
    r_var = radius ** 2 / 4
    t_var = (t1 - t0) ** 2 / 12
    return lift_gaussian(d, t_mean, t_var, r_var, diag)


def cast_rays(tdist, origins, directions, cam_dirs, radii, rand=True, n=7, m=3, std_scale=0.5, **kwargs):
    """Cast rays (cone- or cylinder-shaped) and featurize sections of it.

  Args:
    tdist: float array, the "fencepost" distances along the ray.
    origins: float array, the ray origin coordinates.
    directions: float array, the ray direction vectors.
    radii: float array, the radii (base radii for cones) of the rays.
    ray_shape: string, the shape of the ray, must be 'cone' or 'cylinder'.
    diag: boolean, whether or not the covariance matrices should be diagonal.

  Returns:
    a tuple of arrays of means and covariances.
  """
    means = None
    stds = None

    from extensions import Backend

    if Backend.get_name() == 'dpcpp':
        import random

        MAX_INT = 2 ** 31 - 1
        seed1 = random.randint(0, MAX_INT)
        seed2 = random.randint(0, MAX_INT)
        seed3 = random.randint(0, MAX_INT)

        device = origins.device
        l = list(tdist.size())
        l[len(l) - 1] -= 1

        l.append(6)
        stds = torch.full(l, torch.inf, device = device)
        l.append(3)
        means = torch.full(l, torch.inf, device = device)

        backend = Backend.get_backend()

        backend.synchronize()
        backend.funcs.cast_rays_dpcpp(tdist, origins, directions, cam_dirs, radii, seed1, seed2, seed3, True, 7, 3, 0.5, means, stds)
        backend.synchronize()
        t = 0

    else:
        t0 = tdist[..., :-1, None]
        t1 = tdist[..., 1:, None]
        radii = radii[..., None]

        t_m = (t0 + t1) / 2
        t_d = (t1 - t0) / 2

        j = torch.arange(6, device=tdist.device)
        t = t0 + t_d / (t_d ** 2 + 3 * t_m ** 2) * (t1 ** 2 + 2 * t_m ** 2 + 3 / 7 ** 0.5 * (2 * j / 5 - 1) * (
            (t_d ** 2 - t_m ** 2) ** 2 + 4 * t_m ** 4).sqrt())

        deg = torch.pi / 3 * torch.tensor([0, 2, 4, 3, 5, 1], device=tdist.device, dtype=torch.float)
        deg = torch.broadcast_to(deg, t.shape)
        if rand:
            # randomly rotate and flip
            mask = torch.rand_like(t0[..., 0]) > 0.5
            deg = deg + 2 * torch.pi * torch.rand_like(deg[..., 0])[..., None]
            deg = torch.where(mask[..., None], deg, torch.pi * 5 / 3 - deg)
        else:
            # rotate 30 degree and flip every other pattern
            mask = torch.arange(t.shape[-2], device=tdist.device) % 2 == 0
            mask = torch.broadcast_to(mask, t.shape[:-1])
            deg = torch.where(mask[..., None], deg, deg + torch.pi / 6)
            deg = torch.where(mask[..., None], deg, torch.pi * 5 / 3 - deg)
        means = torch.stack([
            radii * t * torch.cos(deg) / 2 ** 0.5,
            radii * t * torch.sin(deg) / 2 ** 0.5,
            t
        ], dim=-1)
        stds = std_scale * radii * t / 2 ** 0.5

        # two basis in parallel to the image plane
        rand_vec = torch.randn_like(directions)
        ortho1 = F.normalize(torch.cross(directions, rand_vec, dim=-1), dim=-1)
        ortho2 = F.normalize(torch.cross(directions, ortho1, dim=-1), dim=-1)

        # just use directions to be the third vector of the orthonormal basis,
        # while the cross section of cone is parallel to the image plane
        basis_matrix = torch.stack([ortho1, ortho2, directions], dim=-1)
        means = math.matmul(means, basis_matrix[..., None, :, :].transpose(-1, -2))
        means = means + origins[..., None, None, :]
        # import trimesh
        # trimesh.Trimesh(means.reshape(-1, 3).detach().cpu().numpy()).export("test.ply", "ply")

    return means, stds, t


def compute_alpha_weights(sigma_obj, sigma_atten, sigma_bs, sigma_absorb, tdist,
                          dirs, extra_samples=False, opaque_background=False):
    t_delta = tdist[..., 1:] - tdist[..., :-1]
    delta = t_delta * torch.norm(dirs[..., None, :], dim=-1)
    density_delta = sigma_obj * delta
    
    if opaque_background:
        # Equivalent to making the final t-interval infinitely wide.
        density_delta = torch.cat([
            density_delta[..., :-1],
            torch.full_like(density_delta[..., -1:], torch.inf)
        ], dim=-1)
    
    # Object
    obj_alpha = 1 - torch.exp(-density_delta)
    obj_trans = torch.exp(-torch.cat([
        torch.zeros_like(density_delta[..., :1]),
        torch.cumsum(density_delta[..., :-1], dim=-1)
    ], dim=-1))
    obj_weights = obj_alpha * obj_trans
    
    bs_trans = None
    atten_trans = None
    absorb_trans = None
    bs_weights = None
    
    # Extra sampling before the first point of tdist.
    if extra_samples:
        extra_sample_len = 33
        t_media_dist = torch.stack([
            torch.linspace(0, end_value.item(), extra_sample_len) for end_value in tdist[..., 0].detach().reshape(-1)
        ], dim=0).reshape(*tdist.shape[:-1], extra_sample_len).cuda()
        # t_media_dist = torch.linspace(0, tdist[..., 0].detach(), 33, dim=-1)
        t_media_dist_sort = torch.cat([t_media_dist[..., :-1], tdist.detach()], dim=-1)
        
        t_media_delta = t_media_dist_sort[..., 1:] - t_media_dist_sort[..., :-1]
        media_delta = t_media_delta.detach() * torch.norm(dirs[..., None, :], dim=-1)

        # Insert 1 in front of the object trans
        extra_trans = torch.ones_like(t_media_dist)[..., :-1]
        obj_trans = torch.cat([extra_trans, obj_trans], dim=-1)
    else:
        media_delta = t_delta.detach() * torch.norm(dirs[..., None, :], dim=-1)
        
    # absorb field
    if sigma_absorb is not None:
        absorb_density_delta = sigma_absorb * media_delta[..., None]
        absorb_trans = torch.exp(-torch.cat([
            torch.zeros_like(absorb_density_delta[..., :1, :]),
            torch.cumsum(absorb_density_delta[..., :-1, :], dim=-2)
        ], dim=-2))
    
    # Scatter media
    if sigma_bs is not None and sigma_atten is not None:
        bs_density_delta = sigma_bs * media_delta[..., None]
        bs_alpha = 1 - torch.exp(-bs_density_delta)
        
        bs_trans = torch.exp(-torch.cat([
            torch.zeros_like(bs_density_delta[..., :1, :]),
            torch.cumsum(bs_density_delta[..., :-1, :], dim=-2)
        ], dim=-2))
        
        atten_density_delta = sigma_atten * media_delta[..., None]
        atten_trans = torch.exp(-torch.cat([
            torch.zeros_like(atten_density_delta[..., :1, :]),
            torch.cumsum(atten_density_delta[..., :-1, :], dim=-2)
        ], dim=-2))

        bs_weights = bs_alpha * bs_trans
 
    return obj_weights, bs_weights, obj_trans, bs_trans, atten_trans, absorb_trans


def volumetric_rendering(rgbs, c_med, weights, bs_weights, full_trans, atten_trans, absorb_trans,
                         tdist, bg_rgbs, t_far, compute_extras, extras=None, extra_samples=False):
    """Volumetric Rendering Function.

    Args:
        rgbs: color, [batch_size, num_samples, 3]
        c_med: medium's color, [batch_size, 1, 3].
        weights: object weights, [batch_size, num_samples].
        bs_weights: medium's (additive component) weights, [batch_size, num_samples, 3].
        full_trans: full transmittance of object including the extra samples if exist, [batch_size, extra_samples+num_samples].
        trans_atten: medium's attenuation transmission,  exp(-sigma_atten*s_i) , [batch_size, num_samples, 3].
        absorb_trans: absorbing field's transmission, [batch_size, num_samples, 1].
        tdist: [batch_size, num_samples].
        bg_rgbs: the color(s) to use for the background.
        t_far: [batch_size, 1], the distance of the far plane.
        compute_extras: bool, if True, compute extra quantities besides color.
        extras: dict, a set of values along rays to render by alpha compositing.

    Returns:
        rendering: a dict containing an rgb image of size [batch_size, 3], and other
        visualizations if compute_extras=True.
    """
    extra_sample_len = 0
    eps = torch.finfo(rgbs.dtype).eps
    rendering = {}

    acc = weights.sum(dim=-1)
    bg_w = (1 - acc[..., None]).clamp_min(0.)  # The weight of the background.
    
    origin_rgb = (weights[..., None] * rgbs).sum(axis=-2)
    
    if bs_weights is None:
        if absorb_trans is None:
            # Original rendering
            rendering['rgb'] = origin_rgb
        else:
            # absorbing field
            rgb = (weights[..., None] * absorb_trans[..., extra_sample_len-1:, :] * rgbs).sum(axis=-2)
            rendering.update({
                'rgb': rgb,
                'light_rgb': origin_rgb,
                'absorb_trans': absorb_trans
            })
    else:
        if absorb_trans is None:
            # Scatter media
            object_atten_rgb = (weights[..., None] * atten_trans[..., extra_sample_len-1:, :] * rgbs).sum(axis=-2)
            media_rgb = (bs_weights * full_trans[..., None] * c_med).sum(axis=-2)
            rendering.update({
                'J': origin_rgb,
                'bs_rgb': media_rgb,
                'rgb': object_atten_rgb + media_rgb,
                'c_med': c_med,
            })
        else:
            # absorbing field + scatter media
            object_product = weights[..., None] * atten_trans[..., extra_sample_len-1:, :] * rgbs
            media_product = bs_weights * full_trans[..., None] * c_med
            
            absorb_object_rgb = (object_product * absorb_trans[..., extra_sample_len-1:, :]).sum(axis=-2)
            absorb_media_rgb = (media_product * absorb_trans).sum(axis=-2)
            light_object_rgb = object_product.sum(axis=-2)
            light_media_rgb = media_product.sum(axis=-2)

            rendering.update({
                'rgb': absorb_object_rgb + absorb_media_rgb,
                'light_rgb': light_object_rgb + light_media_rgb,
                'J': origin_rgb,
                'bs_rgb': light_media_rgb,
                'c_med': c_med,
                'absorb_trans': absorb_trans
            })
    
    t_mids = 0.5 * (tdist[..., :-1] + tdist[..., 1:])
    depth = (
        torch.clip(
            torch.nan_to_num((weights * t_mids).sum(dim=-1) / acc.clamp_min(eps), torch.inf),
            tdist[..., 0], tdist[..., -1]))
    rendering['depth'] = depth
    rendering['acc'] = acc
    
    if compute_extras:
        if extras is not None:
            for k, v in extras.items():
                if v is not None:
                    rendering[k] = (weights[..., None] * v).sum(dim=-2)

        expectation = lambda x: (weights * x).sum(dim=-1) / acc.clamp_min(eps)
        # For numerical stability this expectation is computing using log-distance.
        rendering['distance_mean'] = (
            torch.clip(
                torch.nan_to_num(torch.exp(expectation(torch.log(t_mids))), torch.inf),
                tdist[..., 0], tdist[..., -1]))

        # Add an extra fencepost with the far distance at the end of each ray, with
        # whatever weight is needed to make the new weight vector sum to exactly 1
        # (`weights` is only guaranteed to sum to <= 1, not == 1).
        t_aug = torch.cat([tdist, t_far], dim=-1)
        weights_aug = torch.cat([weights, bg_w], dim=-1)

        ps = [5, 50, 95]
        distance_percentiles = stepfun.weighted_percentile(t_aug, weights_aug, ps)

        for i, p in enumerate(ps):
            s = 'median' if p == 50 else 'percentile_' + str(p)
            rendering['distance_' + s] = distance_percentiles[..., i]

    return rendering
