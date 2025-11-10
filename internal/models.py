import os
import matplotlib.pyplot as plt
import accelerate
import gin
from internal import coord
from internal import geopoly
from internal import image
from internal import math
from internal import ref_utils
from internal import train_utils
from internal import render
from internal import stepfun
from internal import utils
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils._pytree import tree_map
from tqdm import tqdm
from gridencoder import GridEncoder
try:
    from torch_scatter import segment_coo
except:
    pass

gin.config.external_configurable(math.safe_exp, module='math')


def set_kwargs(self, kwargs):
    for k, v in kwargs.items():
        setattr(self, k, v)


@gin.configurable
class Model(nn.Module):
    """A mip-Nerf360 model containing all MLPs."""
    num_prop_samples: int = 64  # The number of samples for each proposal level.
    num_nerf_samples: int = 32  # The number of samples the final nerf level.
    num_extra_media_samples: int = 0 # The number of extra media samples.
    num_levels: int = 3  # The number of sampling levels (3==2 proposals, 1 nerf).
    bg_intensity_range = (1., 1.)  # The range of background colors.
    anneal_slope: float = 10  # Higher = more rapid annealing.
    stop_level_grad: bool = True  # If True, don't backprop across levels.
    use_viewdirs: bool = True  # If True, use view directions as input.
    raydist_fn = None  # The curve used for ray dists.
    single_jitter: bool = True  # If True, jitter whole rays instead of samples.
    dilation_multiplier: float = 0.5  # How much to dilate intervals relatively.
    dilation_bias: float = 0.0025  # How much to dilate intervals absolutely.
    num_glo_features: int = 0  # GLO vector length, disabled if 0.
    num_glo_embeddings: int = 1000  # Upper bound on max number of train images.
    learned_exposure_scaling: bool = False  # Learned exposure scaling (RawNeRF).
    near_anneal_rate = None  # How fast to anneal in near bound.
    near_anneal_init: float = 0.95  # Where to initialize near bound (in [0, 1]).
    single_mlp: bool = False  # Use the NerfMLP for all rounds of sampling.
    distinct_prop: bool = True  # Use the NerfMLP for all rounds of sampling.
    resample_padding: float = 0.0  # Dirichlet/alpha "padding" on the histogram.
    opaque_background: bool = False  # If true, make the background opaque.
    power_lambda: float = -1.5
    std_scale: float = 0.5
    prop_desired_grid_size = [512, 2048]

    def __init__(self, config=None, **kwargs):
        super().__init__()
        set_kwargs(self, kwargs)
        self.config = config
        self.eps = torch.finfo(torch.float).eps
        self.checkpoint_dir = config.checkpoint_dir

        from extensions import Backend
        Backend.set_backend('dpcpp' if self.config.dpcpp_backend else 'cuda')
        self.backend = Backend.get_backend()
        self.generator = self.backend.get_generator()

        # Construct MLPs. WARNING: Construction order may matter, if MLP weights are
        # being regularized.
        self.nerf_mlp = NerfMLP(num_glo_features=self.num_glo_features,
                                num_glo_embeddings=self.num_glo_embeddings,
                                enable_scatter=self.config.enable_scatter,
                                enable_absorb=self.config.enable_absorb,
                                enable_spatial_media=self.config.enable_spatial_media,
                                consistent_attenuation=self.config.consistent_attenuation,
                                enable_downwell_depth=self.config.enable_downwell_depth)
        if self.config.dpcpp_backend:
            self.generator = self.nerf_mlp.encoder.backend.get_generator()
        else:
            self.generator = None

        if self.single_mlp:
            self.prop_mlp = self.nerf_mlp
        elif not self.distinct_prop:
            self.prop_mlp = PropMLP()
        else:
            for i in range(self.num_levels - 1):
                self.register_module(f'prop_mlp_{i}', PropMLP(grid_disired_resolution=self.prop_desired_grid_size[i]))
        if self.num_glo_features > 0 and not config.zero_glo:
            # Construct/grab GLO vectors for the cameras of each input ray.
            self.glo_vecs = nn.Embedding(self.num_glo_embeddings, self.num_glo_features)

        if self.learned_exposure_scaling:
            # Setup learned scaling factors for output colors.
            max_num_exposures = self.num_glo_embeddings
            # Initialize the learned scaling offsets at 0.
            self.exposure_scaling_offsets = nn.Embedding(max_num_exposures, 3)
            torch.nn.init.zeros_(self.exposure_scaling_offsets.weight)

    def forward(
            self,
            rand,
            batch,
            train_frac,
            compute_extras,
            zero_glo=True,
            step=0,
    ):
        """The mip-NeRF Model.

    Args:
      rand: random number generator (or None for deterministic output).
      batch: util.Rays, a pytree of ray origins, directions, and viewdirs.
      train_frac: float in [0, 1], what fraction of training is complete.
      compute_extras: bool, if True, compute extra quantities besides color.
      zero_glo: bool, if True, when using GLO pass in vector of zeros.

    Returns:
      ret: list, [*(rgb, distance, acc)]
    """
        device = batch['origins'].device
        if self.num_glo_features > 0:
            if not zero_glo:
                # Construct/grab GLO vectors for the cameras of each input ray.
                cam_idx = batch['cam_idx'][..., 0]
                glo_vec = self.glo_vecs(cam_idx.long())
            else:
                glo_vec = torch.zeros(batch['origins'].shape[:-1] + (self.num_glo_features,), device=device)
        else:
            glo_vec = None

        # Define the mapping from normalized to metric ray distance.
        _, s_to_t = coord.construct_ray_warps(self.raydist_fn, batch['near'], batch['far'], self.power_lambda)

        # Initialize the range of (normalized) distances for each ray to [0, 1],
        # and assign that single interval a weight of 1. These distances and weights
        # will be repeatedly updated as we proceed through sampling levels.
        # `near_anneal_rate` can be used to anneal in the near bound at the start
        # of training, eg. 0.1 anneals in the bound over the first 10% of training.
        if self.near_anneal_rate is None:
            init_s_near = 0.
        else:
            init_s_near = np.clip(1 - train_frac / self.near_anneal_rate, 0,
                                  self.near_anneal_init)
        init_s_far = 1.
        sdist = torch.cat([
            torch.full_like(batch['near'], init_s_near),
            torch.full_like(batch['far'], init_s_far)
        ], dim=-1)
        weights = torch.ones_like(batch['near'])
        prod_num_samples = 1

        ray_history = []
        renderings = []
        for i_level in range(self.num_levels): # 2 or 3
            is_prop = i_level < (self.num_levels - 1)
            num_samples = self.num_prop_samples if is_prop else self.num_nerf_samples

            # Dilate by some multiple of the expected span of each current interval,
            # with some bias added in.
            dilation = self.dilation_bias + self.dilation_multiplier * (
                    init_s_far - init_s_near) / prod_num_samples

            # Record the product of the number of samples seen so far.
            prod_num_samples *= num_samples

            # After the first level (where dilation would be a no-op) optionally
            # dilate the interval weights along each ray slightly so that they're
            # overestimates, which can reduce aliasing.
            sampled_sdist = sdist
            use_dilation = self.dilation_bias > 0 or self.dilation_multiplier > 0 # True
            if i_level > 0 and use_dilation:
                sdist, weights = stepfun.max_dilate_weights(
                    sdist,
                    weights,
                    dilation,
                    domain=(init_s_near, init_s_far),
                    renormalize=True)
                sdist = sdist[..., 1:-1]
                weights = weights[..., 1:-1]

            # Optionally anneal the weights as a function of training iteration.
            if self.anneal_slope > 0:
                # Schlick's bias function, see https://arxiv.org/abs/2010.09714
                bias = lambda x, s: (s * x) / ((s - 1) * x + 1)
                anneal = bias(train_frac, self.anneal_slope)
            else:
                anneal = 1.

            # A slightly more stable way to compute weights**anneal. If the distance
            # between adjacent intervals is zero then its weight is fixed to 0.
            logits_resample = torch.where(
                sdist[..., 1:] > sdist[..., :-1],
                anneal * torch.log(weights + self.resample_padding),
                torch.full_like(sdist[..., :-1], -torch.inf))

            # Draw sampled intervals from each ray's current weights.
            sdist = stepfun.sample_intervals(
                rand,
                sdist,
                sampled_sdist,
                logits_resample,
                density if not is_prop else None,
                num_samples,
                single_jitter=self.single_jitter,
                domain=(init_s_near, init_s_far),
                extra_sample_len=self.num_extra_media_samples)

            # Optimization will usually go nonlinear if you propagate gradients
            # through sampling.
            if self.stop_level_grad: # True
                sdist = sdist.detach()

            # Convert normalized distances to metric distances.
            tdist = s_to_t(sdist)

            # Cast our rays, by turning our distance intervals into Gaussians.
            means, stds, ts = render.cast_rays(
                tdist,
                batch['origins'],
                batch['directions'],
                batch['cam_dirs'],
                batch['radii'],
                rand,
                std_scale=self.std_scale)

            # Push our Gaussians through one of our two MLPs.
            mlp = (self.get_submodule(
                f'prop_mlp_{i_level}') if self.distinct_prop else self.prop_mlp) if is_prop else self.nerf_mlp
            ray_results = mlp(
                rand,
                means, stds,
                viewdirs=batch['viewdirs'] if self.use_viewdirs else None,
                imageplane=batch.get('imageplane'),
                glo_vec=None if is_prop else glo_vec,
                exposure=batch.get('exposure_values'),
            )
            
            # Object sigma density.
            density = ray_results['density']
            
            if self.config.gradient_scaling:
                ray_results['rgb'], density = train_utils.GradientScaler.apply(
                    ray_results['rgb'], density, ts.mean(dim=-1))

            # Get the weights used by volumetric rendering (and our other losses).
            weights, bs_weights, full_trans, bs_trans, atten_trans, absorb_trans = \
            render.compute_alpha_weights(
                density,
                ray_results['sigma_atten'],
                ray_results['sigma_bs'],
                ray_results['sigma_absorb'],
                tdist,
                batch['directions'],
                extra_samples=self.config.extra_samples,
                opaque_background=self.opaque_background,
            )
            
            if step > 0 and step % self.config.train_render_every == 0 and not is_prop:
                figure_dir = os.path.dirname(os.path.normpath(self.checkpoint_dir))
                os.makedirs(os.path.join(figure_dir, "rays"), exist_ok=True)
                num_bins = 80
                bins = np.linspace(0, 1, num_bins + 1)
                bin_centers = (bins[:-1] + bins[1:]) / 2
                bar_width = 0.8 * (bins[1] - bins[0])
                
                smid = (sdist[..., :-1] + sdist[..., 1:])/2
                smid_flat = smid.flatten().detach().cpu().numpy()
                
                # # Plot density
                # if self.config.enable_absorb and self.config.enable_spatial_media:
                #     sigma_absorb_flat = ray_results['sigma_absorb'].flatten().detach().cpu().numpy()
                #     density_flat = ray_results['density'].flatten().detach().cpu().numpy()
                #     up_values, _ = np.histogram(smid_flat, bins=bins, weights=density_flat)
                #     down_values, _ = np.histogram(smid_flat, bins=bins, weights=sigma_absorb_flat)

                #     plt.figure(figsize=(10, 5))
                #     plt.bar(bin_centers, up_values, width=bar_width, color='green', label='Obj Sigma', align='center')
                #     plt.bar(bin_centers, -down_values, width=bar_width, color='gray', label='absorb Sigma', align='center')
                #     plt.axhline(0, color='black', linewidth=1)
                #     plt.xlim(0, 1)
                #     plt.xlabel("Sample Positions")
                #     plt.ylabel("Density")
                #     plt.title(f"absorb Density Weight Distribution of a Batch Rays")
                #     plt.legend()
                #     plt.savefig(os.path.join(figure_dir, "rays",
                #                              f"absorb_batch_density_distr={step}.png"),
                #                 bbox_inches="tight")
                    
                #     idx = torch.randint(0, density.shape[0], (1,)).item()
                #     ray_smid = smid[idx].squeeze().detach().cpu().numpy()
                #     ray_density = density[idx].squeeze().detach().cpu().numpy()
                #     ray_absorb = ray_results['sigma_absorb'][idx].squeeze().detach().cpu().numpy()
                    
                #     plt.figure(figsize=(10, 5))
                #     plt.plot(ray_smid, ray_density, label="Obj Sigma")
                #     plt.plot(ray_smid, ray_absorb, label="absorb Sigma")
                #     plt.legend()
                #     plt.xlabel("Sample Position")
                #     plt.ylabel("Sigma")
                #     plt.title(f"Sigma Distribution of a Ray: #{idx}")
                #     plt.yticks()
                #     plt.xticks()
                #     plt.savefig(os.path.join(figure_dir, "rays",
                #                             f"absorb_sigma_stp={step}.png"),
                #                 bbox_inches="tight")
                
                # # spatial-varying scattering media:
                # if self.config.enable_scatter and self.config.enable_spatial_media:
                #     sigma_bs_flat = ray_results['sigma_bs'].mean(dim=-1).flatten().detach().cpu().numpy()
                #     density_flat = ray_results['density'].flatten().detach().cpu().numpy()
                #     up_values, _ = np.histogram(smid_flat, bins=bins, weights=density_flat)
                #     down_values, _ = np.histogram(smid_flat, bins=bins, weights=sigma_bs_flat)

                #     plt.figure(figsize=(10, 5))
                #     plt.bar(bin_centers, up_values, width=bar_width, color='green', label='Obj Sigma', align='center')
                #     plt.bar(bin_centers, -down_values, width=bar_width, color='gray', label='Bs Sigma', align='center')
                #     plt.axhline(0, color='black', linewidth=1)
                #     plt.xlim(0, 1)
                #     plt.xlabel("Sample Positions")
                #     plt.ylabel("Density")
                #     plt.title(f"Bs Density Weight Distribution of a Batch Rays")
                #     plt.legend()
                #     plt.savefig(os.path.join(figure_dir, "rays",
                #                              f"bs_batch_density_distr={step}.png"),
                #                 bbox_inches="tight")
                    
                #     idx = torch.randint(0, density.shape[0], (1,)).item()
                #     ray_smid = smid[idx].squeeze().detach().cpu().numpy()
                #     ray_density = density[idx].squeeze().detach().cpu().numpy()
                #     ray_bs = ray_results['sigma_bs'][idx].squeeze().detach().cpu().numpy()
                    
                #     plt.figure(figsize=(10, 5))
                #     plt.plot(ray_smid, ray_density, color='black', label="Obj Sigma")
                #     plt.plot(ray_smid, ray_bs[:, 0], color='red', label="Bs R")
                #     plt.plot(ray_smid, ray_bs[:, 1], color='green', label="Bs G")
                #     plt.plot(ray_smid, ray_bs[:, 2], color='blue', label="Bs B")
                #     plt.legend()
                #     plt.xlabel("Sample Position")
                #     plt.ylabel("Sigma")
                #     plt.title(f"Sigma Distribution of a Ray: #{idx}")
                #     plt.yticks()
                #     plt.xticks()
                #     plt.savefig(os.path.join(figure_dir, "rays",
                #                             f"bs_sigma_stp={step}.png"),
                #                 bbox_inches="tight")
                
            # Define or sample the background color for each ray.
            if self.bg_intensity_range[0] == self.bg_intensity_range[1]:
                # If the min and max of the range are equal, just take it.
                bg_rgbs = self.bg_intensity_range[0]
            elif rand is None:
                # If rendering is deterministic, use the midpoint of the range.
                bg_rgbs = (self.bg_intensity_range[0] + self.bg_intensity_range[1]) / 2
            else:
                # Sample RGB values from the range for each ray.
                minval = self.bg_intensity_range[0]
                maxval = self.bg_intensity_range[1]
                bg_rgbs = torch.rand(weights.shape[:-1] + (3,), device=device) * (maxval - minval) + minval

            # RawNeRF exposure logic.
            if batch.get('exposure_idx') is not None:
                # Scale output colors by the exposure.
                ray_results['rgb'] *= batch['exposure_values'][..., None, :]
                if self.learned_exposure_scaling:
                    exposure_idx = batch['exposure_idx'][..., 0]
                    # Force scaling offset to always be zero when exposure_idx is 0.
                    # This constraint fixes a reference point for the scene's brightness.
                    mask = exposure_idx > 0
                    # Scaling is parameterized as an offset from 1.
                    scaling = 1 + mask[..., None] * self.exposure_scaling_offsets(exposure_idx.long())
                    ray_results['rgb'] *= scaling[..., None, :]

            # est_cmed = ray_results['ambient_light'] * torch.exp(-ray_results['sigma_atten'].detach() * ray_results['media_depth']) if not is_prop else None
            # if not is_prop:
            #     print(f"Media depth: {ray_results['media_depth'].squeeze().cpu()[0][0].item()}")

            # Render each ray.
            rendering = render.volumetric_rendering(
                ray_results['rgb'],
                ray_results['media_rgb'],
                weights,
                bs_weights,
                full_trans,
                atten_trans,
                absorb_trans,
                tdist,
                bg_rgbs,
                batch['far'],
                compute_extras,
                extras={
                    k: v
                    for k, v in ray_results.items()
                    if k.startswith('normals') or k in ['roughness']
                })
            
            ray_results['full_trans'] = full_trans
            rendering['sigma_obj'] = density
            
            if self.config.enable_scatter and not is_prop:
                rendering['sigma_atten'] = ray_results['sigma_atten']
                rendering['sigma_bs'] = ray_results['sigma_bs']
                if self.config.enable_downwell_depth:
                    rendering['media_depth'] = ray_results['media_depth'] # Vertical depth of scattering media.
                
            if self.config.enable_absorb and not is_prop:
                rendering['sigma_absorb'] = ray_results['sigma_absorb']
                ray_results['absorb_trans'] = absorb_trans

            if compute_extras:
                # Collect some rays to visualize directly. By naming these quantities
                # with `ray_` they get treated differently downstream --- they're
                # treated as bags of rays, rather than image chunks.
                n = self.config.vis_num_rays
                rendering['ray_sdist'] = sdist.reshape([-1, sdist.shape[-1]])[:n, :]
                rendering['ray_weights'] = (
                    weights.reshape([-1, weights.shape[-1]])[:n, :])
                rgb = ray_results['rgb']
                rendering['ray_rgbs'] = (rgb.reshape((-1,) + rgb.shape[-2:]))[:n, :, :]
                rendering['ray_tdist'] = tdist[:n, :]

            if self.training:
                # Compute the hash decay loss for this level.
                idx = mlp.encoder.idx
                param = mlp.encoder.embeddings
                if self.config.dpcpp_backend:
                    ray_results['loss_hash_decay'] = (param ** 2).mean()
                else:
                    loss_hash_decay = segment_coo(param ** 2,
                                                  idx,
                                                  torch.zeros(idx.max() + 1, param.shape[-1], device=param.device),
                                                  reduce='mean'
                                                  ).mean()
                    ray_results['loss_hash_decay'] = loss_hash_decay

            renderings.append(rendering)
            ray_results['tdist'] = tdist.clone()
            ray_results['sdist'] = sdist.clone()
            ray_results['weights'] = weights.clone()
            ray_history.append(ray_results)

        if compute_extras:
            # Because the proposal network doesn't produce meaningful colors, for
            # easier visualization we replace their colors with the final average
            # color.
            weights = [r['ray_weights'] for r in renderings]
            rgbs = [r['ray_rgbs'] for r in renderings]
            final_rgb = torch.sum(rgbs[-1] * weights[-1][..., None], dim=-2)
            avg_rgbs = [
                torch.broadcast_to(final_rgb[:, None, :], r.shape) for r in rgbs[:-1]
            ]
            for i in range(len(avg_rgbs)):
                renderings[i]['ray_rgbs'] = avg_rgbs[i]

        return renderings, ray_history


class MLP(nn.Module):
    """A PosEnc MLP."""
    bottleneck_width: int = 256  # The width of the bottleneck vector.
    net_depth_viewdirs: int = 2  # The depth of the second part of ML.
    net_width_viewdirs: int = 256  # The width of the second part of MLP.
    skip_layer_dir: int = 0  # Add a skip connection to 2nd MLP after Nth layers.
    num_rgb_channels: int = 3  # The number of RGB channels.
    deg_view: int = 4  # Degree of encoding for viewdirs or refdirs.
    use_reflections: bool = False  # If True, use refdirs instead of viewdirs.
    use_directional_enc: bool = False  # If True, use IDE to encode directions.
    # If False and if use_directional_enc is True, use zero roughness in IDE.
    enable_pred_roughness: bool = False
    roughness_bias: float = -1.  # Shift added to raw roughness pre-activation.
    use_specular_tint: bool = False  # If True, predict tint.
    use_n_dot_v: bool = False  # If True, feed dot(n * viewdir) to 2nd MLP.
    bottleneck_noise: float = 0.0  # Std. deviation of noise added to bottleneck.
    density_bias: float = -1.  # Shift added to raw densities pre-activation.
    density_noise: float = 0.  # Standard deviation of noise added to raw density.
    rgb_premultiplier: float = 1.  # Premultiplier on RGB before activation.
    rgb_bias: float = 0.  # The shift added to raw colors pre-activation.
    rgb_padding: float = 0.001  # Padding added to the RGB outputs.
    enable_pred_normals: bool = False  # If True compute predicted normals.
    disable_density_normals: bool = False  # If True don't compute normals.
    disable_rgb: bool = False  # If True don't output RGB.
    warp_fn = 'contract'
    num_glo_features: int = 0  # GLO vector length, disabled if 0.
    num_glo_embeddings: int = 1000  # Upper bound on max number of train images.
    scale_featurization: bool = False
    grid_num_levels: int = 10
    grid_level_interval: int = 2
    grid_level_dim: int = 4
    grid_base_resolution: int = 16
    grid_disired_resolution: int = 8192
    grid_log2_hashmap_size: int = 21
    net_width_glo: int = 128  # The width of the second part of MLP.
    net_depth_glo: int = 2  # The width of the second part of MLP.
    enable_scatter: bool = False # Enable the volume rendering of scatter media.
    enable_absorb: bool = False # Enable the absorbing field.
    enable_spatial_media: bool = False # Use constant media field.
    consistent_attenuation: bool = False # Using same attenuation and backscattering parameters (for haze).
    enable_downwell_depth: bool = False # Using the vertical attenuation model by predicting media depth.
    media_bias: float = -1 # Shift added to raw water densities pre-activation.

    def __init__(self, **kwargs):
        super().__init__()
        set_kwargs(self, kwargs)
        # Make sure that normals are computed if reflection direction is used.
        if self.use_reflections and not (self.enable_pred_normals or
                                         not self.disable_density_normals):
            raise ValueError('Normals must be computed for reflection directions.')

        # Precompute and define viewdir or refdir encoding function.
        if self.use_directional_enc:
            self.dir_enc_fn = ref_utils.generate_ide_fn(self.deg_view)
            dim_dir_enc = self.dir_enc_fn(torch.zeros(1, 3), torch.zeros(1, 1)).shape[-1]
        else:
            def dir_enc_fn(direction, _):
                return coord.pos_enc(
                    direction, min_deg=0, max_deg=self.deg_view, append_identity=True)

            self.dir_enc_fn = dir_enc_fn
            dim_dir_enc = self.dir_enc_fn(torch.zeros(1, 3), None).shape[-1]
        
        self.grid_num_levels = int(
            np.log(self.grid_disired_resolution / self.grid_base_resolution) / np.log(self.grid_level_interval)) + 1
        self.encoder = GridEncoder(input_dim=3,
                                   num_levels=self.grid_num_levels,
                                   level_dim=self.grid_level_dim,
                                   base_resolution=self.grid_base_resolution,
                                   desired_resolution=self.grid_disired_resolution,
                                   log2_hashmap_size=self.grid_log2_hashmap_size,
                                   gridtype='hash',
                                   align_corners=False)
        
        last_dim = self.encoder.output_dim
            
        self.density_layer = nn.Sequential(nn.Linear(last_dim, 64),
                                           nn.ReLU(),
                                           nn.Linear(64,
                                                     1 if self.disable_rgb else self.bottleneck_width))  # Hardcoded to a single channel.
        last_dim = 1 if self.disable_rgb else self.bottleneck_width

        if not self.disable_rgb:
            # Output of the first part of MLP.
            dim_intput_mlp = self.bottleneck_width
            dim_intput_mlp += dim_dir_enc

            if self.num_glo_features > 0:
                last_dim_glo = self.num_glo_features
                for i in range(self.net_depth_glo - 1):
                    self.register_module(f"lin_glo_{i}", nn.Linear(last_dim_glo, self.net_width_glo))
                    last_dim_glo = self.net_width_glo
                self.register_module(f"lin_glo_{self.net_depth_glo - 1}",
                                     nn.Linear(last_dim_glo, self.bottleneck_width * 2))

            last_dim_rgb = dim_intput_mlp
            for i in range(self.net_depth_viewdirs):
                lin = nn.Linear(last_dim_rgb, self.net_width_viewdirs)
                torch.nn.init.kaiming_uniform_(lin.weight)
                self.register_module(f"lin_second_stage_{i}", lin)
                last_dim_rgb = self.net_width_viewdirs
                if i == self.skip_layer_dir:
                    last_dim_rgb += dim_intput_mlp
            self.rgb_layer = nn.Linear(last_dim_rgb, self.num_rgb_channels)
            
            if self.enable_scatter or self.enable_absorb:
                if not self.enable_spatial_media:
                    dim_intput_mlp = dim_dir_enc
                else:
                    self.media_encoder = GridEncoder(input_dim=3,
                                                     num_levels=6,
                                                     level_dim=4,
                                                     base_resolution=16,
                                                     desired_resolution=512,
                                                     log2_hashmap_size=self.grid_log2_hashmap_size,
                                                     gridtype='hash',
                                                     align_corners=False)
                
                    self.media_output_layer = nn.Sequential(nn.Linear(self.media_encoder.output_dim, 64),
                                                            nn.ReLU(),
                                                            nn.Linear(64, self.bottleneck_width))
                
                last_dim_media = dim_intput_mlp
                for i in range(self.net_depth_viewdirs):
                    lin = nn.Linear(last_dim_media, self.net_width_viewdirs)
                    torch.nn.init.kaiming_uniform_(lin.weight)
                    self.register_module(f"lin_media_stage_{i}", lin)
                    last_dim_media = self.net_width_viewdirs
                    if i == self.skip_layer_dir:
                        last_dim_media += dim_intput_mlp
                
                if self.enable_scatter:
                    if self.consistent_attenuation:
                        self.sigma_bs_layer = nn.Linear(last_dim_media, 1)
                    else:
                        self.sigma_atten_layer = nn.Linear(last_dim_media, self.num_rgb_channels)
                        self.sigma_bs_layer = nn.Linear(last_dim_media, self.num_rgb_channels)
                    
                    if self.enable_downwell_depth:    
                        self.media_depth_layer = nn.Linear(last_dim_media, 1)
                        self.light_source = nn.Parameter(torch.zeros(3))
                    else:
                        self.media_rgb_layer = nn.Linear(last_dim_media, self.num_rgb_channels)
                        
                if self.enable_absorb:
                    self.sigma_absorb_layer = nn.Linear(last_dim_media, 1)

    def predict_density(self, encoder, output_layer, means, stds, rand=False, no_warp=False):
        """Helper function to output density."""
        # Encode input positions
        if self.warp_fn is not None and stds is not None and not no_warp:
            means, stds = coord.track_linearize(self.warp_fn, means, stds)
            # contract [-2, 2] to [-1, 1]
            bound = 2
            means = means / bound
            stds = stds / bound
        features = encoder(means, bound=1).unflatten(-1, (encoder.num_levels, -1))
        
        if stds is not None:
            weights = torch.erf(1 / torch.sqrt(8 * stds[..., None] ** 2 * encoder.grid_sizes ** 2))
            features = (features * weights[..., None]).mean(dim=-3).flatten(-2, -1)
    
        x = output_layer(features)
        raw_density = x[..., 0]  # Hardcoded to a single channel.
        # Add noise to regularize the density predictions if needed.
        if rand and (self.density_noise > 0):
            raw_density += self.density_noise * torch.randn_like(raw_density)
        return raw_density, x, means.mean(dim=-2)

    def forward(self,
                rand,
                means, stds,
                viewdirs=None,
                imageplane=None,
                glo_vec=None,
                exposure=None,
                no_warp=False):
        """Evaluate the MLP.

    Args:
      rand: if random .
      means: [..., n, 3], coordinate means.
      stds: [..., n], coordinate stds.
      viewdirs: [..., 3], if not None, this variable will
        be part of the input to the second part of the MLP concatenated with the
        output vector of the first part of the MLP. If None, only the first part
        of the MLP will be used with input x. In the original paper, this
        variable is the view direction.
      imageplane:[batch, 2], xy image plane coordinates
        for each ray in the batch. Useful for image plane operations such as a
        learned vignette mapping.
      glo_vec: [..., num_glo_features], The GLO vector for each ray.
      exposure: [..., 1], exposure value (shutter_speed * ISO) for each ray.

    Returns:
      rgb: [..., num_rgb_channels].
      density: [...].
      normals: [..., 3], or None.
      normals_pred: [..., 3], or None.
      roughness: [..., 1], or None.
    """
        raw_density, x, means_contract = self.predict_density(self.encoder, self.density_layer,
                                                              means, stds, rand=rand, no_warp=no_warp)

        # Apply bias and activation to raw density
        density = F.softplus(raw_density + self.density_bias)

        # media_rgb = None
        sigma_atten = None
        sigma_bs = None
        sigma_absorb = None
        media_rgb = None
        media_depth = None
        ambient_light = None
        
        if self.disable_rgb:
            rgb = torch.zeros(density.shape + (3,), device=density.device)
        else:
            if viewdirs is not None:
                # Output of the first part of MLP.
                if self.bottleneck_width > 0:
                    bottleneck = x
                    # Add bottleneck noise.
                    if rand and (self.bottleneck_noise > 0):
                        bottleneck += self.bottleneck_noise * torch.randn_like(bottleneck)

                    # Append GLO vector if used.
                    if glo_vec is not None:
                        for i in range(self.net_depth_glo):
                            glo_vec = self.get_submodule(f"lin_glo_{i}")(glo_vec)
                            if i != self.net_depth_glo - 1:
                                glo_vec = F.relu(glo_vec)
                        glo_vec = torch.broadcast_to(glo_vec[..., None, :],
                                                     bottleneck.shape[:-1] + glo_vec.shape[-1:])
                        scale, shift = glo_vec.chunk(2, dim=-1)
                        bottleneck = bottleneck * torch.exp(scale) + shift

                    x = [bottleneck]
                else:
                    x = []

                # Encode view directions.
                dir_enc = self.dir_enc_fn(viewdirs, None)
                concat_dir_enc = torch.broadcast_to(
                    dir_enc[..., None, :],
                    bottleneck.shape[:-1] + (dir_enc.shape[-1],))

                # Append view (or reflection) direction encoding to bottleneck vector.
                x.append(concat_dir_enc)

                # Concatenate bottleneck, directional encoding, and GLO.
                x = torch.cat(x, dim=-1)
                # Output of the second part of MLP.
                mlp_inputs = x
                for i in range(self.net_depth_viewdirs):
                    x = self.get_submodule(f"lin_second_stage_{i}")(x)
                    x = F.relu(x)
                    if i == self.skip_layer_dir:
                        x = torch.cat([x, mlp_inputs], dim=-1)
            # If using diffuse/specular colors, then `rgb` is treated as linear
            # specular color. Otherwise it's treated as the color itself.
            rgb = torch.sigmoid(self.rgb_premultiplier *
                                self.rgb_layer(x) +
                                self.rgb_bias)

            # Apply padding, mapping color to [-rgb_padding, 1+rgb_padding].
            rgb = rgb * (1 + 2 * self.rgb_padding) - self.rgb_padding
            
            if self.enable_scatter or self.enable_absorb:
                if self.enable_spatial_media:
                    _, x, _ = self.predict_density(self.media_encoder, self.media_output_layer,
                                                means, stds, rand=rand, no_warp=no_warp)
                    mlp_inputs = [x, concat_dir_enc]
                else:
                    mlp_inputs = [concat_dir_enc]
                    
                mlp_inputs = torch.cat(mlp_inputs, dim=-1)
                x = mlp_inputs
                for i in range(self.net_depth_viewdirs):
                    x = self.get_submodule(f"lin_media_stage_{i}")(x)
                    x = F.relu(x)
                    if i == self.skip_layer_dir:
                        x = torch.cat([x, mlp_inputs], dim=-1)
                
                if self.enable_scatter:
                    # x = x.mean(dim=-2, keepdim=True)
                    sigma_bs = F.softplus(self.sigma_bs_layer(x) + self.media_bias)
                    sigma_atten = F.softplus(self.sigma_atten_layer(x) + self.media_bias)

                    # # Whether use same attenuation parameters.
                    # if not self.consistent_attenuation:
                    #     sigma_atten = F.softplus(self.sigma_atten_layer(x) + self.media_bias)
                    # else:
                    #     sigma_atten = sigma_bs
                    
                    # Whether use the vertical depth model.
                    if self.enable_downwell_depth:
                        media_depth = F.softplus(self.media_depth_layer(x))
                        ambient_light = torch.sigmoid(self.light_source) / 3
                        media_rgb = ambient_light * torch.exp(-sigma_bs.detach() * media_depth)
                    else:
                        media_rgb = torch.sigmoid(self.rgb_premultiplier * self.media_rgb_layer(x) + self.rgb_bias)
                        
                if self.enable_absorb:
                    sigma_absorb = F.softplus(self.sigma_absorb_layer(x))
   
        return dict(
            coord=means_contract,
            density=density,
            rgb=rgb,
            media_rgb=media_rgb,
            media_depth=media_depth,
            ambient_light=ambient_light,
            sigma_atten=sigma_atten,
            sigma_bs=sigma_bs,
            sigma_absorb=sigma_absorb
        )


@gin.configurable
class NerfMLP(MLP):
    pass


@gin.configurable
class PropMLP(MLP):
    pass


@torch.no_grad()
def render_image(model,
                 accelerator: accelerate.Accelerator,
                 batch,
                 rand,
                 train_frac,
                 config,
                 verbose=True,
                 return_weights=False,
                 return_ray_dist=False):
    """Render all the pixels of an image (in test mode).

  Args:
    render_fn: function, jit-ed render function mapping (rand, batch) -> pytree.
    accelerator: used for DDP.
    batch: a `Rays` pytree, the rays to be rendered.
    rand: if random
    config: A Config class.

  Returns:
    rgb: rendered color image.
    disp: rendered disparity image.
    acc: rendered accumulated weights per pixel.
  """
    model.eval()

    height, width = batch['origins'].shape[:2]
    num_rays = height * width
    batch = {k: v.reshape((num_rays, -1)) for k, v in batch.items() if v is not None}

    global_rank = accelerator.process_index
    chunks = []
    idx0s = tqdm(range(0, num_rays, config.render_chunk_size),
                 desc="Rendering chunk", leave=False,
                 disable=not (accelerator.is_main_process and verbose))

    for i_chunk, idx0 in enumerate(idx0s):
        chunk_batch = tree_map(lambda r: r[idx0:idx0 + config.render_chunk_size], batch)
        actual_chunk_size = chunk_batch['origins'].shape[0]
        rays_remaining = actual_chunk_size % accelerator.num_processes
        if rays_remaining != 0:
            padding = accelerator.num_processes - rays_remaining
            chunk_batch = tree_map(lambda v: torch.cat([v, torch.zeros_like(v[-padding:])], dim=0), chunk_batch)
        else:
            padding = 0
        # After padding the number of chunk_rays is always divisible by host_count.
        rays_per_host = chunk_batch['origins'].shape[0] // accelerator.num_processes
        start, stop = global_rank * rays_per_host, (global_rank + 1) * rays_per_host
        chunk_batch = tree_map(lambda r: r[start:stop], chunk_batch)

        with accelerator.autocast():
            chunk_renderings, ray_history = model(rand,
                                                  chunk_batch,
                                                  train_frac=train_frac,
                                                  compute_extras=True,
                                                  zero_glo=True)

        gather = lambda v: accelerator.gather(v.contiguous())[:-padding] \
            if padding > 0 else accelerator.gather(v.contiguous())
        # Unshard the renderings.
        chunk_renderings = tree_map(gather, chunk_renderings)

        # Gather the final pass for 2D buffers and all passes for ray bundles.
        chunk_rendering = chunk_renderings[-1]
        for k in chunk_renderings[0]:
            if k.startswith('ray_'):
                chunk_rendering[k] = [r[k] for r in chunk_renderings]

        if return_weights:
            chunk_rendering['weights'] = gather(ray_history[-1]['weights'])
            chunk_rendering['coord'] = gather(ray_history[-1]['coord'])
            
        if return_ray_dist:
            chunk_rendering['sdist'] = gather(ray_history[-1]['sdist'])
            chunk_rendering['tdist'] = gather(ray_history[-1]['tdist'])
            
        chunk_rendering = tree_map(lambda x: x.detach().cpu(), chunk_rendering)
        chunks.append(chunk_rendering)

    # Concatenate all chunks within each leaf of a single pytree.
    rendering = {}
    for k in chunks[0].keys():
        if isinstance(chunks[0][k], list):
            rendering[k] = []
            for i in range(len(chunks[0][k])):
                rendering[k].append(torch.cat([item[k][i] for item in chunks]))
        else:
            rendering[k] = torch.cat([item[k] for item in chunks])

    for k, z in rendering.items():
        if not k.startswith('ray_'):
            # Reshape 2D buffers into original image shape.
            rendering[k] = z.reshape((height, width) + z.shape[1:])

    # After all of the ray bundles have been concatenated together, extract a
    # new random bundle (deterministically) from the concatenation that is the
    # same size as one of the individual bundles.
    keys = [k for k in rendering if k.startswith('ray_')]
    if keys:
        num_rays = rendering[keys[0]][0].shape[0]
        ray_idx = torch.randperm(num_rays)
        ray_idx = ray_idx[:config.vis_num_rays]
        for k in keys:
            rendering[k] = [r[ray_idx] for r in rendering[k]]
    model.train()
    return rendering
