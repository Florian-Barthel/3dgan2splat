import copy
import numpy as np
import math
import torch
from torch import nn
import dnnlib
import legacy
import sys
import torch.nn.functional as F

from eg3d_utils.triplane_utils import project_onto_planes, generate_planes
from torch_utils import persistence
from training.superresolution import TriplaneSuperresolutionHybrid2X

sys.path.append("./")
sys.path.append("../../gaussian-splatting")

from gaussian_renderer import render_simple
from scene import GaussianModel
from scene.cameras import CustomCam
from training.triplane import TriPlaneGenerator
from gaussian_decoder import decoder
from utils.sh_utils import RGB2SH


@persistence.persistent_class
class GaussianTriplaneDecoder(nn.Module):
    def __init__(
            self,
            num_gaussians_per_axis: int,
            triplane_generator_ckp: str,
            num_channels_per_plane: int = 32,
            learned_feature_dc_weight: float = 0.1,
            learned_opacity_weight: float = 0.1
    ):
        super().__init__()
        self.num_gaussians_per_axis = num_gaussians_per_axis
        self.plane_axes = generate_planes()

        # SPLIT DECODER INTO COLOR/OPACITY AND ROTATION/SCALE + ADD XYZ OFFSET
        self.color_opacity_decoder    = decoder.OSGDecoderSingle(num_channels_per_plane, out_features=4, hidden_dim=1)
        self.scaling_rotation_decoder = decoder.OSGDecoderSingle(num_channels_per_plane, out_features=10, hidden_dim=1)

        self.gaussian_model = GaussianModel(sh_degree=3)
        self._xyz = None
        # self._init_xyz()
        self._init_xyz_random()

        self.learned_feature_dc_weight = learned_feature_dc_weight
        self.learned_opacity_weight = learned_opacity_weight

        # define range for scale dependent of the number of gaussians
        grid_distance = 1 / num_gaussians_per_axis
        self.min_scale = self.gaussian_model.scaling_inverse_activation(torch.tensor(0.05 * grid_distance, device="cuda"))
        max_scale = self.gaussian_model.scaling_inverse_activation(torch.tensor(1, device="cuda"))
        self.scale_range = max_scale - self.min_scale

        print(f"Gaussian min_scale: {self.min_scale}, max_scale:{max_scale}")

        self.triplane_generator = None
        self.original_generator = None
        self.sr_res = 512
        self.triplane_superres = TriplaneSuperresolutionHybrid2X(channels=96, img_resolution=self.sr_res)
        self.setup_triplane_generator(triplane_generator_ckp)

        self.last_w = None

    def setup_triplane_generator(self, triplane_generator_ckp):
        print('Loading networks from "%s"...' % triplane_generator_ckp)
        with dnnlib.util.open_url(triplane_generator_ckp) as fp:
            network_data = legacy.load_network_pkl(fp)
            self.original_generator = network_data['G_ema'].requires_grad_(False)

        self.original_generator.rendering_kwargs["ray_start"] = 2.35
        self.original_generator = self.original_generator.eval().requires_grad_(False)

        self.triplane_generator = TriPlaneGenerator(**self.original_generator.init_kwargs).eval()
        self.triplane_generator.backbone = copy.deepcopy(self.original_generator.backbone)
        self.triplane_generator.backbone.mapping = copy.deepcopy(self.original_generator.backbone.mapping)
        self.triplane_generator.decoder = copy.deepcopy(self.original_generator.decoder)
        self.triplane_generator.superresolution = copy.deepcopy(self.original_generator.superresolution)
        self.triplane_generator.neural_rendering_resolution = self.original_generator.neural_rendering_resolution
        self.triplane_generator.rendering_kwargs = copy.deepcopy(self.original_generator.rendering_kwargs)
        self.triplane_generator.eval().requires_grad_(False)
        self.triplane_generator.decoder.requires_grad_()

    def eg3d_cam(self, extrinsics, fov, size=1):
        focal_length = size / (2 * np.tan(fov / 2))
        intrinsics = torch.tensor([
            [focal_length, 0, 0.5],
            [0, focal_length, 0.5],
            [0, 0, 1]]
        ).to("cuda")
        return torch.cat([extrinsics.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

    def forward(self, w, extrinsic_eg3d, extrinsic_gaus, fov_rad_eg3d, fov_rad_gaus, xyz=None, only_render_eg3d=False, truncation=0.7):
        if xyz is not None:
            self._xyz = xyz

        eg3d_rendering, w = self.render_eg3d(w=w, c=self.eg3d_cam(extrinsic_eg3d, fov_rad_eg3d), truncation=truncation)
        target_image = (eg3d_rendering["image"][0] + 1) / 2
        triplane = eg3d_rendering["planes"].reshape(-1, 96, 256, 256)

        output_features = self.sample_from_planes(triplane)
        eg3d_features = self.original_generator.decoder(output_features, None)
        pretrained_opacity = self.sigma2opacity(eg3d_features["sigma"])
        pretrained_color = self.rgb2gaussiancolor(eg3d_features["rgb"])
        rot_scale_xyz = self.scaling_rotation_decoder(output_features)

        decoded_features = {
            "_features_dc": pretrained_color,
            "_opacity": pretrained_opacity,
            "_rotation": rot_scale_xyz[..., 0:4],
            "_scaling": torch.relu(rot_scale_xyz[..., 4:7]) * -1 - 5,
            "_xyz_offset": torch.tanh(rot_scale_xyz[0, :, 7:] * 0.1) * 0.1 * 0,
            "_xyz": self._xyz
        }

        if only_render_eg3d:
            return target_image, decoded_features

        self.set_gaussian_attributes(decoded_features)
        viewpoint = CustomCam(size=512, fov=fov_rad_gaus, extr=extrinsic_gaus[0])

        bg_color = torch.tensor([1, 1, 1], dtype=torch.float32).to("cuda")
        render_pkg = render_simple(viewpoint_camera=viewpoint, pc=self.gaussian_model, bg_color=bg_color)
        rendering, viewspace_points, visibility_filter = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"]
        rendering = rendering[0:3]

        self.update_filter = visibility_filter
        return rendering, target_image

    def sigma2opacity(self, sigma):
        sigma = F.softplus(sigma - 1)
        sigma = sigma * 1.0 / self.num_gaussians_per_axis
        alpha = 1 - torch.exp(-sigma)
        alpha = self.gaussian_model.inverse_opacity_activation(alpha)
        # alpha[torch.isneginf(alpha)] = -100
        # alpha[torch.isinf(alpha)] = 100
        return alpha

    def rgb2gaussiancolor(self, rgb):
        return rgb
        # return torch.clip(rgb[..., :3], 0, 1)
        return RGB2SH(rgb[..., :3])

    def render_eg3d(self, w=None, c=None):
        with torch.no_grad():
            if self.last_w is not None and torch.equal(w, self.last_w):
                render_object = self.triplane_generator.synthesis(w, c=c, noise_mode='const', use_cached_backbone=True)
            else:
                render_object = self.triplane_generator.synthesis(w, c=c, noise_mode='const', cache_backbone=True)
                self.last_w = w
        return render_object, self.last_w

    def set_gaussian_attributes(self, decoded_features):
        for key, value in decoded_features.items():
            if key == "_features_dc":
                feature_dc = value.permute(1, 0, 2)
                self.gaussian_model.set_color(feature_dc)
                self.gaussian_model._features_rest = torch.zeros((feature_dc.shape[0], 3, (3 + 1) ** 2)).float().cuda().permute(0, 2, 1)
            elif key == "_scaling":
                setattr(self.gaussian_model, key, value[0])
            elif key == "_opacity":
                setattr(self.gaussian_model, key, value[0])
            elif key == "_xyz":
                setattr(self.gaussian_model, key, value)
            elif key == "_rotation":
                setattr(self.gaussian_model, key, value[0])
            else:
                pass

    def sample_from_planes(self, plane_features, mode='nearest', padding_mode='zeros', box_warp=1):
        # box_warp = 1 -> bounding box of triplane spans [-0.5, -0.5, -0.5] to [0.5, 0.5, 0.5]
        assert padding_mode == 'zeros'
        N, n_planes, C, H, W = plane_features.shape
        M, _ = self._xyz.shape
        plane_features = plane_features.view(N * n_planes, C, H, W)
        coordinates = (2 / box_warp) * self._xyz.unsqueeze(0)  # TODO: add specific box bounds
        coordinates = torch.tile(coordinates, [N, 1, 1])
        projected_coordinates = project_onto_planes(self.plane_axes, coordinates).unsqueeze(1)
        output_features = torch.nn.functional.grid_sample(
            plane_features, projected_coordinates.float(), mode=mode,
            padding_mode=padding_mode, align_corners=False).permute(0, 3, 2, 1).reshape(N, n_planes, M, C)
        return output_features

    def _init_xyz_random(self):
        self._xyz = torch.rand([self.num_gaussians_per_axis ** 3, 3], device="cuda") - 0.5

        self.xyz_gradient_accum = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_dirs = torch.zeros((self._xyz.shape[0], 3), device="cuda")
        self.denom = torch.zeros((self._xyz.shape[0], 1), device="cuda")
