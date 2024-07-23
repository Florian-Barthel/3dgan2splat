import copy
import numpy as np
import torch
from torch import nn
import dnnlib
import legacy
import torch.nn.functional as F

from eg3d_utils.triplane_utils import project_onto_planes, generate_planes
from torch_utils import persistence
from training.triplane import TriPlaneGenerator


@persistence.persistent_class
class EG3DWithOpacity(nn.Module):
    def __init__(self, checkpoint: str):
        super().__init__()
        self.plane_axes = generate_planes()
        self.triplane_generator = None
        self.original_generator = None
        self.setup_triplane_generator(checkpoint)
        self.last_w = None

    def setup_triplane_generator(self, triplane_generator_ckp):
        print('Loading networks from "%s"...' % triplane_generator_ckp)
        with dnnlib.util.open_url(triplane_generator_ckp) as fp:
            network_data = legacy.load_network_pkl(fp)
            self.original_generator = network_data['G_ema'].requires_grad_(False).to("cuda")

        self.original_generator.rendering_kwargs["ray_start"] = 2.35
        self.original_generator = self.original_generator.eval().requires_grad_(False)

        # TODO: Find out why it is still not the same output
        self.triplane_generator = TriPlaneGenerator(**self.original_generator.init_kwargs).eval().to("cuda")
        self.triplane_generator.backbone = copy.deepcopy(self.original_generator.backbone)
        self.triplane_generator.backbone.mapping = copy.deepcopy(self.original_generator.backbone.mapping)
        self.triplane_generator.decoder = copy.deepcopy(self.original_generator.decoder)
        self.triplane_generator.superresolution = copy.deepcopy(self.original_generator.superresolution)
        self.triplane_generator.neural_rendering_resolution = self.original_generator.neural_rendering_resolution
        self.triplane_generator.rendering_kwargs = copy.deepcopy(self.original_generator.rendering_kwargs)
        self.triplane_generator.eval().requires_grad_(False)

    def eg3d_cam(self, extrinsics, fov, device="cuda"):
        focal_length = 1 / (2 * np.tan(fov / 2))
        intrinsics = torch.tensor([
            [focal_length, 0, 0.5],
            [0, focal_length, 0.5],
            [0, 0, 1]]
        ).to(device)
        return torch.cat([extrinsics.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

    def forward(self, w, extrinsic, fov):
        c = self.eg3d_cam(extrinsic, fov)
        with torch.no_grad():
            if self.last_w is not None and torch.equal(w, self.last_w):
                eg3d_rendering = self.original_generator.synthesis(w, c=c, noise_mode='const', use_cached_backbone=True)
            else:
                eg3d_rendering = self.original_generator.synthesis(w, c=c, noise_mode='const', cache_backbone=True)
                self.last_w = w
        target_image = (eg3d_rendering["image"][0] + 1) / 2
        return target_image

    def get_pos_color_opacity(self, w, extrinsic, fov, num_gaussians_per_axis):
        c = self.eg3d_cam(extrinsic, fov)
        eg3d_rendering = self.triplane_generator.synthesis(w, c=c, noise_mode='const', use_cached_backbone=True)
        planes = eg3d_rendering["planes"]
        triplane = planes.reshape(-1, 3, 32, 256, 256)
        xyz = torch.rand([num_gaussians_per_axis ** 3, 3], device="cuda") - 0.5
        output_features = self.sample_from_planes(triplane, xyz)
        eg3d_features = self.original_generator.decoder(output_features, None)
        pretrained_color = eg3d_features["rgb"][0, :, :3]
        pretrained_opacity = self.sigma2opacity(eg3d_features["sigma"], num_gaussians_per_axis)[0]
        return xyz, pretrained_color, pretrained_opacity

    def sigma2opacity(self, sigma, num_gaussians_per_axis):
        sigma = F.softplus(sigma - 1)
        sigma = sigma * 1.0 / num_gaussians_per_axis
        alpha = 1 - torch.exp(-sigma)
        return alpha

    def sample_from_planes(self, plane_features, xyz, mode='nearest', box_warp=1):
        # box_warp = 1 -> bounding box of triplane spans [-0.5, -0.5, -0.5] to [0.5, 0.5, 0.5]
        N, n_planes, C, H, W = plane_features.shape
        M, _ = xyz.shape
        plane_features = plane_features.view(N * n_planes, C, H, W)
        coordinates = (2 / box_warp) * xyz.unsqueeze(0)  # TODO: add specific box bounds
        coordinates = torch.tile(coordinates, [N, 1, 1])
        projected_coordinates = project_onto_planes(self.plane_axes, coordinates).unsqueeze(1)
        output_features = torch.nn.functional.grid_sample(
            plane_features, projected_coordinates.float(), mode=mode,
            padding_mode='zeros', align_corners=False).permute(0, 3, 2, 1).reshape(N, n_planes, M, C)
        return output_features
