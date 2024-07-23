import pickle
import os
from pathlib import Path
from typing import List
import numpy as np
import torch

from camera_utils import UniformCameraPoseSampler
from scene.cameras import CustomCam
from utils.run_dir import get_pkl_and_w
from scene.gaussian_model import GaussianModel


class SceneEG3D:
    gaussians: GaussianModel

    def __init__(self, eg3d_model, gaussians: GaussianModel, w_path):
        self.last_pkl = None
        self.gaussians = gaussians
        self.eg3d_model = eg3d_model

        self.cameras_extent = 1
        self.camera_lookat_point = torch.tensor([0, 0, 0], device="cuda")
        self.w_path = w_path
        self.init_gaussians()

    def init_gaussians(self):
        fov_deg = 17
        fov = fov_deg / 360 * 2 * np.pi
        extrinsic = UniformCameraPoseSampler.sample(horizontal_stddev=0, vertical_stddev=0, radius=2.7, device="cuda")
        c = self.eg3d_model.eg3d_cam(extrinsic, fov)
        w = self.get_w(c)
        pos, colors, opacity = self.eg3d_model.get_pos_color_opacity(w, extrinsic, fov, num_gaussians_per_axis=100)
        keep_filter = (opacity > 0.3).squeeze(1)
        self.gaussians.create_from_pos_col(positions=pos[keep_filter], colors=colors[keep_filter])

    def get_camera_and_target(self, fov_deg, extrinsic, size=512):
        fov = fov_deg / 360 * 2 * np.pi
        viewpoint = CustomCam(size=size, fov=fov, extr=extrinsic[0])
        c = self.eg3d_model.eg3d_cam(extrinsics=extrinsic, fov=fov)
        w = self.get_w(c)
        target_image, _planes = self.eg3d_model(w=w, extrinsic=extrinsic, fov=fov)
        return viewpoint, target_image

    def get_w(self, c):
        if self.w_path != self.last_pkl:
            self.last_pkl = self.w_path
            _, w_path = get_pkl_and_w(str(Path(self.w_path).parent))
            self.checkpoint = np.load(w_path)
            self.use_interpolate = "ws" in self.checkpoint.keys()
            if self.use_interpolate:
                self.ws = [torch.tensor(w_).to("cuda") for w_ in self.checkpoint["ws"]]
                self.cs = [torch.tensor(c_).to("cuda") for c_ in self.checkpoint["cs"]]

        if self.use_interpolate:
            return interpolate_w_by_cam(self.ws, self.cs, c, verbose=False).to("cuda")
        else:
            return torch.tensor(self.checkpoint["w"]).to("cuda")

    def save(self, iteration, path):
        point_cloud_path = os.path.join(path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def save_model(self, iteration, path):
        snapshot_data = dict(gaussian_decoder=self.eg3d_model)
        snapshot_pkl = os.path.join(path, f"network-snapshot-{iteration:06d}.pkl")
        print("Saving snapshot to", snapshot_pkl)
        with open(snapshot_pkl, "wb") as f:
            pickle.dump(snapshot_data, f)

def interpolate_w_by_cam(ws: List[torch.tensor], cs: List[torch.tensor], c: torch.tensor, max_angle=np.pi, verbose=False):
    device = ws[0].device
    angle = torch.tensor([CamItem(c).xz_angle().to(device)])
    angle = torch.maximum(angle, torch.tensor(np.pi / 2 - max_angle))
    angle = torch.minimum(angle, torch.tensor(np.pi / 2 + max_angle))
    # angle = torch.clip(angle, np.pi / 2 - max_angle, np.pi / 2 + max_angle)
    cs = torch.tensor([CamItem(cs[i]).xz_angle().to(device) for i in range(len(cs))])

    if angle >= torch.max(cs):
        return ws[-1]

    if angle <= torch.min(cs):
        return ws[0]

    cs_diff = torch.abs(cs - angle)
    closest_index, second_closest_index = torch.argsort(cs_diff)[:2]
    index_left = torch.minimum(closest_index, second_closest_index)
    index_right = torch.maximum(closest_index, second_closest_index)

    total_dist = torch.abs(cs[index_left] - cs[index_right])
    dist_1 = torch.abs(cs[index_left] - angle)
    mag = torch.clip(dist_1 / total_dist, 0, 1).to(device)
    w_int = ws[index_left] * (1 - mag) + ws[index_right] * mag
    if verbose:
        print(f"w{index_left} * {(1 - mag)} + w{index_right} * {mag}")
    return w_int


class CamItem:
    def __init__(self, c: torch.tensor):
        self.c = c

    def extrinsic(self):
        return self.c[0, :16].reshape(4, 4)

    def intrinsic(self):
        return self.c[0, 16:].reshape(3, 3)

    def rotation(self):
        return self.extrinsic()[:-1, :-1]

    def xyz(self, as_numpy=False):
        x = self.extrinsic()[0, -1]
        y = self.extrinsic()[1, -1]
        z = self.extrinsic()[2, -1]
        if as_numpy:
            x = x.cpu().detach().numpy()
            y = y.cpu().detach().numpy()
            z = z.cpu().detach().numpy()
        return [x, y, z]

    def xz_angle(self):
        x, y, z = self.xyz()
        angle = torch.arctan2(z, x)
        return angle

    def direction(self):
        direction = torch.matmul(self.rotation().to("cpu"), torch.tensor([[0.0], [0.0], [-1.0]]).to("cpu"))
        mag = torch.sqrt(direction[0, 0] ** 2 + direction[1, 0] ** 2 + direction[2, 0] ** 2)
        return direction / mag
