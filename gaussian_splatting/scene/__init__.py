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

import pickle
import os
import random
import json
from pathlib import Path
from typing import List

import numpy as np
import torch

from camera_utils import UniformCameraPoseSampler
from scene.cameras import CustomCam
from utils.run_dir import get_pkl_and_w
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON


class Scene:
    gaussians: GaussianModel

    def __init__(
        self, args: ModelParams, gaussians: GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]
    ):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, "rb") as src_file, open(
                os.path.join(self.model_path, "input.ply"), "wb"
            ) as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), "w") as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(
                scene_info.train_cameras, resolution_scale, args
            )
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(
                scene_info.test_cameras, resolution_scale, args
            )

        if self.loaded_iter:
            self.gaussians.load_ply(
                os.path.join(self.model_path, "point_cloud", "iteration_" + str(self.loaded_iter), "point_cloud.ply")
            )
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]


class SceneEG3D:
    gaussians: GaussianModel

    def __init__(self, eg3d_model, gaussians: GaussianModel, w_path=None, ply_file=None):
        self.last_pkl = None
        self.gaussians = gaussians
        self.eg3d_model = eg3d_model.to("cuda")

        self.cameras_extent = 1
        self.radius = 2.7
        self.camera_lookat_point = torch.tensor([0, 0, 0], device="cuda")
        self.z = torch.randn(1, 512).to(device="cuda").float()

        self.w_path = w_path

        fov_deg = 17
        fov = fov_deg / 360 * 2 * np.pi
        extrinsic = self.get_random_extrinsic()
        w = None
        if w_path is not None:
            c = self.eg3d_model.eg3d_cam(extrinsics=extrinsic, fov=fov)
            w = self.get_w(self.w_path, c)
            self.z = None
        _, _ = self.eg3d_model(
            z=self.z, w=w, extrinsic_eg3d=extrinsic, extrinsic_gaus=extrinsic, fov_rad_eg3d=fov, fov_rad_gaus=fov
        )

        if ply_file is not None:
            self.gaussians.load_ply(ply_file)
        else:
            pos = self.eg3d_model._xyz.detach().cpu().numpy()
            colors = self.eg3d_model.gaussian_model._features_dc.detach().cpu().numpy()[:, 0, :]
            opacity = self.eg3d_model.gaussian_model.get_opacity.detach().cpu().numpy()
            keep_filter = (opacity > 0.01).squeeze(1)
            self.gaussians.create_from_pos_col(positions=pos[keep_filter], colors=colors[keep_filter])

    def save(self, iteration, path):
        point_cloud_path = os.path.join(path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def save_model(self, iteration, path):
        snapshot_data = dict(gaussian_decoder=self.eg3d_model)
        snapshot_pkl = os.path.join(path, f"network-snapshot-{iteration:06d}.pkl")
        print("Saving snapshot to", snapshot_pkl)
        with open(snapshot_pkl, "wb") as f:
            pickle.dump(snapshot_data, f)

    def get_random_extrinsic(self, horizontal_stddev=np.pi * 0.5, vertical_stddev=np.pi * 0.5):
        return UniformCameraPoseSampler.sample(
            horizontal_stddev=horizontal_stddev, vertical_stddev=vertical_stddev, radius=self.radius, device="cuda"
        )

    def get_camera_and_target(self, fov_deg=17, extrinsic=None, size=512, xyz=None):
        fov = fov_deg / 360 * 2 * np.pi

        if extrinsic is None:
            extrinsic = self.get_random_extrinsic()
        viewpoint = CustomCam(size=size, fov=fov, extr=extrinsic[0])

        w = None
        if self.w_path is not None:
            self.z = None
            c = self.eg3d_model.eg3d_cam(extrinsics=extrinsic, fov=fov)
            w = self.get_w(self.w_path, c)

        #w = torch.tile(self.eg3d_model.triplane_generator.backbone.mapping.w_avg[None, None, :], [1, 14, 1])
        target_image, decoded_features = self.eg3d_model(
            self.z,
            w=w,
            extrinsic_eg3d=extrinsic,
            extrinsic_gaus=extrinsic,
            fov_rad_eg3d=fov,
            fov_rad_gaus=fov,
            xyz=xyz,
            only_render_eg3d=True,
            truncation=1.0
        )
        return viewpoint, target_image, decoded_features

    def get_w(self, pkl, c):
        if pkl != self.last_pkl:
            self.last_pkl = pkl
            _, w_path = get_pkl_and_w(str(Path(pkl).parent))
            self.checkpoint = np.load(w_path)
            self.use_interpolate = "ws" in self.checkpoint.keys()
            if self.use_interpolate:
                self.ws = [torch.tensor(w_).to("cuda") for w_ in self.checkpoint["ws"]]
                self.cs = [torch.tensor(c_).to("cuda") for c_ in self.checkpoint["cs"]]

        if self.use_interpolate:
            return interpolate_w_by_cam(self.ws, self.cs, c, verbose=False).to("cuda")
        else:
            return torch.tensor(self.checkpoint["w"]).to("cuda")


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
