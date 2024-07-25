import math
import os
import numpy as np
import torch
import lpips
import sys
from tqdm import tqdm

sys.path.append("./eg3d")
sys.path.append("./gaussian_splatting")

from camera_utils import UniformCameraPoseSampler
from gaussian_decoder.eg3d_with_opacity import EG3DWithOpacity
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render_simple
from scene import GaussianModel, SceneEG3D
from utils.general_utils import safe_state
from utils.run_dir import get_pkl_and_w
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def training(opt, pipe, saving_iterations, eg3d_dir):
    eg3d_dir = eg3d_dir.replace(os.sep, '/')
    white_background = False
    tb_writer, path = prepare_output_and_logger(eg3d_dir)
    gaussians = GaussianModel(3)
    lpips_loss = lpips.LPIPS(net="vgg").to("cuda")

    network_path, w_path = get_pkl_and_w(rundir=eg3d_dir, verbose=True)
    eg3d_model = EG3DWithOpacity(network_path)
    scene = SceneEG3D(eg3d_model=eg3d_model, gaussians=gaussians, w_path=w_path)
    # opt.densify_until_iter = 0 # TODO disable when overfitting the mean face

    gaussians.training_setup(opt)

    bg_color = [1, 1, 1]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    first_iter = 1
    for iteration in tqdm(range(first_iter, opt.iterations + 1)):
        gaussians.update_learning_rate(iteration)
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Render GAN
        fov_deg = np.random.rand() * 8 + 7
        lookat_position = torch.tensor([0, 0, np.random.rand() * 0.1], device="cuda")
        extrinsic = UniformCameraPoseSampler.sample(
            horizontal_stddev=math.pi/2 * 0.8,
            vertical_stddev=math.pi/2 * 0.6,
            radius=2.7,
            device="cuda",
            lookat_position=lookat_position
        )
        viewpoint_cam, gt_image = scene.get_camera_and_target(fov_deg=fov_deg, extrinsic=extrinsic)
        bg = torch.rand((3), device="cuda") if opt.random_background else background

        # Render Gaussian Splatting
        render_pkg = render_simple(viewpoint_cam, gaussians, bg_color=bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], \
        render_pkg["visibility_filter"], render_pkg["radii"]
        image = image[:3]

        # Loss
        gt_image = gt_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        perc = lpips_loss(in0=image, in1=gt_image, normalize=True)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + perc * 0.2
        loss.backward()

        with torch.no_grad():
            if iteration % 50 == 0:
                tb_writer.add_scalar("Num Gaussians", gaussians.get_xyz.shape[0], global_step=iteration)

            if iteration % 500 == 0 or iteration == 1:
                tb_writer.add_images("Compare", torch.concat([gt_image, image], dim=1)[None, ...],
                                     global_step=iteration)

            # Log and save
            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration, path)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                     radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

                if iteration % opt.opacity_reset_interval == 0 or (
                        white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)


def prepare_output_and_logger(eg3d_dir):
    out_path = os.path.join("./out/", eg3d_dir.split("/")[-1])

    run_nr = 0
    if os.path.exists(out_path):
        run_nr = len(os.listdir(out_path))
    model_path = os.path.join(out_path, str(run_nr))

    # Set up output folder
    print("Output folder: {}".format(model_path))
    os.makedirs(model_path, exist_ok=True)

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer, model_path


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--eg3d_dir', type=str)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[10_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[10_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet, seed=1)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        op.extract(args),
        pp.extract(args),
        args.save_iterations,
        eg3d_dir=args.eg3d_dir
    )

    # All done
    print("\nTraining complete.")
