import subprocess
import os
import argparse


class Convert:
    def __init__(self, eg3d_dir):
        self.args = []
        self.args.append(f"--eg3d_dir={eg3d_dir}")
        self.python = "python"
        self.path_to_program = "train_eg3d_face.py"
        subprocess.run([self.python, self.path_to_program, *self.args], shell=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("gpu", type=int)
    args = parser.parse_args()
    gpu_index = args.gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_index}"

    dataset_folder = "/home/tmp/barthefl/projects/eg3d/eg3d/out_multi_w_efhq_v1"
    all_folders = sorted(os.listdir(dataset_folder))

    for i in range(0, len(all_folders), 4):
        folder = os.path.join(dataset_folder, all_folders[i + gpu_index])
        print(f"Running on {folder}")
        Convert(eg3d_dir=folder)
