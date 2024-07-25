# 3dgan2splat

This repository turns 3D heads, created by 3D-aware GANs into 3D Gaussian splatting objects.


The EG3D generator has to be stored in a `fintuned_generator.pkl` file and the latent vector has to be
stored in a `final_projected_w.npz` file, which contains a dictionary with the key "w" that holds the 
latent vector (1, 14, 512). Alternatively, multiple latent vectors for different camera views can be used.
In this case the final_projected_w.npz has to contain a dictionary with the keys "ws" and "cs" for the
latent vectors and camera objects respectively.

Run `train_eg3d_face.py --eg3d_dir=path/to/eg3d/finetune/run`.

The finetuned generator and the inverted latent code can be obtained with this repository:
https://github.com/Florian-Barthel/3d-multiview-inversion

The output will be stored in _./out_.