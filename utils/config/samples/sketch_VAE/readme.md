
* Step 1: set `data_root` in `datadet/ali3dfront.yaml`, which points to the directory of processed `*.npz` files (TSDF).
* Step 2: set `latent_dir` as the latent encoding of initial TSDF data from PatchVQGAN, and the `latent_scale` as the reciprocal of latents STD.
* Step 3: set `ckpt` variable in `training` or `testing` section of `root_config.yaml` to load the checkpoint for training or inference. 