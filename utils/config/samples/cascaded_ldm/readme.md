
* Step 1: set `data_root` in `datadet/ali3dfront.yaml`, which points to the directory of processed `*.npz` files (TSDF).
* Step 2: set `latent_dir` as the latent encoding of initial TSDF data from PatchVQGAN, and the `latent_scale` as the reciprocal of latents STD.
* Step 3: set `level` in `model/pyramid_occ_denoiser.yaml` to set the training stage of the cascaded diffusion; set `use_sketch_condition` to use conditional / unconditional diffusion in the 1st stage; set `sketch_embedder/ckpt` as the checkpoint of `SketchVAE` model if `use_sketch_condition=True`; set `first_stage_model/ckpt` as the checkpoint of the PatchVQGAN model.
* [For inference] set `unet_model/LEVEL/ckpt` as the checkpoints of unet models of different diffusion levels.
* [For Training] set `ckpt` variable in `training`  section of `root_config.yaml` for continuous training. 