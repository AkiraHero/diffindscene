
* Step 1: set `data_root` in `datadet/ali3dfront.yaml`, which points to the directory of processed `*.npz` files (TSDF).
* Step 2: enable the `# for training` section or the `# for test` section in `datadet/ali3dfront.yaml` for different data transform in training and inference.
* Step 3: set `ckpt` variable in `training` or `testing` section of `root_config.yaml` to load the checkpoint for training or inference. 