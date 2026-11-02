# Third-Party Source Notices

## SAM 2

SAM2 architecture, propagation, and portions of the training implementation are adapted from [facebookresearch/sam2](https://github.com/facebookresearch/sam2).

Copyright (c) Meta Platforms, Inc. and affiliates. The Apache License 2.0 text is included in [licenses/SAM2.txt](licenses/SAM2.txt). Original file-level attribution is retained. Spatial-SAM modifications include SDF memory, volume input, sparse-slice training, and removal of unused upstream entry points.

## PyTorch 3D U-Net

The local network in `spatialsam/models/sdf_unet.py` is adapted from [wolny/pytorch-3dunet](https://github.com/wolny/pytorch-3dunet). It implements the SDF regression architecture inside this repository; importing or installing `pytorch3dunet` is not required. The network retains layer names compatible with trained checkpoints.

Copyright (c) 2018 Adrian Wolny. The MIT license text is included in [licenses/PyTorch3DUNet.txt](licenses/PyTorch3DUNet.txt). This notice records the origin of retained source code, independently of Python package dependencies.
