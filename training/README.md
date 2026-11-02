# Training

See the [main README](../README.md#quick-start) for data preparation and the training command. `python -m spatialsam train` runs pseudo-label initialization and all training rounds.

## Configuration

The training settings are in [config/task/train.yaml](../config/task/train.yaml), with model and optimizer settings in [config/model/](../config/model/). Use `--rounds` to set the number of rounds, `--gpu` to select a GPU, and `--seed` to set the random seed.

Add `--dry-run` to check inputs and preview the stages. To change a configuration value, append `--set KEY=VALUE` to the command. Named options such as `--rounds` take precedence over `--set`.

## Input Normalization

Use uint8 or uint16 image volumes. Compute the training images' mean and standard deviation after dividing uint8 values by 255 or uint16 values by 65535. Pass these as `--mean` and `--std`, and use the same values for prediction.

## Volume Resolution

The default SAM volume resolution is `1024 × 1024 × 1024`; the SDF U-Net uses half that resolution along each axis. To use a different resolution, append the appropriate override:

| Command | Override |
| --- | --- |
| `train` | `--set 'task.shared.runtime.resolution=[D,H,W]'` |
| `predict` | `--set 'task.resolution=[D,H,W]'` |

Replace `D`, `H`, and `W` with dimensions compatible with U-Net downsampling. Run `--dry-run` after changing the training resolution: reducing the Z dimension can omit annotated slices. If validation reports an omitted slice, increase the Z resolution to retain it.

## Training Schedule

Each round trains the SDF U-Net, generates SDF memory, trains SAM2, and updates pseudo-labels for the next round.

| Round | SDF U-Net epochs | SAM2 epochs |
| --- | ---: | ---: |
| 1 | 30 | 50 |
| 2 onward | 10 | 10 |

Edit `task.pipeline.round_schedule` to change these defaults. Each model starts the next round from its previous weights with a new optimizer.

With `--annotations`, all slices listed in the JSON file are used. With `--labels`, slice selection is controlled by `task.pipeline.bootstrap` in the training configuration.

## Checkpoints and Resume

Each round saves checkpoints under `round_NNN/`:

| Path | Purpose |
| --- | --- |
| `sdf_train/checkpoints/stage.pt` | SDF U-Net weights for inference and the next stage |
| `sdf_train/checkpoints/checkpoint.pt` | SDF training state for resuming training |
| `sam_train/checkpoints/checkpoint.pt` | SAM2 weights and training state |

The SDF stage checkpoint is saved every 2,000 optimizer updates by default. Training-state checkpoints are saved at epoch boundaries. For prediction, select the SDF and SAM2 checkpoints from the same round.

To resume, rerun the original training command with the same output directory. Completed stages are skipped, and interrupted training resumes from the last saved epoch boundary. Changing inputs or settings reruns affected stages.
